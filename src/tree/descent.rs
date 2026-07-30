use core::{mem::MaybeUninit, ptr::NonNull};

use crate::{
    DEFAULT_MAX_LEVELS, Inner, Key, Leaf,
    allocator::{DefaultAllocator, NodeAllocator, Reservation, Reserved},
};

use super::{BPlusTree, Node};

/// The recorded path of one root-to-leaf descent: for each inner level
/// `h`, `path[h]` holds the visited node and the child index routed
/// through. Slots outside the descended range stay uninitialized.
type TreePath<K, V, const M: usize, const H: usize = DEFAULT_MAX_LEVELS> =
    [MaybeUninit<(NonNull<Node<K, V, M>>, usize)>; H];

/// One recorded descent for a key: the tree it walked, the
/// [`path`](Descent::path) it recorded, the leaf it landed at, and the key's
/// slot there. The state shared by ([`Descent::commit_insert`],
/// [`Descent::commit_replace`], and [`Descent::commit_remove`]).
///
/// # Validity
///
/// A descent is a bundle of raw pointers into the tree. It is valid only
/// under the `&mut` borrow of the tree that [`BPlusTree::descend_into`]
/// derived it from, and only until the tree is next mutated — a `commit_*`
/// call ends it (they take `&mut self` purely to avoid moving the
/// recorded path; a committed descent MUST NOT be used again). Until then, the
/// tree may be reached through the descent's pointers ONLY.
pub(crate) struct Descent<
    K: Key,
    V,
    const M: usize,
    A: NodeAllocator<K, V, M> = DefaultAllocator<K, V, M>,
    const H: usize = DEFAULT_MAX_LEVELS,
> {
    /// The descended tree, as the raw handle every other pointer here
    /// descends from. The `commit_*` methods reach the tree's own fields
    /// through this: the scalars (`height`, `len`) via raw projections
    /// disjoint from every recorded pointer's pointee, the `allocator`
    /// (shared, never written) likewise, and the `root` slot only after
    /// the path's last use.
    tree: NonNull<BPlusTree<K, V, M, A, H>>,
    /// The recorded path (see [`TreePath`]); slots `1..=tree.height` are
    /// initialized. Private to this module: only the `commit_*` methods
    /// replay it.
    path: TreePath<K, V, M, H>,
    /// The leaf the descent landed at.
    pub(crate) leaf: NonNull<Leaf<K, V, M>>,
    /// `key`'s slot in that leaf, per [`Leaf::find_key`]: the match if
    /// `exact`, else the insertion point.
    pub(crate) partition: usize,
    /// True iff `partition` holds exactly the sought key.
    pub(crate) exact: bool,
}

// The `commit_*` methods: the back ends of `insert` and `remove`, run
// against a recorded descent. They live on `Descent` rather than on
// `BPlusTree` deliberately — taking `&mut self` on the tree at this
// boundary would freshly retag it and invalidate every pointer the
// descent recorded (see `Descent`'s validity rules). The tree's own
// fields are reached through the descent's raw handle instead: the
// scalars (`height`, `len`) via projections disjoint from every
// recorded pointer's pointee, and the `root` slot only after the
// path's last use.
impl<K: Key, V, const M: usize, A: NodeAllocator<K, V, M>, const H: usize> Descent<K, V, M, A, H> {
    /// Commit a replacement, consuming the descent.
    ///
    /// Safety:
    ///
    /// The caller must ensure that `self.exact` is set.
    #[inline]
    pub(crate) unsafe fn commit_replace(&mut self, val: V) -> V {
        debug_assert!(self.exact);
        // Present: replace the value in place — no structural
        // change, no len change.
        let mut leaf = self.leaf;
        // SAFETY: `exact` vouches that `partition` holds `key`.
        let slot = unsafe { leaf.as_mut() }.val_mut_unchecked(self.partition);
        core::mem::replace(slot, val)
    }

    /// The back half of insertion, for a key known ABSENT: insert
    /// `key`/`val` at the recorded slot, replay the path upward to
    /// house any splits, grow the root if one escapes it, and count the
    /// new pair in `len`.
    ///
    /// Returns the inserted value's slot — final by the time this
    /// returns: nodes never move on the heap, and the upward replay
    /// moves child HANDLES between inner nodes, never leaf contents. The
    /// pointer may be dereferenced for as long as the `&mut` borrow this
    /// descent lives under lasts and the tree is not otherwise touched.
    ///
    /// # Safety
    ///
    /// - This descent came from `BPlusTree::descend(&key)`, the `&mut`
    ///   borrow it was derived under is still live, and the tree has not
    ///   been mutated since.
    /// - `self.exact` is false: `key` is absent, and `self.partition`
    ///   is its insertion point in `self.leaf`.
    /// - `reservation` holds this descent's full allocation bill,
    ///   acquired from this tree's allocator (the wrapper asserts the
    ///   under-provisioned direction; slots from a foreign allocator
    ///   would be handed to this tree's teardown).
    /// - The commit ends the descent: `self` must not be used again
    ///   after this call.
    // `&mut self`, not `self`, and the path read through it in place —
    // although a commit logically consumes the descent, taking it by
    // value compiles to a 1 KiB move at every call boundary (and
    // destructuring `path` out of it to another). `always` because LLVM
    // declines a plain `#[inline]` hint at this size.
    #[inline(always)]
    pub(crate) unsafe fn commit_insert(
        &mut self,
        key: K,
        val: V,
        reservation: &mut Reservation<K, V, M>,
    ) -> NonNull<V> {
        let tree = self.tree.as_ptr();
        let partition = self.partition;
        debug_assert!(!self.exact, "commit_insert requires a vacant descent");

        // SAFETY: as the scalar-access note below — the allocator field is
        // disjoint from every recorded pointer's pointee, so this exclusive
        // reference aliases none of the node mutations below. It is also
        // the commit's only allocator borrow — every helper that allocates
        // or frees receives a reborrow of the wrapper over it — so its
        // exclusivity holds for its whole life.
        let alloc = unsafe { &mut (*tree).allocator };

        // The commit draws every slot from the caller's reservation:
        // the wrapper's `Exhaustion = Infallible` is what satisfies the
        // split helpers' bounds — allocation on this path cannot fail,
        // by type.
        let mut alloc = Reserved::new(reservation, alloc);
        let alloc = &mut alloc;

        // SAFETY: the descent's leaf is live and exclusively reachable
        // under the borrow the caller vouches for.
        let (val_ptr, split) = unsafe { self.leaf.as_mut() }.insert_at(partition, key, val, alloc);

        // If we have a split, we need to insert it into the parent node.
        // We do this by replaying the recorded path from the bottom up.

        let mut split = split.map(|new_child| {
            // SAFETY: `insert_at` hands back a valid, initialized right
            // leaf, and this is its only handle.
            (*unsafe { new_child.as_ref() }.first_key(), Node::from_leaf_ptr(new_child))
        });

        let mut height = 0;
        // SAFETY: for `(*tree).height`, and `len` below — the borrow the
        // caller vouches for keeps the tree alive and exclusive, and the
        // scalar fields are disjoint from every recorded pointer's
        // pointee, so these raw accesses never invalidate the path.
        while height < unsafe { (*tree).height } as usize
            && let Some((sep, new_child)) = split
        {
            // SAFETY: the descent initialized path slots
            // `1..=tree.height` with valid inner nodes.
            let (mut node, child_idx) = unsafe { self.path[height + 1].assume_init() };

            // SAFETY: node is owned by this tree. We have exclusive access to
            // this tree.
            let node = unsafe { node.as_mut().as_inner_mut() };
            split = node
                .insert_child(child_idx, sep, new_child, alloc)
                .map(|(sep, new_inner)| (sep, Node::from_inner_ptr(new_inner)));

            height += 1;
        }

        // If we still have a split, we need to insert it into the root.
        // If the root is full, we need to increase tree height. This is
        // the commit's only touch of the root slot, after the path's
        // last use.
        if let Some((sep, node)) = split {
            // SAFETY: the caller's `&mut` borrow keeps the tree alive and
            // exclusive. The old root moves into the new one exactly once
            // (read, then the slot is overwritten), and the scalar
            // `height` bump is disjoint from every recorded pointee.
            unsafe {
                let old_root = core::ptr::read(&raw const (*tree).root);
                let inner = Inner::from_pair(sep, old_root, node);
                (&raw mut (*tree).root).write(Node::from_inner(inner, alloc));
                (*tree).height += 1;
            }
        }

        // The key was absent (the caller vouches), so this is a new pair.
        // SAFETY: see the scalar-access note above.
        unsafe { (*tree).len += 1 };

        val_ptr
    }

    /// The back half of removal, for a key known PRESENT: remove the
    /// pair at the recorded slot, replay the path upward repairing
    /// whatever the removal left deficient, hoist a lone-child root, and
    /// discount the pair from `len`. Returns the removed pair (the key
    /// comes back out, not just the value).
    ///
    /// # Safety
    ///
    /// - This descent came from `BPlusTree::descend(key)`, the `&mut`
    ///   borrow it was derived under is still live, and the tree has not
    ///   been mutated since.
    /// - `self.exact` is true: `self.partition` holds exactly the
    ///   sought key in `self.leaf`.
    /// - The commit ends the descent: `self` must not be used again
    ///   after this call.
    // `&mut self`, not `self`, and the path read through it in place —
    // although a commit logically consumes the descent, taking it by
    // value compiles to a 1 KiB move at every call boundary (and
    // destructuring `path` out of it to another). `always` because LLVM
    // declines a plain `#[inline]` hint at this size.
    #[inline(always)]
    pub(crate) unsafe fn commit_remove(&mut self) -> (K, V) {
        let tree = self.tree.as_ptr();
        debug_assert!(self.exact, "commit_remove requires an occupied descent");

        // SAFETY: as the scalar-access note below — the allocator field is
        // disjoint from every recorded pointer's pointee, so this exclusive
        // reference aliases none of the node mutations below. It is also
        // the commit's only allocator borrow — every helper that allocates
        // or frees receives a reborrow of it — so its exclusivity holds
        // for its whole life.
        let alloc = unsafe { &mut (*tree).allocator };

        // SAFETY: the descent's leaf is live and exclusively reachable
        // under the borrow the caller vouches for.
        let leaf = unsafe { self.leaf.as_mut() };
        let pair = leaf.remove_at(self.partition);

        // Replay the recorded path upward: repair while the removal (or
        // the repair one level down — a merge shrinks the parent) left
        // the current level's node deficient. The loop stops at the
        // root, which is exempt. NB: a merge at the level below may have
        // freed `leaf`'s node; it is not touched past this read.
        let mut deficient = leaf.is_deficient();
        let mut height = 0;
        // SAFETY: for `(*tree).height` here and `root`/`len` below — the
        // borrow the caller vouches for keeps the tree alive and
        // exclusive; the scalar fields are disjoint from every recorded
        // pointer's pointee, and the root slot is touched only after the
        // path's last use.
        while height < unsafe { (*tree).height } as usize && deficient {
            // SAFETY: the descent initialized path slots
            // `1..=tree.height` with valid inner nodes; height is
            // propagated correctly.
            unsafe {
                let (mut parent, child_idx) = self.path[height + 1].assume_init();
                let parent = parent.as_mut();
                parent.as_inner_mut().rebalance(height as u8 + 1, child_idx, alloc);
                deficient = parent.is_deficient(height as u8 + 1);
            }
            height += 1;
        }

        // If the root is deficient, that's okay. If the root has only 1 child,
        // that child is the new root.
        //
        // SAFETY:
        // height invariant: `(*tree).height` is the height of the root's
        // subtree. The ptr reads are safe because the root slot is owned
        // data, read exactly once and overwritten with its replacement.
        unsafe {
            // if there's exactly 1 child of the root, decompose
            if (*tree).height > 0 && (*tree).root.len((*tree).height) == 1 {
                let new_root =
                    core::ptr::read(&raw const (*tree).root).into_inner(alloc).into_only_child();
                (&raw mut (*tree).root).write(new_root);
                (*tree).height -= 1;
            }
        }

        // The key was present (the caller vouches), so a pair left.
        // SAFETY: see the scalar-access note above.
        unsafe { (*tree).len -= 1 };

        pair
    }

    /// Count the necessary inner splits. If no leaf split is needed, this is
    /// 0. Otherwise it is the count of running non-0 heights that are full —
    /// plus one more when the run reaches the root, since a split escaping
    /// the top grows the tree by a fresh inner (the new root,
    /// [`commit_insert`]'s final allocation).
    ///
    /// This is [`commit_insert`]'s exact inner-allocation bill, computable
    /// before anything mutates. Callable only while the descent is valid
    /// (see the type's validity rules): fresh from `descend`, tree
    /// untouched since.
    ///
    /// [`commit_insert`]: Self::commit_insert
    pub(crate) fn count_inner_splits(&self) -> usize {
        debug_assert!(!self.exact, "the bill is for insertion; a replacement allocates nothing");

        // SAFETY: the descent's leaf is live and readable under the
        // borrow the descent lives within (the type's validity rules).
        if unsafe { self.leaf.as_ref() }.len() < M {
            // The pair fits in the leaf: no split cascade at all.
            return 0;
        }

        // SAFETY: a scalar read through the descent's tree handle,
        // disjoint from every recorded pointer's pointee — as the
        // commit methods' scalar-access note.
        let height = unsafe { (*self.tree.as_ptr()).height } as usize;

        // Climb the recorded path exactly as the commit's replay would.
        let mut count = 0;
        while count < height {
            // SAFETY: the descent initialized path slots
            // `1..=tree.height` with valid nodes, live and readable
            // under the same borrow; slots `1..` sit above the leaf
            // level, so the cast is an `Inner` by the depth-type
            // invariant.
            let inner = unsafe {
                let (node, _) = self.path[count + 1].assume_init();
                node.as_ref().as_inner()
            };

            if inner.len() < M {
                // This ancestor absorbs the split: the cascade stops.
                return count;
            }
            count += 1;
        }

        // The split escaped the top of the path: growing the tree
        // costs one more inner — the new root.
        count + 1
    }
}

impl<K: Key, V, const M: usize, A: NodeAllocator<K, V, M>, const H: usize>
    BPlusTree<K, V, M, A, H>
{
    /// Descend from the root to the leaf, determining the path by appling `f`
    /// to each inner node. Recording the path — each visited inner node and
    /// the child index routed through — in `path[h]` for every inner level
    /// `h`, and returning the leaf.
    #[inline]
    #[track_caller]
    fn descend_recording_with(
        &mut self,
        f: impl Fn(&Inner<K, V, M>) -> usize,
        path: &mut TreePath<K, V, M, H>,
    ) -> NonNull<Node<K, V, M>> {
        let mut height = self.height as usize;
        let mut node = NonNull::from_mut(&mut self.root);

        while height > 0 {
            // SAFETY: height is non-0, the pointer is valid. Root is always ok.
            let n = unsafe { node.as_mut().as_inner_mut() };

            let idx = f(n);

            path[height].write((node, idx));

            node = NonNull::from_mut(&mut n.children_mut()[idx]);

            height -= 1;
        }

        node
    }

    /// As [`Self::descend_recording_with`], but building the descent in
    /// caller-provided storage — the way in for the plain ops. Returning a
    /// [`Descent`] by value compiles to a 1 KiB copy per call (Rust guarantees
    /// no NRVO), which is pure loss for
    /// [`insert`](Self::insert)/[`remove`](Self::remove), which commit it on
    /// the spot.
    ///
    // `always` because LLVM declines a plain `#[inline]` hint at this
    // size, and out of line every caller pays call-boundary stack
    // traffic for the slot.
    #[inline(always)]
    #[track_caller]
    pub(crate) fn descend_into_with<'s>(
        &mut self,
        f: impl Fn(&Inner<K, V, M>) -> usize,
        slot: &'s mut MaybeUninit<Descent<K, V, M, A, H>>,
    ) -> &'s mut Descent<K, V, M, A, H> {
        let mut tree: NonNull<BPlusTree<K, V, M, A, H>> = NonNull::from_mut(self);

        // Initialize field by field, in place: writing a whole `Descent`
        // value into the slot would be exactly the copy this out-param
        // exists to avoid. `path` is `MaybeUninit` slots and needs no
        // writes.
        let ptr = slot.as_mut_ptr();
        // SAFETY: `ptr` is the caller's `MaybeUninit` slot — the field
        // projections are in bounds, and raw writes into uninitialized
        // storage need no prior validity.
        unsafe {
            (&raw mut (*ptr).tree).write(tree);
            (&raw mut (*ptr).leaf).write(NonNull::dangling());
            (&raw mut (*ptr).partition).write(0);
            (&raw mut (*ptr).exact).write(false);
        }
        // SAFETY: every always-initialized field was written above;
        // `path`'s slots are `MaybeUninit` and carry no validity
        // requirement.
        let descent = unsafe { slot.assume_init_mut() };

        // Re-derive the tree reference from the raw handle the descent
        // will carry, so every pointer recorded below descends from
        // `tree` — NOT from the `&mut self` at this function's boundary,
        // whose tag family dies at the next `&mut self` method call (see
        // the field docs on `Descent::tree`).
        // SAFETY: `tree` is `self`: live and exclusively borrowed.
        let mut node = unsafe { tree.as_mut() }.descend_recording_with(f, &mut descent.path);

        // SAFETY: `descend_recording` ran the descent to the bottom, so
        // the node in hand is the leaf (height invariant).
        let leaf = unsafe { node.as_mut().as_leaf_mut() };

        descent.leaf = NonNull::from_mut(leaf);

        descent
    }

    /// Descend into the tree by key, writing the path to caller provided
    /// storage.
    #[inline(always)]
    #[track_caller]
    pub(crate) fn descend_into<'s>(
        &mut self,
        key: &K,
        slot: &'s mut MaybeUninit<Descent<K, V, M, A, H>>,
    ) -> &'s mut Descent<K, V, M, A, H> {
        let descent = self.descend_into_with(|inner| inner.child_idx_for_key(key), slot);

        (descent.partition, descent.exact) = unsafe { descent.leaf.as_ref() }.probe(key);

        descent
    }

    /// Descend into the tree, always taking the first child, writing the path
    /// to caller provided storage.
    #[inline(always)]
    #[track_caller]
    pub(crate) fn descend_into_first<'s>(
        &mut self,
        slot: &'s mut MaybeUninit<Descent<K, V, M, A, H>>,
    ) -> &'s mut Descent<K, V, M, A, H> {
        let descent = self.descend_into_with(|_| 0, slot);
        descent.partition = 0;
        descent.exact = true;
        descent
    }

    /// Descend into the tree, always taking the last child, writing the path
    /// to caller provided storage.
    #[inline(always)]
    #[track_caller]
    pub(crate) fn descend_into_last<'s>(
        &mut self,
        slot: &'s mut MaybeUninit<Descent<K, V, M, A, H>>,
    ) -> &'s mut Descent<K, V, M, A, H> {
        let descent = self.descend_into_with(|inner| inner.len() - 1, slot);
        descent.partition = unsafe { descent.leaf.as_ref().len() - 1 };
        descent.exact = true;
        descent
    }
}
