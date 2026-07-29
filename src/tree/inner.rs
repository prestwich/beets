use core::mem::MaybeUninit;
use core::ptr::NonNull;

#[cfg(debug_assertions)]
use crate::NodeKind;
use crate::allocator::{NodeAllocator, SlotAllocator};
use crate::{Key, Node};

// TODO:
// - perf: remove is the board's worst cell vs BTreeMap (see perf.md).
//   Before further tuning, profile the remove/churn path (samply or
//   cargo flamegraph) to see what the eager per-level rebalance check
//   actually costs.

// SAFETY:
// - `child_count` counts CHILDREN. The first `child_count` slots of
//   `children` MUST be initialized, owned handles, and the first
//   `child_count - 1` slots of `keys` MUST be initialized (n children
//   have n - 1 separators between them).
// - All other slots of both arrays MUST NOT be treated as initialized.
//
// Functionality:
// - The key prefix MUST be strictly sorted, and `keys[i]` is the
//   separator between `children[i]` and `children[i + 1]` — the minimum
//   key of the subtree under `children[i + 1]`, so routing sends
//   `key < keys[i]` to its left and `key >= keys[i]` to its right.
// - Every child roots a subtree of the same height (the depth-type
//   invariant; see `Node`).
// - Occupancy: a non-root inner holds `MIN_OCCUPANCY..=M` children; the
//   root holds `2..=M`, dipping to 1 only transiently inside `remove`'s
//   root shrink, which hoists the lone child before returning.
/// An inner (routing) node: child handles and the separator keys
/// between them.
///
/// Public only so allocator bounds can name it (the [`NodeAllocator`]
/// alias); every field and method is crate-private, and it never
/// appears in a usable public signature.
#[cfg_attr(debug_assertions, repr(C))]
pub struct Inner<K: Key, V, const M: usize> {
    /// Debug-only kind tag. MUST stay the first field: the erased cast
    /// accessors on [`Node`] read it through the pointer before knowing the
    /// pointee's type (hence the debug-only `repr(C)`).
    #[cfg(debug_assertions)]
    kind: NodeKind,

    child_count: usize,

    keys: [MaybeUninit<K>; M],
    children: [MaybeUninit<Node<K, V, M>>; M],
}

impl<K: Key, V, const M: usize> core::fmt::Debug for Inner<K, V, M> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Inner").field("child_count", &self.child_count).finish_non_exhaustive()
    }
}

/// Bitwise-copy a run of `$count` children from `$src` (starting at child
/// index `$src_at`) into `$dst` (starting at child index `$dst_at`), together
/// with the `$count - 1` separator keys BETWEEN those children.
///
/// The key ranges are one shorter than the child ranges by design: a run of
/// `n` adjacent subtrees has exactly `n - 1` separators strictly inside it,
/// and they sit at the SAME start index in the key arrays (the separator
/// between children `j` and `j + 1` lives at key index `j`, so the run
/// starting at child `s` has internal separators at key indices
/// `s .. s + n - 1`). The fencepost separator at a run's boundary is not part
/// of the run and travels separately — promoted to the parent on a split,
/// demoted from the parent on a merge, or written by hand between piecewise
/// runs. This macro deliberately never copies a fencepost.
///
/// These are copies, not moves: afterwards both nodes hold the bits — child
/// handles included — so exactly one side must count each child in its
/// `child_count`. A handle counted twice double-frees on teardown; counted
/// zero times, it leaks. That bookkeeping stays with the caller.
///
/// Expands to raw, unchecked copies, so the call site must be inside an
/// `unsafe` block, upholding:
///
/// - `$count >= 1` (a run is at least one child; `$count == 1` copies one
///   child and no keys);
/// - `$src` and `$dst` are distinct nodes (the copies must not overlap);
/// - both child ranges are in bounds: `$src_at + $count <= M` and
///   `$dst_at + $count <= M` (the key ranges, being one shorter, are then in
///   bounds a fortiori).
///
/// `$src`/`$dst` should be plain place expressions like `self` or `right`
/// (they are evaluated more than once).
///
/// For runs whose source and destination may overlap (shifts within one
/// node), or that carry each child's left separator instead of only the
/// interior keys, use `shift_run!` (defined below).
macro_rules! copy_run {
    ($src:expr, $src_at:expr => $dst:expr, $dst_at:expr; $count:expr) => {{
        let (src_at, dst_at, count) = ($src_at, $dst_at, $count);
        core::ptr::copy_nonoverlapping(
            $src.children.as_ptr().add(src_at),
            $dst.children.as_mut_ptr().add(dst_at),
            count,
        );
        core::ptr::copy_nonoverlapping(
            $src.keys.as_ptr().add(src_at),
            $dst.keys.as_mut_ptr().add(dst_at),
            count - 1,
        );
    }};
}

/// Overlap-tolerant counterpart of [`copy_run!`] for sliding a run WITHIN
/// one node, in two forms. (Cross-node runs never overlap; they stay
/// `copy_run!`'s business. Single-node also keeps miri happy: one mutable
/// pointer is derived per array, where a `$src`/`$dst` pair naming the
/// same node would stack conflicting borrows.)
///
/// The first form has `copy_run!`'s exact run semantics — `$count`
/// children plus the `$count - 1` separators BETWEEN them, fenceposts
/// never copied — but expands to [`ptr::copy`](core::ptr::copy), so the ranges may overlap:
/// e.g. a rotation prepending a borrowed child
/// (`shift_run!(self, 0 => 1; count)`) or a donor closing over its
/// departed first child (`shift_run!(self, 1 => 0; count)`).
///
/// The second form, `shift_run!(..; with left seps)`, moves `$count`
/// children together with each child's LEFT separator: `$count` keys
/// starting ONE INDEX BELOW the children (the separator between children
/// `j - 1` and `j` lives at key index `j - 1`). This is the shape of a
/// shift that opens or closes a paired key+child slot mid-node — the
/// shift-insert making room for a `sep`/`child` pair, or a remove/merge
/// closing the vacated pair — where the run's left key is not a fencepost
/// at all: it belongs to the moved run, travelling into (or over) the
/// slot being opened (or closed). Requires `$src_at >= 1` and
/// `$dst_at >= 1` (child 0 has no left separator).
///
/// Everything else carries over from [`copy_run!`]: raw, unchecked
/// bitwise copies (call from `unsafe` with both child ranges in bounds —
/// against the node's TRUE occupancy, not just `M`), after which both
/// ranges hold the child-handle bits and each handle must stay counted
/// exactly once. `$count >= 1` in the first form (it copies `$count - 1`
/// keys); the second form accepts `$count == 0` (an empty shift, so
/// tail-position call sites need no special-casing).
macro_rules! shift_run {
    ($node:expr, $src_at:expr => $dst_at:expr; $count:expr) => {{
        let (src_at, dst_at, count) = ($src_at, $dst_at, $count);
        let children = $node.children.as_mut_ptr();
        let keys = $node.keys.as_mut_ptr();
        core::ptr::copy(children.add(src_at), children.add(dst_at), count);
        core::ptr::copy(keys.add(src_at), keys.add(dst_at), count - 1);
    }};
    ($node:expr, $src_at:expr => $dst_at:expr; $count:expr; with left seps) => {{
        let (src_at, dst_at, count) = ($src_at, $dst_at, $count);
        let children = $node.children.as_mut_ptr();
        let keys = $node.keys.as_mut_ptr();
        core::ptr::copy(children.add(src_at), children.add(dst_at), count);
        core::ptr::copy(keys.add(src_at - 1), keys.add(dst_at - 1), count);
    }};
}

impl<K: Key, V, const M: usize> Inner<K, V, M> {
    // NOTE: an associated const is only evaluated where it is USED — the
    // definition alone checks nothing. `new` must keep its
    // `const { Self::__FANOUT }` reference or this assert silently never
    // runs (the `compile_fail` doctest on `BPlusTree` pins this; it has
    // been broken twice by dropping the reference — don't make it three).
    const __FANOUT: () = {
        assert!(M == K::FANOUT);
    };

    /// The number of keys in a post-split left node.
    const LEFT_COUNT: usize = M.div_ceil(2) - 1;

    /// The number of keys in a post-split right node.
    const RIGHT_COUNT: usize = M - Self::LEFT_COUNT;

    /// Minimum children per NON-ROOT inner node — the classical-
    /// rebalancing occupancy invariant (the root is exempt, down to 2).
    /// Coherent with splitting (a split leaves `LEFT_COUNT + 1 == ⌈M/2⌉`
    /// children on the left and at least as many on the right) and with
    /// merging (a deficient node at `MIN_OCCUPANCY - 1` plus a minimal
    /// sibling is `2⌈M/2⌉ - 1 <= M` children, so a merge of an
    /// at-minimum pair always fits).
    pub(crate) const MIN_OCCUPANCY: usize = M.div_ceil(2);

    /// Instantiate an empty inner node.
    ///
    /// The node is not yet initialized, and must not be passed to anything
    /// that expects it to be initialized.
    pub(crate) fn new() -> Self {
        // Every inner node is born here; see `assert_fanout_floor` for why
        // a too-small M must be a compile error. The `__FANOUT` reference is
        // load-bearing: without a use, the `M == K::FANOUT` assert never
        // evaluates (see the note at its definition). Do not remove it.
        const { crate::assert_fanout_floor(M) };
        const { Self::__FANOUT };

        Self {
            #[cfg(debug_assertions)]
            kind: NodeKind::Inner,
            child_count: 0,
            keys: [MaybeUninit::uninit(); M],
            children: [const { MaybeUninit::uninit() }; M],
        }
    }

    pub(crate) fn from_pair(sep: K, left: Node<K, V, M>, right: Node<K, V, M>) -> Self {
        let mut this = Self::new();

        this.keys[0].write(sep);
        this.children[0].write(left);
        this.children[1].write(right);
        this.child_count = 2;

        this
    }

    /// Instantiate an inner holding just `child` as child 0 — the seed a
    /// bulk loader grows with [`Self::raw_append_child`]. One child is
    /// below even the root's occupancy floor, so the node must not join a
    /// tree until at least a second child lands.
    pub(crate) fn from_first_child(child: Node<K, V, M>) -> Self {
        let mut this = Self::new();

        this.children[0].write(child);
        this.child_count = 1;

        this
    }

    /// Append `child` as the new last child, with `sep` — the minimum key
    /// of `child`'s subtree — as the separator on its left. Bulk-load
    /// building block: chunks assemble left to right, so every pair after
    /// a node's first lands here.
    ///
    /// # Panics
    ///
    /// - if the node is full;
    /// - if the node has no child yet (child 0 has no left separator —
    ///   seed it directly);
    /// - (debug) if `sep` does not sort strictly after the present keys.
    pub(crate) fn raw_append_child(&mut self, sep: K, child: Node<K, V, M>) {
        assert!(self.child_count >= 1, "seed child 0 before appending");
        debug_assert!(self.key_count() == 0 || self.keys_ref().last().unwrap() < &sep);

        self.keys[self.key_count()].write(sep);
        self.children[self.child_count].write(child);
        self.child_count += 1;
    }

    /// Tear down this inner node and everything below it, dropping every
    /// descendant value exactly once.
    ///
    /// # Panic during teardown
    ///
    /// NOT panic-safe. Children and keys are dropped one at a time; if a
    /// value's [`Drop`] (or a deeper [`drop_subtree`](Node::drop_subtree)) unwinds partway through,
    /// every child and key not yet reached is abandoned — the remainder of
    /// the subtree leaks, and the leaf chain below it is never spliced. Worse,
    /// this runs from [`BPlusTree`](crate::BPlusTree)'s [`Drop`] glue: a panic escaping here while
    /// the thread is already unwinding is a double-panic and aborts the
    /// process.
    pub(crate) fn drop_subtree<A: NodeAllocator<K, V, M>>(self, height: u8, alloc: &mut A) {
        debug_assert!(height > 0, "Inner::drop_subtree called at leaf");

        let Self {
            child_count,
            keys,
            children,
            #[cfg(debug_assertions)]
                kind: _,
        } = self;

        // Wrinkle: Inner nodes store 1 more pointer than they do key,
        // as start and end of the ranges can be inferrred.

        // SAFETY:
        // `child_count` guarantees initialization
        // The `assume_init_drop` calls are safe because:
        // - EXACTLY child_count children are initialized
        // - EXACTLY child_count - 1 keys are initialized
        // - AT LEAST child_count
        unsafe {
            let mut iter = children.into_iter();
            iter.by_ref().take(child_count - 1).zip(keys).for_each(|(child, _)| {
                // NB: not necessary to drop keys.
                child.assume_init().drop_subtree(height - 1, alloc);
            });
            iter.next().unwrap().assume_init().drop_subtree(height - 1, alloc);
        }
    }

    #[inline(always)]
    fn key_count(&self) -> usize {
        // Strict on purpose: zero-child inners no longer exist under the
        // min-occupancy invariant, so an underflow here is a real bug.
        self.child_count - 1
    }

    /// The occupancy of this node: its child count. Per-node, not
    /// subtree-wide — this is what a parent's underflow check reads
    /// through [`Node::len`] after a recursive `remove` returns.
    #[inline(always)]
    pub(crate) fn len(&self) -> usize {
        self.child_count
    }

    /// True if this node is below the minimum occupancy a non-root inner
    /// must keep ([`Self::MIN_OCCUPANCY`]) — the parent's rebalance
    /// trigger after a recursive `remove` returns. The root is exempt and
    /// must not be judged by this.
    pub(crate) fn is_deficient(&self) -> bool {
        self.child_count < Self::MIN_OCCUPANCY
    }

    pub(crate) fn keys_ref(&self) -> &[K] {
        // SAFETY: `child_count` guarantees initialization
        unsafe { self.keys[..self.key_count()].assume_init_ref() }
    }

    fn keys_mut(&mut self) -> &mut [K] {
        let kc = self.key_count();
        // SAFETY: `child_count` guarantees initialization
        unsafe { self.keys[..kc].assume_init_mut() }
    }

    pub(crate) fn children_ref(&self) -> &[Node<K, V, M>] {
        // SAFETY: `child_count` guarantees initialization
        unsafe { self.children[..self.child_count].assume_init_ref() }
    }

    // pub(crate): `Node::drop_subtree` walks the children to tear each one
    // down at `height - 1`.
    pub(crate) fn children_mut(&mut self) -> &mut [Node<K, V, M>] {
        // SAFETY: `child_count` guarantees initialization
        unsafe { self.children[..self.child_count].assume_init_mut() }
    }

    /// Merge the child at `child_idx + 1` into the child at `child_idx`,
    /// closing up the vacated child and key slots. The separator between
    /// the pair travels with the merge, per [`Node::merge`]: discarded when
    /// the children are leaves, demoted into the merged node when they are
    /// inners.
    ///
    /// # Safety
    ///
    /// - `height` must be the true height of the subtree rooted at this
    ///   node (necessarily > 0: this node is an `Inner`; both members of
    ///   the merged pair root subtrees of `height - 1`).
    /// - Both slots of the pair must be live: `child_idx + 1 < child_count`.
    /// - The merged occupancy must fit one node, and above the leaf level
    ///   both members must hold at least one child of their own — the
    ///   per-level bounds of [`Leaf::merge`](crate::Leaf::merge) and [`Inner::merge`], whose
    ///   preconditions this call forwards.
    unsafe fn merge_sibling_into<A: NodeAllocator<K, V, M>>(
        &mut self,
        height: u8,
        child_idx: usize,
        alloc: &mut A,
    ) {
        debug_assert!(height > 0);
        debug_assert!(child_idx + 1 < self.child_count);
        debug_assert!(
            // Safety: dbg and height propagation.
            unsafe {
                self.children[child_idx].assume_init_ref().len(height - 1)
                    + self.children[child_idx + 1].assume_init_ref().len(height - 1)
                    <= M
            },
            "Merging would overflow a node"
        );
        // SAFETY: per this fn's contract both pair slots are live, so the
        // sibling handle and the separator read from initialized slots
        // (each read exactly once: the shift below closes both). `merge`'s
        // preconditions are forwarded by this call's own; the shift moves
        // initialized slots (`child_idx + 2..child_count` with their left
        // separators), and the count decrement retires the vacated tail.
        unsafe {
            let sibling = self.children[child_idx + 1].assume_init_read();
            let sep = self.keys[child_idx].assume_init_read();

            self.children_mut()[child_idx].merge(height - 1, sep, sibling, alloc);

            let count = self.child_count - child_idx - 2;
            shift_run!(self, child_idx + 2 => child_idx + 1; count; with left seps);
        }

        self.child_count -= 1;
    }

    pub(crate) fn rebalance<A: NodeAllocator<K, V, M>>(
        &mut self,
        height: u8,
        child_idx: usize,
        alloc: &mut A,
    ) {
        debug_assert!(self.child_count > child_idx);
        // SAFETY: the caller threads the true height, so every child
        // roots a subtree of height `height - 1`.
        debug_assert!(unsafe { self.children_ref()[child_idx].is_deficient(height - 1) });

        // Cases:
        // - Right Sibling has > MIN -> Steal Right
        // - Left Sibling has > MIN -> Steal Left
        // - Neither ->
        //    - right exists -> merge right
        //    - left exists -> merge left
        //    - None:
        //      It is the only child. This can only happen if the tree is about
        //      to shrink 1 level. Handled in tree.rs.
        //
        // The separator between the pair lives in THIS node's keys — at
        // `child_idx` for the right sibling, `child_idx - 1` for the left
        // (which is why each steal branch is bounds-guarded on that
        // index). The steal consumes it and returns its replacement,
        // written over the old one in place: no slots close, no counts
        // change in this node. A merge instead reads the separator itself
        // (it travels with the merge), so the branches below need no
        // plumbing.

        // right sibling exists, and would not be made deficient.
        if child_idx + 1 < self.child_count
            // SAFETY: the guard keeps the index in the live range, and
            // the sibling roots a subtree of height `height - 1`.
            && unsafe { self.children_ref()[child_idx + 1].len(height - 1) } > Self::MIN_OCCUPANCY
        {
            let sep = self.keys_ref()[child_idx];
            let (low, high) = self.children_mut().split_at_mut(child_idx + 1);
            // SAFETY: same-parent adjacent siblings share the true height
            // `height - 1`; `sep` is the separator between them; the donor
            // is strictly above its minimum and the receiver, deficient,
            // has room.
            let sep = unsafe { low[child_idx].steal_from_right(height - 1, sep, &mut high[0]) };
            self.keys_mut()[child_idx] = sep;
            return;
        }

        // left sibling exists, and would not be made deficient.
        if child_idx > 0
            // SAFETY: as the right-sibling probe above, mirrored.
            && unsafe { self.children_ref()[child_idx - 1].len(height - 1) } > Self::MIN_OCCUPANCY
        {
            let sep = self.keys_ref()[child_idx - 1];
            let (low, high) = self.children_mut().split_at_mut(child_idx);
            // SAFETY: as the right-steal above, mirrored.
            let sep = unsafe { high[0].steal_from_left(height - 1, sep, &mut low[child_idx - 1]) };
            self.keys_mut()[child_idx - 1] = sep;
            return;
        }

        // Merge.
        if self.child_count > child_idx + 1 {
            // merge right
            // SAFETY: the guard gives both pair slots live; heights
            // thread through; the union fits — the deficient child
            // (`MIN_OCCUPANCY - 1`) plus a sibling at its minimum (both
            // steals declined) stays within `M`.
            unsafe { self.merge_sibling_into(height, child_idx, alloc) };
        } else {
            // merge left
            // SAFETY: as the right merge, mirrored — `child_idx > 0`
            // here, since every non-root inner has >= 2 children and the
            // right guard failed.
            unsafe { self.merge_sibling_into(height, child_idx - 1, alloc) };
        }
    }

    /// Find the child idx that should/could contain `key`.
    pub(crate) fn child_idx_for_key(&self, key: &K) -> usize {
        // Subtlety:
        // We route on `<=` (first key strictly greater wins), as the inner
        // node stores the low child of the next leaf. This is different
        // from finding the exact insertion point in the leaf nodes using
        // `<`.
        //
        // Branchless linear count (A/B history in perf.md),
        // mirroring `Leaf::find_key`: the child index is the number of
        // separators at or below `key`. See `find_key` for why this
        // shape won.
        self.keys_ref().iter().map(|existing| usize::from(existing <= key)).sum()
    }

    /// Get the child into which we would insert the `key`
    // pub(crate): traversal lives on `Node` under the union design; Inner
    // only routes.
    pub(crate) fn child_for_key(&self, key: &K) -> &Node<K, V, M> {
        &self.children_ref()[self.child_idx_for_key(key)]
    }

    /// Get the child into which we would insert the `key`
    // pub(crate): traversal lives on `Node` under the union design; Inner
    // only routes.
    pub(crate) fn child_for_key_mut(&mut self, key: &K) -> &mut Node<K, V, M> {
        let idx = self.child_idx_for_key(key);
        &mut self.children_mut()[idx]
    }

    /// Shift-insert `sep`/`child` without checking occupancy: `child`
    /// lands at CHILD index `child_idx` and `sep` — its left separator —
    /// at key index `child_idx - 1`, with the children at `child_idx..`
    /// (each traveling with its own left separator) shifting up one slot.
    /// The mirror of [`Leaf::insert_unchecked`], indexed in child slots
    /// throughout: a new child never lands at slot 0, so its left
    /// separator always exists.
    ///
    /// [`Leaf::insert_unchecked`]: crate::Leaf
    ///
    /// # Safety Preconditions
    ///
    /// - `self.child_count < M` (room for one more child);
    /// - `1 <= child_idx <= self.child_count` — for a split of the child
    ///   at index `i`, the new right sibling lands at `i + 1`
    ///   (`child_idx == child_count` appends at the end);
    /// - ordering: `sep` slots into sorted key order at key index
    ///   `child_idx - 1`, and `child` roots a subtree of the same height
    ///   as its new neighbors, holding exactly the keys `>= sep` below
    ///   the next separator.
    #[track_caller]
    #[inline(always)]
    pub(crate) unsafe fn insert_child_unchecked(
        &mut self,
        child_idx: usize,
        sep: K,
        child: Node<K, V, M>,
    ) {
        debug_assert!(self.child_count < M);
        debug_assert!((1..=self.child_count).contains(&child_idx));

        // SAFETY: the moved run `child_idx..child_count` is inside the
        // live prefix and its destination stays within `M`
        // (`child_count < M`); `child_idx >= 1`, so every moved child has
        // a left separator to travel with. An append
        // (`child_idx == child_count`) is the empty shift.
        unsafe {
            shift_run!(self, child_idx => child_idx + 1; self.child_count - child_idx; with left seps)
        };

        self.keys[child_idx - 1].write(sep);
        self.children[child_idx].write(child);
        self.child_count += 1;
    }

    /// Adopt `child`, the new right sibling produced when one of this
    /// node's children split; `sep` is the separator between the split
    /// child and `child` (`key < sep` routes to the old child, `key >= sep`
    /// to the new one). Splits this node if it is already full.
    ///
    /// Returns the promoted separator and this node's own new right sibling
    /// if the adoption forced a split. (The separator travels with the
    /// split because the caller holds only erased handles — and on an inner
    /// split the middle key is pushed up out of the node, not copied.)
    pub(crate) fn splitting_insert_child<A: SlotAllocator<Self>>(
        &mut self,
        partition: usize,
        sep: K,
        child: Node<K, V, M>,
        alloc: &mut A,
    ) -> (K, NonNull<Self>) {
        debug_assert!(self.child_count == M);
        debug_assert!(partition <= M);
        debug_assert!(partition == self.child_idx_for_key(&sep));

        let mut right = Self::new();
        let mut promoted = sep;

        if partition < Self::LEFT_COUNT {
            // SAFETY: the node is full (`child_count == M`, asserted), so
            // every source slot — keys `0..M - 1`, children `0..M` — is
            // initialized, and the fresh `right`'s slots are vacant. The
            // count adjustments retire the moved right half exactly once,
            // and `insert_child_unchecked`'s contract holds: room
            // (`M - RIGHT_COUNT < M`) and ordering (`partition` is `sep`'s
            // routing index, asserted).
            unsafe {
                // The promoted is the key that WILL BE the last key of Left,
                // after we insert the new child.
                promoted = self.keys[Self::LEFT_COUNT - 1].assume_init_read();

                // Copy the right half to the right
                copy_run!(
                    self, Self::LEFT_COUNT => right, 0; Self::RIGHT_COUNT
                );

                // insert the new child
                self.child_count -= Self::RIGHT_COUNT;
                self.insert_child_unchecked(partition + 1, sep, child);
            }

            self.child_count = Self::LEFT_COUNT;
            right.child_count = Self::RIGHT_COUNT;
        } else if partition == Self::LEFT_COUNT {
            // The new child belongs in the right node. The separator is
            // promoted.
            // SAFETY: full node — every source slot is initialized; the
            // fresh `right`'s slots are vacant and are filled left to
            // right (the new child at 0, the copied run above it).
            unsafe {
                // write the new child and key
                right.children[0].write(child);
                right.keys[0] = self.keys[Self::LEFT_COUNT];

                // Copy the right half to the right, above the key we just
                // inserted
                copy_run!(self, Self::LEFT_COUNT + 1 => right, 1; Self::RIGHT_COUNT - 1);
            }
        } else {
            // The new child belongs in the right node. Some other key is
            // promoted.

            // The new child's child index in `right`.
            let insertion = partition - Self::LEFT_COUNT;

            // SAFETY: full node — every source slot is initialized; the
            // fresh `right`'s slots are vacant. `insertion >= 1` in this
            // branch, so the by-hand pair lands in bounds, between the
            // two copied runs.
            unsafe {
                // The promoted key is the key that WILL BE the first in the
                // right child.
                promoted = self.keys[Self::LEFT_COUNT].assume_init_read();

                // First: the run strictly between the promoted key and
                // the insertion point.
                copy_run!(self, Self::LEFT_COUNT + 1 => right, 0; insertion);

                // Then: the new pair, by hand — `sep` is the fencepost
                // on the new child's left.
                right.keys[insertion - 1].write(sep);
                right.children[insertion].write(child);

                // Then: the displaced fencepost that sat at OLD
                // keys[partition] (now on the new child's right), then the
                // run above the insertion point.
                if partition < M - 1 {
                    right.keys[insertion] = self.keys[partition];
                    copy_run!(self, partition + 1 => right, insertion + 1; M - partition - 1);
                }
            }
        };

        self.child_count = Self::LEFT_COUNT + 1;
        right.child_count = M - Self::LEFT_COUNT;

        (promoted, alloc.allocate(right))
    }

    /// Adopt `sep`/`child` — the new right sibling produced when the child
    /// at `partition` split — dispatching on occupancy: shift-insert in
    /// place when this node has room, split via
    /// [`Self::splitting_insert_child`] when it is full. Returns the
    /// promoted separator and this node's own new right sibling in the
    /// split case, `None` otherwise.
    ///
    /// This is the only correct entry point for adopting a child split:
    /// [`splitting_insert_child`](Self::splitting_insert_child) ALWAYS splits, and
    /// [`insert_child_unchecked`](Self::insert_child_unchecked) never checks room — the occupancy
    /// decision lives here.
    pub(crate) fn insert_child<A: SlotAllocator<Self>>(
        &mut self,
        partition: usize,
        sep: K,
        child: Node<K, V, M>,
        alloc: &mut A,
    ) -> Option<(K, NonNull<Self>)> {
        debug_assert!(partition == self.child_idx_for_key(&sep));

        // The split child sits at `partition`; its new right sibling
        // lands one child slot above.
        if self.child_count < M {
            // SAFETY: the branch condition is the room precondition; the
            // assert above puts `partition + 1` in `1..=child_count`; the
            // ordering precondition is inherited from the split that
            // produced `sep`/`child`.
            unsafe { self.insert_child_unchecked(partition + 1, sep, child) };
            None
        } else {
            Some(self.splitting_insert_child(partition, sep, child, alloc))
        }
    }

    /// Fold `other` — the immediate right sibling of `self` under their
    /// shared parent — into `self`, the structural inverse of
    /// [`Self::splitting_insert_child`]: demote `sep`, the separator that
    /// sat between the two nodes in the parent, into the key gap, then
    /// append every child of `other` after `self`'s.
    ///
    /// The separator is DEMOTED, not dropped — the mirror image of promotion
    /// on split, and the opposite of the leaf-level merge (where the parent
    /// discards it). The counts force this: `a + b` children need
    /// `a + b - 1` keys, and the two nodes bring only `(a - 1) + (b - 1)`
    /// between them.
    ///
    /// No `height` parameter: merging moves handles without ever
    /// dereferencing them, so it never needs to know what they point at.
    ///
    /// The caller (the parent, from `remove`'s underflow path) remains
    /// responsible for what this does not touch: removing `other`'s child
    /// slot and the demoted separator from its own arrays. Note the caller
    /// reaches this through an erased [`Node`] pair, so it needs the
    /// `into_inner` conversion — currently private to `tree.rs`; widening
    /// it (or dispatching the whole merge from `Node`) is part of the
    /// remove plumbing.
    ///
    /// # Safety
    ///
    /// - `other` must be `self`'s immediate right sibling under the same
    ///   parent, with `sep` the separator between them: every key in
    ///   `self`'s subtree is `< sep`, and every key in `other`'s subtree is
    ///   `>= sep`.
    /// - Both nodes must sit at the same height (their children root
    ///   subtrees of equal height — the depth-type invariant for the merged
    ///   node).
    /// - Both must hold at least one child, and the union must fit:
    ///   `self.child_count + other.child_count <= M`.
    /// - `other` must own its subtree; ownership of every child transfers
    ///   to `self`, and `other`'s allocation must already have been
    ///   reclaimed by the caller (it arrives by value).
    pub(crate) unsafe fn merge(&mut self, sep: K, other: Self) {
        debug_assert!(self.child_count + other.child_count <= M, "Merging would overflow a node");
        debug_assert!(self.key_count() == 0 || self.keys_ref().last().unwrap() < &sep,);
        debug_assert!(other.key_count() == 0 || &sep <= other.keys_ref().first().unwrap());

        // 1. The separator is demoted to a key
        self.keys[self.key_count()].write(sep);

        // 2. We copy in all the sibling's nodes.
        // SAFETY: per this fn's contract the union fits (asserted), so
        // the destination run starting at `self.child_count` is vacant;
        // `other`'s first `child_count` slots are initialized, and
        // `other` arrives by value — ownership of the copied handles
        // transfers without a double-free path.
        unsafe {
            copy_run!(other, 0 => self, self.child_count; other.child_count);
        }

        // 3. We increase our count.
        self.child_count += other.child_count;
    }

    /// Consume a single-child inner — the state a root reaches when a
    /// cascade merge collapses its last pair of children — and return that
    /// child, discarding the shell. The root-shrink (hoist) primitive;
    /// only the tree layer calls it, and only on the root, since under the
    /// min-occupancy invariant no other node can be single-child.
    ///
    /// # Panics
    ///
    /// Debug-asserts `child_count == 1`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `child_count` is exactly 1.
    pub(crate) fn into_only_child(self) -> Node<K, V, M> {
        debug_assert_eq!(self.child_count, 1);

        // SAFETY: per this fn's contract `child_count == 1`, so child 0
        // is initialized; `self` is consumed by value, so the handle
        // moves out exactly once.
        unsafe { self.children[0].assume_init_read() }
    }

    /// Rotate one child in from `right`, `self`'s immediate right sibling
    /// under their shared parent: `sep` — the parent separator between the
    /// pair — demotes to `self`'s new LAST key; `right`'s first child
    /// moves across to `self`'s new last child slot; and `right`'s first
    /// key promotes out as the returned replacement separator, which the
    /// caller writes over the old one in place.
    ///
    /// The borrow half of classical rebalancing at inner level — a true
    /// rotation through the parent. Getting the demote/promote pairing
    /// backwards mis-routes keys silently; the unit tests pin it.
    ///
    /// # Safety
    ///
    /// - Same-parent adjacency and ordering: every key under `self` is
    ///   `< sep`, every key under `right` is `>= sep`.
    /// - Both nodes' children root subtrees of equal height.
    /// - Room and donor liveness: `self.child_count < M` and
    ///   `right.child_count >= 2` (the donor must remain a node; under
    ///   the C policy the caller only borrows from a sibling strictly
    ///   above its minimum, which implies both).
    pub(crate) unsafe fn rotate_from_right(&mut self, sep: K, right: &mut Self) -> K {
        // SAFETY: the donor has at least 2 children, so key 0 is live.
        let promotion = unsafe { right.keys[0].assume_init_read() };

        // SAFETY: the donor has >= 2 children (contract), so child 0 is
        // live; the shift below closes the vacated slot, so the handle
        // moves exactly once.
        self.children[self.child_count].write(unsafe { right.children[0].assume_init_read() });

        // The demoted separator becomes our new last key.
        self.keys[self.key_count()].write(sep);

        // Shift all of rights children down into the empty slot
        // SAFETY: slots `1..child_count` (and their left separators) are
        // initialized, and the overlapping shift is `ptr::copy`; slot 0's
        // old handle was moved out above.
        unsafe {
            shift_run!(right, 1 => 0; right.child_count - 1);
        };

        self.child_count += 1;
        right.child_count -= 1;
        promotion
    }

    /// Mirror of [`Self::rotate_from_right`]: rotate one child in from
    /// `left`, `self`'s immediate LEFT sibling. `sep` demotes to `self`'s
    /// new FIRST key; `left`'s last child moves to `self`'s new first
    /// child slot; `left`'s last key promotes out as the returned
    /// replacement separator.
    ///
    /// # Safety
    ///
    /// - Same-parent adjacency and ordering: every key under `left` is
    ///   `< sep`, every key under `self` is `>= sep`.
    /// - Both nodes' children root subtrees of equal height.
    /// - `self.child_count < M` and `left.child_count >= 2`.
    pub(crate) unsafe fn rotate_from_left(&mut self, sep: K, left: &mut Self) -> K {
        // SAFETY: `self` has room (`child_count < M`, contract) and its
        // slots `0..child_count` are initialized; the overlapping shift
        // is `ptr::copy`, and the vacated slot 0 is overwritten below.
        unsafe {
            shift_run!(self, 0 => 1; self.child_count);
        }

        self.keys[0].write(sep);
        // SAFETY: the donor has >= 2 children (contract), so its last
        // child is live; the count decrement below retires the slot, so
        // the handle moves exactly once.
        self.children[0].write(unsafe { left.children[left.child_count - 1].assume_init_read() });

        // SAFETY: the donor has at least 2 children, so its last key is live.
        let promotion = unsafe { left.keys[left.key_count() - 1].assume_init_read() };

        self.child_count += 1;
        left.child_count -= 1;

        promotion
    }
}

#[cfg(test)]
#[path = "../tests/inner.rs"]
mod tests;
