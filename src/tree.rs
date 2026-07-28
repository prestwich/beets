use core::{mem::MaybeUninit, ptr::NonNull};

use crate::allocator::{Global, NodeAllocator, SlotAllocator};
use crate::{Inner, Key, Leaf, MAX_LEVELS, Slabs, iter};

// TODO:
// - perf: last-touched-leaf cache for point reads (cf. sweep_bptree's
//   `try_cache`): remember the leaf the previous get landed at together
//   with its key-range bounds, and let a probe that falls inside the
//   range skip the descent entirely. Design questions to settle first:
//   the cache must be written under `&self` (sweep uses a relaxed
//   AtomicUsize node id; our handle is a NonNull, and a Cell would cost
//   `Sync`); every structural mutation must invalidate or re-validate
//   it; and the payoff is workload-shaped — big for sequential/skewed
//   key streams, ~nil for uniform-random probes (benches would need a
//   locality-heavy get workload to see it at all).

/// Debug-only node-kind tag. In debug builds it is the first field of both
/// [`Leaf`] and [`Inner`] (which are `repr(C)` there), so the cast accessors on
/// [`Node`] can soundly read it through an erased pointer — before knowing
/// which type they point at — and assert the height-inferred kind against
/// reality. In release builds the tag, the asserts, and their cost do not
/// exist.
#[cfg(debug_assertions)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NodeKind {
    Leaf,
    Inner,
}

/// An untyped handle to a heap-allocated tree node.
///
/// # The depth-type invariant
///
/// There is no tag: which field is live is inferred from position in the
/// tree. A `Node` rooting a subtree of height `h` points at an [`Inner`] if
/// `h > 0` and a [`Leaf`] if `h == 0`. This is sound because the tree is
/// perfectly height-balanced — every leaf sits at the same depth — and the
/// height changes at exactly one place: the root.
///
/// Every `unsafe` method on this type takes a `height` parameter, and its
/// safety contract is the same sentence: **`height` must be the true height
/// of the subtree rooted at this node.** A wrong height reinterprets one
/// node type as the other.
///
/// # Ownership
///
/// A `Node` owns its subtree, but has no drop glue: dropping the handle
/// silently leaks the subtree. Teardown is explicit, via
/// [`Node::drop_subtree`].
pub(crate) union Node<K: Key, V, const M: usize> {
    inner: NonNull<Inner<K, V, M>>,
    leaf: NonNull<Leaf<K, V, M>>,
}

/// Generate the typed views of the erased handle for one pointee kind —
/// the shared borrow, the exclusive borrow, and the by-value conversion
/// (which retires the node's slot to the allocator; every free in the
/// crate funnels through it). Every accessor checks the pointee's
/// debug-only kind tag before trusting the caller's cast; all three
/// share one safety contract, stated once here: **the caller must
/// ensure `self` points at the named kind** — it is not enforced by the
/// type system.
macro_rules! cast_accessors {
    ($Kind:ident, $field:ident: $as_ref:ident, $as_mut:ident, $into:ident) => {
        /// # Safety
        ///
        /// The caller must ensure that `self` is a
        #[doc = concat!("`", stringify!($Kind), "`.")]
        /// This is not enforced by the type system.
        #[inline(always)]
        #[track_caller]
        unsafe fn $as_ref(&self) -> &$Kind<K, V, M> {
            #[cfg(debug_assertions)]
            self.assert_kind(NodeKind::$Kind);
            // SAFETY: this accessor's contract — the caller vouches the
            // pointee is a live node of this kind (debug-asserted above);
            // the shared borrow of the handle keeps it valid.
            unsafe { self.$field.as_ref() }
        }

        /// # Safety
        ///
        /// The caller must ensure that `self` is a
        #[doc = concat!("`", stringify!($Kind), "`.")]
        /// This is not enforced by the type system.
        #[inline(always)]
        #[track_caller]
        unsafe fn $as_mut(&mut self) -> &mut $Kind<K, V, M> {
            #[cfg(debug_assertions)]
            self.assert_kind(NodeKind::$Kind);
            // SAFETY: this accessor's contract — the caller vouches the
            // pointee is a live node of this kind (debug-asserted above);
            // the handle owns its subtree, so the exclusive borrow of the
            // handle makes the pointee exclusively reachable.
            unsafe { self.$field.as_mut() }
        }

        /// # Safety
        ///
        /// The caller must ensure that `self` is a
        #[doc = concat!("`", stringify!($Kind), "`,")]
        /// which is not enforced by the type system, and that `alloc` is
        /// the allocator this node came from.
        #[track_caller]
        unsafe fn $into<A: SlotAllocator<$Kind<K, V, M>>>(self, alloc: &mut A) -> $Kind<K, V, M> {
            #[cfg(debug_assertions)]
            self.assert_kind(NodeKind::$Kind);
            // SAFETY: the pointer came from `alloc` (caller vouches) and
            // `self` is consumed — no other path to this slot remains.
            unsafe { alloc.deallocate(self.$field) }
        }
    };
}

/// Dispatch a niladic method on [`Node`] to whichever pointee `height`
/// implies: a single cast — no routing key, no recursion, no threading
/// of `height` down a descent (that is [`delegate!`]'s business). Each
/// listed method must exist on both [`Leaf`] and [`Inner`] with the written
/// signature. Doc comments and other attributes carry over.
///
/// The generated method is `unsafe`: `height` must be the true height
/// of the subtree rooted at this node, per the type's safety contract.
macro_rules! dispatch {
    () => {};
    (
        $(#[$attr:meta])*
        fn $name:ident(&self) -> $ret:ty;
        $($rest:tt)*
    ) => {
        $(#[$attr])*
        ///
        /// # Safety
        ///
        /// `height` must be the height of the subtree rooted at this node.
        pub(crate) unsafe fn $name(&self, height: u8) -> $ret {
            if height == 0 {
                // SAFETY: height 0 ⇒ this node is a leaf (caller vouches
                // for `height`).
                unsafe { self.as_leaf().$name() }
            } else {
                // SAFETY: height > 0 ⇒ this node is inner (caller vouches
                // for `height`).
                unsafe { self.as_inner().$name() }
            }
        }
        dispatch! { $($rest)* }
    };
}

impl<K: Key, V, const M: usize> Node<K, V, M> {
    pub(crate) fn from_inner<A: SlotAllocator<Inner<K, V, M>>>(
        inner: Inner<K, V, M>,
        alloc: &mut A,
    ) -> Self {
        Self { inner: alloc.allocate(inner) }
    }

    /// Wrap an already-boxed leaf: the right sibling handed back by
    /// [`Leaf::insert_at`] on split.
    pub(crate) fn from_leaf_ptr(leaf: NonNull<Leaf<K, V, M>>) -> Self {
        Self { leaf }
    }

    /// Wrap an already-boxed inner node: the right sibling handed back by
    /// [`Inner::splitting_insert_child`] on split.
    pub(crate) fn from_inner_ptr(inner: NonNull<Inner<K, V, M>>) -> Self {
        Self { inner }
    }

    /// Read the debug-only kind tag through the erased pointer.
    #[cfg(debug_assertions)]
    fn kind(&self) -> NodeKind {
        // SAFETY: in debug builds both pointee types are repr(C) with
        // NodeKind as their first field, so this read is valid whichever
        // type the handle actually points at. (Union fields share offset
        // 0; which field we read the pointer from is immaterial.)
        unsafe { *self.leaf.as_ptr().cast::<NodeKind>() }
    }

    /// Debug-only teeth for the depth-type invariant: every cast accessor
    /// checks the height-inferred kind against the pointee's tag.
    #[cfg(debug_assertions)]
    #[track_caller]
    fn assert_kind(&self, expected: NodeKind) {
        assert_eq!(
            self.kind(),
            expected,
            "depth-type invariant violated: cast to {expected:?}, but the pointee's tag \
             disagrees — a wrong height was passed down this call path"
        );
    }

    cast_accessors! { Leaf, leaf: as_leaf, as_leaf_mut, into_leaf }
    cast_accessors! { Inner, inner: as_inner, as_inner_mut, into_inner }

    dispatch! {
        /// The occupancy of THIS node — not the entry count of the subtree
        /// under it (that's `BPlusTree.len`, per the decided note above):
        /// key/value pairs for a leaf, children for an inner. This is the
        /// number the parent reads after a recursive `remove` returns, to
        /// decide whether the child underflowed and needs merging.
        fn len(&self) -> usize;

        /// True if this node is below the minimum occupancy the classical-
        /// rebalancing invariant demands of a NON-ROOT node of its kind
        /// (pairs for a leaf, children for an inner) — the parent's rebalance
        /// trigger after a recursive `remove` returns. The root is exempt and
        /// must not be judged by this.
        fn is_deficient(&self) -> bool;
    }

    /// Fold `other` — the immediate right sibling of `self` under their
    /// shared parent — into `self`, consuming and freeing `other`'s node.
    /// `sep` is the separator that sat between the two in the parent; its
    /// fate diverges by level, which is why it travels with the merge: at
    /// the leaf level it is discarded (the leaf keys carry the data, and
    /// `K: Copy` leaves nothing to drop), at inner levels it is demoted
    /// into the merged node's key gap.
    ///
    /// The caller (the parent, on `remove`'s underflow path) remains
    /// responsible for its own bookkeeping: removing `other`'s child slot
    /// and the separator from its arrays.
    ///
    /// (Hand-written rather than a `delegate!` entry: it consumes a second
    /// node of the same kind by value, and there is no routing key and no
    /// recursion — `height` is consumed by a single cast.)
    ///
    /// # Safety
    ///
    /// - `height` must be the true height of the subtrees rooted at BOTH
    ///   nodes (as same-parent siblings they necessarily share it).
    /// - `other` must be `self`'s immediate right sibling under the same
    ///   parent, with `sep` the separator between them: every key in
    ///   `self`'s subtree is `< sep`, every key in `other`'s is `>= sep`.
    /// - `other` must own its subtree; ownership of its contents transfers
    ///   to `self`, its node is freed here, and no other handle may be used
    ///   to reach it afterward.
    /// - The merged occupancy must fit; see [`Leaf::merge`] and
    ///   [`Inner::merge`] for each level's precise bound (each level's
    ///   remaining preconditions apply through this call).
    pub(crate) unsafe fn merge<A: NodeAllocator<K, V, M>>(
        &mut self,
        height: u8,
        sep: K,
        other: Self,
        alloc: &mut A,
    ) {
        if height == 0 {
            // SAFETY: height 0 ⇒ both nodes are leaves (caller vouches for
            // `height`, shared by both siblings). `other` is moved out of
            // and its slot reclaimed by `into_leaf`; `Leaf::merge`'s
            // remaining preconditions are the caller's, forwarded. The
            // separator is discarded — nothing to drop, `K: Copy`.
            unsafe { self.as_leaf_mut().merge(other.into_leaf(alloc)) }
        } else {
            // SAFETY: height > 0 ⇒ both nodes are inner (caller vouches for
            // `height`, shared by both siblings). `other` is moved out of
            // and its slot reclaimed by `into_inner`; `Inner::merge`'s
            // remaining preconditions are the caller's, forwarded.
            unsafe { self.as_inner_mut().merge(sep, other.into_inner(alloc)) }
        }
    }

    /// Borrow one unit from `right`, `self`'s immediate right sibling
    /// under their shared parent — a pair at the leaf level, a child at
    /// inner levels. Returns the replacement separator, which the caller
    /// writes over the old one in place. `sep` is the current separator
    /// between the pair: consumed (demoted) by the inner rotation, unused
    /// at the leaf level, where the new separator is minted from the
    /// donor's own contents.
    ///
    /// (Hand-written rather than a `delegate!` entry: it takes a second
    /// node of the same kind, and `height` is consumed by a single cast —
    /// no routing key, no recursion.)
    ///
    /// # Safety
    ///
    /// - `height` must be the true height of the subtrees rooted at BOTH
    ///   nodes (as same-parent siblings they necessarily share it).
    /// - `right` must be `self`'s immediate right sibling under the same
    ///   parent, with `sep` the separator between them; each level's
    ///   remaining preconditions ([`Leaf::steal_from_right`],
    ///   [`Inner::rotate_from_right`]) apply through this call.
    pub(crate) unsafe fn steal_from_right(&mut self, height: u8, sep: K, right: &mut Self) -> K {
        if height == 0 {
            // The separator is unused at this level — nothing to drop,
            // `K: Copy`.
            let _ = sep;
            // SAFETY: height 0 ⇒ both nodes are leaves (caller vouches
            // for `height`, shared by the pair).
            unsafe { self.as_leaf_mut().steal_from_right(right.as_leaf_mut()) }
        } else {
            // SAFETY: height > 0 ⇒ both nodes are inner (caller vouches
            // for `height`, shared by the pair).
            unsafe { self.as_inner_mut().rotate_from_right(sep, right.as_inner_mut()) }
        }
    }

    /// Mirror of [`Node::steal_from_right`]: borrow one unit from `left`,
    /// `self`'s immediate LEFT sibling, `sep` being the separator between
    /// `left` and `self`.
    ///
    /// # Safety
    ///
    /// As [`Node::steal_from_right`], mirrored; each level's remaining
    /// preconditions ([`Leaf::steal_from_left`],
    /// [`Inner::rotate_from_left`]) apply through this call.
    pub(crate) unsafe fn steal_from_left(&mut self, height: u8, sep: K, left: &mut Self) -> K {
        if height == 0 {
            // The separator is unused at this level.
            let _ = sep;
            // SAFETY: height 0 ⇒ both nodes are leaves (caller vouches
            // for `height`, shared by the pair).
            unsafe { self.as_leaf_mut().steal_from_left(left.as_leaf_mut()) }
        } else {
            // SAFETY: height > 0 ⇒ both nodes are inner (caller vouches
            // for `height`, shared by the pair).
            unsafe { self.as_inner_mut().rotate_from_left(sep, left.as_inner_mut()) }
        }
    }

    /// Tear down the subtree rooted at this node: free every descendant
    /// node, dropping the values they hold, exactly once. `Node` has no
    /// drop glue — a handle that goes out of scope without passing through
    /// this method leaks its whole subtree.
    ///
    /// # Safety
    ///
    /// - `height` must be the height of the subtree rooted at this node.
    /// - This handle must own the subtree: it must not have been torn down
    ///   already, and no other handle may be used to reach it afterward.
    /// - No leaf outside this subtree may hold a sibling link into it: the
    ///   caller must have spliced the leaf chain past this subtree first (or
    ///   be tearing down everything that links into it, as whole-tree drop
    ///   does). Teardown never touches `next`, so a stale inbound link
    ///   dangles silently until iteration walks into it.
    ///
    /// # Panic during teardown
    ///
    /// NOT panic-safe. A value [`Drop`] that unwinds partway through abandons
    /// every node and value not yet reached in this subtree — they leak. This
    /// is also the path taken by [`BPlusTree`]'s `Drop`, so a panic escaping it
    /// during an already-unwinding drop double-panics and aborts.
    pub(crate) unsafe fn drop_subtree<A: NodeAllocator<K, V, M>>(self, height: u8, alloc: &mut A) {
        if height == 0 {
            // SAFETY:
            // - `height` is 0, so `self` is a `Leaf`.
            // - `self` is moved out of, so no other code can access it.
            unsafe {
                drop(self.into_leaf(alloc));
            }
        } else {
            // SAFETY:
            // - `height` > 0, so `self` is an `Inner`.
            unsafe {
                self.into_inner(alloc).drop_subtree(height, alloc);
            }
        }
    }
}

/// A B+Tree.
///
/// The fanout `M` must be exactly [`Key::FANOUT`] for `K` — trees are
/// instantiated as `BPlusTree<K, V, { K::FANOUT }>`, and a mismatched
/// `M` must be rejected at compile time, where the tree's nodes are
/// born:
///
/// ```compile_fail
/// // <u64 as beets::Key>::FANOUT is not 7 — this must not compile.
/// let tree: beets::BPlusTree<u64, u64, 7> = beets::BPlusTree::new();
/// ```
///
/// The tree is [`Send`] exactly when its parts are; a non-`Send`
/// constituent must deny it:
///
/// ```compile_fail
/// // Rc is not Send — a tree holding Rc values must not be either.
/// fn require_send<T: Send>() {}
/// require_send::<beets::BPlusTree<u64, core::rc::Rc<u8>, { <u64 as beets::Key>::FANOUT }>>();
/// ```
///
/// [`Sync`] likewise — and `Send` parts alone must not make a
/// shareable tree:
///
/// ```compile_fail
/// // Cell is Send but not Sync — a tree of Cell values must not be Sync.
/// fn require_sync<T: Sync>() {}
/// require_sync::<beets::BPlusTree<u64, core::cell::Cell<u8>, { <u64 as beets::Key>::FANOUT }>>();
/// ```
pub struct BPlusTree<K: Key, V, const M: usize, A: NodeAllocator<K, V, M> = Slabs<K, V, M, Global>>
{
    // The handle IS the pointer; boxing it would be double indirection.
    root: Node<K, V, M>,
    height: u8,
    len: usize,

    // Declared last so any by-value teardown order is values-first; the
    // real guarantee is `Drop`, which walks the tree through `&mut self
    // .allocator` before the field itself drops.
    allocator: A,
}

// SAFETY: sending the tree sends exclusive ownership of everything it
// reaches. The `NonNull`s that suppress the auto-impl (the root
// union's node pointers and the leaf chain) all target nodes this tree
// allocated from its own `allocator` field and never shares: no node
// is reachable from two trees, and every alias the crate creates
// (descents, iterators, entries) is borrow-bound, so none outlives a
// move. Nothing in the tree is tied to its birth thread; moving it
// moves the nodes' `K`/`V` payloads and the allocator along with it,
// which is exactly what the three `Send` bounds sign for.
unsafe impl<K, V, const M: usize, A> Send for BPlusTree<K, V, M, A>
where
    K: Key + Send,
    V: Send,
    A: NodeAllocator<K, V, M> + Send,
{
}

// SAFETY: sharing `&BPlusTree` shares a read-only tree. Every `&self`
// method is a pure read of the node graph — descents, gets, iteration;
// none mutates node memory through the `NonNull`s — and no `&self`
// path can reach the allocator: [`SlotAllocator`]'s receivers are
// `&mut self`, unreachable through a shared borrow, and nothing hands
// out `&A`. The tree itself has no interior mutability, so while
// shared borrows exist, no thread can write anything a reader
// dereferences. What readers DO reach — `&K`s and `&V`s — is what the
// `Sync` bounds sign for (`A: Sync` is defensive; no `&self` path
// reads it today).
//
// Every future `&self` feature re-signs this contract; a
// `&self`-written cache (the leaf-cache TODO atop this file) is the
// standing example of what would break it.
unsafe impl<K, V, const M: usize, A> Sync for BPlusTree<K, V, M, A>
where
    K: Key + Sync,
    V: Sync,
    A: NodeAllocator<K, V, M> + Sync,
{
}

/// Walk from the root down to a leaf, choosing one child per inner
/// level: the loop body runs exactly `$tree.height` times, so by the
/// depth-type invariant the node in hand is an [`Inner`] on every
/// iteration and a [`Leaf`] after the last — each cast inside is justified
/// by the tree layer's height invariant (see the impl block below).
/// `ref`/`mut` picks the borrow flavor; `$inner` names the current inner
/// node inside `$pick`, which must evaluate to the child to descend
/// into.
/// The recorded path of one root-to-leaf descent: for each inner level
/// `h`, `path[h]` holds the visited node and the child index routed
/// through. Slots outside the descended range stay uninitialized.
type TreePath<K, V, const M: usize> = [MaybeUninit<(NonNull<Node<K, V, M>>, usize)>; MAX_LEVELS];

macro_rules! descend {
    ($tree:expr, ref |$inner:ident| $pick:expr) => {{
        let mut node = &$tree.root;
        for _ in 0..$tree.height {
            // SAFETY: if we're in this loop, the node sits above the leaf
            // level (height invariant).
            let $inner = unsafe { node.as_inner() };
            node = $pick;
        }
        // SAFETY: `height` levels below the root is the leaf level.
        unsafe { node.as_leaf() }
    }};
    ($tree:expr, mut |$inner:ident| $pick:expr) => {{
        let mut node = &mut $tree.root;
        for _ in 0..$tree.height {
            // SAFETY: if we're in this loop, the node sits above the leaf
            // level (height invariant).
            let $inner = unsafe { node.as_inner_mut() };
            node = $pick;
        }
        // SAFETY: `height` levels below the root is the leaf level.
        unsafe { node.as_leaf_mut() }
    }};
}

/// One recorded descent for a key: the tree it walked, the [`path`](Descent::path) it
/// recorded, the leaf it landed at, and the key's slot there. The state
/// shared by the two commit halves ([`Descent::commit_insert`],
/// [`Descent::commit_remove`]) and, above them, the entry API
/// (`entry.rs`): search once, then mutate through what the search
/// recorded — no re-traversal.
///
/// # Validity
///
/// A descent is a bundle of raw pointers into the tree. It is valid only
/// under the `&mut` borrow of the tree that [`BPlusTree::descend_into`]
/// derived it from, and only until the tree is next mutated — a commit
/// half ends it (they take `&mut self` purely to avoid moving the
/// recorded path; a committed descent must not be used again). Until then, the tree may be reached through the
/// descent's pointers ONLY. In Stacked Borrows terms: every pointer here
/// descends from `tree`, and any fresh `&mut` retag of the tree — a
/// re-derived path from the root, or the tree passing a method boundary
/// by `&mut` — pops the whole family. That is why the commit halves live
/// on `Descent` (not on [`BPlusTree`], whose `&mut self` at the call
/// boundary would be exactly such a retag), and why holders carry the
/// originating borrow as `PhantomData<&'a mut BPlusTree>` rather than a
/// live reference field, which would be retagged at each of THEIR
/// method boundaries.
pub(crate) struct Descent<K: Key, V, const M: usize, A: NodeAllocator<K, V, M> = Global> {
    /// The descended tree, as the raw handle every other pointer here
    /// descends from. The commit halves reach the tree's own fields
    /// through this: the scalars (`height`, `len`) via raw projections
    /// disjoint from every recorded pointer's pointee, the `allocator`
    /// (shared, never written) likewise, and the `root` slot only after
    /// the path's last use.
    tree: NonNull<BPlusTree<K, V, M, A>>,
    /// The recorded path (see [`TreePath`]); slots `1..=tree.height` are
    /// initialized. Private to this module: only the commit halves
    /// replay it.
    path: TreePath<K, V, M>,
    /// The leaf the descent landed at.
    pub(crate) leaf: NonNull<Leaf<K, V, M>>,
    /// `key`'s slot in that leaf, per [`Leaf::find_key`]: the match if
    /// `exact`, else the insertion point.
    pub(crate) partition: usize,
    /// True iff `partition` holds exactly the sought key.
    pub(crate) exact: bool,
}

// The commit halves: the back ends of `insert` and `remove`, run
// against a recorded descent. They live on `Descent` rather than on
// `BPlusTree` deliberately — taking `&mut self` on the tree at this
// boundary would freshly retag it and invalidate every pointer the
// descent recorded (see `Descent`'s validity rules). The tree's own
// fields are reached through the descent's raw handle instead: the
// scalars (`height`, `len`) via projections disjoint from every
// recorded pointer's pointee, and the `root` slot only after the
// path's last use.
impl<K: Key + Ord, V, const M: usize, A: NodeAllocator<K, V, M>> Descent<K, V, M, A> {
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
    /// - The commit ends the descent: `self` must not be used again
    ///   after this call.
    // `&mut self`, not `self`, and the path read through it in place —
    // although a commit logically consumes the descent, taking it by
    // value compiles to a 1 KiB move at every call boundary (and
    // destructuring `path` out of it to another). `always` because LLVM
    // declines a plain `#[inline]` hint at this size.
    #[inline(always)]
    pub(crate) unsafe fn commit_insert(&mut self, key: K, val: V) -> NonNull<V> {
        let tree = self.tree.as_ptr();
        let partition = self.partition;
        debug_assert!(!self.exact, "commit_insert requires a vacant descent");

        // SAFETY: as the scalar-access note below — the allocator field is
        // disjoint from every recorded pointer's pointee, so this exclusive
        // reference aliases none of the node mutations below. It is also
        // the commit's only allocator borrow — every helper that allocates
        // or frees receives a reborrow of it — so its exclusivity holds
        // for its whole life.
        let alloc = unsafe { &mut (*tree).allocator };

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

        // If the root is deficient, that's okay. If it's a single entry, we
        // collapse.
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
}

impl<K: Key, V, const M: usize, A: NodeAllocator<K, V, M>> Drop for BPlusTree<K, V, M, A> {
    // Panic during teardown: NOT panic-safe. Teardown walks the tree dropping
    // values as it goes; a value `Drop` that unwinds leaks every node and
    // value not yet reached. Because this is itself `Drop` glue, such a panic
    // during an already-unwinding drop double-panics and aborts.
    fn drop(&mut self) {
        if const { Self::CAN_SKIP_DROP } {
            return;
        }

        // SAFETY:
        // - `root` is a node.
        // - `height` is maintained as the exact height of `root`'s subtree
        //   (it changes only where the root grows or shrinks).
        // - `root` is read exactly once and never touched again: the tree is
        //   mid-drop and `Node` has no drop glue of its own.
        // - The walk finishes before `self.allocator` itself drops — the
        //   values-first teardown order the allocator contract demands.
        unsafe { core::ptr::read(&self.root).drop_subtree(self.height, &mut self.allocator) }
    }
}

// The type-level invariant every method below signs: `height` is exactly
// the height of `root`'s subtree, and `len` is exactly the number of pairs
// in it. `height` changes in exactly two places — `insert`'s root grow and
// `remove`'s root shrink — and every unsafe `Node` call justifies its
// height argument by this invariant.
impl<K: Key + Ord, V, const M: usize, A: NodeAllocator<K, V, M>> BPlusTree<K, V, M, A> {
    /// A heuristic max height.
    pub const MAX_HEIGHT: usize = ((usize::BITS - 2) / M.div_ceil(2).ilog(2)) as usize;

    pub const CAN_SKIP_DROP: bool = !core::mem::needs_drop::<V>()
        && <A as SlotAllocator<Leaf<K, V, M>>>::OWNS_ALL
        && <A as SlotAllocator<Inner<K, V, M>>>::OWNS_ALL;

    /// Creates a tree whose root is a single empty leaf.
    pub fn new() -> Self
    where
        A: Default,
    {
        Self::new_in(A::default())
    }

    /// As [`Self::new`], but allocating nodes from `allocator` for the
    /// tree's whole life.
    pub fn new_in(mut allocator: A) -> Self {
        let root = Node::from_leaf_ptr(allocator.allocate(Leaf::new(None)));
        Self { root, height: 0, len: 0, allocator }
    }

    /// The number of key/value pairs in the tree.
    pub fn len(&self) -> usize {
        self.len
    }

    /// True if the tree holds no pairs.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get a reference to the first leaf of the tree.
    pub(crate) fn first_leaf(&self) -> &Leaf<K, V, M> {
        descend!(self, ref |inner| &inner.children_ref()[0])
    }

    /// Get a mutable reference to the first leaf of the tree.
    pub(crate) fn first_leaf_mut(&mut self) -> &mut Leaf<K, V, M> {
        descend!(self, mut |inner| &mut inner.children_mut()[0])
    }

    /// Get a reference to the last leaf of the tree.
    pub(crate) fn last_leaf(&self) -> &Leaf<K, V, M> {
        descend!(self, ref |inner| inner.children_ref().last().expect("no empty inner nodes"))
    }

    /// Find the leaf whose range contains the key. That leaf may or may not
    /// contain a value at that key
    pub(crate) fn find_leaf(&self, key: &K) -> &Leaf<K, V, M> {
        descend!(self, ref |inner| inner.child_for_key(key))
    }

    /// Find the leaf whose range contains the key. That leaf may or may not
    /// contain a value at that key
    pub(crate) fn find_leaf_mut(&mut self, key: &K) -> &mut Leaf<K, V, M> {
        descend!(self, mut |inner| inner.child_for_key_mut(key))
    }

    /// Get a reference to the value for `key`, if it is present.
    pub fn get(&self, key: &K) -> Option<&V> {
        self.find_leaf(key).get(key)
    }

    /// Get a mutable reference to the value for `key`, if it is present.
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        self.find_leaf_mut(key).get_mut(key)
    }

    /// True if `key` is present.
    pub fn contains_key(&self, key: &K) -> bool {
        self.get(key).is_some()
    }

    /// Descend from the root to the leaf whose range contains `key`,
    /// recording the path — each visited inner node and the child index
    /// routed through — in `path[h]` for every inner level `h`, and
    /// returning the leaf. The shared front half of [`insert`](Self::insert) (which
    /// replays the path upward to adopt splits) and [`remove`](Self::remove) (which
    /// replays it to rebalance whatever the removal left deficient);
    /// levels at and below the leaf are left untouched.
    #[inline]
    #[track_caller]
    fn descend_recording(
        &mut self,
        key: &K,
        path: &mut TreePath<K, V, M>,
    ) -> NonNull<Node<K, V, M>> {
        let mut height = self.height as usize;
        let mut node = NonNull::from_mut(&mut self.root);

        while height > 0 {
            // SAFETY: height is non-0, the pointer is valid. Root is always ok.
            let n = unsafe { node.as_mut().as_inner_mut() };
            let child_idx = n.child_idx_for_key(key);
            path[height].write((node, child_idx));

            node = NonNull::from_mut(&mut n.children_mut()[child_idx]);

            height -= 1;
        }

        node
    }

    /// As [`Self::descend_recording`], but building the descent in caller-provided
    /// storage — the way in for the plain ops. Returning a [`Descent`] by
    /// value compiles to a 1 KiB copy per call (Rust guarantees no NRVO),
    /// which is fine for the entry API, whose entry stores the descent by
    /// value anyway, and pure loss for [`insert`](Self::insert)/[`remove`](Self::remove), which commit it
    /// on the spot.
    // `always` because LLVM declines a plain `#[inline]` hint at this
    // size, and out of line every caller pays call-boundary stack
    // traffic for the slot.
    #[inline(always)]
    #[track_caller]
    pub(crate) fn descend_into<'s>(
        &mut self,
        key: &K,
        slot: &'s mut MaybeUninit<Descent<K, V, M, A>>,
    ) -> &'s mut Descent<K, V, M, A> {
        let mut tree = NonNull::from_mut(self);

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
        let mut node = unsafe { tree.as_mut() }.descend_recording(key, &mut descent.path);

        // SAFETY: `descend_recording` ran the descent to the bottom, so
        // the node in hand is the leaf (height invariant).
        let leaf = unsafe { node.as_mut().as_leaf_mut() };
        (descent.partition, descent.exact) = leaf.probe(key);
        descent.leaf = NonNull::from_mut(leaf);

        descent
    }

    /// Insert a key-value pair, returning the previous value if the key was
    /// already present.
    pub fn insert(&mut self, key: K, val: V) -> Option<V> {
        let mut slot = MaybeUninit::uninit();
        let descent = self.descend_into(&key, &mut slot);

        if descent.exact {
            // SAFETY:
            //
            // `descent.exact`` is set
            return Some(unsafe { descent.commit_replace(val) });
        }

        // SAFETY: the descent is fresh from `descend` under this borrow,
        // the tree untouched since, and `exact` is false.
        unsafe { descent.commit_insert(key, val) };
        None
    }

    /// Remove `key`, returning its value if it was present.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        let mut slot = MaybeUninit::uninit();
        let descent = self.descend_into(key, &mut slot);
        if !descent.exact {
            return None;
        }

        // SAFETY: the descent is fresh from `descend` under this borrow,
        // the tree untouched since, and `exact` is true.
        Some(unsafe { descent.commit_remove() }.1)
    }

    /// Drop every pair, resetting to the empty tree.
    pub fn clear(&mut self) {
        if const { Self::CAN_SKIP_DROP } {
            // SAFETY:
            // A::OWNS_ALL is checked by CAN_SKIP_DROP
            // We immediately invalidate the tree by overwriting the root
            // We kinow that V has no drop glue (checked by CAN_SKIP_DROP
            unsafe {
                SlotAllocator::<Leaf<K, V, M>>::clear_all(&mut self.allocator);
                SlotAllocator::<Inner<K, V, M>>::clear_all(&mut self.allocator);
            }
        } else {
            // SAFETY:
            // - `height` is the impl-block invariant — exactly the
            // height of `root`'s subtree. `root` is overwritten with a fresh
            // node immediately below, before anything can read it.
            unsafe {
                let tree = core::ptr::read(&self.root);
                tree.drop_subtree(self.height, &mut self.allocator);
            }
        }

        self.root = Node::from_leaf_ptr(self.allocator.allocate(Leaf::new(None)));
        self.height = 0;
        self.len = 0;
    }

    /// Assemble a tree directly from its parts — the bulk loader's
    /// (`bulk.rs`) way in, since the fields are private to this module.
    ///
    /// # Safety
    ///
    /// The caller signs this impl block's invariant: `height` must be
    /// exactly the height of `root`'s subtree, and `len` exactly the
    /// number of pairs in it. A wrong height reinterprets node types
    /// throughout the tree (see [`Node`]); a wrong `len` misreports but
    /// is not unsound. Additionally, every node of `root`'s subtree must
    /// have been allocated from `allocator`.
    pub(crate) unsafe fn from_parts(
        root: Node<K, V, M>,
        height: u8,
        len: usize,
        allocator: A,
    ) -> Self {
        Self { root, height, len, allocator }
    }

    /// Iterate over all KV pairs.
    pub fn iter<'a>(&'a self) -> iter::FullIterator<'a, K, V, M> {
        iter::FullIterator::new(self)
    }

    /// Iterate over the pairs whose keys fall in `range`, in ascending
    /// key order.
    pub fn range<'a, R: core::ops::RangeBounds<K>>(&'a self, range: R) -> iter::Range<'a, K, V, M> {
        iter::Range::new(self, range)
    }

    /// Iterate over key/value pairs with mutable values.
    pub fn iter_mut<'a>(&'a mut self) -> iter::FullIteratorMut<'a, K, V, M> {
        iter::FullIteratorMut::new(self)
    }

    /// Iterate over the pairs whose keys fall in `range`, in ascending
    /// key order, with mutable values.
    pub fn range_mut<'a, R: core::ops::RangeBounds<K>>(
        &'a mut self,
        range: R,
    ) -> iter::RangeMut<'a, K, V, M> {
        iter::RangeMut::new(self, range)
    }

    /// The minimum-key pair, or `None` if the tree is empty.
    pub fn first_key_value(&self) -> Option<(&K, &V)> {
        // Only the empty tree's root leaf is empty; any other first
        // leaf holds its subtree's minimum at index 0.
        let leaf = self.first_leaf();
        (leaf.len() > 0).then(|| leaf.kv_ref_unchecked(0))
    }

    /// The maximum-key pair, or `None` if the tree is empty.
    pub fn last_key_value(&self) -> Option<(&K, &V)> {
        let leaf = self.last_leaf();
        leaf.len().checked_sub(1).map(|last| leaf.kv_ref_unchecked(last))
    }

    /// Iterate over the keys, in ascending order.
    pub fn keys(&self) -> impl core::iter::Iterator<Item = &K> {
        self.iter().map(|(k, _)| k)
    }

    /// Iterate over the values, in ascending key order.
    pub fn values(&self) -> impl core::iter::Iterator<Item = &V> {
        self.iter().map(|(_, v)| v)
    }

    /// Iterate over the values mutably, in ascending key order.
    pub fn values_mut(&mut self) -> impl core::iter::Iterator<Item = &mut V> {
        self.iter_mut().map(|(_, v)| v)
    }
}

impl<K: Key + Ord, V, const M: usize, A: NodeAllocator<K, V, M> + Default> Default
    for BPlusTree<K, V, M, A>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V, const M: usize, A: NodeAllocator<K, V, M>> core::fmt::Debug for BPlusTree<K, V, M, A>
where
    K: Key + core::fmt::Debug,
    V: core::fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl<K: Key + Ord, V, const M: usize, A: NodeAllocator<K, V, M> + Default> FromIterator<(K, V)>
    for BPlusTree<K, V, M, A>
{
    /// Builds through the bulk loader, not an insert loop: collect, sort,
    /// dedup, then [`BPlusTree::from_sorted_iter`]. The loaded tree is
    /// fully packed (every leaf at `M` pairs, up to the tail) where
    /// repeated [`insert`](Self::insert)s settle around ~2/3 occupancy — denser and
    /// shallower for the same pairs.
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let mut pairs: alloc::vec::Vec<(K, V)> = iter.into_iter().collect();
        // Stable sort, so duplicate keys stay in arrival order for the
        // dedup below.
        pairs.sort_by_key(|pair| pair.0);
        // `from_sorted_iter` demands strictly ascending keys. Collapse
        // each duplicate run to one pair with `insert`'s overwrite
        // semantics — the first-arrived key, the last-arrived value.
        // (`dedup_by` keeps the first element of a run and passes the
        // later one on the LEFT; the swap walks the newest value into
        // the kept slot.)
        pairs.dedup_by(|later, kept| {
            let dup = later.0 == kept.0;
            if dup {
                core::mem::swap(&mut later.1, &mut kept.1);
            }
            dup
        });
        Self::from_sorted_iter(pairs)
    }
}

impl<K: Key + Ord, V, const M: usize, A: NodeAllocator<K, V, M>> Extend<(K, V)>
    for BPlusTree<K, V, M, A>
{
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        for (key, val) in iter {
            self.insert(key, val);
        }
    }
}

// The testutils surface (the differential harness plus the invariant
// net and its node-level views) lives under this module so it can reach
// the tree's private fields; `lib.rs` re-exports it as `crate::harness`
// for the in-crate tests and, under the `testutils` feature, the fuzz
// targets.
#[cfg(any(test, feature = "testutils"))]
#[path = "tests/harness.rs"]
pub mod harness;

#[cfg(test)]
#[path = "tests/tree.rs"]
mod tests;
