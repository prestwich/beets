//! Bulk-loading: build a tree from a pre-sorted stream in one pass.
//!
//! [`BPlusTree::from_sorted_iter`] lives here with its scaffolding
//! ([`LevelState`]). The loader is the one code path that assembles a tree
//! bottom-up rather than mutating one top-down.
//!
//! How this works:
//! - Each level contains
//!     - An in-progress node
//!     - A complete node.
//!
//! When an in-progress node fills, it is promoted to complete.
//! If that displaces a complete node, that node is pushed into the in-progress
//! node at the next layer up.
//!
//! When the value stream terminates, any deficient (see
//! [`Node::is_deficient`]) in-progress nodes borrow from the complete node,
//! which is always their left sibling. This ensures the resulting tree is
//! tightly left-packed at every level, and no deficient right nodes exist.

use core::{
    cell::RefCell,
    ops::{Index, IndexMut},
    ptr::NonNull,
};
use core::convert::Infallible;

use crate::{BPlusTree, DEFAULT_MAX_LEVELS, Inner, Key, Leaf, Node, allocator::NodeAllocator};

/// A [`Leaf`] node that has been constructed but not yet enrolled in any tree.
///
/// This type exists to ensure leaves cannot be leaked by panics during bulk
/// loading: it holds (a shared handle to) the allocator the leaf came
/// from, so its [`Drop`] can return the slot while the loader goes on
/// allocating through the same allocator.
pub(crate) struct Unyielded<'a, K: Key, V, const M: usize, A: NodeAllocator<K, V, M>>(
    NonNull<Leaf<K, V, M>>,
    &'a RefCell<A>,
);

impl<K: Key, V, const M: usize, A: NodeAllocator<K, V, M>> Drop for Unyielded<'_, K, V, M, A> {
    fn drop(&mut self) {
        // SAFETY: the pending is used only during from_fn iter construction
        // and is known to be totally owned while it exists; its slot came
        // from the held allocator.
        drop(unsafe { self.1.borrow_mut().dealloc_leaf(self.0) })
    }
}

impl<'a, K: Key, V, const M: usize, A: NodeAllocator<K, V, M>> Unyielded<'a, K, V, M, A> {
    /// Create a new [`Unyielded`] by moving a leaf into `alloc`.
    fn leaking_new(leaf: Leaf<K, V, M>, alloc: &'a RefCell<A>) -> Self {
        // SAFETY: sorted-load surface — infallible allocators cannot
        // exhaust, and the fallible load (`try_from_sorted_iter_in`)
        // pre-flights before drawing.
        Self(unsafe { alloc.borrow_mut().alloc_leaf_unchecked(leaf) }, alloc)
    }

    /// Convert into a [`Node`]. This is the intended way to pass off ownership
    /// of the inner data.
    ///
    /// [`ManuallyDrop`](core::mem::ManuallyDrop) defuses the guard: exactly one of the two exits —
    /// this transfer or [`Drop`] — may reclaim the leaf.
    fn into_node(self) -> Node<K, V, M> {
        let this = core::mem::ManuallyDrop::new(self);
        Node::from_leaf_ptr(this.0)
    }

    /// Shared view of the held leaf.
    fn as_ref(&self) -> &Leaf<K, V, M> {
        // SAFETY: the guard holds the leaf's only handle for its whole
        // lifetime (leaked at birth, sole owner until `into_node` or
        // drop), so borrowing through the pointer at `&self`'s lifetime
        // is sound.
        unsafe { self.0.as_ref() }
    }

    /// Exclusive view of the held leaf.
    fn as_mut(&mut self) -> &mut Leaf<K, V, M> {
        // SAFETY: as `as_ref`, and `&mut self` makes the borrow unique.
        unsafe { self.0.as_mut() }
    }

    /// Set the held leaf's sibling link to another un-yielded leaf.
    fn set_next(&mut self, next: &Self) {
        self.as_mut().set_next(Some(next.0));
    }
}

/// One inner level's in-progress state during a bulk load
/// ([`BPlusTree::from_sorted_iter`]): the chunk currently filling, and the
/// completed chunk held back behind it.
///
/// The hold-back is the load's timing rule, shared with
/// [`Leaf::drain_sorted_iter`]: a node moves up to its parent level only
/// once a full successor chunk exists, so the stream's short final chunk
/// always finds its left neighbor still here — un-emitted and exactly `M`
/// wide — to borrow from. Each held pointer is an allocator slot the
/// treepath owns until the node is emitted upward or crowned the root.
#[derive(Clone, Copy)]
struct LevelState<K: Key, V, const M: usize> {
    /// The held-back completed chunk, always exactly `M` children wide.
    pending: Option<(K, NonNull<Inner<K, V, M>>)>,
    /// The chunk currently filling: `1..M` children.
    current: Option<(K, NonNull<Inner<K, V, M>>)>,
}

impl<K: Key, V, const M: usize> Default for LevelState<K, V, M> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Key, V, const M: usize> LevelState<K, V, M> {
    const fn new() -> Self {
        Self { pending: None, current: None }
    }
}

struct TreeProgress<
    'a,
    K: Key,
    V,
    const M: usize,
    A: NodeAllocator<K, V, M>,
    const H: usize = DEFAULT_MAX_LEVELS,
> {
    states: [LevelState<K, V, M>; H],
    len: usize,
    /// The allocator every chunk comes from — a shared handle, so the
    /// unwind `Drop` below can free through it while the loader (and
    /// the leaf guards) go on using the same allocator; each access
    /// takes a transient `borrow_mut`.
    alloc: &'a RefCell<A>,
}

impl<Idx, K: Key, V, const M: usize, A: NodeAllocator<K, V, M>, const H: usize> Index<Idx>
    for TreeProgress<'_, K, V, M, A, H>
where
    [LevelState<K, V, M>; H]: Index<Idx>,
{
    type Output = <[LevelState<K, V, M>; H] as Index<Idx>>::Output;

    fn index(&self, index: Idx) -> &Self::Output {
        &self.states[index]
    }
}

impl<Idx, K: Key, V, const M: usize, A: NodeAllocator<K, V, M>, const H: usize> IndexMut<Idx>
    for TreeProgress<'_, K, V, M, A, H>
where
    [LevelState<K, V, M>; H]: IndexMut<Idx>,
{
    fn index_mut(&mut self, index: Idx) -> &mut Self::Output {
        &mut self.states[index]
    }
}

impl<K: Key, V, const M: usize, A: NodeAllocator<K, V, M>, const H: usize> Drop
    for TreeProgress<'_, K, V, M, A, H>
{
    /// Unwind teardown. The source iterator is caller code and runs
    /// between every pair, so a load can panic while the treepath holds
    /// chunks — each one the sole handle to a live subtree of leaves
    /// full of values. Dropping the treepath reclaims them: a level-`h`
    /// chunk roots a subtree of height `h + 1` (leaves sit at height
    /// 0), and `take()`ing each slot makes a double-drop structurally
    /// impossible.
    ///
    /// The success path stays free: `build` `take()`s every slot it
    /// emits and never touches levels above the root, so a completed
    /// load reaches this drop with every slot `None` — the integration
    /// pin that a load frees nothing still holds.
    fn drop(&mut self) {
        for (h, state) in self.states.iter_mut().enumerate() {
            let h = (h + 1) as u8;
            if let Some((_sep, node)) = state.current.take() {
                // SAFETY: `take` empties the slot, so this is the sole
                // surviving handle (leaked at birth, never emitted); a
                // level-`h` (0-indexed) chunk roots a subtree of height
                // `h + 1`; every node in it came from `self.alloc`.
                unsafe {
                    Node::from_inner_ptr(node).drop_subtree(h, &mut *self.alloc.borrow_mut());
                }
            }
            if let Some((_sep, node)) = state.pending.take() {
                // SAFETY: as `current` above — sole handle, height
                // `h + 1`, same allocator.
                unsafe {
                    Node::from_inner_ptr(node).drop_subtree(h, &mut *self.alloc.borrow_mut());
                }
            }
        }
    }
}

impl<'a, K: Key, V, const M: usize, A: NodeAllocator<K, V, M>, const H: usize>
    TreeProgress<'a, K, V, M, A, H>
{
    const __LEVEL_CAP: () =
        assert!(H >= 1 && H <= usize::BITS as usize, "the level cap H must be in 1..=usize::BITS");

    fn new(alloc: &'a RefCell<A>) -> Self {
        const { Self::__LEVEL_CAP };

        Self { states: [const { LevelState::new() }; H], len: 0, alloc }
    }

    /// Push one `(subtree-min key, node)` pair into the state at level
    /// `height`, carrying upward: the pair lands in the level's filling chunk,
    /// and when that chunk completes (its `M`th child) it displaces the
    /// level's held-back node, which climbs to level `h + 1` — [`insert`](BPlusTree::insert)'s
    /// split cascade, inverted. Amortized O(1): level `h` is touched once
    /// per `Mʰ` pushes.
    ///
    /// Panics — leak-free, see the assert below — if the carry would
    /// open level `H - 1`: the level-cap contract, under which the
    /// finished tree may occupy at most `H` levels.
    fn push(&mut self, mut height: usize, mut key: K, mut node: Node<K, V, M>) {
        // Copied out (`&'a RefCell<A>` is `Copy`) so the level borrow
        // below and the allocator can be used together.
        let alloc = self.alloc;
        if height == 0 {
            // SAFETY: The node is valid and owned by us. The height is 0.
            // Reading its len is sound.
            self.len += unsafe { node.len(0) };
        }
        loop {
            let lvl = &mut self[height];

            let Some((_, current)) = &mut lvl.current else {
                // If there is no current `Inner` at this layer, we make a new
                // `Inner` containing the incoming node as its first child.
                lvl.current =
                    // SAFETY: sorted-load surface — infallible
                    // allocators cannot exhaust, and the fallible load
                    // (`try_from_sorted_iter_in`) pre-flights before
                    // drawing.
                    Some((key, unsafe {
                        alloc.borrow_mut().alloc_inner_unchecked(Inner::from_first_child(node))
                    }));
                // Occupying level `H - 1` would stand the finished tree at `H
                // + 1` levels. This would
                assert!(
                    height < H - 1,
                    "bulk load would exceed the level cap H. HINT: the `H` paramater on the tree is not sufficient to hold the number of leaves you're loading."
                );
                return;
            };

            // Otherwise we're going to append the incoming node to the current
            // inner.

            // SAFETY: `current` was leaked at birth and not yet emitted —
            // this treepath slot holds the only handle to it.
            let current = unsafe { current.as_mut() };
            current.raw_append_child(key, node);

            // If the current inner is not full, we're done.
            if current.len() < M {
                return;
            }

            // Now the inner node is full. So we move it to the "completed"
            // reserve.
            let full = lvl.current.take();
            let Some((up_key, ptr)) = core::mem::replace(&mut lvl.pending, full) else {
                return;
            };

            // So we'll bump it up and run this loop again.
            key = up_key;
            node = Node::from_inner_ptr(ptr);
            height += 1;
        }
    }

    // Traverse our state, ensuring that the tail of each level is not
    // deficient. If it is deficient, we rotate from the complete node at that
    // level. If there is no complete node, the deficient node is the root.
    //
    // Our completion states are either:
    // - We reach a level with an in-progress and no complete node.
    // - We reach a level with neither an in-progress or complete node.
    //
    // Returns the assembled tree's parts (root, height, len) rather than
    // the tree itself: the caller owns the allocator this treepath only
    // borrows, and moves it into `BPlusTree::from_parts` once this borrow
    // ends. The parts contract is `from_parts`'s, checked at the two
    // return sites below.

    fn build(mut self) -> (Node<K, V, M>, u8, usize) {
        let mut h = 0;

        loop {
            let lvl = &mut self[h];
            if let (Some((_, donor)), Some((tail_key, tail))) = (&mut lvl.pending, &mut lvl.current)
            {
                // Short tail: only the level's LAST chunk can sit below
                // MIN_OCCUPANCY, and the held-back donor before it is a
                // full `M` children. Rotate children in from it until
                // the tail meets the occupancy invariant —
                // `rotate_from_left`'s contract lines up exactly: the
                // separator between the two IS the tail's up-key, and
                // each promotion IS its replacement (the moved child's
                // subtree minimum). At most `MIN_OCCUPANCY - 1`
                // rotations leave the donor at `⌊M/2⌋ + 1 >=
                // MIN_OCCUPANCY`, so both nodes end legal (M >= 3).
                // SAFETY: this treepath slot holds the only handle to its
                // live node (leaked at birth, not yet emitted).
                let donor = unsafe { donor.as_mut() };
                // SAFETY: as `donor`; a distinct slot and node, so the
                // two exclusive borrows are disjoint.
                let tail = unsafe { tail.as_mut() };
                while tail.is_deficient() {
                    // SAFETY: both treepath slots hold the only handles to
                    // their nodes (leaked at birth, not yet emitted).
                    // `donor` and `tail` are adjacent chunks of one
                    // sorted stream — everything under `donor` is
                    // < `tail_key`, everything under `tail` is
                    // >= `tail_key` — rooting equal-height subtrees, and
                    // the arithmetic above keeps `tail` short of `M` and
                    // the donor at 2 or more children.
                    *tail_key = unsafe { tail.rotate_from_left(*tail_key, donor) };
                }
            }

            // In-bounds by the level cap: occupied levels stop at `H - 2`
            // (the refusals here and in `push`), so `h + 1` tops out at
            // `H - 1` — the top slot never holds a chunk and serves as
            // the fold's always-empty lookahead.
            let above_untouched = self[h + 1].pending.is_none() && self[h + 1].current.is_none();
            let lvl = &mut self[h];
            // The fold's level-cap refusal, BEFORE the level's chunks move
            // into locals: residue that must climb into level `h + 1`
            // roots the tree at height `h + 2` or more. Refusing while
            // every chunk still sits in the treepath lets its `Drop`
            // reclaim them on unwind — `push`'s own refusal would fire
            // one chunk too late, with the already-taken sibling
            // unwinding as a raw, glue-free handle: a leak.
            let will_climb = lvl.pending.is_some() && (lvl.current.is_some() || !above_untouched);
            assert!(!will_climb || h + 1 < H - 1, "bulk load would exceed the level cap H");
            let pending = lvl.pending.take();
            let current = lvl.current.take();
            match (pending, current) {
                // The level's lone node is the root. It never filled, so
                // it never carried — nothing can live above it.
                (None, Some((_, only))) => {
                    debug_assert!(above_untouched, "a level below the top cannot be single-node");
                    // Emitted with >= 2 children: level 0 starts from the
                    // two probed leaves, and a higher level only exists
                    // once >= 2 nodes have been pushed into it.
                    // SAFETY: the slot holds the sole handle to a live node.
                    debug_assert!(unsafe { only.as_ref() }.len() >= 2);
                    // A level-`h` inner roots a subtree of height `h + 1`
                    // (its leaves all sit at height 0), and `len` is the
                    // tally of every drained pair.
                    return (Node::from_inner_ptr(only), h as u8 + 1, self.len);
                }
                // Exactly one full chunk and nothing above: the root.
                (Some((_, only)), None) if above_untouched => {
                    // As above — a level-`h` inner roots a subtree of
                    // height `h + 1`, and `len` is exact.
                    return (Node::from_inner_ptr(only), h as u8 + 1, self.len);
                }
                // Two or more nodes on this level (counting those already
                // carried up): emit the residue and fold the next level.
                (Some((pk, p)), current) => {
                    self.push(h + 1, pk, Node::from_inner_ptr(p));
                    if let Some((ck, c)) = current {
                        self.push(h + 1, ck, Node::from_inner_ptr(c));
                    }
                    h += 1;
                }
                // Level 0 was seeded with the two probed leaves, and the
                // fold only climbs into levels it has emitted into.
                (None, None) => unreachable!("the fold visits only populated levels"),
            }
        }
    }
}

impl<K: Key + Ord, V, const M: usize, A: NodeAllocator<K, V, M>, const H: usize>
    BPlusTree<K, V, M, A, H>
{
    /// Bulk-load a tree from a stream of pairs sorted strictly ascending
    /// by key, in one pass: chunk the pairs into a linked leaf chain,
    /// then build each level of inners over the one below, bottom-up,
    /// until a level is a single node — the root.
    ///
    /// The whole load is streaming and allocates nothing beyond the
    /// tree's own nodes. Live memory is O(M · H).
    ///
    /// # Panics
    ///
    /// Panics if the loaded tree would occupy more than `H` levels —
    /// the level cap (see [`BPlusTree`]'s type-level docs). The refusal
    /// unwinds cleanly: every value already drawn from the stream is
    /// dropped exactly once.
    ///
    /// Debug builds assert that the keys are strictly ascending (no
    /// duplicates). Release builds skip the checks and quietly builds an
    /// incorrect tree that misroutes lookups.
    pub fn from_sorted_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self
    where
        A: Default + NodeAllocator<K, V, M, Exhaustion = Infallible>,
    {
        Self::from_sorted_iter_in(iter, A::default())
    }

    /// As [`Self::from_sorted_iter_in`], but reporting allocator
    /// exhaustion instead of panicking. On `Err`, every node built so
    /// far has been torn down through the loader's guard machinery,
    /// every pair already drawn from `iter` has been dropped (a
    /// streaming load cannot hand them back), and the emptied allocator
    /// comes back to the caller.
    ///
    /// # Panics
    ///
    /// As [`Self::from_sorted_iter_in`]: the `H`-level cap still panics
    /// — that is a structural refusal, not an allocation failure.
    pub fn try_from_sorted_iter_in<I: IntoIterator<Item = (K, V)>>(
        iter: I,
        allocator: A,
    ) -> Result<Self, A> {
        // Level 0: chunk the pairs into the linked leaf chain. Sibling
        // links and the leaf-level tail repair live in the adapter;
        // `len` is tallied leaf by leaf as they stream out. The loader
        // machinery borrows `allocator`; each early return below first
        // ends the borrows (the guards via `into_node`, the adapter via
        // `drop`) and then moves `allocator` into the finished tree.
        //
        // The `RefCell` reconciles the trait's `&mut self` receivers
        // with the loader's sharing: the adapter, its yielded guards,
        // and the treepath all hold the allocator at once, and each
        // access takes a transient `borrow_mut`. Borrow discipline —
        // what keeps the flag from ever being contended: no borrow is
        // held while the caller's iterator runs (the only code that can
        // unwind mid-load), accesses never nest (nothing inside an
        // allocator call touches the cell), and the unwind `Drop`s run
        // strictly one at a time. `into_inner` at each return needs no
        // borrow bookkeeping: it consumes the cell, so the borrow
        // checker itself proves every guard is gone by then.
        let allocator = RefCell::new(allocator);

        // The drain never yields an empty leaf, so `first_key` cannot
        // panic. (The tuples below are destructured in full: a guard left
        // partially moved would keep its borrow of `allocator` alive in
        // dropck's eyes, blocking the moves at the returns.)
        let mut leaves = Leaf::drain_sorted_iter(iter.into_iter(), &allocator)
            .map(|leaf| (*leaf.as_ref().first_key(), leaf));

        // If there is no first leaf, then return an empty tree
        let Some((first_key, first_leaf)) = leaves.next() else {
            drop(leaves);
            return Self::try_new_in(allocator.into_inner());
        };

        // If there's no second leaf, return the first leaf as the root node.
        let Some((second_key, second_leaf)) = leaves.next() else {
            let len = first_leaf.as_ref().len();
            let root = first_leaf.into_node();
            drop(leaves);
            // SAFETY: a lone leaf roots a height-0 tree, `len` is the
            // tally of every drained pair, and the leaf came from
            // `allocator`.
            return unsafe { Ok(Self::from_parts(root, 0, len, allocator.into_inner())) };
        };

        // Stream every leaf into the treepath's level 0;
        // carries climb on their own as chunks fill (see `TreeProgress::push`).
        let mut tree: TreeProgress<'_, K, V, M, A, H> = TreeProgress::new(&allocator);

        tree.push(0, first_key, first_leaf.into_node());
        tree.push(0, second_key, second_leaf.into_node());
        for (key, node) in leaves {
            tree.push(0, key, node.into_node());
        }

        let (root, height, len) = tree.build();
        // SAFETY: `build` returns exactly `from_parts`'s contract (see its
        // return sites), and every node was allocated from `allocator`.
        unsafe { Ok(Self::from_parts(root, height, len, allocator.into_inner())) }
    }

    /// As [`Self::from_sorted_iter`], but allocating nodes from
    /// `allocator` for the tree's whole life.
    // TODO(scaffold): becomes the thin panicking wrapper over
    // `try_from_sorted_iter_in` once that lands.
    pub fn from_sorted_iter_in<I: IntoIterator<Item = (K, V)>>(iter: I, allocator: A) -> Self
    where
        A: NodeAllocator<K, V, M, Exhaustion = Infallible>,
    {
        match Self::try_from_sorted_iter_in(iter, allocator) {
            Ok(tree) => tree,
            Err(_) => unreachable!(),
        }
    }
}

impl<K: Key, V, const M: usize> Leaf<K, V, M> {
    /// Drain a sorted iterator of items into a chain of leaves.
    ///
    /// Chunks the items into full leaves of `M` pairs. Occupancy: when
    /// several leaves are yielded, every one holds `MIN_OCCUPANCY..=M`
    /// pairs — a short tail is repaired by borrowing from its left
    /// neighbor before either is yielded. A lone yielded leaf may hold
    /// fewer (down to 0) — legal only for the root, which is what a lone
    /// leaf is about to become.
    ///
    /// Sibling links are set here: each leaf's `next` points at the leaf
    /// yielded after it, and the final leaf's `next` is `None`. Each
    /// yielded leaf lives in a slot of `alloc` that the caller now owns.
    ///
    /// The caller MUST ensure the items are strictly sorted by key (no
    /// duplicates).
    pub(crate) fn drain_sorted_iter<'a, I, A>(
        mut iter: I,
        alloc: &'a RefCell<A>,
    ) -> impl Iterator<Item = Unyielded<'a, K, V, M, A>> + use<'a, I, K, V, M, A>
    where
        I: Iterator<Item = (K, V)>,
        A: NodeAllocator<K, V, M>,
    {
        // One leaf of delay: a leaf is yielded only once its successor
        // exists (or the source is exhausted), so its `next` is already
        // final and the caller never observes a link mutate behind a
        // pointer it holds.
        let mut pending: Option<Unyielded<'a, K, V, M, A>> = None;
        core::iter::from_fn(move || {
            // Loops at most twice: only the very first chunk finds no
            // pending leaf to yield.
            loop {
                let mut leaf_contents = iter.by_ref().take(M);

                // Source exhausted: flush the pending leaf, whose
                // `next` stays `None`. `this` was never leaked and
                // drops here, so further calls allocate nothing and
                // keep returning `None`.
                let Some((key, val)) = leaf_contents.next() else {
                    return pending.take();
                };

                // empty the remaining data into the Leaf
                let this = Self::from_first_item(None, key, val);
                let mut this = leaf_contents.fold(this, |mut acc, (k, v)| {
                    acc.raw_append(k, v);
                    acc
                });

                if this.is_deficient()
                    && let Some(prev) = pending.as_mut()
                {
                    // Short tail: a non-full chunk with a predecessor can
                    // only be the last, and its predecessor is a full `M`
                    // pairs. Steal from it until the tail meets the
                    // occupancy invariant; at most `MIN_OCCUPANCY - 1`
                    // steals leave the donor at `⌊M/2⌋ + 1 >=
                    // MIN_OCCUPANCY`, so both leaves end legal (M >= 3).
                    //
                    let prev = prev.as_mut();
                    while this.is_deficient() {
                        // SAFETY: the stream ordering puts every key in
                        // `prev` below every key in `this`, and the
                        // arithmetic above keeps the donor strictly
                        // above minimum for every steal.
                        unsafe { this.steal_from_left(prev) };
                    }
                }

                let this = Unyielded::leaking_new(this, alloc);

                if let Some(mut prev) = pending.replace(this) {
                    prev.set_next(pending.as_ref().expect("replace ensures populated"));
                    return Some(prev);
                }
            }
        })
    }
}

#[cfg(test)]
#[path = "../tests/bulk.rs"]
mod tests;
