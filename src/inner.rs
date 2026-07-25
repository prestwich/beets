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

    /// Test-only constructor: assemble an inner from `children` and the
    /// `children.len() - 1` separators between them, so fixtures in other
    /// modules' tests (the fields are private to this one) can build
    /// multi-level trees.
    #[cfg(test)]
    pub(crate) fn test_from_parts(keys: Vec<K>, children: Vec<Node<K, V, M>>) -> Self {
        assert!((2..=M).contains(&children.len()), "an inner node holds 2..=M children");
        assert_eq!(keys.len() + 1, children.len(), "n children need n - 1 separators");

        let mut node = Self::new();
        node.child_count = children.len();
        for (i, child) in children.into_iter().enumerate() {
            node.children[i].write(child);
        }
        for (i, k) in keys.into_iter().enumerate() {
            node.keys[i].write(k);
        }
        node
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

    /// Test-only views for invariant checking from other modules' tests.
    #[cfg(test)]
    pub(crate) fn test_keys(&self) -> &[K] {
        self.keys_ref()
    }

    #[cfg(test)]
    pub(crate) fn test_children(&self) -> &[Node<K, V, M>] {
        self.children_ref()
    }

    fn keys_ref(&self) -> &[K] {
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

    /// Get a reference to a value in the subtree rooted at this inner node,
    /// if it is present: route to the child owning `key` and recurse at
    /// `height - 1`.
    ///
    /// Test-only — the production descent is iterative (`tree.rs`);
    /// this module's tests assert reachability through it.
    ///
    /// # Safety
    ///
    /// `height` must be the true height of the subtree rooted at this node
    /// (necessarily > 0: this node is an `Inner`; the child it routes to
    /// roots a subtree of `height - 1`).
    #[cfg(test)]
    pub(crate) unsafe fn get(&self, height: u8, key: &K) -> Option<&V> {
        let child = self.child_for_key(key);
        // SAFETY: height propagation.
        unsafe { child.get(height - 1, key) }
    }

    /// Remove `key` from the subtree rooted at this inner node, if present:
    /// route to the child owning `key` and recurse at `height - 1`.
    /// Recursive remove: route to the child, then rebalance it if the
    /// removal left it deficient.
    ///
    /// Test-only — the production descent is iterative (`tree.rs`);
    /// the node-layer tests drive `rebalance` through it.
    ///
    /// # Safety
    ///
    /// `height` must be the true height of the subtree rooted at this node
    /// (necessarily > 0: this node is an `Inner`; the child it routes to
    /// roots a subtree of `height - 1`).
    #[cfg(test)]
    #[track_caller]
    pub(crate) unsafe fn remove<A: NodeAllocator<K, V, M>>(
        &mut self,
        height: u8,
        key: &K,
        alloc: &mut A,
    ) -> Option<V> {
        debug_assert!(height > 0);
        // Route once and reuse the index — `child_for_key_mut` would
        // repeat the identical search.
        let child_idx = self.child_idx_for_key(key);
        let child = &mut self.children_mut()[child_idx];
        // SAFETY:
        // Same safety assumptions documented on the function.
        let val = unsafe { child.remove(height - 1, key, alloc) };

        // SAFETY: height propagation.
        if unsafe { child.is_deficient(height - 1) } {
            self.rebalance(height, child_idx, alloc);
        }

        val
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
mod tests {
    //! Contract tests for subtrees rooted at an `Inner`: teardown, sibling
    //! merges, and the remove/underflow path.
    //!
    //! Contract pinned: `Node::drop_subtree`, called at the subtree's true
    //! height, drops every node and every value in the subtree exactly
    //! once — nothing leaked, nothing dropped twice, nothing outside the
    //! initialized prefix touched. `Inner` itself has no drop glue BY
    //! DESIGN: it cannot know its children's types, so teardown is driven
    //! from above with the height in hand.
    //!
    //! Instrumentation: `Key` requires `Copy`, so keys can't observe their own
    //! drops; the children carry the counters instead. Each child is a
    //! heap-allocated leaf holding one `Counted` value, so a child dropped
    //! exactly once moves the live counter by exactly one. Run these under
    //! `cargo miri test` as well as plain `cargo test` — Miri turns any
    //! violation of the memory-safety half of the contract into a clean
    //! report instead of whatever a plain run degenerates into, and its leak
    //! checker is the backstop for the "exactly once" half.

    use alloc::sync::Arc;
    use core::sync::atomic::{AtomicIsize, Ordering::Relaxed};

    use super::*;
    use crate::test_util::{Counted, IMIN, LMIN, M, counted_leaf, inner_with_occupancies, leak};
    use crate::{Global, Leaf};

    /// A leaf-child holding a single counted value, boxed and erased the
    /// same way the tree hands leaves to their parents.
    fn counted_leaf_child(k: u64, live: &Arc<AtomicIsize>) -> Node<u64, Counted, M> {
        Node::from_leaf_ptr(counted_leaf(k, 1, live, None))
    }

    /// An `Inner` with `n` children and the matching `n - 1` separator
    /// keys. Child `i` holds the key `base + 10 * i`; separator `i` is
    /// the min key of child `i + 1`, per the crate's separator
    /// convention.
    fn inner_with_children(n: usize, base: u64, live: &Arc<AtomicIsize>) -> Inner<u64, Counted, M> {
        let keys: Vec<u64> = (1..n as u64).map(|i| base + 10 * i).collect();
        let children: Vec<Node<u64, Counted, M>> =
            (0..n as u64).map(|i| counted_leaf_child(base + 10 * i, live)).collect();
        Inner::test_from_parts(keys, children)
    }

    /// Tearing down a height-1 subtree must drop each of the inner node's
    /// children exactly once, whatever its occupancy: swept from the
    /// 2-child minimum to a full `M` children.
    #[test]
    fn drop_subtree_at_height_one_drops_every_child_exactly_once() {
        for n in 2..=M {
            let live = Arc::new(AtomicIsize::new(0));

            let node = Node::from_inner(inner_with_children(n, 0, &live), &mut Global);
            assert_eq!(live.load(Relaxed), n as isize, "one live value per child before teardown");

            // SAFETY: `node` roots an inner with leaf children (height 1)
            // and owns the subtree.
            unsafe { node.drop_subtree(1, &mut Global) };
            assert_eq!(
                live.load(Relaxed),
                0,
                "drop_subtree(1) over {n} children must drop each child exactly once \
                 (positive = leak, negative = double-drop)"
            );
        }
    }

    /// Tearing down a height-2 subtree (inner of inners of leaves) must
    /// recurse through the middle layer and drop every value exactly once.
    /// Pins that teardown is genuinely recursive — the union design has
    /// no typed `Drop` to recurse for it; `drop_subtree` must do the
    /// walk itself.
    #[test]
    fn drop_subtree_at_height_two_drops_the_whole_subtree_exactly_once() {
        let live = Arc::new(AtomicIsize::new(0));

        // Two height-1 subtrees over disjoint key ranges (0.. and 100..),
        // separated by the right subtree's min key.
        let left = Node::from_inner(inner_with_children(2, 0, &live), &mut Global);
        let right = Node::from_inner(inner_with_children(2, 100, &live), &mut Global);

        let root = Node::from_inner(Inner::test_from_parts(vec![100], vec![left, right]), &mut Global);
        assert_eq!(live.load(Relaxed), 4, "one live value per grandchild before teardown");

        // SAFETY: `root` roots an inner of inners of leaves (height 2) and
        // owns the subtree.
        unsafe { root.drop_subtree(2, &mut Global) };
        assert_eq!(
            live.load(Relaxed),
            0,
            "drop_subtree(2) must recurse through the middle layer and drop every \
             value exactly once (positive = leak, negative = double-drop)"
        );
    }

    // ── merge / remove fixtures ────────────────────────────────────────

    /// Like [`inner_with_children`], but with the leaf sibling chain wired:
    /// child `i`'s leaf links to child `i + 1`'s, and the last child's leaf
    /// links to `tail`. Accepts `n == 1` (a keyless single-child node).
    /// Returns the node plus the leaf pointers in key order, so assertions
    /// can walk the chain afterwards.
    #[allow(clippy::type_complexity)]
    fn linked_inner_with_children(
        n: usize,
        base: u64,
        live: &Arc<AtomicIsize>,
        tail: Option<NonNull<Leaf<u64, Counted, M>>>,
    ) -> (Inner<u64, Counted, M>, Vec<NonNull<Leaf<u64, Counted, M>>>) {
        assert!((1..=M).contains(&n), "1..=M children");

        // Build the leaves right-to-left so each can link to its successor.
        let mut ptrs: Vec<NonNull<Leaf<u64, Counted, M>>> = Vec::with_capacity(n);
        let mut next = tail;
        for i in (0..n).rev() {
            let ptr = counted_leaf(base + 10 * i as u64, 1, live, next);
            ptrs.push(ptr);
            next = Some(ptr);
        }
        ptrs.reverse();

        // Assembled by hand rather than through `test_from_parts`, whose
        // 2-child floor would reject the keyless n == 1 node the merge
        // minimum-occupancy test needs.
        let mut keys = [MaybeUninit::uninit(); M];
        let mut children: [MaybeUninit<Node<u64, Counted, M>>; M] =
            core::array::from_fn(|_| MaybeUninit::uninit());
        for (i, ptr) in ptrs.iter().enumerate() {
            children[i].write(Node::from_leaf_ptr(*ptr));
            if i > 0 {
                keys[i - 1].write(base + 10 * i as u64);
            }
        }

        (
            Inner {
                #[cfg(debug_assertions)]
                kind: NodeKind::Inner,
                child_count: n,
                keys,
                children,
            },
            ptrs,
        )
    }

    /// Walk the leaf chain from `head`, collecting each visited leaf's
    /// `(len, first_key)` (`None` first key for an empty leaf). `max_hops`
    /// is a cycle guard.
    ///
    /// # Safety
    ///
    /// Every leaf reachable from `head` along `next` must be live.
    unsafe fn walk_chain(
        head: NonNull<Leaf<u64, Counted, M>>,
        max_hops: usize,
    ) -> Vec<(usize, Option<u64>)> {
        let mut out = Vec::new();
        let mut cur = Some(head);
        while let Some(ptr) = cur {
            assert!(out.len() < max_hops, "the leaf chain must terminate (cycle suspected)");
            // SAFETY: the caller vouches every chained leaf is live.
            let leaf = unsafe { ptr.as_ref() };
            let first = (leaf.len() != 0).then(|| *leaf.first_key());
            out.push((leaf.len(), first));
            cur = leaf.next();
        }
        out
    }

    // ── Inner::merge ───────────────────────────────────────────────────

    /// Merging two siblings must concatenate their children in key order,
    /// with the separator that sat between them in the parent becoming a
    /// key of the merged node — afterwards every key reaches its value
    /// through the merged node alone, and teardown stays exactly-once.
    #[test]
    fn merge_concatenates_children_and_demotes_the_separator() {
        let live = Arc::new(AtomicIsize::new(0));
        let (right, rptrs) = linked_inner_with_children(2, 100, &live, None);
        let (mut left, _) = linked_inner_with_children(2, 0, &live, Some(rptrs[0]));

        // SAFETY: `right` is `left`'s immediate right sibling by
        // construction — adjacent disjoint ranges separated by 100 (the min
        // key under `right`), same height (1), 2 + 2 <= M, and `right`
        // owns its subtree.
        unsafe { left.merge(100, right) };

        assert_eq!(left.len(), 4, "the merged node must hold both sides' children");
        assert_eq!(
            left.keys_ref(),
            &[10, 100, 110],
            "the merged node's keys must be left's keys, the demoted separator, then right's"
        );
        for k in [0u64, 10, 100, 110] {
            // SAFETY: the merged node roots a height-1 subtree.
            let got = unsafe { left.get(1, &k) };
            assert!(got.is_some_and(|v| v.0 == k), "key {k} must remain reachable after the merge");
        }

        assert_eq!(live.load(Relaxed), 4, "a merge must not drop any value");
        // SAFETY: the merged node owns all four leaves (height 1).
        unsafe { Node::from_inner(left, &mut Global).drop_subtree(1, &mut Global) };
        assert_eq!(
            live.load(Relaxed),
            0,
            "teardown after a merge must drop every value exactly once \
             (positive = leak, negative = double-drop)"
        );
    }

    /// A single-child node brings no keys of its own to a merge; merging
    /// two of them must still succeed, producing a two-child node whose
    /// only key is the demoted separator.
    #[test]
    fn merge_succeeds_at_the_one_child_minimum() {
        let live = Arc::new(AtomicIsize::new(0));
        let (right, rptrs) = linked_inner_with_children(1, 100, &live, None);
        let (mut left, _) = linked_inner_with_children(1, 0, &live, Some(rptrs[0]));

        // SAFETY: adjacent single-child siblings over disjoint ranges
        // separated by 100, same height (1), 1 + 1 <= M, `right` owns its
        // subtree. Both members hold at least one child, as the contract
        // requires — no stronger occupancy is demanded of a merge input.
        unsafe { left.merge(100, right) };

        assert_eq!(left.len(), 2, "the merged node must hold both children");
        assert_eq!(
            left.keys_ref(),
            &[100],
            "the demoted separator must be the merged node's only key"
        );
        for k in [0u64, 100] {
            // SAFETY: the merged node roots a height-1 subtree.
            let got = unsafe { left.get(1, &k) };
            assert!(got.is_some_and(|v| v.0 == k), "key {k} must remain reachable after the merge");
        }

        // SAFETY: the merged node owns both leaves (height 1).
        unsafe { Node::from_inner(left, &mut Global).drop_subtree(1, &mut Global) };
        assert_eq!(
            live.load(Relaxed),
            0,
            "teardown after a merge must drop every value exactly once \
             (positive = leak, negative = double-drop)"
        );
    }

    /// The fit bound is inclusive: a merge may fill the surviving node to
    /// exactly `M` children (and so `M - 1` keys), with every key still
    /// reachable and teardown exactly-once.
    #[test]
    fn merge_fills_a_node_to_exact_capacity() {
        let live = Arc::new(AtomicIsize::new(0));
        let a = M / 2;
        let b = M - a;
        let (right, rptrs) = linked_inner_with_children(b, 1000, &live, None);
        let (mut left, _) = linked_inner_with_children(a, 0, &live, Some(rptrs[0]));

        // SAFETY: adjacent siblings over disjoint ranges separated by 1000,
        // same height (1), a + b == M <= M, `right` owns its subtree.
        unsafe { left.merge(1000, right) };

        assert_eq!(left.len(), M, "the merged node must hold exactly M children");
        assert_eq!(left.keys_ref().len(), M - 1, "M children need exactly M - 1 keys");
        for k in [0, 10 * (a as u64 - 1), 1000, 1000 + 10 * (b as u64 - 1)] {
            // SAFETY: the merged node roots a height-1 subtree.
            let got = unsafe { left.get(1, &k) };
            assert!(got.is_some_and(|v| v.0 == k), "key {k} must remain reachable after the merge");
        }

        // SAFETY: the merged node owns all M leaves (height 1).
        unsafe { Node::from_inner(left, &mut Global).drop_subtree(1, &mut Global) };
        assert_eq!(
            live.load(Relaxed),
            0,
            "teardown after a capacity-filling merge must drop every value exactly once \
             (positive = leak, negative = double-drop)"
        );
    }

    // ── Inner::rotate_from_{right,left} ────────────────────────────────

    /// Rotating in from the right sibling must: demote the parent
    /// separator to the receiver's LAST key, move the donor's FIRST child
    /// across, and promote the donor's first key out as the replacement
    /// separator — with every key still routing to its value on the
    /// correct side, and nothing dropped.
    #[test]
    fn rotate_from_right_demotes_sep_takes_first_child_promotes_first_key() {
        let live = Arc::new(AtomicIsize::new(0));
        let (mut donor, dptrs) = linked_inner_with_children(3, 100, &live, None);
        let (mut recv, _) = linked_inner_with_children(2, 0, &live, Some(dptrs[0]));

        // SAFETY: adjacent same-parent siblings over disjoint ranges
        // separated by 100 (the donor subtree's min); both height 1; the
        // receiver has room and the donor keeps >= 1 child.
        let new_sep = unsafe { recv.rotate_from_right(100, &mut donor) };

        assert_eq!(new_sep, 110, "the donor's first key must promote out as the new separator");
        assert_eq!(recv.len(), 3, "the receiver must gain exactly one child");
        assert_eq!(donor.len(), 2, "the donor must lose exactly one child");
        assert_eq!(
            recv.test_keys(),
            &[10, 100],
            "the old separator must demote to the receiver's LAST key"
        );
        assert_eq!(donor.test_keys(), &[120], "the donor's keys must close over the promoted one");
        for (node, k) in [(&recv, 0u64), (&recv, 10), (&recv, 100), (&donor, 110), (&donor, 120)] {
            // SAFETY: each node roots a height-1 subtree.
            let got = unsafe { node.get(1, &k) };
            assert!(got.is_some_and(|v| v.0 == k), "key {k} must route to its value");
        }
        assert_eq!(live.load(Relaxed), 5, "a rotation must not drop any value");

        // SAFETY: each node owns its (rearranged) height-1 subtree.
        unsafe { Node::from_inner(recv, &mut Global).drop_subtree(1, &mut Global) };
        unsafe { Node::from_inner(donor, &mut Global).drop_subtree(1, &mut Global) };
        assert_eq!(
            live.load(Relaxed),
            0,
            "teardown after a rotation must drop every value exactly once \
             (positive = leak, negative = double-drop)"
        );
    }

    /// Rotating in from the left sibling must: demote the parent separator
    /// to the receiver's FIRST key, move the donor's LAST child across,
    /// and promote the donor's last key out as the replacement separator.
    #[test]
    fn rotate_from_left_demotes_sep_takes_last_child_promotes_last_key() {
        let live = Arc::new(AtomicIsize::new(0));
        let (mut recv, rptrs) = linked_inner_with_children(2, 100, &live, None);
        let (mut donor, _) = linked_inner_with_children(3, 0, &live, Some(rptrs[0]));

        // SAFETY: adjacent same-parent siblings over disjoint ranges
        // separated by 100 (the receiver subtree's min); both height 1;
        // the receiver has room and the donor keeps >= 1 child.
        let new_sep = unsafe { recv.rotate_from_left(100, &mut donor) };

        assert_eq!(new_sep, 20, "the donor's last key must promote out as the new separator");
        assert_eq!(recv.len(), 3, "the receiver must gain exactly one child");
        assert_eq!(donor.len(), 2, "the donor must lose exactly one child");
        assert_eq!(
            recv.test_keys(),
            &[100, 110],
            "the old separator must demote to the receiver's FIRST key"
        );
        assert_eq!(donor.test_keys(), &[10], "the donor's keys must close over the promoted one");
        for (node, k) in [(&donor, 0u64), (&donor, 10), (&recv, 20), (&recv, 100), (&recv, 110)] {
            // SAFETY: each node roots a height-1 subtree.
            let got = unsafe { node.get(1, &k) };
            assert!(got.is_some_and(|v| v.0 == k), "key {k} must route to its value");
        }
        assert_eq!(live.load(Relaxed), 5, "a rotation must not drop any value");

        // SAFETY: each node owns its (rearranged) height-1 subtree.
        unsafe { Node::from_inner(recv, &mut Global).drop_subtree(1, &mut Global) };
        unsafe { Node::from_inner(donor, &mut Global).drop_subtree(1, &mut Global) };
        assert_eq!(
            live.load(Relaxed),
            0,
            "teardown after a rotation must drop every value exactly once \
             (positive = leak, negative = double-drop)"
        );
    }

    // ── remove under the min-occupancy invariant ───────────────────────
    //
    // These tests pin the C (classical rebalancing) contract for
    // `Inner::remove`: after any remove, every node in the subtree
    // satisfies its minimum occupancy (checked by
    // `Node::check_invariants`), a deficient child is repaired by
    // borrowing from a sibling above its minimum — merging only when both
    // members sit at the minimum — and the leaf chain, reachability, and
    // drop-exactly-once accounting all survive the repair.

    /// A root over `left` and `right` height-1 subtrees, separated by
    /// `sep` (the right subtree's min key).
    fn root_over(
        left: Inner<u64, Counted, M>,
        right: Inner<u64, Counted, M>,
        sep: u64,
    ) -> Inner<u64, Counted, M> {
        Inner::test_from_parts(
            vec![sep],
            vec![Node::from_inner(left, &mut Global), Node::from_inner(right, &mut Global)],
        )
    }

    /// Removing below the minimum from a leaf whose RIGHT sibling is above
    /// its own minimum must repair by borrowing: no leaf is freed, both
    /// leaves end at or above the minimum, and the invariants hold.
    #[test]
    fn remove_from_a_minimal_leaf_borrows_from_a_richer_right_sibling() {
        let live = Arc::new(AtomicIsize::new(0));
        let (inner, ptrs) = inner_with_occupancies(&[LMIN, LMIN + 1], 0, &live, None);
        let total = (2 * LMIN + 1) as isize;
        let mut node = Node::from_inner(inner, &mut Global);

        // SAFETY: `node` roots a height-1 subtree; 1 is its true height.
        let got = unsafe { node.remove(1, &0, &mut Global) };
        assert!(got.is_some_and(|v| v.0 == 0), "removing present key 0 must return its value");
        assert_eq!(live.load(Relaxed), total - 1, "exactly the removed value must drop");

        // Borrow, not merge: both leaves survive, both at or above MIN.
        // SAFETY: a borrow frees no leaf, so the fixture pointers are live.
        let hops = unsafe { walk_chain(ptrs[0], 3) };
        assert_eq!(hops.len(), 2, "a borrow must not free either leaf: {hops:?}");
        assert!(
            hops.iter().all(|(len, _)| *len >= LMIN),
            "both leaves must end at or above MIN_OCCUPANCY: {hops:?}"
        );

        // SAFETY: height-1 subtree, judged as a root (2 children).
        unsafe { node.check_invariants(1, true) };

        // The boundary that moved: the donor's old first key must now
        // route LEFT of the updated separator.
        for k in [10u64, 1_000, 1_010] {
            // SAFETY: height-1 subtree.
            let got = unsafe { node.get(1, &k) };
            assert!(got.is_some_and(|v| v.0 == k), "surviving key {k} must remain reachable");
        }
        // SAFETY: height-1 subtree.
        assert!(unsafe { node.get(1, &0) }.is_none(), "removed key 0 must be absent");

        // SAFETY: `node` owns the subtree.
        unsafe { node.drop_subtree(1, &mut Global) };
        assert_eq!(
            live.load(Relaxed),
            0,
            "teardown must drop every survivor exactly once \
             (positive = leak, negative = double-drop)"
        );
    }

    /// Removing below the minimum from a leaf with no right sibling must
    /// borrow from its richer LEFT sibling.
    #[test]
    fn remove_from_a_minimal_leaf_borrows_from_a_richer_left_sibling() {
        let live = Arc::new(AtomicIsize::new(0));
        let (inner, ptrs) = inner_with_occupancies(&[LMIN + 1, LMIN], 0, &live, None);
        let total = (2 * LMIN + 1) as isize;
        let mut node = Node::from_inner(inner, &mut Global);

        // SAFETY: `node` roots a height-1 subtree; 1 is its true height.
        let got = unsafe { node.remove(1, &1_000, &mut Global) };
        assert!(
            got.is_some_and(|v| v.0 == 1_000),
            "removing present key 1000 must return its value"
        );
        assert_eq!(live.load(Relaxed), total - 1, "exactly the removed value must drop");

        // SAFETY: a borrow frees no leaf, so the fixture pointers are live.
        let hops = unsafe { walk_chain(ptrs[0], 3) };
        assert_eq!(hops.len(), 2, "a borrow must not free either leaf: {hops:?}");
        assert!(
            hops.iter().all(|(len, _)| *len >= LMIN),
            "both leaves must end at or above MIN_OCCUPANCY: {hops:?}"
        );

        // SAFETY: height-1 subtree, judged as a root (2 children).
        unsafe { node.check_invariants(1, true) };

        // The donor's old last key crossed the boundary rightward.
        for k in [0u64, 10 * LMIN as u64, 1_010] {
            // SAFETY: height-1 subtree.
            let got = unsafe { node.get(1, &k) };
            assert!(got.is_some_and(|v| v.0 == k), "surviving key {k} must remain reachable");
        }
        // SAFETY: height-1 subtree.
        assert!(unsafe { node.get(1, &1_000) }.is_none(), "removed key 1000 must be absent");

        // SAFETY: `node` owns the subtree.
        unsafe { node.drop_subtree(1, &mut Global) };
        assert_eq!(
            live.load(Relaxed),
            0,
            "teardown must drop every survivor exactly once \
             (positive = leak, negative = double-drop)"
        );
    }

    /// A borrow rewrites exactly ONE separator: the boundary between the
    /// receiver and its donor. With a third child in the node, the
    /// separator above the rewritten one is live — and must keep its
    /// value, so keys under it stay reachable and the key order holds.
    #[test]
    fn borrowing_from_the_right_leaves_the_separator_above_intact() {
        let live = Arc::new(AtomicIsize::new(0));
        // Child 0 will go deficient; child 1 is the richer right donor;
        // child 2 exists only to keep a live separator (2000) above the
        // rewritten boundary.
        let (mut inner, _ptrs) = inner_with_occupancies(&[LMIN, LMIN + 1, LMIN], 0, &live, None);

        // SAFETY: `inner` roots a height-1 subtree; 1 is its true height.
        let got = unsafe { inner.remove(1, &0, &mut Global) };
        assert!(got.is_some_and(|v| v.0 == 0), "removing present key 0 must return its value");

        assert_eq!(
            inner.test_keys(),
            &[1_010, 2_000],
            "only the receiver/donor boundary separator may change: the donor's new \
             first key replaces 1000, and the separator 2000 above it must survive"
        );

        let node = Node::from_inner(inner, &mut Global);
        // SAFETY: height-1 subtree, judged as a root.
        unsafe { node.check_invariants(1, true) };

        // The keys of the child above the borrow must remain reachable.
        for k in [2_000u64, 2_010] {
            // SAFETY: height-1 subtree.
            let got = unsafe { node.get(1, &k) };
            assert!(got.is_some_and(|v| v.0 == k), "surviving key {k} must remain reachable");
        }

        // SAFETY: `node` owns the subtree.
        unsafe { node.drop_subtree(1, &mut Global) };
        assert_eq!(
            live.load(Relaxed),
            0,
            "teardown must drop every survivor exactly once \
             (positive = leak, negative = double-drop)"
        );
    }

    /// Mirror of the right-borrow pin for a LEFT borrow away from the
    /// node's edge: the deficient middle child borrows from its richer
    /// left sibling (its right sibling is at minimum and cannot donate),
    /// rewriting the boundary below it — and the separator above the
    /// receiver must keep its value.
    #[test]
    fn borrowing_from_the_left_leaves_the_separator_above_intact() {
        let live = Arc::new(AtomicIsize::new(0));
        // Child 1 will go deficient; child 0 is the richer left donor;
        // child 2 (at minimum, no donor) keeps a live separator (2000)
        // above the rewritten boundary.
        let (mut inner, _ptrs) = inner_with_occupancies(&[LMIN + 1, LMIN, LMIN], 0, &live, None);

        // SAFETY: `inner` roots a height-1 subtree; 1 is its true height.
        let got = unsafe { inner.remove(1, &1_000, &mut Global) };
        assert!(
            got.is_some_and(|v| v.0 == 1_000),
            "removing present key 1000 must return its value"
        );

        assert_eq!(
            inner.test_keys(),
            &[10 * LMIN as u64, 2_000],
            "only the donor/receiver boundary separator may change: the donor's moved \
             last key replaces 1000, and the separator 2000 above it must survive"
        );

        let node = Node::from_inner(inner, &mut Global);
        // SAFETY: height-1 subtree, judged as a root.
        unsafe { node.check_invariants(1, true) };

        // The keys of the child above the borrow must remain reachable.
        for k in [2_000u64, 2_010] {
            // SAFETY: height-1 subtree.
            let got = unsafe { node.get(1, &k) };
            assert!(got.is_some_and(|v| v.0 == k), "surviving key {k} must remain reachable");
        }

        // SAFETY: `node` owns the subtree.
        unsafe { node.drop_subtree(1, &mut Global) };
        assert_eq!(
            live.load(Relaxed),
            0,
            "teardown must drop every survivor exactly once \
             (positive = leak, negative = double-drop)"
        );
    }

    /// When the deficient leaf and its sibling BOTH sit at the minimum,
    /// they merge. At the head position this exercises the chain-critical
    /// direction: the head leaf's address must survive (an outside
    /// predecessor could link into it), and walking from it must find
    /// every surviving pair, in order.
    #[test]
    fn remove_merges_minimal_leaves_and_preserves_the_chain() {
        let live = Arc::new(AtomicIsize::new(0));
        let (inner, ptrs) = inner_with_occupancies(&[LMIN, LMIN, LMIN], 0, &live, None);
        let total = (3 * LMIN) as isize;
        let mut node = Node::from_inner(inner, &mut Global);

        // SAFETY: `node` roots a height-1 subtree; 1 is its true height.
        let got = unsafe { node.remove(1, &0, &mut Global) };
        assert!(got.is_some_and(|v| v.0 == 0), "removing present key 0 must return its value");
        assert_eq!(live.load(Relaxed), total - 1, "exactly the removed value must drop");

        // Merge, not borrow: one leaf slot closes; the head survives.
        // SAFETY: the head leaf's address must outlive the merge; the rest
        // of the chain is whatever remove left reachable from it.
        let hops = unsafe { walk_chain(ptrs[0], 4) };
        assert_eq!(hops.len(), 2, "minimal siblings must merge into one leaf: {hops:?}");
        let pairs: usize = hops.iter().map(|(len, _)| len).sum();
        assert_eq!(pairs, 3 * LMIN - 1, "the chain must hold every surviving pair exactly once");
        let firsts: Vec<u64> = hops.iter().filter_map(|(_, first)| *first).collect();
        assert!(firsts.windows(2).all(|w| w[0] < w[1]), "chain order must hold: {firsts:?}");

        // SAFETY: height-1 subtree, judged as a root (2 children).
        unsafe { node.check_invariants(1, true) };

        for k in [10u64, 1_000, 2_000] {
            // SAFETY: height-1 subtree.
            let got = unsafe { node.get(1, &k) };
            assert!(got.is_some_and(|v| v.0 == k), "surviving key {k} must remain reachable");
        }

        // SAFETY: `node` owns the subtree.
        unsafe { node.drop_subtree(1, &mut Global) };
        assert_eq!(
            live.load(Relaxed),
            0,
            "teardown must drop every survivor exactly once \
             (positive = leak, negative = double-drop)"
        );
    }

    /// The mirror merge at the TAIL position: the deficient last leaf
    /// folds into its left sibling.
    #[test]
    fn remove_merges_minimal_leaves_at_the_tail() {
        let live = Arc::new(AtomicIsize::new(0));
        let (inner, ptrs) = inner_with_occupancies(&[LMIN, LMIN, LMIN], 0, &live, None);
        let total = (3 * LMIN) as isize;
        let mut node = Node::from_inner(inner, &mut Global);

        // SAFETY: `node` roots a height-1 subtree; 1 is its true height.
        let got = unsafe { node.remove(1, &2_000, &mut Global) };
        assert!(
            got.is_some_and(|v| v.0 == 2_000),
            "removing present key 2000 must return its value"
        );
        assert_eq!(live.load(Relaxed), total - 1, "exactly the removed value must drop");

        // SAFETY: the head leaf is untouched by a tail-side merge.
        let hops = unsafe { walk_chain(ptrs[0], 4) };
        assert_eq!(hops.len(), 2, "minimal siblings must merge into one leaf: {hops:?}");
        let pairs: usize = hops.iter().map(|(len, _)| len).sum();
        assert_eq!(pairs, 3 * LMIN - 1, "the chain must hold every surviving pair exactly once");

        // SAFETY: height-1 subtree, judged as a root (2 children).
        unsafe { node.check_invariants(1, true) };

        for k in [0u64, 1_000, 2_010] {
            // SAFETY: height-1 subtree.
            let got = unsafe { node.get(1, &k) };
            assert!(got.is_some_and(|v| v.0 == k), "surviving key {k} must remain reachable");
        }
        // SAFETY: height-1 subtree.
        assert!(unsafe { node.get(1, &2_000) }.is_none(), "removed key 2000 must be absent");

        // SAFETY: `node` owns the subtree.
        unsafe { node.drop_subtree(1, &mut Global) };
        assert_eq!(
            live.load(Relaxed),
            0,
            "teardown must drop every survivor exactly once \
             (positive = leak, negative = double-drop)"
        );
    }

    /// A leaf merge that leaves an inner node deficient must cascade: the
    /// inner borrows a child from its richer sibling (a rotation through
    /// the root separator), leaving every node at or above its minimum.
    #[test]
    fn remove_cascades_a_borrow_across_inner_nodes() {
        let live = Arc::new(AtomicIsize::new(0));
        // A: IMIN minimal leaves — one leaf merge inside makes A itself
        // deficient. B: IMIN + 1 minimal leaves, the donor.
        let (b, bptrs) = inner_with_occupancies(&vec![LMIN; IMIN + 1], 1_000_000, &live, None);
        let (a, _) = inner_with_occupancies(&[LMIN; IMIN], 0, &live, Some(bptrs[0]));
        let total = ((2 * IMIN + 1) * LMIN) as isize;
        let mut root = root_over(a, b, 1_000_000);

        // SAFETY: `root` roots a height-2 subtree; 2 is its true height.
        let got = unsafe { root.remove(2, &0, &mut Global) };
        assert!(got.is_some_and(|v| v.0 == 0), "removing present key 0 must return its value");
        assert_eq!(live.load(Relaxed), total - 1, "exactly the removed value must drop");

        assert_eq!(root.len(), 2, "the root must keep both inner children (borrow, not merge)");

        let root = Node::from_inner(root, &mut Global);
        // SAFETY: height-2 subtree, judged as a root.
        unsafe { root.check_invariants(2, true) };

        // The child that crossed sides and the donor's remainder both
        // stay reachable through the root.
        for k in [10u64, 1_000_000, 1_001_000] {
            // SAFETY: height-2 subtree.
            let got = unsafe { root.get(2, &k) };
            assert!(got.is_some_and(|v| v.0 == k), "surviving key {k} must remain reachable");
        }
        // SAFETY: height-2 subtree.
        assert!(unsafe { root.get(2, &0) }.is_none(), "removed key 0 must be absent");

        // SAFETY: `root` owns the subtree.
        unsafe { root.drop_subtree(2, &mut Global) };
        assert_eq!(
            live.load(Relaxed),
            0,
            "teardown must drop every survivor exactly once \
             (positive = leak, negative = double-drop)"
        );
    }

    /// When both inner siblings sit at the minimum, the cascade merges
    /// them, leaving the root a single child — whose subtree must be fully
    /// valid. Hoisting that child (root shrink) is the tree layer's job
    /// and is out of scope here.
    #[test]
    fn remove_cascades_a_merge_across_minimal_inner_nodes() {
        let live = Arc::new(AtomicIsize::new(0));
        let (b, bptrs) = inner_with_occupancies(&[LMIN; IMIN], 1_000_000, &live, None);
        let (a, _) = inner_with_occupancies(&[LMIN; IMIN], 0, &live, Some(bptrs[0]));
        let total = (2 * IMIN * LMIN) as isize;
        let mut root = root_over(a, b, 1_000_000);

        // SAFETY: `root` roots a height-2 subtree; 2 is its true height.
        let got = unsafe { root.remove(2, &0, &mut Global) };
        assert!(got.is_some_and(|v| v.0 == 0), "removing present key 0 must return its value");
        assert_eq!(live.load(Relaxed), total - 1, "exactly the removed value must drop");

        assert_eq!(
            root.len(),
            1,
            "minimal inner siblings must merge, leaving the root a single child \
             for the tree layer to hoist"
        );
        // The merged child is a fully valid NON-root inner.
        // SAFETY: the root's child roots a height-1 subtree.
        unsafe { root.test_children()[0].check_invariants(1, false) };

        for k in [10u64, 1_000_000, 1_000_000 + 1_000 * (IMIN as u64 - 1)] {
            // SAFETY: height-2 subtree.
            let got = unsafe { root.get(2, &k) };
            assert!(got.is_some_and(|v| v.0 == k), "surviving key {k} must remain reachable");
        }

        // SAFETY: `root` owns the height-2 subtree (a 1-child root still
        // owns everything below it).
        unsafe { Node::from_inner(root, &mut Global).drop_subtree(2, &mut Global) };
        assert_eq!(
            live.load(Relaxed),
            0,
            "teardown must drop every survivor exactly once \
             (positive = leak, negative = double-drop)"
        );
    }

    // ── the write path: insert_unchecked / insert_child /
    //    splitting_insert_child ─────────────────────────────────────────
    //
    // The scene is always the same: a child leaf of this node has just
    // split — the new right leaf is already spliced into the LEAF CHAIN
    // (that is `Leaf::splitting_insert`'s job) — and this node must now
    // adopt the (separator, new child) pair: in place when it has room,
    // splitting itself when full. Contract pinned, whichever path runs:
    // the node counts the new child, every key in the subtree routes to
    // its leaf, tree child order agrees with the leaf chain
    // (`check_invariants`), and teardown drops every value exactly once.
    // Run these under `cargo miri test` too — the memory-safety half of
    // the contract (no reads or writes outside the initialized prefixes)
    // is Miri's department.

    /// A leaf holding the `LMIN` pairs `base, base + 1, ..`, boxed and
    /// leaked, linked to `next` — occupancy-legal for a non-root leaf, so
    /// `check_invariants` can judge the whole subtree.
    fn scene_leaf(
        base: u64,
        live: &Arc<AtomicIsize>,
        next: Option<NonNull<Leaf<u64, Counted, M>>>,
    ) -> NonNull<Leaf<u64, Counted, M>> {
        let mut leaf: Leaf<u64, Counted, M> = Leaf::new(next);
        for j in 0..LMIN as u64 {
            leaf.raw_append(base + j, Counted::new(base + j, live));
        }
        leak(leaf)
    }

    /// The canonical adoption scene: a parent over `n` leaves (leaf `i`
    /// holds the keys `100·i ..`, separators `100·i` per the crate
    /// convention), with a FRESH leaf (keys `100·split_idx + 50 ..`)
    /// already spliced into the leaf chain immediately after child
    /// `split_idx` — exactly the state a leaf split leaves behind before
    /// the parent has adopted the new sibling. Returns the parent, the
    /// separator (the new leaf's min key), and the new leaf as an erased
    /// child, ready to adopt at `partition == split_idx`.
    fn adoption_scene(
        n: usize,
        split_idx: usize,
        live: &Arc<AtomicIsize>,
    ) -> (Inner<u64, Counted, M>, u64, Node<u64, Counted, M>) {
        assert!((2..=M).contains(&n), "an inner node holds 2..=M children");
        assert!(split_idx < n);

        let sep = 100 * split_idx as u64 + 50;

        // Build right-to-left so each leaf links to its chain successor.
        let mut next = None;
        let mut new_leaf = None;
        let mut ptrs: Vec<NonNull<Leaf<u64, Counted, M>>> = Vec::with_capacity(n);
        for i in (0..n).rev() {
            if i == split_idx {
                let ptr = scene_leaf(sep, live, next);
                new_leaf = Some(ptr);
                next = Some(ptr);
            }
            let ptr = scene_leaf(100 * i as u64, live, next);
            ptrs.push(ptr);
            next = Some(ptr);
        }
        ptrs.reverse();

        let keys: Vec<u64> = (1..n as u64).map(|i| 100 * i).collect();
        let children: Vec<Node<u64, Counted, M>> =
            ptrs.iter().map(|p| Node::from_leaf_ptr(*p)).collect();

        (Inner::test_from_parts(keys, children), sep, Node::from_leaf_ptr(new_leaf.unwrap()))
    }

    /// Every key the scene's `n + 1` leaves hold, for routing sweeps.
    fn scene_keys(n: usize, split_idx: usize) -> Vec<u64> {
        (0..n as u64)
            .map(|i| 100 * i)
            .chain([100 * split_idx as u64 + 50])
            .flat_map(|base| (0..LMIN as u64).map(move |j| base + j))
            .collect()
    }

    /// The separators the parent must hold after adopting the scene's
    /// split without splitting itself: the old separators plus the new
    /// one, in sorted order.
    fn adopted_keys(n: usize, split_idx: usize) -> Vec<u64> {
        let mut keys: Vec<u64> =
            (1..n as u64).map(|i| 100 * i).chain([100 * split_idx as u64 + 50]).collect();
        keys.sort_unstable();
        keys
    }

    /// Adopting a split into a node with room must: count the new child,
    /// slot the separator into sorted key order, keep tree child order in
    /// agreement with the leaf chain, leave every key routable, and drop
    /// every value exactly once on teardown. Swept over every split
    /// position, at a small occupancy and at the fullest occupancy that
    /// still has room (`M - 1` children).
    #[test]
    fn insert_unchecked_adopts_a_split_at_every_position() {
        for n in [3, M - 1] {
            for split_idx in 0..n {
                let live = Arc::new(AtomicIsize::new(0));
                let (mut parent, sep, new_child) = adoption_scene(n, split_idx, &live);

                // SAFETY: the node has room (`n < M`); the new child's
                // slot `split_idx + 1` is in `1..=child_count` because
                // `split_idx == child_idx_for_key(&sep) < n` by the
                // scene's construction; and the pair is ordered — `sep`
                // is the new leaf's min key, strictly between its
                // neighbors' ranges.
                unsafe { parent.insert_child_unchecked(split_idx + 1, sep, new_child) };

                assert_eq!(
                    parent.len(),
                    n + 1,
                    "adopting a split must grow the child count by exactly one \
                     (n={n}, split_idx={split_idx})"
                );
                assert_eq!(
                    parent.test_keys(),
                    &adopted_keys(n, split_idx)[..],
                    "after adoption the keys must be the old separators plus the new one, \
                     in sorted order (n={n}, split_idx={split_idx})"
                );

                let node = Node::from_inner(parent, &mut Global);
                // SAFETY: height-1 subtree, judged as a root.
                unsafe { node.check_invariants(1, true) };

                for k in scene_keys(n, split_idx) {
                    // SAFETY: height-1 subtree.
                    let got = unsafe { node.get(1, &k) };
                    assert!(
                        got.is_some_and(|v| v.0 == k),
                        "key {k} must route to its leaf after the adoption \
                         (n={n}, split_idx={split_idx})"
                    );
                }

                // SAFETY: `node` owns the subtree.
                unsafe { node.drop_subtree(1, &mut Global) };
                assert_eq!(
                    live.load(Relaxed),
                    0,
                    "teardown after an adoption must drop every value exactly once \
                     (positive = leak, negative = double-drop; n={n}, split_idx={split_idx})"
                );
            }
        }
    }

    /// `insert_child` on a node with room must adopt in place and report
    /// no split — same contract as the direct `insert_unchecked` test,
    /// through the dispatch.
    #[test]
    fn insert_child_with_room_adopts_in_place() {
        let live = Arc::new(AtomicIsize::new(0));
        let (mut parent, sep, new_child) = adoption_scene(3, 1, &live);

        let split = parent.insert_child(1, sep, new_child, &mut Global);
        assert!(split.is_none(), "a node with room must adopt without splitting");
        assert_eq!(parent.len(), 4, "adopting a split must grow the child count by exactly one");
        assert_eq!(
            parent.test_keys(),
            &adopted_keys(3, 1)[..],
            "after adoption the keys must be the old separators plus the new one, in order"
        );

        let node = Node::from_inner(parent, &mut Global);
        // SAFETY: height-1 subtree, judged as a root.
        unsafe { node.check_invariants(1, true) };
        for k in scene_keys(3, 1) {
            // SAFETY: height-1 subtree.
            let got = unsafe { node.get(1, &k) };
            assert!(got.is_some_and(|v| v.0 == k), "key {k} must route to its leaf");
        }

        // SAFETY: `node` owns the subtree.
        unsafe { node.drop_subtree(1, &mut Global) };
        assert_eq!(
            live.load(Relaxed),
            0,
            "teardown after an adoption must drop every value exactly once \
             (positive = leak, negative = double-drop)"
        );
    }

    /// `insert_child` on a FULL node must split it and hand back the
    /// promoted separator and new right sibling for the caller to adopt
    /// in turn — after which the reassembled two-level tree is fully
    /// valid and every key still routes.
    #[test]
    fn insert_child_when_full_splits() {
        let live = Arc::new(AtomicIsize::new(0));
        let (mut parent, sep, new_child) = adoption_scene(M, M / 2, &live);

        let (promoted, right) = parent
            .insert_child(M / 2, sep, new_child, &mut Global)
            .expect("a full node must split to adopt");

        // Adopt the split the way the parent's parent (or the tree root)
        // would.
        let root = Node::from_inner(
            Inner::from_pair(
                promoted,
                Node::from_inner(parent, &mut Global),
                Node::from_inner_ptr(right),
            ),
            &mut Global,
        );
        // SAFETY: height-2 subtree, judged as a root.
        unsafe { root.check_invariants(2, true) };
        for k in scene_keys(M, M / 2) {
            // SAFETY: height-2 subtree.
            let got = unsafe { root.get(2, &k) };
            assert!(
                got.is_some_and(|v| v.0 == k),
                "key {k} must route to its leaf after the split"
            );
        }

        // SAFETY: `root` owns the subtree.
        unsafe { root.drop_subtree(2, &mut Global) };
        assert_eq!(
            live.load(Relaxed),
            0,
            "teardown after a split must drop every value exactly once \
             (positive = leak, negative = double-drop)"
        );
    }

    /// The heart of the split contract, swept over every insertion point
    /// `0..M`: the two halves hold all `M + 1` children between them,
    /// near-balanced and both at or above `MIN_OCCUPANCY`; the left keys,
    /// then the promoted separator, then the right keys are exactly the
    /// old separators plus the new one, in order (every key lands in
    /// exactly one of the three destinations); the reassembled tree is
    /// fully valid with every key routable; and teardown drops every
    /// value exactly once — so every child handle ended up owned by
    /// exactly one node.
    #[test]
    fn splitting_insert_child_covers_every_insertion_point() {
        for split_idx in 0..M {
            let live = Arc::new(AtomicIsize::new(0));
            let (mut parent, sep, new_child) = adoption_scene(M, split_idx, &live);

            let (promoted, right_ptr) =
                parent.splitting_insert_child(split_idx, sep, new_child, &mut Global);

            {
                // SAFETY: the split hands back a live, exclusively-owned
                // right sibling; this borrow ends before the move below.
                let right = unsafe { right_ptr.as_ref() };

                assert_eq!(
                    parent.len() + right.len(),
                    M + 1,
                    "the halves must hold all M + 1 children between them \
                     (inserting at {split_idx})"
                );
                assert!(
                    parent.len().abs_diff(right.len()) <= 1,
                    "the split must be near-balanced: left={}, right={} \
                     (inserting at {split_idx})",
                    parent.len(),
                    right.len()
                );
                assert!(
                    !parent.is_deficient() && !right.is_deficient(),
                    "both halves must satisfy MIN_OCCUPANCY: left={}, right={} \
                     (inserting at {split_idx})",
                    parent.len(),
                    right.len()
                );

                let mut combined: Vec<u64> = parent.test_keys().to_vec();
                combined.push(promoted);
                combined.extend_from_slice(right.test_keys());
                assert_eq!(
                    combined,
                    adopted_keys(M, split_idx),
                    "left keys, then the promoted separator, then right keys must be \
                     exactly the old separators plus the new one, in order \
                     (inserting at {split_idx})"
                );
            }

            let root = Node::from_inner(
                Inner::from_pair(
                    promoted,
                    Node::from_inner(parent, &mut Global),
                    Node::from_inner_ptr(right_ptr),
                ),
                &mut Global,
            );
            // SAFETY: height-2 subtree, judged as a root.
            unsafe { root.check_invariants(2, true) };
            for k in scene_keys(M, split_idx) {
                // SAFETY: height-2 subtree.
                let got = unsafe { root.get(2, &k) };
                assert!(
                    got.is_some_and(|v| v.0 == k),
                    "key {k} must route to its leaf after the split (inserting at {split_idx})"
                );
            }

            // SAFETY: `root` owns the subtree.
            unsafe { root.drop_subtree(2, &mut Global) };
            assert_eq!(
                live.load(Relaxed),
                0,
                "teardown after a split must drop every value exactly once — each child \
                 handle must have ended up owned by exactly one node \
                 (positive = leak, negative = double-drop; inserting at {split_idx})"
            );
        }
    }

    /// `from_pair` — the root-grow constructor — must produce a fully
    /// valid two-child node: both children counted, the separator its
    /// only key, every key routable to the correct side, teardown
    /// exactly-once.
    #[test]
    fn from_pair_builds_a_valid_two_child_node() {
        let live = Arc::new(AtomicIsize::new(0));
        let right_leaf = scene_leaf(100, &live, None);
        let left_leaf = scene_leaf(0, &live, Some(right_leaf));

        let node = Node::from_inner(
            Inner::from_pair(100, Node::from_leaf_ptr(left_leaf), Node::from_leaf_ptr(right_leaf)),
            &mut Global,
        );

        // SAFETY: height-1 subtree, judged as a root.
        unsafe { node.check_invariants(1, true) };
        for base in [0u64, 100] {
            for j in 0..LMIN as u64 {
                let k = base + j;
                // SAFETY: height-1 subtree.
                let got = unsafe { node.get(1, &k) };
                assert!(
                    got.is_some_and(|v| v.0 == k),
                    "key {k} must route through the separator to its side"
                );
            }
        }

        // SAFETY: `node` owns the subtree.
        unsafe { node.drop_subtree(1, &mut Global) };
        assert_eq!(
            live.load(Relaxed),
            0,
            "teardown must drop both children's values exactly once \
             (positive = leak, negative = double-drop)"
        );
    }

    /// `child_idx_for_key` must route by the separator convention — a
    /// separator IS its right child's minimum key. Probes below every
    /// separator, at each separator exactly, inside the gaps, and above
    /// everything. The equality probes are the load-bearing ones: a key
    /// EQUAL to a separator lives under the separator's RIGHT child.
    #[test]
    fn child_idx_for_key_routes_hits_gaps_and_extremes() {
        let live = Arc::new(AtomicIsize::new(0));
        // Three children over the ranges 0.., 100.., and 200.. — so the
        // separators are 100 and 200, each its right child's min key.
        let right = scene_leaf(200, &live, None);
        let mid = scene_leaf(100, &live, Some(right));
        let left = scene_leaf(0, &live, Some(mid));
        let node = Inner::test_from_parts(
            vec![100, 200],
            vec![Node::from_leaf_ptr(left), Node::from_leaf_ptr(mid), Node::from_leaf_ptr(right)],
        );

        assert_eq!(node.child_idx_for_key(&0), 0, "a key below every separator routes to child 0");
        assert_eq!(node.child_idx_for_key(&99), 0, "a gap key routes to the child covering it");
        assert_eq!(
            node.child_idx_for_key(&100),
            1,
            "a key EQUAL to a separator must route to the separator's RIGHT child — \
             the separator is that child's minimum key"
        );
        assert_eq!(node.child_idx_for_key(&150), 1, "a gap key routes to the child covering it");
        assert_eq!(
            node.child_idx_for_key(&199),
            1,
            "a key just below a separator routes to the separator's LEFT child"
        );
        assert_eq!(
            node.child_idx_for_key(&200),
            2,
            "a key equal to the LAST separator must route to the last child"
        );
        assert_eq!(
            node.child_idx_for_key(&u64::MAX),
            2,
            "a key above every separator routes to the last child"
        );

        // SAFETY: `node` owns its three leaves (height 1).
        unsafe { Node::from_inner(node, &mut Global).drop_subtree(1, &mut Global) };
        assert_eq!(
            live.load(Relaxed),
            0,
            "teardown must drop every value exactly once \
             (positive = leak, negative = double-drop)"
        );
    }
}
