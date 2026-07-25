use core::{mem::MaybeUninit, ptr::NonNull};

use crate::{Key, allocator::SlotAllocator};

#[cfg(debug_assertions)]
use crate::NodeKind;

// TODO:

// - perf: keys AND values are both inline in the node. For large V that
//   bloats the node past NODE_BUDGET's assumptions; add a fat-value
//   config to benches/vs_btreemap.rs (currently all-u64), then benchmark
//   inline-V vs boxed-V at several V sizes.

// - perf: audit the #[inline(always)] annotations against the bench.

// SAFETY:
// - The first `occupied` elements of `keys` and `values` MUST be initialized.
// - All other values MUST NOT be initialized.
//
// Functionality:
// - The Keys slice MUST be in sorted order.
/// A leaf node: the tree's key/value pairs, in sorted order, chained
/// left-to-right for iteration.
///
/// Public only so allocator bounds can name it (the
/// [`NodeAllocator`](crate::NodeAllocator) alias); every field and
/// method is crate-private, and it never appears in a usable public
/// signature.
#[cfg_attr(debug_assertions, repr(C))]
pub struct Leaf<K: Key, V, const M: usize> {
    /// Debug-only kind tag. MUST stay the first field: the erased cast
    /// accessors on [`Node`](crate::Node) read it through the pointer before knowing the
    /// pointee's type (hence the debug-only `repr(C)`).
    #[cfg(debug_assertions)]
    kind: NodeKind,

    occupied: usize,

    keys: [MaybeUninit<K>; M],
    vals: [MaybeUninit<V>; M],

    // non-owned ptr to other.
    next: Option<NonNull<Self>>,
}

impl<K: Key, V, const M: usize> core::fmt::Debug for Leaf<K, V, M> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Leaf").field("occupied", &self.occupied).finish_non_exhaustive()
    }
}

impl<K: Key, V, const M: usize> Drop for Leaf<K, V, M> {
    // Panic during teardown: NOT panic-safe. Values are dropped in order; if
    // one value's `Drop` unwinds, every value at a higher index (and the key
    // beside it) is abandoned and leaks. And since this is `Drop` glue, a
    // value that panics while the thread is already unwinding double-panics
    // and aborts the process.
    fn drop(&mut self) {
        let Self { occupied, vals, .. } = self;

        // SAFETY: indice < occupied are initialized
        vals.iter_mut().take(*occupied).for_each(|val| unsafe { val.assume_init_drop() });
    }
}

/// Bitwise-copy `$count` key/value pairs from `$src` (starting at `$src_at`)
/// into `$dst` (starting at `$dst_at`).
///
/// These are copies, not moves: afterwards both leaves hold the bits, so
/// exactly one side must count each pair in its `occupied` — that
/// bookkeeping stays with the caller.
///
/// Expands to raw, unchecked copies, so the call site must be inside an
/// `unsafe` block, upholding:
///
/// - `$src` and `$dst` are distinct leaves (the copies must not overlap);
/// - both ranges are in bounds: `$src_at + $count <= M` and
///   `$dst_at + $count <= M`.
///
/// `$src`/`$dst` should be plain place expressions like `self` or `right`
/// (they are evaluated more than once).
macro_rules! copy_pairs {
    ($src:expr, $src_at:expr => $dst:expr, $dst_at:expr; $count:expr) => {{
        let (src_at, dst_at, count) = ($src_at, $dst_at, $count);
        core::ptr::copy_nonoverlapping(
            $src.keys.as_ptr().add(src_at),
            $dst.keys.as_mut_ptr().add(dst_at),
            count,
        );
        core::ptr::copy_nonoverlapping(
            $src.vals.as_ptr().add(src_at),
            $dst.vals.as_mut_ptr().add(dst_at),
            count,
        );
    }};
}

/// Overlap-tolerant counterpart of [`copy_pairs!`] for sliding a run of
/// pairs WITHIN one leaf: `$count` key/value pairs move from `$src_at`
/// to `$dst_at`, and the ranges may overlap ([`ptr::copy`](core::ptr::copy)) — the shape of
/// a shift-insert opening a slot mid-leaf, a remove closing one, or a
/// steal closing the donor over its departed edge pair. (Cross-leaf runs
/// never overlap; they stay `copy_pairs!`'s business. Single-leaf also
/// keeps miri happy: one mutable pointer is derived per array, where a
/// `$src`/`$dst` pair naming the same leaf would stack conflicting
/// borrows.)
///
/// Everything else carries over from [`copy_pairs!`]: raw, unchecked
/// bitwise copies (call from `unsafe` with both ranges in bounds against
/// the leaf's true occupancy), after which overlapped slots hold the
/// bits twice — exactly one live slot must count each pair in
/// `occupied`, and that bookkeeping stays with the caller. `$count == 0`
/// is the empty shift.
macro_rules! shift_pairs {
    ($node:expr, $src_at:expr => $dst_at:expr; $count:expr) => {{
        let (src_at, dst_at, count) = ($src_at, $dst_at, $count);
        let keys = $node.keys.as_mut_ptr();
        let vals = $node.vals.as_mut_ptr();
        core::ptr::copy(keys.add(src_at), keys.add(dst_at), count);
        core::ptr::copy(vals.add(src_at), vals.add(dst_at), count);
    }};
}

impl<K: Key, V, const M: usize> Leaf<K, V, M> {
    // NOTE: an associated const is only evaluated where it is USED — the
    // definition alone checks nothing. `new` must keep its
    // `const { Self::__FANOUT }` reference or this assert silently never
    // runs (the `compile_fail` doctest on `BPlusTree` pins this; it has
    // been broken twice by dropping the reference — don't make it three).
    const __FANOUT: () = {
        assert!(M == K::FANOUT);
    };

    /// Midpoint of the fanout set. Used during splitting.
    const LEFT_COUNT: usize = M.div_ceil(2);

    /// The minimum number of leaves we copy out to split.
    const RIGHT_COUNT: usize = M - Self::LEFT_COUNT;

    /// Minimum pairs per NON-ROOT leaf — the classical-rebalancing
    /// occupancy invariant (a root leaf is exempt, down to 0). Coherent
    /// with splitting (a split leaves `LEFT_COUNT == ⌈M/2⌉` pairs on the
    /// left and `RIGHT_COUNT + 1 >= ⌈M/2⌉` on the right) and with merging
    /// (a deficient leaf at `MIN_OCCUPANCY - 1` plus a minimal sibling is
    /// `2⌈M/2⌉ - 1 <= M` pairs, so a merge of an at-minimum pair always
    /// fits).
    pub(crate) const MIN_OCCUPANCY: usize = M.div_ceil(2);

    // Instantiate an empty leaf, with the `next` ptr.
    pub(crate) fn new(next: Option<NonNull<Self>>) -> Self {
        // Every leaf is born here; see `assert_fanout_floor` for why a
        // too-small M must be a compile error. The `__FANOUT` reference is
        // load-bearing: without a use, the `M == K::FANOUT` assert never
        // evaluates (see the note at its definition). Do not remove it.
        const { crate::assert_fanout_floor(M) };
        const { Self::__FANOUT };

        Self {
            #[cfg(debug_assertions)]
            kind: NodeKind::Leaf,
            occupied: 0,
            keys: [MaybeUninit::uninit(); M],
            // SAFETY:
            // uninit MaybeUninit array is validly uninit
            vals: [const { MaybeUninit::uninit() }; M],
            next,
        }
    }

    /// Instantiate an empty leaf, with the `next` ptr, and the first item.
    pub(crate) fn from_first_item(next: Option<NonNull<Self>>, key: K, val: V) -> Self {
        let mut this = Self::new(next);
        this.raw_append(key, val);
        this
    }

    /// Return the number of key/value pairs in this leaf.
    pub(crate) fn len(&self) -> usize {
        self.occupied
    }

    /// True if this leaf is below the minimum occupancy a non-root leaf
    /// must keep ([`Self::MIN_OCCUPANCY`]) — the parent's rebalance
    /// trigger after a remove. The root leaf is exempt and must not be
    /// judged by this.
    pub(crate) fn is_deficient(&self) -> bool {
        self.occupied < Self::MIN_OCCUPANCY
    }

    /// Set the sibling link. Bulk-load building block (`bulk.rs`): the
    /// loader finalizes each leaf's link to its successor before yielding
    /// it (the field itself stays private).
    pub(crate) fn set_next(&mut self, next: Option<NonNull<Self>>) {
        self.next = next;
    }

    /// Test-only read of the sibling link, for chain-walking assertions in
    /// other modules' tests (the field itself stays private).
    pub(crate) fn next(&self) -> Option<NonNull<Self>> {
        self.next
    }

    /// Test-only view of the keys, for invariant checking from other
    /// modules' tests.
    #[cfg(test)]
    pub(crate) fn test_keys(&self) -> &[K] {
        self.keys_ref()
    }

    // # Panics
    //
    // - if the leaf is empty.
    //
    // pub(crate): `Node::insert` reads the fresh split sibling's min key as
    // the separator while it still holds the typed pointer, before erasing.
    pub(crate) fn first_key(&self) -> &K {
        self.keys_ref().first().unwrap()
    }

    #[track_caller]
    pub(crate) fn keys_ref(&self) -> &[K] {
        // SAFETY: `occupied` guarantees initialization
        unsafe { self.keys[..self.occupied].assume_init_ref() }
    }

    #[track_caller]
    fn vals_mut(&mut self) -> &mut [V] {
        // SAFETY: `occupied`` guarantees initialization
        unsafe { self.vals[..self.occupied].assume_init_mut() }
    }

    #[track_caller]
    fn vals_ref(&self) -> &[V] {
        // SAFETY: `occupied` guarantees initialization
        unsafe { self.vals[..self.occupied].assume_init_ref() }
    }

    /// Get a KV pair by index. Panics if out of range.
    pub(crate) fn kv_ref_unchecked(&self, idx: usize) -> (&K, &V) {
        (&self.keys_ref()[idx], &self.vals_ref()[idx])
    }

    /// Value-only counterpart of [`Self::kv_ref_unchecked`]: the value in
    /// slot `idx`. Panics if out of range.
    // The entry API's occupied views read through this; the allow
    // leaves when their bodies land.
    #[allow(dead_code)]
    pub(crate) fn val_ref_unchecked(&self, idx: usize) -> &V {
        &self.vals_ref()[idx]
    }

    /// Mutable counterpart of [`Self::val_ref_unchecked`]: the value in
    /// slot `idx`, for in-place replacement via the tree's replace path.
    /// Panics if out of range.
    pub(crate) fn val_mut_unchecked(&mut self, idx: usize) -> &mut V {
        &mut self.vals_mut()[idx]
    }

    /// Split the occupied pairs from index `from` onward into per-pair
    /// iterators — keys shared, values mutable — zipped. The two arrays
    /// are disjoint fields, so the borrows coexist, and the zip's items
    /// are the disjoint `(&K, &mut V)` borrows [`IteratorMut`](crate::iter::IteratorMut) yields: one
    /// whole-leaf `&mut` is spent here, then never re-created over pairs
    /// already handed out.
    ///
    /// # Panics
    ///
    /// If `from > self.len()`.
    pub(crate) fn pairs_mut_from(
        &mut self,
        from: usize,
    ) -> core::iter::Zip<core::slice::Iter<'_, K>, core::slice::IterMut<'_, V>> {
        // SAFETY: `occupied` guarantees initialization of both ranges.
        let keys = unsafe { self.keys[from..self.occupied].assume_init_ref() };
        // SAFETY: as above; the arrays are disjoint fields, so the shared
        // and exclusive borrows coexist.
        let vals = unsafe { self.vals[from..self.occupied].assume_init_mut() };
        keys.iter().zip(vals.iter_mut())
    }

    /// Find the first element of the array that is >= key
    #[track_caller]
    pub(crate) fn find_key(&self, key: &K) -> usize {
        // Branchless linear count (A/B history in perf.md):
        // the partition point is the number of keys below `key`, so
        // count them all. No early exit and no data-dependent branch:
        // the whole prefix is touched, but the loop auto-vectorizes and
        // nothing mispredicts, which beat the early-exit scan, chunked
        // hybrids, and (branchless) binary search at every fanout on the
        // tree-level readout.
        self.keys_ref().iter().map(|k| usize::from(k < key)).sum()
    }

    /// Locate `key` in one probe: its partition point, and whether the
    /// slot there is an exact hit — [`Self::find_key`] plus the hit
    /// check.
    pub(crate) fn probe(&self, key: &K) -> (usize, bool) {
        let partition = self.find_key(key);
        (partition, partition < self.occupied && self.keys_ref()[partition] == *key)
    }

    /// Debug-build check of the struct invariant: the occupied prefix of
    /// `keys` is strictly increasing (sorted, no duplicates). Run after every
    /// mutation; compiles to nothing in release builds.
    #[track_caller]
    #[inline(always)]
    fn debug_assert_sorted(&self) {
        debug_assert!(
            self.keys_ref().windows(2).all(|w| w[0] < w[1]),
            "leaf keys must be strictly sorted.",
        );
    }

    /// # Panics
    ///
    /// - if leaf is full
    #[track_caller]
    #[inline(always)]
    #[allow(dead_code)]
    pub(crate) fn raw_append(&mut self, key: K, val: V) {
        // Write at the end.
        self.keys[self.occupied].write(key);
        self.vals[self.occupied].write(val);
        self.occupied += 1;

        self.debug_assert_sorted();
    }

    /// Insert without checking for duplicates.
    ///
    /// # Safety Preconditions
    ///
    /// - partition <= self.occupied < M
    #[track_caller]
    #[inline(always)]
    unsafe fn insert_unchecked(&mut self, partition: usize, key: K, val: V) {
        debug_assert!(self.occupied < M);
        debug_assert!(partition <= self.occupied);

        // SAFETY:
        // The moved run `partition..occupied` is inside the live prefix,
        // and its destination stays within `M` (`occupied < M`).
        unsafe {
            shift_pairs!(self, partition => partition + 1; self.occupied - partition);
        }

        self.keys[partition].write(key);
        self.vals[partition].write(val);
        self.occupied += 1;

        self.debug_assert_sorted();
    }

    // Split and insert at a partition point. Returns the new right leaf.
    //
    // The key MUST NOT be a duplicate.
    //
    // # Panics
    //
    // - if self.occupied != M
    // - if partition is > M
    fn splitting_insert<A: SlotAllocator<Self>>(
        &mut self,
        partition: usize,
        key: K,
        val: V,
        alloc: &mut A,
    ) -> NonNull<Self> {
        debug_assert_eq!(self.occupied, M);
        debug_assert!(partition <= M);

        let mut right = Self::new(self.next);

        if partition < Self::LEFT_COUNT {
            // The new pair lands in the left leaf. Hand the top
            // `RIGHT_COUNT + 1` pairs to `right`, then shift-insert into the
            // now-short left. Two steps, but unlike the right-lands branch
            // there is nothing to fuse: the copy-out covers
            // `copy_idx..M` and the shift covers `partition..copy_idx`,
            // disjoint ranges, so each pair already moves exactly once
            // (`M - partition` moves — the minimum).
            let copy_idx = Self::LEFT_COUNT - 1;

            // SAFETY:
            // `occupied == M`, so `copy_idx..M` is initialized and in bounds.
            // `right` is a fresh, distinct leaf with room for
            // `M - copy_idx = RIGHT_COUNT + 1` pairs at offset 0. `occupied`
            // on both sides is set immediately below, counting each pair
            // exactly once.
            unsafe {
                copy_pairs!(self, copy_idx => right, 0; Self::RIGHT_COUNT + 1);
            }
            self.occupied = copy_idx;
            right.occupied = Self::RIGHT_COUNT + 1;

            // SAFETY:
            // `partition < LEFT_COUNT`, so `partition <= copy_idx
            // == self.occupied`, and `self.occupied == LEFT_COUNT - 1 < M`.
            unsafe { self.insert_unchecked(partition, key, val) };
        } else {
            // The new pair lands in the right leaf. Fuse the insert into the
            // copy-out, so nothing moves twice:
            // right = self[LEFT_COUNT..partition] ++ [new] ++ self[partition..M].
            let insertion = partition - Self::LEFT_COUNT;

            // SAFETY:
            // `occupied == M` and `LEFT_COUNT <= partition <= M`, so both
            // source ranges are initialized and in bounds. `right` is a
            // fresh, distinct leaf; its highest written slot is
            // `insertion + 1 + (M - partition) = RIGHT_COUNT + 1 <= M`.
            // `occupied` on both sides is set immediately below, counting
            // each copied pair (and the new one) exactly once.
            unsafe {
                copy_pairs!(self, Self::LEFT_COUNT => right, 0; insertion);
                right.keys[insertion].write(key);
                right.vals[insertion].write(val);
                copy_pairs!(self, partition => right, insertion + 1; M - partition);
            }
            self.occupied = Self::LEFT_COUNT;
            right.occupied = Self::RIGHT_COUNT + 1;
        }

        self.debug_assert_sorted();
        right.debug_assert_sorted();

        let right = alloc.allocate(right);
        self.next = Some(right);
        right
    }

    /// Fold the immediate right sibling `other` back into `self`, the
    /// structural inverse of [`Self::splitting_insert`]: append every pair
    /// from `other` after `self`'s, splice `other` out of the leaf chain, and
    /// reclaim its allocation. After this returns `other` is freed and must
    /// not be reached again.
    ///
    /// The caller (an [`Inner`](crate::Inner)) is responsible for the parent-side bookkeeping
    /// this does not touch: dropping the separator key that sat between the
    /// two children and removing `other`'s child slot.
    ///
    /// # Safety
    ///
    /// - `other` must be `self`'s immediate right sibling: `self.next ==
    ///   Some(other)`, `other` owns its subtree, and no other handle reaches
    ///   it afterward.
    /// - Every key in `other` must compare greater than every key in `self`
    ///   (they hold disjoint, adjacent key ranges).
    /// - The merged occupancy must fit: `self.occupied + other.occupied <= M`.
    pub(crate) unsafe fn merge(&mut self, mut other: Self) {
        debug_assert!(self.occupied + other.occupied <= M);
        debug_assert!(
            self.occupied == 0
                || other.occupied == 0
                || self.keys_ref().last().unwrap() < other.keys_ref().first().unwrap()
        );

        // If other is empty, Just splice it out of the chain and return.
        if other.occupied == 0 {
            self.next = other.next;
            return;
        }

        // If self is empty, just take over other. This automatically splices
        // the in-bound pointer.
        if self.occupied == 0 {
            *self = other;
            return;
        }

        // Copy first (M - self.occupied elements). This ensures that we get
        // the right node's high tail.
        // SAFETY:
        // `self.occupied + (M - self.occupied) == M`.
        unsafe {
            copy_pairs!(other, 0 => self, self.occupied; M - self.occupied);
        }
        self.occupied += other.occupied;
        other.occupied = 0;
        self.next = other.next;

        self.debug_assert_sorted();

        drop(other);
    }

    /// Take the FIRST pair of `right` — `self`'s immediate right sibling —
    /// and append it to `self`. Returns the replacement separator for the
    /// parent to write over the old one in place: `right`'s NEW first key.
    ///
    /// The borrow half of classical rebalancing at leaf level. The chain
    /// is untouched, and nothing is allocated, freed, or dropped.
    ///
    /// The caller must uphold (unchecked): `right` is `self`'s immediate
    /// chain successor under the same parent, and every key in `right` is
    /// greater than every key in `self`. Under the C policy the caller
    /// only borrows from a sibling strictly above its minimum, so `self`
    /// has room and `right` keeps at least one pair.
    ///
    /// # Panics
    ///
    /// Debug-asserts that `self` is not full and `right` has at least 2
    /// pairs.
    ///
    /// # Safety
    ///
    /// The caller must uphold (unchecked): `right` is `self`'s immediate
    /// right sibling, and every key in `right` is greater than every key in
    /// left. And that
    pub(crate) unsafe fn steal_from_right(&mut self, right: &mut Self) -> K {
        debug_assert!(self.occupied < M);
        debug_assert!(right.occupied > Self::MIN_OCCUPANCY);
        debug_assert!(right.keys_ref()[0] > self.keys_ref()[self.occupied - 1]);

        // SAFETY:
        // `self.occupied < M` and `right.occupied >= 2`.
        // Because stealing is always an append, copying rights tail will
        // always be preserve our ascending K invariant
        unsafe {
            // copy the donor's pairs onto self's tail.
            copy_pairs!(right, 0 => self, self.occupied; M - self.occupied);

            // close the donor over its departed first pair
            shift_pairs!(right, 1 => 0; right.occupied - 1);
        }

        self.occupied += 1;
        right.occupied -= 1;

        self.debug_assert_sorted();

        right.keys_ref()[0]
    }

    /// Mirror of [`Self::steal_from_right`]: take the LAST pair of `left`
    /// — `self`'s immediate left sibling — and prepend it to `self`.
    /// Returns the replacement separator: the moved key itself (it is
    /// `self`'s new minimum).
    ///
    /// The caller must uphold (unchecked): `left` is the leaf whose `next`
    /// is `self`, under the same parent, and every key in `left` is less
    /// than every key in `self`.
    ///
    /// # Panics
    ///
    /// Debug-asserts that `self` is not full and `left` has at least 2
    /// pairs.
    pub(crate) unsafe fn steal_from_left(&mut self, left: &mut Self) -> K {
        debug_assert!(self.occupied < M);
        debug_assert!(left.occupied > Self::MIN_OCCUPANCY);
        debug_assert!(left.keys_ref()[left.occupied - 1] < self.keys_ref()[0]);

        // SAFETY:
        // `self.occupied < M` and `left.occupied >= 2`.
        unsafe {
            // open slot 0 by shifting all of self's pairs up one position
            shift_pairs!(self, 0 => 1; self.occupied);

            // copy the donor's last pair into the opened slot
            copy_pairs!(left, left.occupied - 1 => self, 0; 1);
        }

        self.occupied += 1;
        left.occupied -= 1;

        self.debug_assert_sorted();

        self.keys_ref()[0]
    }

    /// Index-addressed insertion: insert `key`/`val` at `partition` — the
    /// slot [`Self::find_key`] returns for a key that is NOT already
    /// present — splitting if the leaf is full. The caller has already
    /// searched and ruled out a duplicate; no re-search happens here.
    ///
    /// Returns the inserted value's slot, and the new right sibling if the
    /// leaf split. The slot pointer is final when this returns: leaf
    /// contents move only within this call, never during the caller's
    /// upward split propagation (which moves child handles, not pairs).
    ///
    /// # Panics
    ///
    /// In debug builds, if `partition` is not `key`'s insertion point or
    /// `key` is already present.
    pub(crate) fn insert_at<A: SlotAllocator<Self>>(
        &mut self,
        partition: usize,
        key: K,
        val: V,
        alloc: &mut A,
    ) -> (NonNull<V>, Option<NonNull<Self>>) {
        debug_assert_eq!(
            partition,
            self.find_key(&key),
            "partition must be the key's insertion point"
        );
        debug_assert!(
            partition >= self.occupied || self.keys_ref()[partition] != key,
            "the key must not already be present"
        );

        // check if full
        if self.occupied == M {
            // Full leaf, split it. Which side the new pair landed on
            // follows the split policy: `partition < LEFT_COUNT` keeps it
            // in `self` at `partition`; otherwise it sits in the new right
            // sibling at `partition - LEFT_COUNT`.
            let mut right = self.splitting_insert(partition, key, val, alloc);
            let val_ptr = if partition < Self::LEFT_COUNT {
                debug_assert!(self.keys_ref()[partition] == key);
                NonNull::from(&mut self.vals_mut()[partition])
            } else {
                // SAFETY: `splitting_insert` hands back a live right
                // sibling that the caller exclusively owns until it links
                // it into the tree.
                let r = unsafe { right.as_mut() };
                debug_assert!(r.keys_ref()[partition - Self::LEFT_COUNT] == key);
                NonNull::from(&mut r.vals_mut()[partition - Self::LEFT_COUNT])
            };
            return (val_ptr, Some(right));
        }

        // otherwise, simple insert
        // This also covers empty leaf case.
        // SAFETY:
        // `self.occupied < M` and `partition <= self.occupied`, so the
        // insert is safe.
        unsafe {
            self.insert_unchecked(partition, key, val);
        }
        (NonNull::from(&mut self.vals_mut()[partition]), None)
    }

    /// Insert `key`/`val`, deciding on its own whether a split is needed:
    /// replace in place when the key is already present, plain insert when
    /// there is room, split when the leaf is full.
    ///
    /// Test-only — production insertion is descend/commit (the tree
    /// searches once and hands the slot to [`Self::insert_at`]): the leaf
    /// tests pin the self-contained contract through it.
    #[cfg(test)]
    pub(crate) fn insert<A: SlotAllocator<Self>>(
        &mut self,
        key: K,
        val: V,
        alloc: &mut A,
    ) -> (Option<V>, Option<NonNull<Self>>) {
        let partition = self.find_key(&key);

        // check for duplicate
        if partition < self.occupied && self.keys_ref()[partition] == key {
            let old = core::mem::replace(self.val_mut_unchecked(partition), val);
            return (Some(old), None);
        }

        let (_, split) = self.insert_at(partition, key, val, alloc);
        (None, split)
    }

    /// Get a reference to a value, if it is present.
    pub(crate) fn get(&self, key: &K) -> Option<&V> {
        let (partition, exact) = self.probe(key);
        exact.then(|| &self.vals_ref()[partition])
    }

    /// Mutable mirror of [`Self::get`]: a reference to the value for
    /// `key`, if present.
    pub(crate) fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        let (partition, exact) = self.probe(key);
        exact.then(move || &mut self.vals_mut()[partition])
    }

    /// Index-addressed removal: remove and return the pair in slot `idx`.
    /// The caller has already searched; no re-search happens here.
    ///
    /// # Panics
    ///
    /// If `idx` is not an occupied slot.
    pub(crate) fn remove_at(&mut self, idx: usize) -> (K, V) {
        assert!(idx < self.occupied, "remove_at must target an occupied slot");

        let count = self.occupied - idx - 1;

        // SAFETY:
        // `idx < occupied`, so both slot reads are of initialized pairs
        // (and the key, being `Copy`, has no drop glue to double-run).
        // The remainder is a slice-wise shift closing the vacated slot:
        // idx + 1 + count == occupied <= M. `occupied` then shrinks, so
        // exactly one live slot counts each survivor.
        unsafe {
            let key = self.keys[idx].assume_init_read();
            let val = self.vals[idx].assume_init_read();

            shift_pairs!(self, idx + 1 => idx; count);

            self.occupied -= 1;

            self.debug_assert_sorted();
            (key, val)
        }
    }

    /// Remove a key from the leaf, if it exists.
    ///
    /// Test-only since removal went descend/commit (the tree searches
    /// once and hands the slot to [`Self::remove_at`]): the leaf tests
    /// and [`Node`](crate::Node)'s test-only recursive `remove` still drive it.
    #[cfg(test)]
    pub(crate) fn remove(&mut self, key: &K) -> Option<V> {
        let partition = self.find_key(key);
        if partition >= self.occupied || &self.keys_ref()[partition] != key {
            return None;
        }
        Some(self.remove_at(partition).1)
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for the `Leaf` primitives, in isolation from the tree:
    //! `find_key`, `insert_unchecked`, `splitting_insert`, and `get`.
    //!
    //! Contract pinned for `find_key` + `insert_unchecked` below capacity:
    //! inserting a new key at its partition point keeps `keys` sorted with
    //! `vals` in step; all `M` slots are usable before any split happens;
    //! values are dropped exactly once. Duplicate detection and replacement are
    //! the caller's job now — `find_key` hands back the match index, and the
    //! (future) tree level decides whether to replace or insert.
    //!
    //! Contract pinned for `splitting_insert` on a full leaf, at the u64
    //! fanout: the two leaves together hold exactly the old entries plus the
    //! new one, in order with left's keys below right's, both sides
    //! near-balanced (midpoint policy) with room to spare, and values still
    //! drop exactly once. (`splitting_insert_tests` pins the same contract at
    //! other fanouts, plus the separator convention: after a split the parent
    //! stores the right sibling's minimum key and routes `key < separator`
    //! left, `key >= separator` right.)
    //!
    //! Contract pinned for the sibling chain: `splitting_insert` heap-allocates
    //! the right leaf and splices it in — the left leaf's `next` points at the
    //! returned leaf, and the right leaf takes over the left's old successor
    //! (or `None`). The returned pointer owns the right leaf; these tests
    //! reclaim it with `own` so drop accounting stays exact.
    //!
    //! Contract pinned for `get`: `Some(&value)` for every stored key, `None`
    //! for anything else — whether the probe falls below, between, or above the
    //! stored keys, or the leaf is empty.
    //!
    //! Contract pinned for `remove`: removing a stored key returns its value
    //! and closes the gap (survivors sorted, values in step, `occupied` down
    //! by one); misses return `None` and leave the leaf untouched; every
    //! position of a leaf is removable, including every position of a *full*
    //! leaf; removed values drop exactly once, via the returned handle.
    //!
    //! Contract pinned for `drain_sorted_iter`: a sorted stream is chunked
    //! into leaves of `M` pairs, the chain arrives pre-linked in yield
    //! order ending in `None`, the iterator terminates (no empty tail
    //! leaves, `None` forever once exhausted), and drained values drop
    //! exactly once, via the leaves. Occupancy: a short tail borrows from
    //! its left neighbor before either is yielded, so every leaf of a
    //! multi-leaf drain meets `MIN_OCCUPANCY`; a lone leaf (the
    //! root-to-be, which is exempt) is passed through unrepaired.

    use alloc::boxed::Box;
    use alloc::sync::Arc;
    use core::cell::RefCell;
    use core::sync::atomic::{AtomicIsize, Ordering::Relaxed};

    use super::*;
    use crate::Global;
    use crate::test_util::{Counted, LMIN, M, entries, own, shuffled, v};

    /// Insert a key expected to be new into a leaf expected to have room, the
    /// way a caller would: partition via `find_key`, then `insert_unchecked`.
    fn insert_new(l: &mut Leaf<u64, u64, M>, k: u64) {
        let partition = l.find_key(&k);
        assert!(
            partition == l.occupied || l.keys_ref()[partition] != k,
            "key {k} unexpectedly present"
        );
        // SAFETY: `find_key` returns `partition <= occupied`, and the caller
        // guarantees the leaf has room (`occupied < M`).
        unsafe { l.insert_unchecked(partition, k, v(k)) };
    }

    /// Shuffled inserts land at the front, middle, and back (and into an empty
    /// leaf); keys must be sorted with values in step after every step.
    #[test]
    fn inserts_stay_sorted_with_values_in_step() {
        let mut l: Leaf<u64, u64, M> = Leaf::new(None);
        for k in shuffled(M as u64 - 1) {
            insert_new(&mut l, k);
            assert!(
                l.keys_ref().windows(2).all(|w| w[0] < w[1]),
                "keys must stay sorted after every insert: {:?}",
                l.keys_ref()
            );
        }
        let expected: Vec<_> = (0..M as u64 - 1).map(|k| (k, v(k))).collect();
        assert_eq!(entries(&l), expected);
    }

    /// `find_key` is the partition point: the index of the key itself when
    /// present, and of the first larger key (or `occupied`) when absent.
    #[test]
    fn find_key_locates_hits_and_gaps() {
        let mut l: Leaf<u64, u64, M> = Leaf::new(None);
        for k in 0..M as u64 - 1 {
            insert_new(&mut l, 2 * k + 100);
        }
        for i in 0..M as u64 - 1 {
            let k = 2 * i + 100;
            assert_eq!(l.find_key(&k), i as usize, "stored key {k}");
            assert_eq!(l.find_key(&(k + 1)), i as usize + 1, "gap above {k}");
        }
        assert_eq!(l.find_key(&0), 0, "below the smallest key");
        assert_eq!(l.find_key(&u64::MAX), l.occupied, "above the largest key");
    }

    /// All `M` slots fill without splitting; `splitting_insert` only becomes
    /// necessary on the insert *after* the leaf is full.
    #[test]
    fn fill_every_slot() {
        let mut l: Leaf<u64, u64, M> = Leaf::new(None);
        for k in shuffled(M as u64) {
            insert_new(&mut l, k);
        }
        assert_eq!(l.occupied, M);
        let expected: Vec<_> = (0..M as u64).map(|k| (k, v(k))).collect();
        assert_eq!(entries(&l), expected);
    }

    /// Fill a leaf with the odd keys `1, 3, .., 2M-1`, then split-insert one new
    /// even key. Sweeping `pos` over `0..=M` lands the new key before the first
    /// stored key, between every adjacent pair, and after the last — every
    /// possible insertion point, each from a fresh full leaf. After the split
    /// the two leaves in slot order must hold exactly the old entries plus the
    /// new one, sorted (which also proves left's keys sit below right's), and
    /// the halves must be near-balanced with room to spare on both sides.
    #[test]
    fn split_covers_every_insertion_point() {
        let stored: Vec<u64> = (0..M as u64).map(|k| 2 * k + 1).collect();
        for pos in 0..=M as u64 {
            let mut l: Leaf<u64, u64, M> = Leaf::new(None);
            for &k in &stored {
                insert_new(&mut l, k);
            }

            let new_key = 2 * pos;
            let partition = l.find_key(&new_key);
            let right_ptr = l.splitting_insert(partition, new_key, v(new_key), &mut Global);
            let right = own(right_ptr);

            assert_eq!(
                l.next,
                Some(right_ptr),
                "after a split, the left leaf's next must point at the new right leaf \
                 (inserting {new_key})"
            );
            assert_eq!(
                right.next, None,
                "a leaf with no successor must split into a right leaf with no successor \
                 (inserting {new_key})"
            );

            let mut combined = entries(&l);
            combined.extend(entries(&right));
            let mut expected: Vec<_> =
                stored.iter().copied().chain([new_key]).map(|k| (k, v(k))).collect();
            expected.sort_unstable();
            assert_eq!(
                combined, expected,
                "left then right must hold exactly the old entries plus key {new_key}, in order"
            );

            assert_eq!(l.occupied + right.occupied, M + 1);
            assert!(
                l.occupied.abs_diff(right.occupied) <= 1,
                "split must be near-balanced: left={}, right={} (inserting {new_key})",
                l.occupied,
                right.occupied
            );
            assert!(
                l.occupied < M && right.occupied < M,
                "both halves must have room to spare: left={}, right={} (inserting {new_key})",
                l.occupied,
                right.occupied
            );
        }
    }

    /// Splitting a leaf that already has a successor must splice the new right
    /// leaf into the middle of the chain: left's `next` points at the new
    /// leaf, and the new leaf inherits left's old successor — walking `next`
    /// from the left leaf reaches the right leaf, then the old successor, so
    /// no suffix of the leaf chain is orphaned.
    #[test]
    fn split_splices_into_sibling_chain() {
        let successor: Box<Leaf<u64, u64, M>> = Box::new(Leaf::new(None));
        let successor_ptr = NonNull::from(successor.as_ref());

        let mut l: Leaf<u64, u64, M> = Leaf::new(Some(successor_ptr));
        for k in 0..M as u64 {
            l.raw_append(2 * k + 1, v(2 * k + 1));
        }

        let new_key = 0;
        let right_ptr = l.splitting_insert(l.find_key(&new_key), new_key, v(new_key), &mut Global);
        let right = own(right_ptr);

        assert_eq!(
            l.next,
            Some(right_ptr),
            "after a split, the left leaf's next must point at the new right leaf"
        );
        assert_eq!(
            right.next,
            Some(successor_ptr),
            "the right leaf must inherit the left leaf's old successor"
        );
    }

    /// Even keys with a gap on each side of the range, so every probe class
    /// exists: exact hits, and misses below, between, and above the stored keys.
    #[test]
    fn get_hits_stored_keys_only() {
        let mut l: Leaf<u64, u64, M> = Leaf::new(None);
        let keys: Vec<u64> = (0..M as u64 - 1).map(|k| 2 * k + 100).collect();
        for &k in &keys {
            insert_new(&mut l, k);
        }
        for &k in &keys {
            assert_eq!(l.get(&k), Some(&v(k)), "stored key {k} must be found");
            assert_eq!(l.get(&(k + 1)), None, "key {} was never inserted", k + 1);
        }
        assert_eq!(l.get(&0), None, "below the smallest key");
        assert_eq!(l.get(&99), None, "just below the smallest key");
        assert_eq!(l.get(&u64::MAX), None, "above the largest key");
    }

    #[test]
    fn get_on_empty_leaf() {
        let l: Leaf<u64, u64, M> = Leaf::new(None);
        assert_eq!(l.get(&0), None);
        assert_eq!(l.get(&u64::MAX), None);
    }

    /// Removing a stored key must return its value and close the gap: after
    /// every removal the survivors are exactly the not-yet-removed entries,
    /// sorted with values in step, and a second probe for the removed key
    /// misses. Draining in shuffled order hits removals at the front, middle,
    /// and back, down to the empty leaf.
    #[test]
    fn remove_returns_value_and_closes_the_gap() {
        const N: u64 = M as u64 - 1;
        let mut l: Leaf<u64, u64, M> = Leaf::new(None);
        for k in 0..N {
            insert_new(&mut l, k);
        }

        let mut remaining: Vec<u64> = (0..N).collect();
        for k in shuffled(N) {
            assert_eq!(l.remove(&k), Some(v(k)), "removing stored key {k} must return its value");
            assert_eq!(l.remove(&k), None, "a second removal of {k} must miss");
            remaining.retain(|&r| r != k);
            let expected: Vec<_> = remaining.iter().map(|&r| (r, v(r))).collect();
            assert_eq!(
                entries(&l),
                expected,
                "survivors must be the unremoved entries, sorted with values in step \
                 (just removed {k})"
            );
        }
        assert_eq!(l.occupied, 0, "draining every key must empty the leaf");
    }

    /// Probes that miss — below, between, and above the stored keys, and on an
    /// empty leaf — return `None` and leave the leaf untouched.
    #[test]
    fn remove_miss_leaves_leaf_untouched() {
        let mut empty: Leaf<u64, u64, M> = Leaf::new(None);
        assert_eq!(empty.remove(&0), None, "removing from an empty leaf");

        let mut l: Leaf<u64, u64, M> = Leaf::new(None);
        let keys: Vec<u64> = (0..M as u64 - 1).map(|k| 2 * k + 100).collect();
        for &k in &keys {
            insert_new(&mut l, k);
        }
        let before = entries(&l);

        assert_eq!(l.remove(&0), None, "below the smallest key");
        assert_eq!(l.remove(&101), None, "between stored keys");
        assert_eq!(l.remove(&u64::MAX), None, "above the largest key");
        assert_eq!(entries(&l), before, "missed removes must not disturb the leaf");
    }

    /// Every position of a *full* leaf must be removable: with all `M` slots
    /// occupied, removing the key at each position in turn returns its value
    /// and leaves the other `M - 1` entries in order. (Under Miri this also
    /// checks the memory-safety contract of removal at full occupancy.)
    #[test]
    fn remove_at_every_position_of_a_full_leaf() {
        for target in 0..M as u64 {
            let mut l: Leaf<u64, u64, M> = Leaf::new(None);
            for k in 0..M as u64 {
                l.raw_append(k, v(k));
            }

            assert_eq!(
                l.remove(&target),
                Some(v(target)),
                "removing key {target} from a full leaf must return its value"
            );
            assert_eq!(l.occupied, M - 1);
            let expected: Vec<_> =
                (0..M as u64).filter(|&k| k != target).map(|k| (k, v(k))).collect();
            assert_eq!(
                entries(&l),
                expected,
                "survivors must close the gap in order (removed {target} from a full leaf)"
            );
        }
    }

    #[test]
    fn values_drop_exactly_once() {
        const N: u64 = M as u64 - 1;
        let live = Arc::new(AtomicIsize::new(0));
        {
            let mut l: Leaf<u64, Counted, M> = Leaf::new(None);
            for k in shuffled(N) {
                let partition = l.find_key(&k);
                // SAFETY: `find_key` returns `partition <= occupied`, and only
                // `N = M - 1` keys are inserted, so the leaf always has room.
                unsafe { l.insert_unchecked(partition, k, Counted::new(v(k), &live)) };
            }
            assert_eq!(live.load(Relaxed), N as isize, "one live value per stored key");
        }
        assert_eq!(
            live.load(Relaxed),
            0,
            "dropping the leaf must drop every value exactly once \
             (positive = leak, negative = double-drop)"
        );
    }

    /// `remove` transfers ownership of the value to the caller: dropping the
    /// returned value drops it exactly once, the survivors drop exactly once
    /// when the leaf drops, and nothing drops twice or leaks. Removals at the
    /// front, middle, and back, starting from a full leaf.
    #[test]
    fn remove_drops_values_exactly_once() {
        let live = Arc::new(AtomicIsize::new(0));
        {
            let mut l: Leaf<u64, Counted, M> = Leaf::new(None);
            for k in 0..M as u64 {
                l.raw_append(k, Counted::new(v(k), &live));
            }

            let mut expect_live = M as isize;
            for k in [0, M as u64 / 2, M as u64 - 1] {
                let removed = l.remove(&k).expect("stored key must come out");
                assert_eq!(
                    live.load(Relaxed),
                    expect_live,
                    "removal itself must not drop anything (removed {k})"
                );
                drop(removed);
                expect_live -= 1;
                assert_eq!(
                    live.load(Relaxed),
                    expect_live,
                    "dropping the returned value must drop it exactly once (removed {k})"
                );
            }
        }
        assert_eq!(
            live.load(Relaxed),
            0,
            "dropping the leaf must drop each survivor exactly once \
             (positive = leak, negative = double-drop)"
        );
    }

    /// Splitting moves values, never duplicates or loses them: after the split
    /// exactly `M + 1` values are live, and dropping both halves drops each
    /// exactly once. Exercised with the new key landing in the left half, the
    /// right half, and past the end.
    #[test]
    fn split_drops_values_exactly_once() {
        for pos in [0u64, M as u64 - 1, M as u64] {
            let live = Arc::new(AtomicIsize::new(0));
            {
                let mut l: Leaf<u64, Counted, M> = Leaf::new(None);
                for k in 0..M as u64 {
                    l.raw_append(2 * k + 1, Counted::new(v(2 * k + 1), &live));
                }

                let new_key = 2 * pos;
                let partition = l.find_key(&new_key);
                let right = own(l.splitting_insert(
                    partition,
                    new_key,
                    Counted::new(v(new_key), &live),
                    &mut Global,
                ));
                assert_eq!(
                    live.load(Relaxed),
                    M as isize + 1,
                    "one live value per stored key after splitting on key {new_key}"
                );
                drop(right);
            }
            assert_eq!(
                live.load(Relaxed),
                0,
                "dropping both halves must drop every value exactly once \
                 (positive = leak, negative = double-drop; split on key {})",
                2 * pos
            );
        }
    }

    /// `drain_sorted_iter` chunks a sorted stream into full leaves,
    /// pre-linked in yield order: each leaf's `next` is the leaf yielded
    /// after it, and the final leaf's is `None`. The 3-pair tail here is
    /// short, so the final two leaves must arrive rebalanced: the tail
    /// brought up to `MIN_OCCUPANCY`, its neighbor down by the difference.
    #[test]
    fn drain_chunks_and_links_in_order() {
        const TAIL: usize = 3;
        const N: u64 = (2 * M + TAIL) as u64;
        let alloc = RefCell::new(Global);
        let yielded: Vec<_> =
            Leaf::<u64, u64, M>::drain_sorted_iter((0..N).map(|k| (k, v(k))), &alloc).collect();
        assert_eq!(yielded.len(), 3, "{N} pairs at fanout {M} must make exactly 3 leaves");

        let ptrs: Vec<_> = yielded.iter().map(|u| u.as_ptr()).collect();
        let leaves: Vec<_> = yielded.into_iter().map(|u| u.into_leaf()).collect();
        assert_eq!(leaves[0].next, Some(ptrs[1]), "leaf 0 must link to leaf 1");
        assert_eq!(leaves[1].next, Some(ptrs[2]), "leaf 1 must link to leaf 2");
        assert_eq!(leaves[2].next, None, "the final leaf must not link onward");

        let want = [M, M - (LMIN - TAIL), LMIN];
        let mut expect = 0u64;
        for (i, leaf) in leaves.iter().enumerate() {
            assert_eq!(leaf.len(), want[i], "leaf {i} occupancy");
            for pair in entries(leaf) {
                assert_eq!(pair, (expect, v(expect)), "pairs must stay in stream order");
                expect += 1;
            }
        }
        assert_eq!(expect, N, "every pair must land in exactly one leaf");
    }

    /// The worst ragged tail — one pair past an exact multiple — must be
    /// repaired: every leaf of a multi-leaf drain meets `MIN_OCCUPANCY`,
    /// with stream order and the chain both intact.
    #[test]
    fn drain_repairs_a_deficient_tail() {
        const N: u64 = 2 * M as u64 + 1;
        let alloc = RefCell::new(Global);
        let yielded: Vec<_> =
            Leaf::<u64, u64, M>::drain_sorted_iter((0..N).map(|k| (k, v(k))), &alloc).collect();
        assert_eq!(yielded.len(), 3, "{N} pairs at fanout {M} must make exactly 3 leaves");

        let ptrs: Vec<_> = yielded.iter().map(|u| u.as_ptr()).collect();
        let leaves: Vec<_> = yielded.into_iter().map(|u| u.into_leaf()).collect();
        assert_eq!(
            leaves.iter().map(|l| l.len()).collect::<Vec<_>>(),
            [M, M - (LMIN - 1), LMIN],
            "the tail must be brought up to MIN_OCCUPANCY from its neighbor"
        );

        let mut expect = 0u64;
        for leaf in &leaves {
            for pair in entries(leaf) {
                assert_eq!(pair, (expect, v(expect)), "repair must preserve stream order");
                expect += 1;
            }
        }
        assert_eq!(expect, N, "every pair must land in exactly one leaf");

        assert_eq!(leaves[0].next, Some(ptrs[1]), "the chain must survive the repair");
        assert_eq!(leaves[1].next, Some(ptrs[2]), "the chain must survive the repair");
        assert_eq!(leaves[2].next, None, "the final leaf must not link onward");
    }

    /// A lone short chunk has no neighbor to borrow from and needs none:
    /// it is the root-to-be, exempt from `MIN_OCCUPANCY`, and passes
    /// through unrepaired.
    #[test]
    fn drain_passes_a_lone_short_leaf_through() {
        let alloc = RefCell::new(Global);
        let mut yielded: Vec<_> =
            Leaf::<u64, u64, M>::drain_sorted_iter((0..2u64).map(|k| (k, v(k))), &alloc).collect();
        assert_eq!(yielded.len(), 1);
        let leaf = yielded.pop().expect("just asserted one leaf").into_leaf();
        assert_eq!(leaf.len(), 2);
        assert_eq!(entries(&leaf), vec![(0, v(0)), (1, v(1))]);
        assert_eq!(leaf.next, None);
    }

    /// `drain_sorted_iter` terminates: an exact multiple of `M` yields only
    /// full leaves (no empty tail), an empty source yields nothing, and an
    /// exhausted iterator keeps returning `None`.
    #[test]
    fn drain_terminates_without_empty_tail() {
        let alloc = RefCell::new(Global);
        let mut it =
            Leaf::<u64, u64, M>::drain_sorted_iter((0..2 * M as u64).map(|k| (k, v(k))), &alloc);
        // `take` guards the test against regressing to an unbounded drain.
        let yielded: Vec<_> = it.by_ref().take(5).collect();
        assert_eq!(yielded.len(), 2, "an exact multiple of M must make only full leaves");
        assert!(it.next().is_none(), "an exhausted drain must keep returning None");

        let leaves: Vec<_> = yielded.into_iter().map(|u| u.into_leaf()).collect();
        assert!(leaves.iter().all(|l| l.len() == M), "both leaves must be full");
        assert_eq!(leaves[1].next, None, "the final leaf must not link onward");

        let mut empty = Leaf::<u64, u64, M>::drain_sorted_iter(core::iter::empty(), &alloc);
        assert!(empty.next().is_none(), "an empty source must yield no leaves");
        assert!(empty.next().is_none(), "and must stay exhausted");
    }

    /// Values flow through `drain_sorted_iter` into the leaves without
    /// dropping — the drain itself drops nothing — and every value drops
    /// exactly once when the leaves drop.
    #[test]
    fn drain_values_drop_exactly_once() {
        const N: u64 = M as u64 + 2;
        let live = Arc::new(AtomicIsize::new(0));
        {
            let alloc = RefCell::new(Global);
            let items = (0..N).map(|k| (k, Counted::new(v(k), &live)));
            let leaves: Vec<_> = Leaf::<u64, Counted, M>::drain_sorted_iter(items, &alloc)
                .map(|u| u.into_leaf())
                .collect();
            assert_eq!(leaves.len(), 2);
            assert_eq!(live.load(Relaxed), N as isize, "one live value per drained pair");
        }
        assert_eq!(
            live.load(Relaxed),
            0,
            "dropping the leaves must drop every value exactly once \
             (positive = leak, negative = double-drop)"
        );
    }
}

#[cfg(test)]
mod merge_tests {
    //! Unit tests for `Leaf::merge`, in isolation from the tree.
    //!
    //! Contract pinned (from `merge`'s doc comment): folding the immediate
    //! right sibling back into `self` appends every pair from the sibling
    //! after `self`'s (merged entries sorted, values in step, occupancies
    //! summed), splices the sibling out of the leaf chain — the left leaf
    //! takes over the sibling's successor — and reclaims the sibling's
    //! allocation. This must hold across occupancy shapes: both sides
    //! populated (up to an exactly-full merged leaf), an empty left side,
    //! and an empty right side. Values move, never drop: the merge itself
    //! drops nothing, and every value from both sides drops exactly once
    //! when the merged leaf drops.

    use alloc::boxed::Box;
    use alloc::sync::Arc;
    use core::sync::atomic::{AtomicIsize, Ordering::Relaxed};

    use super::*;
    use crate::test_util::{Counted, M, entries, v};

    /// Drive `merge` the way a parent would: heap-allocate the right
    /// sibling, link it in as `left`'s successor, then reclaim it by value
    /// and fold it in.
    fn merge_into<V>(left: &mut Leaf<u64, V, M>, right: Leaf<u64, V, M>) {
        let right_ptr = NonNull::from(Box::leak(Box::new(right)));
        left.next = Some(right_ptr);
        // SAFETY: `right` is `left`'s immediate right sibling, reclaimed by
        // value with no other handle to it. Callers keep the key ranges
        // disjoint (left strictly below right) and the merged occupancy
        // within `M`.
        unsafe { left.merge(*Box::from_raw(right_ptr.as_ptr())) }
    }

    /// Merging appends the right sibling's pairs after the left's: the
    /// merged leaf holds exactly both sides' entries, sorted with values in
    /// step, occupancies summed. Swept over every exactly-full shape
    /// (`a + b == M`) plus a couple of loose fits.
    #[test]
    fn merge_concatenates_both_sides_in_order() {
        let mut shapes: Vec<(usize, usize)> = (1..M).map(|a| (a, M - a)).collect();
        shapes.extend([(1, 1), (2, 3)]);

        for (a, b) in shapes {
            let mut left: Leaf<u64, u64, M> = Leaf::new(None);
            for k in 0..a as u64 {
                left.raw_append(k, v(k));
            }
            let mut right: Leaf<u64, u64, M> = Leaf::new(None);
            for k in 0..b as u64 {
                right.raw_append(100 + k, v(100 + k));
            }

            merge_into(&mut left, right);

            assert_eq!(
                left.occupied,
                a + b,
                "merged occupancy must be the sum of both sides (left={a}, right={b})"
            );
            let expected: Vec<_> =
                (0..a as u64).chain((0..b as u64).map(|k| 100 + k)).map(|k| (k, v(k))).collect();
            assert_eq!(
                entries(&left),
                expected,
                "merged leaf must hold both sides' entries in order (left={a}, right={b})"
            );
        }
    }

    /// Merging splices the right sibling out of the leaf chain: the left
    /// leaf takes over the sibling's successor — a live leaf, or `None` at
    /// the end of the chain.
    #[test]
    fn merge_takes_over_the_right_siblings_successor() {
        let successor: Box<Leaf<u64, u64, M>> = Box::new(Leaf::new(None));
        let successor_ptr = NonNull::from(successor.as_ref());

        let mut left: Leaf<u64, u64, M> = Leaf::new(None);
        left.raw_append(0, v(0));
        let mut right: Leaf<u64, u64, M> = Leaf::new(Some(successor_ptr));
        right.raw_append(100, v(100));

        merge_into(&mut left, right);
        assert_eq!(
            left.next,
            Some(successor_ptr),
            "after a merge the left leaf's next must be the right sibling's old successor"
        );

        let mut tail: Leaf<u64, u64, M> = Leaf::new(None);
        tail.raw_append(200, v(200));
        merge_into(&mut left, tail);
        assert_eq!(
            left.next, None,
            "merging away the last leaf in the chain must leave the left leaf with no successor"
        );
    }

    /// An empty right sibling merges away like any other — and the chain
    /// must still be spliced: the left leaf takes over the empty sibling's
    /// successor, with its own entries untouched.
    #[test]
    fn merge_of_an_empty_right_sibling_still_splices_the_chain() {
        let successor: Box<Leaf<u64, u64, M>> = Box::new(Leaf::new(None));
        let successor_ptr = NonNull::from(successor.as_ref());

        let mut left: Leaf<u64, u64, M> = Leaf::new(None);
        for k in 0..3 {
            left.raw_append(k, v(k));
        }
        let before = entries(&left);

        let right: Leaf<u64, u64, M> = Leaf::new(Some(successor_ptr));
        merge_into(&mut left, right);

        assert_eq!(
            entries(&left),
            before,
            "merging an empty sibling must not disturb the left leaf's entries"
        );
        assert_eq!(
            left.next,
            Some(successor_ptr),
            "after a merge the left leaf's next must be the right sibling's old successor — \
             even when that sibling is empty"
        );
    }

    /// Merging into an empty left leaf takes over the sibling wholesale:
    /// its entries and its successor both come across.
    #[test]
    fn merge_into_an_empty_left_sibling_takes_over_contents_and_successor() {
        let successor: Box<Leaf<u64, u64, M>> = Box::new(Leaf::new(None));
        let successor_ptr = NonNull::from(successor.as_ref());

        let mut left: Leaf<u64, u64, M> = Leaf::new(None);
        let mut right: Leaf<u64, u64, M> = Leaf::new(Some(successor_ptr));
        for k in 0..3 {
            right.raw_append(100 + k, v(100 + k));
        }
        let expected = entries(&right);

        merge_into(&mut left, right);

        assert_eq!(
            entries(&left),
            expected,
            "an empty left leaf must end up holding exactly the sibling's entries"
        );
        assert_eq!(
            left.next,
            Some(successor_ptr),
            "after a merge the left leaf's next must be the right sibling's old successor"
        );
    }

    /// Merging moves values, never drops or duplicates them: the live count
    /// is unchanged by the merge itself, and dropping the merged leaf drops
    /// every value from both sides exactly once.
    #[test]
    fn merge_drops_values_exactly_once() {
        let live = Arc::new(AtomicIsize::new(0));
        {
            let mut left: Leaf<u64, Counted, M> = Leaf::new(None);
            for k in 0..3 {
                left.raw_append(k, Counted::new(v(k), &live));
            }
            let mut right: Leaf<u64, Counted, M> = Leaf::new(None);
            for k in 0..2 {
                right.raw_append(100 + k, Counted::new(v(100 + k), &live));
            }
            assert_eq!(live.load(Relaxed), 5, "one live value per stored key before the merge");

            merge_into(&mut left, right);
            assert_eq!(live.load(Relaxed), 5, "the merge itself must not drop any value");
        }
        assert_eq!(
            live.load(Relaxed),
            0,
            "dropping the merged leaf must drop every value from both sides exactly once \
             (positive = leak, negative = double-drop)"
        );
    }
}

#[cfg(test)]
mod splitting_insert_tests {
    //! Unit tests for `Leaf::splitting_insert`, in isolation from the tree.
    //!
    //! Contract pinned: splitting a full leaf at any insertion point yields two
    //! leaves that together hold exactly the old entries plus the new one, in
    //! order, near-balanced (per the midpoint policy: occupancies differ by at
    //! most one) and each with room to spare for further inserts. The split must
    //! hold at every fanout a legal key size can produce — `Key::SIZE` may be
    //! anything in `1..128`, so `M` ranges from 3 to 56 — not just the fanout of
    //! a `u64` key.
    //!
    //! Also pinned: the separator convention. After a split, the parent stores
    //! the right sibling's minimum key (read off the split's actual result, not
    //! computed beforehand) and routes lookups `key < separator` to the left
    //! child and `key >= separator` to the right. Every entry must be findable
    //! in the leaf that convention routes to.

    use super::*;
    use crate::Global;
    use crate::test_util::{M, own, v};

    /// A full leaf holding the odd keys `1, 3, .., 2M-1`, so that sweeping the
    /// even keys `0, 2, .., 2M` lands a new key at every possible insertion
    /// point.
    fn full_leaf<const N: usize>() -> Leaf<u64, u64, N> {
        let mut l = Leaf::new(None);
        for k in 0..N as u64 {
            l.raw_append(2 * k + 1, v(2 * k + 1));
        }
        l
    }

    /// `[u8; N]` keys let the tests hit fanouts other than a u64's. `FANOUT` is
    /// pinned by const assert so the key sizes track `NODE_BUDGET`.
    fn bkey<const N: usize>(k: u8) -> [u8; N] {
        [k; N]
    }

    /// An 80-byte key: `512 / (80 + 8)` = fanout 5.
    const _: () = assert!(<[u8; 80] as Key>::FANOUT == 5);

    /// A 121-byte key: the smallest fanout a legal key size (`SIZE < 128`) can
    /// produce, `512 / (121 + 8)` = 3.
    const _: () = assert!(<[u8; 121] as Key>::FANOUT == 3);

    /// Splitting a full odd-fanout leaf must stay near-balanced at every
    /// insertion point: the two halves hold `M + 1` entries total, differing in
    /// occupancy by at most one.
    #[test]
    fn split_is_near_balanced_at_odd_fanout() {
        const M5: usize = 5;
        for pos in 0..=M5 as u8 {
            let mut l: Leaf<[u8; 80], u64, M5> = Leaf::new(None);
            for k in 0..M5 as u8 {
                l.raw_append(bkey(2 * k + 1), v(k as u64));
            }

            let new_key = bkey(2 * pos);
            let partition = l.find_key(&new_key);
            let right = own(l.splitting_insert(partition, new_key, v(pos as u64), &mut Global));

            assert_eq!(l.occupied + right.occupied, M5 + 1);
            assert!(
                l.occupied.abs_diff(right.occupied) <= 1,
                "split must be near-balanced: left={}, right={} (inserting at partition {partition})",
                l.occupied,
                right.occupied
            );
        }
    }

    /// Splitting must leave room to spare in *both* halves — a sibling that
    /// comes out of a split already full defeats the point of splitting. Pinned
    /// at the minimum legal fanout, where headroom is scarcest.
    #[test]
    fn split_leaves_room_in_both_halves_at_minimum_fanout() {
        const M3: usize = 3;
        for pos in 0..=M3 as u8 {
            let mut l: Leaf<[u8; 121], u64, M3> = Leaf::new(None);
            for k in 0..M3 as u8 {
                l.raw_append(bkey(2 * k + 1), v(k as u64));
            }

            let new_key = bkey(2 * pos);
            let partition = l.find_key(&new_key);
            let right = own(l.splitting_insert(partition, new_key, v(pos as u64), &mut Global));

            assert!(
                l.occupied < M3 && right.occupied < M3,
                "both halves must have room to spare: left={}, right={} (inserting at partition {partition})",
                l.occupied,
                right.occupied
            );
        }
    }

    /// A parent splitting a full child issues the split-insert, then stores the
    /// right sibling's first key as the separator. Routing by that separator
    /// (`key < separator` goes left, `key >= separator` goes right) must find
    /// every one of the `M + 1` entries in the leaf it routes to — most
    /// pointedly the separator key itself, which lives in the right leaf and is
    /// the first casualty of an off-by-one in the routing comparison. Swept
    /// over every insertion point.
    #[test]
    fn separator_routes_every_entry_after_split() {
        for pos in 0..=M as u64 {
            let mut l = full_leaf::<M>();

            let new_key = 2 * pos;
            let partition = l.find_key(&new_key);
            let right = own(l.splitting_insert(partition, new_key, v(new_key), &mut Global));
            let separator = right.keys_ref()[0];

            assert_eq!(
                right.get(&separator),
                Some(&v(separator)),
                "the separator key itself must be served by the right leaf \
                 (inserted {new_key} at partition {partition})"
            );

            let all_keys = (0..M as u64).map(|k| 2 * k + 1).chain([new_key]);
            for k in all_keys {
                let routed = if k < separator { &l } else { &*right };
                assert_eq!(
                    routed.get(&k),
                    Some(&v(k)),
                    "key {k} lost after split: separator {separator} routes it to the \
                     {} leaf, which does not hold it (inserted {new_key} at partition {partition})",
                    if k < separator { "left" } else { "right" },
                );
            }
        }
    }
}

#[cfg(test)]
mod insert_tests {
    //! Unit tests for `Leaf::insert`, the self-contained entry point that
    //! decides on its own between replacing, inserting, and splitting.
    //!
    //! Contract pinned (from `insert`'s doc comment): replace in place when
    //! the key is already present, plain insert when the key is new and there
    //! is room, split when the key is new and the leaf is full. The return
    //! pair reports (replaced value, new right sibling). Concretely:
    //!
    //! - A plain insert returns `(None, None)`, keeps keys sorted with values
    //!   in step, and can fill all `M` slots.
    //! - A splitting insert returns the new right sibling; the two halves
    //!   hold exactly the old entries plus the new one, in order,
    //!   near-balanced, each with room to spare, and the sibling chain is
    //!   spliced.
    //! - Inserting a key the leaf already holds replaces its value in place:
    //!   occupancy unchanged, the new value served, the displaced value
    //!   handed back to the caller (who thereby owns its one drop) — and a
    //!   full leaf must not split over a key it already holds.

    use alloc::sync::Arc;
    use core::sync::atomic::{AtomicIsize, Ordering::Relaxed};

    use super::*;
    use crate::Global;
    use crate::test_util::{Counted, M, entries, own, shuffled, v};

    /// A second value for the same key, distinct from `v(k)`, for replacements.
    fn v2(k: u64) -> u64 {
        v(k) ^ 0xF00D
    }

    /// New keys inserted in shuffled order below capacity report no split and
    /// keep the leaf sorted with values in step; all `M` slots fill before
    /// any split becomes necessary.
    #[test]
    fn insert_fills_every_slot_without_splitting() {
        let mut l: Leaf<u64, u64, M> = Leaf::new(None);
        for k in shuffled(M as u64) {
            let (replaced, split) = l.insert(k, v(k), &mut Global);
            assert!(replaced.is_none(), "key {k} is new — there is no value to replace");
            assert!(split.is_none(), "no split may occur while the leaf has room (inserting {k})");
        }
        assert_eq!(l.occupied, M, "all M slots must be usable");
        let expected: Vec<_> = (0..M as u64).map(|k| (k, v(k))).collect();
        assert_eq!(entries(&l), expected, "keys sorted with values in step");
    }

    /// Inserting a new key into a full leaf must split: the call returns the
    /// new right sibling, the two halves hold exactly the old entries plus
    /// the new one in order, near-balanced with room to spare, and the
    /// sibling chain is spliced. Swept over every insertion point.
    #[test]
    fn insert_into_a_full_leaf_splits() {
        let stored: Vec<u64> = (0..M as u64).map(|k| 2 * k + 1).collect();
        for pos in 0..=M as u64 {
            let mut l: Leaf<u64, u64, M> = Leaf::new(None);
            for &k in &stored {
                l.raw_append(k, v(k));
            }

            let new_key = 2 * pos;
            let (replaced, split) = l.insert(new_key, v(new_key), &mut Global);
            assert!(replaced.is_none(), "key {new_key} is new — there is no value to replace");
            let right_ptr = split.expect("a full leaf must split on a key it does not hold");
            let right = own(right_ptr);

            assert_eq!(
                l.next,
                Some(right_ptr),
                "left's next must point at the new right leaf (inserting {new_key})"
            );

            let mut combined = entries(&l);
            combined.extend(entries(&right));
            let mut expected: Vec<_> =
                stored.iter().copied().chain([new_key]).map(|k| (k, v(k))).collect();
            expected.sort_unstable();
            assert_eq!(
                combined, expected,
                "left then right must hold exactly the old entries plus key {new_key}, in order"
            );

            assert_eq!(l.occupied + right.occupied, M + 1);
            assert!(
                l.occupied.abs_diff(right.occupied) <= 1,
                "split must be near-balanced: left={}, right={} (inserting {new_key})",
                l.occupied,
                right.occupied
            );
            assert!(
                l.occupied < M && right.occupied < M,
                "both halves must have room to spare: left={}, right={} (inserting {new_key})",
                l.occupied,
                right.occupied
            );
        }
    }

    /// Inserting a key the leaf already holds must replace its value in
    /// place: no split reported, occupancy unchanged, the new value served,
    /// every other entry untouched. Checked at the front, middle, and back of
    /// a partially-full leaf.
    #[test]
    fn insert_replaces_existing_key_in_place() {
        const N: u64 = M as u64 - 1;
        for target in [0, N / 2, N - 1] {
            let mut l: Leaf<u64, u64, M> = Leaf::new(None);
            for k in 0..N {
                l.raw_append(k, v(k));
            }

            let (replaced, split) = l.insert(target, v2(target), &mut Global);
            assert!(split.is_none(), "replacing the value of stored key {target} must not split");
            assert_eq!(
                replaced,
                Some(v(target)),
                "the displaced value of stored key {target} must be handed back"
            );
            assert_eq!(
                l.occupied, N as usize,
                "replacing the value of stored key {target} must not change occupancy"
            );
            assert_eq!(
                l.get(&target),
                Some(&v2(target)),
                "stored key {target} must serve the value it was last given"
            );
            let expected: Vec<_> =
                (0..N).map(|k| (k, if k == target { v2(k) } else { v(k) })).collect();
            assert_eq!(
                entries(&l),
                expected,
                "other entries must be untouched (replaced {target})"
            );
        }
    }

    /// A full leaf asked to store a key it already holds must also replace in
    /// place — never split: no new sibling, occupancy still `M`, the new
    /// value served.
    #[test]
    fn insert_of_existing_key_must_not_split_a_full_leaf() {
        for target in [0, M as u64 / 2, M as u64 - 1] {
            let mut l: Leaf<u64, u64, M> = Leaf::new(None);
            for k in 0..M as u64 {
                l.raw_append(k, v(k));
            }

            let (replaced, split) = l.insert(target, v2(target), &mut Global);
            assert!(split.is_none(), "a full leaf must not split over stored key {target}");
            assert_eq!(
                replaced,
                Some(v(target)),
                "the displaced value of stored key {target} must be handed back"
            );
            assert_eq!(l.occupied, M, "occupancy must stay M (replaced {target})");
            assert_eq!(
                l.get(&target),
                Some(&v2(target)),
                "stored key {target} must serve the value it was last given"
            );
        }
    }

    /// A replacement transfers ownership of the displaced value to the
    /// caller: replacing drops nothing by itself, dropping the returned value
    /// drops it exactly once, and the survivors drop exactly once with the
    /// leaf — nothing leaks, nothing double-drops.
    #[test]
    fn replaced_values_drop_exactly_once() {
        const N: u64 = M as u64 - 1;
        let live = Arc::new(AtomicIsize::new(0));
        {
            let mut l: Leaf<u64, Counted, M> = Leaf::new(None);
            for k in 0..N {
                l.raw_append(k, Counted::new(v(k), &live));
            }
            assert_eq!(live.load(Relaxed), N as isize, "one live value per stored key");

            let target = N / 2;
            let (replaced, split) = l.insert(target, Counted::new(v2(target), &live), &mut Global);
            assert!(split.is_none(), "replacing the value of stored key {target} must not split");
            let old = replaced.expect("the displaced value of a stored key must be handed back");
            assert_eq!(
                live.load(Relaxed),
                N as isize + 1,
                "replacement itself must not drop anything: the displaced value is now \
                 owned by the caller"
            );
            drop(old);
            assert_eq!(
                live.load(Relaxed),
                N as isize,
                "dropping the returned value must drop it exactly once"
            );
        }
        assert_eq!(
            live.load(Relaxed),
            0,
            "dropping the leaf must drop every value exactly once \
             (positive = leak, negative = double-drop)"
        );
    }

    /// The insert path as a whole moves values, never duplicates or loses
    /// them: `M + 1` distinct keys through `insert` (forcing exactly one
    /// split), then both halves dropped — every value drops exactly once.
    #[test]
    fn insert_path_drops_values_exactly_once() {
        let live = Arc::new(AtomicIsize::new(0));
        {
            let mut l: Leaf<u64, Counted, M> = Leaf::new(None);
            let mut right = None;
            for k in 0..=M as u64 {
                let (replaced, split) = l.insert(k, Counted::new(v(k), &live), &mut Global);
                assert!(replaced.is_none(), "key {k} is new — there is no value to replace");
                if let Some(ptr) = split {
                    assert!(right.is_none(), "only one split can occur in M + 1 inserts");
                    right = Some(own(ptr));
                }
            }
            assert!(right.is_some(), "M + 1 distinct keys cannot fit in one leaf");
            assert_eq!(live.load(Relaxed), M as isize + 1, "one live value per stored key");
        }
        assert_eq!(
            live.load(Relaxed),
            0,
            "dropping both halves must drop every value exactly once \
             (positive = leak, negative = double-drop)"
        );
    }
}

#[cfg(test)]
mod index_addressed_tests {
    //! Unit tests for the index-addressed primitives the descend/commit
    //! split (and, above it, the entry API) addresses leaves through:
    //! `insert_at`, `remove_at`, and the by-slot value accessors.
    //!
    //! Contract pinned for `insert_at`: the caller has already searched,
    //! so the pair lands at the given partition with no re-search — and
    //! the returned slot pointer addresses the inserted value at EVERY
    //! partition, on both the no-split path and both sides of a full
    //! leaf's split.
    //!
    //! Contract pinned for `remove_at`: the slot's pair comes back and
    //! the survivors close ranks, at every position.

    use alloc::vec::Vec;

    use super::*;
    use crate::Global;
    use crate::test_util::{M, entries, own, v};

    /// Below capacity, `insert_at` shift-inserts at the given partition,
    /// and the returned slot pointer addresses the new value — at every
    /// position.
    #[test]
    fn insert_at_returns_the_inserted_values_slot_below_capacity() {
        for p in 0..M {
            // M - 1 odd keys; the even probe key lands at position `p`.
            let mut l: Leaf<u64, u64, M> = Leaf::new(None);
            for k in 0..(M - 1) as u64 {
                l.raw_append(2 * k + 1, v(2 * k + 1));
            }

            let key = 2 * p as u64;
            let (val_ptr, split) = l.insert_at(p, key, v(key), &mut Global);

            assert!(split.is_none(), "a leaf below capacity must not split (partition {p})");
            assert_eq!(l.len(), M, "the insert must add one pair (partition {p})");
            // SAFETY: the slot pointer addresses a live pair in `l`.
            assert_eq!(
                unsafe { *val_ptr.as_ref() },
                v(key),
                "the returned slot must hold the inserted value (partition {p})"
            );
            assert_eq!(l.get(&key), Some(&v(key)), "the pair must be served (partition {p})");
        }
    }

    /// `insert_at` on a full leaf splits, and the returned slot pointer
    /// must address the inserted value on whichever side it landed —
    /// swept across every partition, covering both sides and the
    /// boundary.
    #[test]
    fn insert_at_on_a_full_leaf_reports_the_slot_across_the_split() {
        for p in 0..=M {
            // Full leaf: keys 10, 20, ..., 10·M, a gap at every position.
            let mut l: Leaf<u64, u64, M> = Leaf::new(None);
            for k in 1..=M as u64 {
                l.raw_append(10 * k, v(10 * k));
            }

            let key = 10 * p as u64 + 5;
            let (val_ptr, split) = l.insert_at(p, key, v(key), &mut Global);
            let right_ptr = split.expect("inserting into a full leaf must split");

            // SAFETY: the slot pointer addresses a live pair in `l` or in
            // the just-returned right sibling, both alive here.
            assert_eq!(
                unsafe { *val_ptr.as_ref() },
                v(key),
                "the returned slot must hold the inserted value (partition {p})"
            );

            assert_eq!(
                l.next(),
                Some(right_ptr),
                "the split must splice the right sibling into the chain (partition {p})"
            );

            let right = own(right_ptr);
            let got: Vec<(u64, u64)> = entries(&l).into_iter().chain(entries(&right)).collect();
            let mut want: Vec<(u64, u64)> = (1..=M as u64).map(|k| (10 * k, v(10 * k))).collect();
            want.insert(p, (key, v(key)));
            assert_eq!(
                got, want,
                "the two leaves together must hold the old pairs plus the new one, \
                 in order (partition {p})"
            );
        }
    }

    /// `remove_at` returns the slot's pair and the survivors close
    /// ranks, at every position of a full leaf.
    #[test]
    fn remove_at_returns_the_pair_and_closes_the_gap() {
        for idx in 0..M {
            let mut l: Leaf<u64, u64, M> = Leaf::new(None);
            for k in 0..M as u64 {
                l.raw_append(10 * k, v(10 * k));
            }

            let pair = l.remove_at(idx);
            assert_eq!(
                pair,
                (10 * idx as u64, v(10 * idx as u64)),
                "remove_at must return the slot's pair (slot {idx})"
            );
            assert_eq!(l.len(), M - 1, "remove_at must shrink the leaf by one pair (slot {idx})");

            let want: Vec<(u64, u64)> =
                (0..M as u64).filter(|k| *k != idx as u64).map(|k| (10 * k, v(10 * k))).collect();
            assert_eq!(entries(&l), want, "the survivors must close ranks in order (slot {idx})");
        }
    }

    /// The by-slot value accessors address the same slots as
    /// `kv_ref_unchecked`, and writes through the mutable one stick.
    #[test]
    fn val_accessors_address_the_slot() {
        let mut l: Leaf<u64, u64, M> = Leaf::new(None);
        for k in 0..4u64 {
            l.raw_append(k, v(k));
        }

        for idx in 0..4 {
            assert_eq!(
                l.val_ref_unchecked(idx),
                l.kv_ref_unchecked(idx).1,
                "val_ref_unchecked must view slot {idx}'s value"
            );
        }
        *l.val_mut_unchecked(2) = 999;
        assert_eq!(l.get(&2), Some(&999), "a val_mut_unchecked write must be visible to get");
    }
}

#[cfg(test)]
mod steal_tests {
    //! Unit tests for `Leaf::steal_from_right` / `steal_from_left` — the
    //! borrow half of classical rebalancing at leaf level, in isolation
    //! from the tree.
    //!
    //! Contract pinned (from the doc comments): exactly one pair crosses
    //! the boundary — the donor's edge pair, keeping both sides sorted
    //! with values in step and occupancies adjusted by one each way; the
    //! returned key is the correct replacement separator (the right
    //! side's new minimum); the sibling chain is untouched; and no value
    //! is dropped, duplicated, or leaked — everything drops exactly once
    //! when the leaves do.
    //!
    //! Occupancies follow the C policy: the receiver is deficient
    //! (`MIN_OCCUPANCY - 1`), the donor strictly above its minimum
    //! (`MIN_OCCUPANCY + 1`), so a steal lands both sides exactly at
    //! `MIN_OCCUPANCY`.

    use alloc::sync::Arc;
    use core::sync::atomic::{AtomicIsize, Ordering::Relaxed};

    use super::*;
    use crate::test_util::{Counted, LMIN as MIN, M, entries, v};

    /// `count` keys `base, base + 10, base + 20, ..` — what
    /// `leaf_of(base, count)` holds, in order.
    fn keys(base: u64, count: usize) -> impl Iterator<Item = u64> {
        (0..count as u64).map(move |i| base + i * 10)
    }

    /// The (key, value) pairs expected for `ks`.
    fn pairs(ks: impl IntoIterator<Item = u64>) -> Vec<(u64, u64)> {
        ks.into_iter().map(|k| (k, v(k))).collect()
    }

    /// A leaf holding `keys(base, count)`, values from `v`.
    fn leaf_of(base: u64, count: usize) -> Leaf<u64, u64, M> {
        let mut leaf: Leaf<u64, u64, M> = Leaf::new(None);
        for k in keys(base, count) {
            leaf.raw_append(k, v(k));
        }
        leaf
    }

    /// Stealing from the right sibling moves exactly the donor's FIRST
    /// pair to the receiver's end, and returns the donor's new first key
    /// as the replacement separator.
    #[test]
    fn steal_from_right_moves_the_donors_first_pair_and_returns_its_new_min() {
        let mut left = leaf_of(0, MIN - 1); // deficient receiver
        let mut right = leaf_of(10_000, MIN + 1); // donor strictly above minimum
        let right_ptr = NonNull::from(&right);
        left.next = Some(right_ptr);

        // SAFETY: `right` is `left`'s chain successor and all its keys are
        // greater than all of `left`'s.
        let sep = unsafe { left.steal_from_right(&mut right) };

        assert_eq!(sep, 10_010, "the replacement separator must be the donor's new first key");
        assert_eq!(
            entries(&left),
            pairs(keys(0, MIN - 1).chain([10_000])),
            "the receiver must gain exactly the donor's first pair, at its end, values in step"
        );
        assert_eq!(
            entries(&right),
            pairs(keys(10_010, MIN)),
            "the donor must lose exactly its first pair, closing the gap, values in step"
        );
        assert_eq!(left.next, Some(right_ptr), "a steal must not touch the sibling chain");
        assert!(!left.is_deficient(), "the steal must lift the receiver out of deficiency");
        assert!(!right.is_deficient(), "the steal must not make the donor deficient");
    }

    /// Stealing from the left sibling moves exactly the donor's LAST pair
    /// to the receiver's front, and returns the moved key itself (the
    /// receiver's new minimum) as the replacement separator.
    #[test]
    fn steal_from_left_moves_the_donors_last_pair_and_returns_the_moved_key() {
        let mut left = leaf_of(0, MIN + 1); // donor strictly above minimum
        let mut right = leaf_of(10_000, MIN - 1); // deficient receiver
        let right_ptr = NonNull::from(&right);
        left.next = Some(right_ptr);

        // SAFETY: `left` is the leaf whose `next` is `right` and all its
        // keys are less than all of `right`'s.
        let sep = unsafe { right.steal_from_left(&mut left) };

        let moved = MIN as u64 * 10; // the donor's last key
        assert_eq!(sep, moved, "the replacement separator must be the moved key itself");
        assert_eq!(
            entries(&right),
            pairs([moved].into_iter().chain(keys(10_000, MIN - 1))),
            "the receiver must gain exactly the donor's last pair, at its front, values in step"
        );
        assert_eq!(
            entries(&left),
            pairs(keys(0, MIN)),
            "the donor must lose exactly its last pair, values in step"
        );
        assert_eq!(left.next, Some(right_ptr), "a steal must not touch the sibling chain");
        assert!(!right.is_deficient(), "the steal must lift the receiver out of deficiency");
        assert!(!left.is_deficient(), "the steal must not make the donor deficient");
    }

    /// A steal from a donor at its minimum legal occupancy
    /// (`MIN_OCCUPANCY + 1`) and a steal into a nearly-full receiver both
    /// stay in bounds.
    #[test]
    fn steal_works_at_the_occupancy_extremes() {
        // Donor at MIN + 1 (the least it can hold and still donate): one
        // steal leaves it exactly at the minimum.
        let mut left = leaf_of(0, MIN - 1);
        let mut right = leaf_of(10_000, MIN + 1);
        // SAFETY: sibling/key-order preconditions hold by construction.
        let sep = unsafe { left.steal_from_right(&mut right) };
        assert_eq!(sep, 10_010);
        assert_eq!(left.len(), MIN);
        assert_eq!(right.len(), MIN, "a minimum-legal donor must end exactly at MIN_OCCUPANCY");
        assert!(!right.is_deficient(), "a steal must never leave the donor deficient");

        // Receiver at M - 1 pairs: the leaf-level contract (`occupied < M`)
        // permits the steal, filling the receiver to exactly M.
        let mut receiver = leaf_of(0, M - 1);
        let mut donor = leaf_of(1_000_000, MIN + 1);
        // SAFETY: sibling/key-order preconditions hold by construction.
        let sep = unsafe { receiver.steal_from_right(&mut donor) };
        assert_eq!(sep, 1_000_010);
        assert_eq!(receiver.len(), M, "a steal may fill the receiver to exactly M");
        assert_eq!(*receiver.keys_ref().last().unwrap(), 1_000_000);
        assert_eq!(donor.len(), MIN);
    }

    /// Steals move values without dropping, duplicating, or leaking any:
    /// the live count is unchanged across steals in both directions, and
    /// everything drops exactly once when the leaves do.
    #[test]
    fn steals_drop_values_exactly_once() {
        let live = Arc::new(AtomicIsize::new(0));
        {
            let mut left: Leaf<u64, Counted, M> = Leaf::new(None);
            for k in keys(0, MIN) {
                left.raw_append(k, Counted::new(k, &live));
            }
            let mut right: Leaf<u64, Counted, M> = Leaf::new(None);
            for k in keys(10_000, MIN + 1) {
                right.raw_append(k, Counted::new(k, &live));
            }
            let total = (2 * MIN + 1) as isize;
            assert_eq!(live.load(Relaxed), total, "one live value per stored key");

            // SAFETY: sibling/key-order preconditions hold by construction;
            // the first steal makes `left` the strictly-above-minimum donor
            // for the second.
            unsafe { left.steal_from_right(&mut right) };
            assert_eq!(live.load(Relaxed), total, "a right-steal must not drop any value");
            // SAFETY: as above.
            unsafe { right.steal_from_left(&mut left) };
            assert_eq!(live.load(Relaxed), total, "a left-steal must not drop any value");
        }
        assert_eq!(
            live.load(Relaxed),
            0,
            "dropping both leaves must drop every value exactly once \
             (positive = leak, negative = double-drop)"
        );
    }
}
