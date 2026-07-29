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
}

#[cfg(test)]
#[path = "../tests/leaf.rs"]
mod tests;

#[cfg(test)]
#[path = "../tests/leaf_merge.rs"]
mod merge_tests;

#[cfg(test)]
#[path = "../tests/leaf_splitting_insert.rs"]
mod splitting_insert_tests;

#[cfg(test)]
#[path = "../tests/leaf_insert.rs"]
mod insert_tests;

#[cfg(test)]
#[path = "../tests/leaf_index_addressed.rs"]
mod index_addressed_tests;

#[cfg(test)]
#[path = "../tests/leaf_steal.rs"]
mod steal_tests;
