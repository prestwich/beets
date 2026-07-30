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

use alloc::{boxed::Box, sync::Arc};
use core::{
    cell::RefCell,
    sync::atomic::{AtomicIsize, Ordering::Relaxed},
};

use super::*;
use crate::{
    Global,
    test_util::{Counted, LMIN, M, entries, own, shuffled, v},
};

impl<K: Key, V, const M: usize> Leaf<K, V, M> {
    /// Insert `key`/`val`, deciding on its own whether a split is needed:
    /// replace in place when the key is already present, plain insert when
    /// there is room, split when the leaf is full.
    ///
    /// Test-only — production insertion is descend/commit (the tree
    /// searches once and hands the slot to [`Self::insert_at`]): the leaf
    /// tests pin the self-contained contract through it.
    pub(crate) fn insert<A: NodeAllocator<K, V, M, Exhaustion = core::convert::Infallible>>(
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

    /// Remove a key from the leaf, if it exists.
    ///
    /// Test-only since removal went descend/commit (the tree searches
    /// once and hands the slot to [`Self::remove_at`]): the leaf tests
    /// and [`Node`](crate::Node)'s test-only recursive `remove` still drive it.
    pub(crate) fn remove(&mut self, key: &K) -> Option<V> {
        let partition = self.find_key(key);
        if partition >= self.occupied || &self.keys_ref()[partition] != key {
            return None;
        }
        Some(self.remove_at(partition).1)
    }
}

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
        let expected: Vec<_> = (0..M as u64).filter(|&k| k != target).map(|k| (k, v(k))).collect();
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
