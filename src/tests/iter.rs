//! Contract tests for the iterator family, driven entirely through
//! the public API (`iter`, `iter_mut`, `iter_mut_from_key`,
//! `iter_range`, `keys`, `values`, `values_mut`, `into_iter`).
//!
//! The shared contract: iteration yields pairs in ascending key
//! order, exactly the pairs the call promises — all `len()` of them
//! for the full iterators, the in-range window for `iter_range` —
//! whatever shape the tree is in and however it was built. The
//! mutable iterators additionally promise every yielded `&mut V` is
//! the pair's real value: writes through it must be visible to
//! every later read. `into_iter` additionally promises that whatever
//! it hasn't yielded yet is still owned by the tree inside it: it
//! must be dropped exactly once, whether that's through exhaustion
//! or through dropping the iterator early.

use alloc::{sync::Arc, vec::Vec};
use core::ops::Bound;
use core::sync::atomic::{AtomicIsize, Ordering::Relaxed};

use crate::{
    BPlusTree,
    test_util::{Counted, M, v},
};

/// A tree of `n` pairs grown by scattered inserts, so leaf
/// occupancies vary (a sorted load would pack every leaf full).
fn grown(n: u64) -> BPlusTree<u64, u64, M> {
    let mut tree = BPlusTree::new();
    for i in 0..n {
        let k = (i * 7919) % n; // coprime stride: a permutation
        tree.insert(k, v(k));
    }
    tree
}

/// An empty tree iterates as the empty sequence.
#[test]
fn iter_of_an_empty_tree_yields_nothing() {
    let tree: BPlusTree<u64, u64, M> = BPlusTree::new();
    assert_eq!(tree.iter().count(), 0, "an empty tree must yield no pairs");
}

/// A height-0 tree yields its pairs in ascending key order with the
/// right values, however they were inserted.
#[test]
fn iter_yields_a_root_leafs_pairs_in_order() {
    let mut tree: BPlusTree<u64, u64, M> = BPlusTree::new();
    for k in (0..M as u64).rev() {
        tree.insert(k, v(k));
    }
    let got: Vec<(u64, u64)> = tree.iter().map(|(k, val)| (*k, *val)).collect();
    let want: Vec<(u64, u64)> = (0..M as u64).map(|k| (k, v(k))).collect();
    assert_eq!(got, want, "iteration must yield every pair in ascending key order");
}

/// Iteration must cross every leaf boundary: a multi-level tree
/// yields exactly `len()` pairs, in ascending key order, from both
/// construction paths (bulk-loaded and insert-grown).
#[test]
fn iter_walks_the_whole_tree_in_order() {
    let n = (M * M + 1) as u64;
    let loaded: BPlusTree<u64, u64, M> = BPlusTree::from_sorted_iter((0..n).map(|k| (k, v(k))));

    for (tree, how) in [(&loaded, "bulk-loaded"), (&grown(n), "insert-grown")] {
        let mut expect = 0u64;
        for (k, val) in tree.iter() {
            assert_eq!(*k, expect, "iteration must visit every key in ascending order ({how})");
            assert_eq!(*val, v(expect), "each key must carry its own value ({how})");
            expect += 1;
        }
        assert_eq!(expect, n, "iteration must yield exactly len() pairs ({how})");
    }
}

/// `iter().len()` must report the pairs REMAINING, per
/// `ExactSizeIterator`'s contract: full at the start, shrinking as
/// pairs are consumed, zero at exhaustion.
#[test]
fn iter_len_reports_the_remaining_pairs() {
    let n = 2 * M + 1;
    let tree: BPlusTree<u64, u64, M> =
        BPlusTree::from_sorted_iter((0..n as u64).map(|k| (k, v(k))));

    let mut it = tree.iter();
    assert_eq!(it.len(), n, "a fresh iterator's len must be the tree's len");
    it.next();
    assert_eq!(it.len(), n - 1, "len must shrink as pairs are consumed");
    for _ in it.by_ref().take(M) {}
    assert_eq!(it.len(), n - 1 - M, "len must track consumption across a leaf hop");
    for _ in it.by_ref() {}
    assert_eq!(it.len(), 0, "an exhausted iterator's len must be zero");
}

// ── iter_mut / values_mut ───────────────────────────────────────────

/// `iter_mut` visits every pair in ascending key order, and each
/// yielded `&mut V` is the pair's real value: the writes must be
/// visible to every later read.
#[test]
fn iter_mut_mutates_every_pair_in_order() {
    let n = (M * M + 1) as u64;
    let loaded: BPlusTree<u64, u64, M> = BPlusTree::from_sorted_iter((0..n).map(|k| (k, v(k))));

    for (mut tree, how) in [(loaded, "bulk-loaded"), (grown(n), "insert-grown")] {
        let mut expect = 0u64;
        for (k, val) in tree.iter_mut() {
            assert_eq!(*k, expect, "iter_mut must visit every key in ascending order ({how})");
            assert_eq!(*val, v(expect), "each key must carry its own value ({how})");
            *val += 1;
            expect += 1;
        }
        assert_eq!(expect, n, "iter_mut must yield exactly len() pairs ({how})");

        for (k, val) in tree.iter() {
            assert_eq!(*val, v(*k) + 1, "the write through key {k}'s &mut must stick ({how})");
        }
    }
}

/// `iter_mut` on an empty tree yields nothing.
#[test]
fn iter_mut_of_an_empty_tree_yields_nothing() {
    let mut tree: BPlusTree<u64, u64, M> = BPlusTree::new();
    assert_eq!(tree.iter_mut().count(), 0, "an empty tree must yield no pairs");
}

/// `values_mut` walks values in ascending key order and its writes
/// stick.
#[test]
fn values_mut_mutates_every_value() {
    let n = 2 * M as u64 + 1;
    let mut tree: BPlusTree<u64, u64, M> = BPlusTree::from_sorted_iter((0..n).map(|k| (k, v(k))));

    for val in tree.values_mut() {
        *val ^= 0xF00D;
    }
    for (k, val) in tree.iter() {
        assert_eq!(*val, v(*k) ^ 0xF00D, "the write through key {k}'s value must stick");
    }
}

// ── keys / values ───────────────────────────────────────────────────

/// `keys` and `values` are the pair iteration, projected: same
/// order, same count, matching halves.
#[test]
fn keys_and_values_project_the_pairs_in_order() {
    let n = 2 * M as u64 + 1;
    let tree: BPlusTree<u64, u64, M> = BPlusTree::from_sorted_iter((0..n).map(|k| (k, v(k))));

    let keys: Vec<u64> = tree.keys().copied().collect();
    let want_keys: Vec<u64> = (0..n).collect();
    assert_eq!(keys, want_keys, "keys() must yield every key in ascending order");

    let values: Vec<u64> = tree.values().copied().collect();
    let want_values: Vec<u64> = (0..n).map(v).collect();
    assert_eq!(values, want_values, "values() must yield every value in key order");
}

/// The iterator family is fused: each type declares
/// `FusedIterator` (the helper's bound makes that a compile-time
/// check), and an exhausted iterator keeps returning `None`.
#[test]
fn the_iterator_family_is_fused() {
    fn fused<I: core::iter::FusedIterator>(it: I) -> I {
        it
    }

    let n = 2 * M as u64 + 1;
    let mut tree: BPlusTree<u64, u64, M> = BPlusTree::from_sorted_iter((0..n).map(|k| (k, v(k))));

    let mut it = fused(tree.iter());
    for _ in it.by_ref() {}
    assert!(it.next().is_none(), "an exhausted iter must stay exhausted");
    assert!(it.next().is_none(), "and stay that way");

    let mut it = fused(tree.range(1..2 * M as u64));
    for _ in it.by_ref() {}
    assert!(it.next().is_none(), "an exhausted iter_range must stay exhausted");
    assert!(it.next().is_none(), "and stay that way");

    let mut it = fused(tree.iter_mut());
    for _ in it.by_ref() {}
    assert!(it.next().is_none(), "an exhausted iter_mut must stay exhausted");
    assert!(it.next().is_none(), "and stay that way");
}

// ── iter_range ──────────────────────────────────────────────────────

/// A `start..` range on a present key starts at that key, inclusive,
/// and yields the complete sorted tail — wherever in its leaf the
/// key sits.
#[test]
fn range_from_a_present_key_yields_the_tail_from_that_key() {
    let n = (M * M + 1) as u64;
    let tree: BPlusTree<u64, u64, M> = BPlusTree::from_sorted_iter((0..n).map(|k| (k, v(k))));

    // Starts spanning leaf interiors and boundaries.
    for start in [0, 1, M as u64 - 1, M as u64, n / 2, n - 1] {
        let got: Vec<u64> = tree.range(start..).map(|(k, _)| *k).collect();
        let want: Vec<u64> = (start..n).collect();
        assert_eq!(got, want, "the tail from present key {start} must be complete");
    }
}

/// A `start..` range on an absent key starts at the next greater
/// key; an explicitly excluded start skips an exact hit.
#[test]
fn range_start_bounds_are_honored() {
    // Store the even keys, probe odd ones between them.
    let n = 3 * M as u64;
    let tree: BPlusTree<u64, u64, M> =
        BPlusTree::from_sorted_iter((0..n).map(|k| (2 * k, v(2 * k))));

    for probe in [1, M as u64 * 2 - 1, n - 1] {
        assert_eq!(
            tree.range(probe..).next().map(|(k, _)| *k),
            Some(probe + 1),
            "the tail from absent key {probe} must start at its successor"
        );
    }
    assert_eq!(
        tree.range((Bound::Excluded(4u64), Bound::Unbounded)).next().map(|(k, _)| *k),
        Some(6),
        "an excluded start must skip its exact hit"
    );
}

/// A bounded window yields exactly the keys inside it: the end bound
/// excludes (`..end`) or includes (`..=end`) its endpoint.
#[test]
fn range_end_bounds_are_honored() {
    let n = (M * M + 1) as u64;
    let tree: BPlusTree<u64, u64, M> = BPlusTree::from_sorted_iter((0..n).map(|k| (k, v(k))));

    // Windows spanning leaf interiors and boundaries.
    for (start, end) in [(1, 5), (0, M as u64), (M as u64 - 1, M as u64 + 1), (n / 2, n - 1)] {
        let got: Vec<u64> = tree.range(start..end).map(|(k, _)| *k).collect();
        let want: Vec<u64> = (start..end).collect();
        assert_eq!(got, want, "{start}..{end} must yield exactly the window");

        let got: Vec<u64> = tree.range(start..=end).map(|(k, _)| *k).collect();
        let want: Vec<u64> = (start..=end).collect();
        assert_eq!(got, want, "{start}..={end} must include its endpoint");
    }
}

/// The unbounded range is the full iteration, and ranges that cover
/// no keys — past the maximum, or inverted — yield nothing.
#[test]
fn range_degenerate_cases() {
    let n = 2 * M as u64;
    let tree: BPlusTree<u64, u64, M> = BPlusTree::from_sorted_iter((0..n).map(|k| (k, v(k))));

    assert_eq!(tree.range(..).count(), n as usize, ".. must yield every pair");
    assert_eq!(tree.range(n..).count(), 0, "a range above every key must yield nothing");
    assert_eq!(
        tree.range(u64::MAX..).count(),
        0,
        "a range at the key-space maximum must yield nothing"
    );
    assert_eq!(tree.range(5..5).count(), 0, "an empty window must yield nothing");
    #[allow(clippy::reversed_empty_ranges)]
    let inverted = 7..3;
    assert_eq!(tree.range(inverted).count(), 0, "an inverted range must yield nothing");

    let empty: BPlusTree<u64, u64, M> = BPlusTree::new();
    assert_eq!(empty.range(0..).count(), 0, "an empty tree must yield nothing");
}

/// An excluded start above every stored key is an empty range: it
/// must yield nothing.
#[test]
fn range_excluded_start_above_the_maximum_yields_nothing() {
    let n = 2 * M as u64;
    let tree: BPlusTree<u64, u64, M> = BPlusTree::from_sorted_iter((0..n).map(|k| (k, v(k))));
    assert_eq!(
        tree.range((Bound::Excluded(n + 5), Bound::Unbounded)).count(),
        0,
        "an excluded start above every key must yield nothing"
    );
}

// ── range_mut ───────────────────────────────────────────────────────

/// `range_mut` yields exactly the in-window pairs, in ascending key
/// order, and its writes stick — pairs outside the window must be
/// untouched.
#[test]
fn range_mut_mutates_exactly_the_window() {
    let n = (M * M + 1) as u64;
    // Windows spanning leaf interiors and boundaries.
    for (start, end) in [(1, 5), (0, M as u64), (M as u64 - 1, M as u64 + 1), (n / 2, n - 1)] {
        let mut tree: BPlusTree<u64, u64, M> =
            BPlusTree::from_sorted_iter((0..n).map(|k| (k, v(k))));

        let mut expect = start;
        for (k, val) in tree.range_mut(start..end) {
            assert_eq!(*k, expect, "{start}..{end} must visit the window in order");
            *val += 1;
            expect += 1;
        }
        assert_eq!(expect, end, "{start}..{end} must visit the whole window");

        for (k, val) in tree.iter() {
            let want = if (start..end).contains(k) { v(*k) + 1 } else { v(*k) };
            assert_eq!(*val, want, "only the window {start}..{end} may change (key {k})");
        }
    }
}

/// `range_mut` honors the same bound conventions as `range`: an
/// inclusive end includes its endpoint, and an excluded start skips
/// an exact hit.
#[test]
fn range_mut_bounds_are_honored() {
    let n = 2 * M as u64;
    let mut tree: BPlusTree<u64, u64, M> = BPlusTree::from_sorted_iter((0..n).map(|k| (k, v(k))));

    for (_, val) in tree.range_mut(2..=5) {
        *val += 1;
    }
    for (_, val) in tree.range_mut((Bound::Excluded(M as u64), Bound::Unbounded)) {
        *val += 10;
    }

    for (k, val) in tree.iter() {
        let mut want = v(*k);
        if (2..=5).contains(k) {
            want += 1;
        }
        if *k > M as u64 {
            want += 10;
        }
        assert_eq!(*val, want, "bounds must be honored exactly (key {k})");
    }
}

/// Ranges that cover no keys mutate nothing: empty and inverted
/// windows, starts above every key (included or excluded), and the
/// empty tree.
#[test]
fn range_mut_degenerate_cases_mutate_nothing() {
    let n = 2 * M as u64;
    let mut tree: BPlusTree<u64, u64, M> = BPlusTree::from_sorted_iter((0..n).map(|k| (k, v(k))));

    assert_eq!(tree.range_mut(5..5).count(), 0, "an empty window must yield nothing");
    #[allow(clippy::reversed_empty_ranges)]
    let inverted = 7..3;
    assert_eq!(tree.range_mut(inverted).count(), 0, "an inverted range must yield nothing");
    assert_eq!(tree.range_mut(n..).count(), 0, "a range above every key must yield nothing");
    assert_eq!(
        tree.range_mut((Bound::Excluded(n + 5), Bound::Unbounded)).count(),
        0,
        "an excluded start above every key must yield nothing"
    );

    for (k, val) in tree.iter() {
        assert_eq!(*val, v(*k), "degenerate ranges must not mutate anything (key {k})");
    }

    let mut empty: BPlusTree<u64, u64, M> = BPlusTree::new();
    assert_eq!(empty.range_mut(0..).count(), 0, "an empty tree must yield nothing");
}

// ── into_iter ───────────────────────────────────────────────────────

/// Consuming a tree by value yields every pair in ascending key
/// order, exactly `len()` of them — the same order and coverage
/// `iter()` promises, whatever the tree's shape or how it was built.
#[test]
fn into_iter_yields_every_pair_in_ascending_order() {
    let n = (M * M + 1) as u64;
    let loaded: BPlusTree<u64, u64, M> = BPlusTree::from_sorted_iter((0..n).map(|k| (k, v(k))));

    for (tree, how) in [(loaded, "bulk-loaded"), (grown(n), "insert-grown")] {
        let mut expect = 0u64;
        for (k, val) in tree {
            assert_eq!(k, expect, "into_iter must visit every key in ascending order ({how})");
            assert_eq!(val, v(expect), "each key must carry its own value ({how})");
            expect += 1;
        }
        assert_eq!(expect, n, "into_iter must yield exactly len() pairs ({how})");
    }
}

/// Consuming an empty tree yields the empty sequence.
#[test]
fn into_iter_of_an_empty_tree_yields_nothing() {
    let tree: BPlusTree<u64, u64, M> = BPlusTree::new();
    assert_eq!(tree.into_iter().count(), 0, "an empty tree must yield no pairs");
}

/// `into_iter().len()` must report the pairs REMAINING, per
/// `ExactSizeIterator`'s contract: full at the start, shrinking as
/// pairs are consumed, zero at exhaustion.
#[test]
fn into_iter_len_reports_the_remaining_pairs() {
    let n = 2 * M + 1;
    let tree: BPlusTree<u64, u64, M> =
        BPlusTree::from_sorted_iter((0..n as u64).map(|k| (k, v(k))));

    let mut it = tree.into_iter();
    assert_eq!(it.len(), n, "a fresh into_iter's len must be the tree's len");
    it.next();
    assert_eq!(it.len(), n - 1, "len must shrink as pairs are consumed");
    for _ in it.by_ref().take(M) {}
    assert_eq!(it.len(), n - 1 - M, "len must track consumption across a leaf hop");
    for _ in it.by_ref() {}
    assert_eq!(it.len(), 0, "an exhausted into_iter's len must be zero");
}

/// `into_iter` is fused: it declares `FusedIterator` (the helper's
/// bound makes that a compile-time check), and an exhausted iterator
/// keeps returning `None`.
#[test]
fn into_iter_is_fused() {
    fn fused<I: core::iter::FusedIterator>(it: I) -> I {
        it
    }

    let n = 2 * M as u64 + 1;
    let tree: BPlusTree<u64, u64, M> = BPlusTree::from_sorted_iter((0..n).map(|k| (k, v(k))));

    let mut it = fused(tree.into_iter());
    for _ in it.by_ref() {}
    assert!(it.next().is_none(), "an exhausted into_iter must stay exhausted");
    assert!(it.next().is_none(), "and stay that way");
}

/// Whatever `into_iter` hasn't yielded yet is still owned by the tree
/// riding inside it: dropping it early — mid-consumption, before
/// exhaustion — must drop each of those not-yet-yielded values
/// exactly once. Pairs already taken out are the taker's
/// responsibility, not the iterator's, so they must be untouched by
/// that drop.
#[test]
fn into_iter_drop_drops_the_unyielded_values_exactly_once() {
    let n = (M * M + 1) as u64;
    let live = Arc::new(AtomicIsize::new(0));
    let mut tree: BPlusTree<u64, Counted, M> = BPlusTree::new();
    for k in 0..n {
        tree.insert(k, Counted::new(k, &live));
    }
    assert_eq!(live.load(Relaxed), n as isize, "one live value per inserted key");

    let mut it = tree.into_iter();
    let taken: Vec<(u64, Counted)> = (&mut it).take((n / 3) as usize).collect();
    assert_eq!(
        live.load(Relaxed),
        n as isize,
        "taking pairs out of the iterator must not itself drop anything"
    );

    drop(it);
    assert_eq!(
        live.load(Relaxed),
        taken.len() as isize,
        "dropping a not-yet-exhausted into_iter must drop exactly the values \
         it hadn't yielded, once each (positive = leak, negative = double-drop)"
    );

    drop(taken);
    assert_eq!(live.load(Relaxed), 0, "the values already taken out must drop exactly once too");
}

/// Fully draining `into_iter` to exhaustion — never dropping it
/// early — must also drop every value exactly once: the empty tree
/// left behind still has to fall, and it must fall clean.
#[test]
fn into_iter_full_drain_drops_every_value_exactly_once() {
    let n = (M * M + 1) as u64;
    let live = Arc::new(AtomicIsize::new(0));
    let mut tree: BPlusTree<u64, Counted, M> = BPlusTree::new();
    for k in 0..n {
        tree.insert(k, Counted::new(k, &live));
    }

    let drained: Vec<(u64, Counted)> = tree.into_iter().collect();
    assert_eq!(drained.len(), n as usize, "draining must yield every pair");
    assert_eq!(
        live.load(Relaxed),
        n as isize,
        "values moved out by a full drain must still be live, held by the collection"
    );

    drop(drained);
    assert_eq!(
        live.load(Relaxed),
        0,
        "dropping the drained pairs must drop every value exactly once \
         (positive = leak, negative = double-drop)"
    );
}
