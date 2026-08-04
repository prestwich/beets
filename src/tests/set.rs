//! Contract tests for `BPlusSet`, the sorted-set wrapper over
//! `BPlusTree`. Everything here drives the public API only — `Set`
//! has no private-field fixtures of its own, and its logic is a thin
//! value-dropping translation layer over `BPlusTree`; the tree's own
//! structural invariants (splits, merges, rebalancing) are pinned in
//! `tree.rs`. `alloc::collections::BTreeSet` stands in as the
//! reference model throughout, the same role `BTreeMap` plays for the
//! tree's own churn/differential tests.

use super::*;
use crate::test_util::{M, xorshift};

// ── construction / basic reads ──────────────────────────────────────

#[test]
fn new_set_is_empty_and_reads_miss() {
    let set: BPlusSet<u64, M> = BPlusSet::new();
    assert_eq!(set.len(), 0, "a new set holds no keys");
    assert!(set.is_empty());
    assert!(!set.contains(&0), "a new set must miss on any key");
    assert_eq!(set.get(&0), None);
    assert_eq!(set.first(), None);
    assert_eq!(set.last(), None);
}

#[test]
fn default_constructs_an_empty_set() {
    let set: BPlusSet<u64, M> = BPlusSet::default();
    assert!(set.is_empty());
    assert_eq!(set.len(), 0);
}

// ── insert / remove / contains / get ────────────────────────────────

/// `insert` reports whether the key is new to the set: `true` the
/// first time a key is inserted, `false` on every later insert of the
/// same key — which leaves the set unchanged, since there is no
/// second value to overwrite.
#[test]
fn insert_reports_whether_the_key_was_new() {
    let mut set: BPlusSet<u64, M> = BPlusSet::new();
    assert!(set.insert(7), "the first insert of a key must report it as new");
    assert!(!set.insert(7), "re-inserting an already-present key must report it as not new");
    assert_eq!(set.len(), 1, "re-inserting a present key must not change the set's size");
}

/// `remove` reports whether the key was present: `true` on a hit,
/// `false` on a miss, and a miss must leave the set untouched.
#[test]
fn remove_reports_whether_the_key_was_present() {
    let mut set: BPlusSet<u64, M> = BPlusSet::new();
    set.insert(7);

    assert!(set.remove(&7), "removing a present key must report true");
    assert_eq!(set.len(), 0, "a hit must decrement len");
    assert!(!set.remove(&7), "removing an already-absent key must report false");
    assert_eq!(set.len(), 0, "a miss must not change len");
}

#[test]
fn contains_reflects_membership() {
    let mut set: BPlusSet<u64, M> = BPlusSet::new();
    assert!(!set.contains(&1), "an empty set contains nothing");

    set.insert(1);
    assert!(set.contains(&1), "an inserted key must be reported present");
    assert!(!set.contains(&2), "an absent key must be reported missing");

    set.remove(&1);
    assert!(!set.contains(&1), "a removed key must be reported missing");
}

/// `get` returns a reference to the set's own stored copy of `key`,
/// or `None` if it is absent.
#[test]
fn get_returns_a_reference_to_the_stored_key() {
    let mut set: BPlusSet<u64, M> = BPlusSet::new();
    assert_eq!(set.get(&5), None, "an absent key has no stored copy to return");

    set.insert(5);
    assert_eq!(set.get(&5), Some(&5), "a present key's stored copy must equal the key");
    assert_eq!(set.get(&6), None, "a different absent key must still miss");
}

// ── clear ────────────────────────────────────────────────────────────

#[test]
fn clear_resets_to_an_empty_set() {
    let mut set: BPlusSet<u64, M> = (0..300u64).collect();
    assert_eq!(set.len(), 300);

    set.clear();
    assert!(set.is_empty());
    assert_eq!(set.len(), 0);
    assert!(!set.contains(&0));

    set.insert(1);
    assert_eq!(set.len(), 1, "the set must remain usable after clear");
}

// ── first / last / pop_first / pop_last ─────────────────────────────

/// `first`/`last`: `None` on the empty set, the extreme keys
/// otherwise — tracking growth and removal.
#[test]
fn first_and_last_return_the_extreme_keys() {
    let mut set: BPlusSet<u64, M> = BPlusSet::new();
    assert_eq!(set.first(), None, "an empty set has no first key");
    assert_eq!(set.last(), None, "an empty set has no last key");

    set.insert(5);
    assert_eq!(set.first(), Some(&5), "a lone key is both extremes");
    assert_eq!(set.last(), Some(&5), "a lone key is both extremes");

    let n = (M * M + 1) as u64;
    let mut set: BPlusSet<u64, M> = BPlusSet::from_sorted_iter(0..n);
    assert_eq!(set.first(), Some(&0), "first must be the minimum key");
    assert_eq!(set.last(), Some(&(n - 1)), "last must be the maximum key");

    set.remove(&0);
    set.remove(&(n - 1));
    assert_eq!(set.first(), Some(&1), "first must track removal of the minimum");
    assert_eq!(set.last(), Some(&(n - 2)), "last must track removal of the maximum");
}

#[test]
fn pop_on_an_empty_set_returns_none() {
    let mut set: BPlusSet<u64, M> = BPlusSet::new();
    assert_eq!(set.pop_first(), None, "a fresh set has no first key to pop");
    assert_eq!(set.pop_last(), None, "a fresh set has no last key to pop");

    set.insert(1);
    set.remove(&1);
    assert_eq!(set.pop_first(), None, "an emptied set has no first key to pop");
    assert_eq!(set.pop_last(), None, "an emptied set has no last key to pop");
    assert_eq!(set.len(), 0, "a pop miss must not change len");
}

#[test]
fn popping_the_lone_key_empties_the_set() {
    for pop_last in [false, true] {
        let mut set: BPlusSet<u64, M> = BPlusSet::new();
        set.insert(7);

        let got = if pop_last { set.pop_last() } else { set.pop_first() };
        assert_eq!(got, Some(7), "the lone key is both extremes (pop_last={pop_last})");
        assert!(set.is_empty(), "popping the lone key must empty the set (pop_last={pop_last})");
    }
}

/// `pop_first` removes and returns the minimum key — the key `first`
/// reports — decrementing `len` and promoting the next key to
/// minimum.
#[test]
fn pop_first_returns_the_minimum_key() {
    let n = (M * M + 1) as u64;
    let mut set: BPlusSet<u64, M> = BPlusSet::from_sorted_iter(0..n);

    assert_eq!(set.pop_first(), Some(0), "pop_first must return the minimum key");
    assert_eq!(set.len(), (n - 1) as usize, "a pop hit must decrement len");
    assert!(!set.contains(&0), "the popped key must be gone from the set");
    assert_eq!(set.first(), Some(&1), "the next key up must become the minimum");
}

/// `pop_last` removes and returns the maximum key — the key `last`
/// reports — decrementing `len` and demoting the next key down to
/// maximum.
#[test]
fn pop_last_returns_the_maximum_key() {
    let n = (M * M + 1) as u64;
    let mut set: BPlusSet<u64, M> = BPlusSet::from_sorted_iter(0..n);

    assert_eq!(set.pop_last(), Some(n - 1), "pop_last must return the maximum key");
    assert_eq!(set.len(), (n - 1) as usize, "a pop hit must decrement len");
    assert!(!set.contains(&(n - 1)), "the popped key must be gone from the set");
    assert_eq!(set.last(), Some(&(n - 2)), "the next key down must become the maximum");
}

// ── iteration ────────────────────────────────────────────────────────

#[test]
fn iter_yields_keys_in_ascending_order() {
    const N: u64 = 2_000;
    let mut set: BPlusSet<u64, M> = BPlusSet::new();
    for i in 0..N {
        set.insert((i * 37) % N); // coprime stride: a shuffled bijection
    }
    assert_eq!(set.len(), N as usize);
    assert!(set.iter().copied().eq(0..N), "iter must yield every key in ascending order");
}

#[test]
fn iter_len_matches_the_sets_len() {
    let set: BPlusSet<u64, M> = BPlusSet::from_sorted_iter(0..500u64);
    assert_eq!(set.iter().len(), set.len(), "iter must be exact-sized");
}

#[test]
fn range_yields_keys_within_bounds_in_ascending_order() {
    let n = (M * M + 1) as u64;
    let set: BPlusSet<u64, M> = BPlusSet::from_sorted_iter(0..n);
    let (lo, hi) = (n / 4, 3 * n / 4);
    assert!(
        set.range(lo..hi).copied().eq(lo..hi),
        "range must yield exactly the keys within the given bounds, in ascending order"
    );
}

#[test]
fn into_iter_for_ref_matches_iter() {
    let set: BPlusSet<u64, M> = BPlusSet::from_sorted_iter(0..500u64);
    assert!((&set).into_iter().eq(set.iter()), "&set's IntoIterator must match iter()");
}

#[test]
fn debug_formats_like_btreeset() {
    use alloc::{collections::BTreeSet, format};

    let n = 2 * M as u64 + 3;
    let set: BPlusSet<u64, M> = BPlusSet::from_sorted_iter(0..n);
    let model: BTreeSet<u64> = (0..n).collect();
    assert_eq!(
        format!("{set:?}"),
        format!("{model:?}"),
        "Debug must render the keys in ascending order, debug_set-shaped"
    );

    let empty: BPlusSet<u64, M> = BPlusSet::new();
    assert_eq!(format!("{empty:?}"), "{}", "an empty set must render as an empty set");
}

// ── FromIterator / from_sorted_iter / Extend ────────────────────────

#[test]
fn from_iterator_collects_distinct_keys() {
    let set: BPlusSet<u64, M> = (0..300u64).chain(0..300u64).collect();
    assert_eq!(set.len(), 300, "duplicate keys in the input must collapse to one");
    assert!(set.iter().copied().eq(0..300u64));
}

#[test]
fn from_sorted_iter_builds_correct_sets_at_awkward_sizes() {
    let m = M as u64;
    #[rustfmt::skip]
    let sizes = [
        0, 1, 2, m - 1, m, m + 1, 2 * m, 2 * m + 1,
        m * m, m * m + 1, m * m + m + 3,
    ];
    for n in sizes {
        let set: BPlusSet<u64, M> = BPlusSet::from_sorted_iter(0..n);
        assert_eq!(set.len(), n as usize, "len must count the loaded keys (n={n})");
        assert!(set.iter().copied().eq(0..n), "iter must yield every loaded key in order (n={n})");
        assert!(!set.contains(&n), "an unloaded key must miss (n={n})");
    }
}

#[test]
fn extend_adds_every_key() {
    let mut set: BPlusSet<u64, M> = BPlusSet::new();
    set.insert(1);
    set.extend(2..300u64);
    assert_eq!(set.len(), 299);
    assert!(set.iter().copied().eq(1..300u64));
}

// ── churn ─────────────────────────────────────────────────────────────

/// A deterministic insert/remove churn mirrored against
/// `alloc::collections::BTreeSet`. Every operation must agree with the
/// model — return values and length — with the final contents
/// matching too.
#[test]
fn churn_mirrors_btreeset() {
    use alloc::collections::BTreeSet;

    let mut set: BPlusSet<u64, M> = BPlusSet::new();
    let mut model: BTreeSet<u64> = BTreeSet::new();
    let mut state: u64 = 0x5EED_CAFE_F00D_D00D;

    for step in 0..1_500u64 {
        let r = xorshift(&mut state);
        let key = r % 200;
        if (r >> 32) % 5 < 3 {
            assert_eq!(
                set.insert(key),
                model.insert(key),
                "insert({key}) must agree with the model at step {step}"
            );
        } else {
            assert_eq!(
                set.remove(&key),
                model.remove(&key),
                "remove({key}) must agree with the model at step {step}"
            );
        }
        assert_eq!(set.len(), model.len(), "len must agree with the model at step {step}");
    }

    assert!(set.iter().copied().eq(model.iter().copied()), "final contents must match the model");
}

// ── set algebra ──────────────────────────────────────────────────────

#[test]
fn is_subset_and_is_superset_agree_with_the_model() {
    use alloc::collections::BTreeSet;

    let a: BPlusSet<u64, M> = (0..50u64).collect();
    let b: BPlusSet<u64, M> = (0..100u64).collect();
    let c: BPlusSet<u64, M> = (200..250u64).collect();

    let ma: BTreeSet<u64> = (0..50u64).collect();
    let mb: BTreeSet<u64> = (0..100u64).collect();
    let mc: BTreeSet<u64> = (200..250u64).collect();

    assert_eq!(a.is_subset(&b), ma.is_subset(&mb), "a strict subset must report true");
    assert_eq!(b.is_subset(&a), mb.is_subset(&ma), "a strict superset must not report as a subset");
    assert_eq!(a.is_subset(&a), ma.is_subset(&ma), "a set must be a subset of itself");
    assert_eq!(a.is_subset(&c), ma.is_subset(&mc), "disjoint sets are not subsets of one another");

    assert_eq!(b.is_superset(&a), mb.is_superset(&ma), "is_superset must mirror is_subset");
    assert_eq!(a.is_superset(&b), ma.is_superset(&mb), "is_superset must mirror is_subset");
}

#[test]
fn is_disjoint_reflects_shared_membership() {
    let a: BPlusSet<u64, M> = (0..50u64).collect();
    let b: BPlusSet<u64, M> = (200..250u64).collect();
    let c: BPlusSet<u64, M> = (40..90u64).collect();

    assert!(a.is_disjoint(&b), "sets with no shared keys must report disjoint");
    assert!(!a.is_disjoint(&c), "sets sharing at least one key must not report disjoint");
    assert!(!a.is_disjoint(&a), "a non-empty set is never disjoint from itself");
}

#[test]
fn union_yields_keys_from_either_set_without_duplicates() {
    use alloc::collections::BTreeSet;

    let a: BPlusSet<u64, M> = (0..150u64).collect();
    let b: BPlusSet<u64, M> = (100..250u64).collect();
    let ma: BTreeSet<u64> = (0..150u64).collect();
    let mb: BTreeSet<u64> = (100..250u64).collect();

    assert!(
        a.union(&b).eq(ma.union(&mb)),
        "union must yield every key present in either set, in ascending order, without duplicates"
    );
}

#[test]
fn intersection_yields_keys_common_to_both_sets() {
    use alloc::collections::BTreeSet;

    let a: BPlusSet<u64, M> = (0..150u64).collect();
    let b: BPlusSet<u64, M> = (100..250u64).collect();
    let ma: BTreeSet<u64> = (0..150u64).collect();
    let mb: BTreeSet<u64> = (100..250u64).collect();

    assert!(
        a.intersection(&b).eq(ma.intersection(&mb)),
        "intersection must yield exactly the keys present in both sets, in ascending order"
    );
}

#[test]
fn difference_yields_keys_only_in_the_first_set() {
    use alloc::collections::BTreeSet;

    let a: BPlusSet<u64, M> = (0..150u64).collect();
    let b: BPlusSet<u64, M> = (100..250u64).collect();
    let ma: BTreeSet<u64> = (0..150u64).collect();
    let mb: BTreeSet<u64> = (100..250u64).collect();

    assert!(
        a.difference(&b).eq(ma.difference(&mb)),
        "difference must yield exactly the keys in the first set but not the second, in ascending order"
    );
}

#[test]
fn symmetric_difference_yields_keys_in_exactly_one_set() {
    use alloc::collections::BTreeSet;

    let a: BPlusSet<u64, M> = (0..150u64).collect();
    let b: BPlusSet<u64, M> = (100..250u64).collect();
    let ma: BTreeSet<u64> = (0..150u64).collect();
    let mb: BTreeSet<u64> = (100..250u64).collect();

    assert!(
        a.symmetric_difference(&b).eq(ma.symmetric_difference(&mb)),
        "symmetric_difference must yield exactly the keys in exactly one of the two sets, in ascending order"
    );
}
