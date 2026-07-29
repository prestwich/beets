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
use crate::{
    Global,
    test_util::{Counted, M, entries, own, shuffled, v},
};

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
        assert_eq!(entries(&l), expected, "other entries must be untouched (replaced {target})");
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
