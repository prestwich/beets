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
