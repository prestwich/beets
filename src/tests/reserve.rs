//! Contract tests for reserve-then-commit's moving parts: the slot bag
//! ([`Reservation`]) — acquire debits the allocator, release credits it
//! back exactly, takes transfer ownership — and the wrapper allocator
//! ([`Reserved`]) — draws come from the bag, never the backing.
//!
//! The fixed arena is the observer throughout: its `available()` is
//! exact, so every slot the bag holds is visible as a debit.

use super::*;
use crate::{NodeStorage, allocator::FixedNodes, test_util::M};

/// Acquisition debits the allocator per pool; release credits every
/// still-held slot back, leaving both the bag and the arena exactly as
/// they started — the rollback path in miniature.
#[test]
fn reserve_then_release_is_a_perfect_round_trip() {
    let mut storage = NodeStorage::<u64, u64, M, 2, 3>::new();
    let mut arena = FixedNodes::new(&mut storage);

    let mut bag = Reservation::<u64, u64, M>::new();
    assert!(bag.is_empty(), "a fresh bag holds nothing");

    bag.reserve_leaf(&mut arena).expect("a leaf slot exists");
    bag.reserve_inner(&mut arena).expect("an inner slot exists");
    bag.reserve_inner(&mut arena).expect("a second inner slot exists");

    assert!(!bag.is_empty(), "the bag holds what it acquired");
    assert_eq!(arena.leaf_available(), 1, "one leaf slot debited");
    assert_eq!(arena.inner_available(), 1, "two inner slots debited");

    bag.release(&mut arena);
    assert!(bag.is_empty(), "release empties the bag");
    assert_eq!(arena.leaf_available(), 2, "every leaf slot credited back");
    assert_eq!(arena.inner_available(), 3, "every inner slot credited back");
}

/// A failed acquisition leaves the bag releasable and the failure
/// clean: what was already acquired comes back; nothing is lost.
#[test]
fn a_failed_acquisition_rolls_back_cleanly() {
    let mut storage = NodeStorage::<u64, u64, M, 1, 1>::new();
    let mut arena = FixedNodes::new(&mut storage);

    let mut bag = Reservation::<u64, u64, M>::new();
    bag.reserve_inner(&mut arena).expect("the inner slot exists");
    assert!(bag.reserve_inner(&mut arena).is_err(), "the pool has no second inner slot");

    bag.release(&mut arena);
    assert_eq!(arena.inner_available(), 1, "the one acquired slot must come back");
}

/// Takes hand out exactly what was reserved — each slot once, then
/// `None` — and taken slots BELONG to the taker: release afterwards
/// credits only the remainder.
#[test]
fn takes_transfer_ownership_out_of_the_bag() {
    let mut storage = NodeStorage::<u64, u64, M, 2, 2>::new();
    let mut arena = FixedNodes::new(&mut storage);

    let mut bag = Reservation::<u64, u64, M>::new();
    bag.reserve_leaf(&mut arena).expect("a leaf slot exists");
    bag.reserve_inner(&mut arena).expect("an inner slot exists");
    bag.reserve_inner(&mut arena).expect("a second inner slot exists");

    let leaf = bag.take_leaf().expect("the reserved leaf is takeable");
    assert!(bag.take_leaf().is_none(), "only one leaf was reserved");
    let _inner = bag.take_inner().expect("a reserved inner is takeable");

    // One inner still in the bag; the taken slots are ours now.
    bag.release(&mut arena);
    assert_eq!(arena.leaf_available(), 1, "the TAKEN leaf slot stays out");
    assert_eq!(arena.inner_available(), 1, "one inner taken (out), one released (back)");

    // Return the taken slots by hand, closing the loop.
    // SAFETY: both came from this arena via the bag, never initialized,
    // retired exactly once here.
    unsafe {
        arena.dealloc_leaf_uninit(leaf);
        arena.dealloc_inner_uninit(_inner);
    }
    assert_eq!(arena.leaf_available(), 2);
    assert_eq!(arena.inner_available(), 2);
}

/// The wrapper serves the commit FROM THE BAG: a draw through
/// [`Reserved`] pops the reserved slot — the same address — and the
/// backing allocator's availability does not move.
#[test]
fn the_wrapper_draws_from_the_bag_not_the_backing() {
    let mut storage = NodeStorage::<u64, u64, M, 2, 2>::new();
    let mut arena = FixedNodes::new(&mut storage);

    let mut bag = Reservation::<u64, u64, M>::new();
    bag.reserve_leaf(&mut arena).expect("a leaf slot exists");
    let available_after_reserve = arena.leaf_available();

    let mut wrapper = Reserved::new(&mut bag, &mut arena);
    let drawn = wrapper.try_alloc_leaf_uninit();
    let Ok(drawn) = drawn; // Exhaustion = Infallible: no Err arm exists

    assert_eq!(
        wrapper.leaf_available(),
        available_after_reserve,
        "a draw from the bag must not touch the backing's availability"
    );
    assert!(bag.is_empty(), "the draw consumed the bag");

    // SAFETY: from this arena via the bag, never initialized, retired
    // exactly once.
    unsafe { arena.dealloc_leaf_uninit(drawn) };
}
