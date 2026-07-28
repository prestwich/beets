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
