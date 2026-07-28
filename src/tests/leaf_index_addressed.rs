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
