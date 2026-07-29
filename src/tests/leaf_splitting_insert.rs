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
use crate::{
    Global,
    test_util::{M, own, v},
};

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
