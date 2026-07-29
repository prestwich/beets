//! Shared machinery for tests across modules: the counted value, the
//! u64-fanout constants, deterministic pseudo-randomness, and the
//! leaf/inner fixture builders every layer's tests assemble trees
//! from.

use alloc::{boxed::Box, sync::Arc, vec::Vec};
use core::{
    ptr::NonNull,
    sync::atomic::{AtomicIsize, Ordering::Relaxed},
};

use crate::{Inner, Key, Leaf, Node};

/// Box `value` and leak it, returning the raw handle — the fixture
/// builders' node-allocation idiom, equivalent to
/// `crate::Global.allocate(value)`. Non-test code allocates through
/// `SlotAllocator` only; a `leak(` outside this module is a bypass.
#[inline]
pub(crate) fn leak<T>(value: T) -> NonNull<T> {
    NonNull::from(Box::leak(Box::new(value)))
}

/// The fanout the u64-keyed tests run at.
pub(crate) const M: usize = <u64 as Key>::FANOUT;

/// Minimum pairs per non-root leaf at that fanout.
pub(crate) const LMIN: usize = Leaf::<u64, u64, M>::MIN_OCCUPANCY;

/// Minimum children per non-root inner at that fanout.
pub(crate) const IMIN: usize = Inner::<u64, u64, M>::MIN_OCCUPANCY;

/// Key-derived but distinct value, so tests catch key/value drift.
pub(crate) fn v(k: u64) -> u64 {
    k.wrapping_mul(31) ^ 0xBEE7
}

/// Deterministic permutation of `0..n`. No `rand` dependency.
pub(crate) fn shuffled(n: u64) -> Vec<u64> {
    let mut ks: Vec<u64> = (0..n).collect();
    ks.sort_by_key(|k| k.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    ks
}

/// Deterministic xorshift for churn tests. No `rand` dependency.
pub(crate) fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// The initialized (key, value) pairs of a leaf, in slot order.
pub(crate) fn entries<V: Copy>(leaf: &Leaf<u64, V, M>) -> Vec<(u64, V)> {
    (0..leaf.len())
        .map(|i| {
            let (k, val) = leaf.kv_ref_unchecked(i);
            (*k, *val)
        })
        .collect()
}

/// Reclaim ownership of a leaked leaf (a split's right sibling, a
/// fixture), so a test can inspect it and drop it without leaking.
pub(crate) fn own<K: Key, V, const N: usize>(ptr: NonNull<Leaf<K, V, N>>) -> Box<Leaf<K, V, N>> {
    // SAFETY: the pointer is a leaked `Box`; the test takes sole
    // ownership and frees it exactly once.
    unsafe { Box::from_raw(ptr.as_ptr()) }
}

/// A value that counts live instances: a leak leaves the counter positive,
/// a double-drop drives it negative.
pub(crate) struct Counted(pub(crate) u64, Arc<AtomicIsize>);

impl Counted {
    pub(crate) fn new(x: u64, live: &Arc<AtomicIsize>) -> Self {
        live.fetch_add(1, Relaxed);
        Counted(x, Arc::clone(live))
    }
}

impl Drop for Counted {
    fn drop(&mut self) {
        self.1.fetch_sub(1, Relaxed);
    }
}

/// A leaked counted-leaf handle, as the fixtures hand them around.
pub(crate) type CountedLeafPtr = NonNull<Leaf<u64, Counted, M>>;

/// A chained leaf holding `occ` pairs `base + 10 * j` of `Counted`.
pub(crate) fn counted_leaf(
    base: u64,
    occ: usize,
    live: &Arc<AtomicIsize>,
    next: Option<CountedLeafPtr>,
) -> CountedLeafPtr {
    let mut leaf: Leaf<u64, Counted, M> = Leaf::new(next);
    for j in 0..occ {
        let k = base + 10 * j as u64;
        leaf.raw_append(k, Counted::new(k, live));
    }
    leak(leaf)
}

/// Build a height-1 inner whose leaf `i` holds `occs[i]` pairs; leaf
/// `i`'s keys are `base + 1_000 * i + 10 * j`, so sibling ranges stay
/// disjoint through any rebalancing. Separators are each leaf's first
/// key; the chain is wired left-to-right, ending at `tail`. Returns
/// the node plus the leaf pointers in key order.
pub(crate) fn inner_with_occupancies(
    occs: &[usize],
    base: u64,
    live: &Arc<AtomicIsize>,
    tail: Option<CountedLeafPtr>,
) -> (Inner<u64, Counted, M>, Vec<CountedLeafPtr>) {
    let n = occs.len();
    assert!((2..=M).contains(&n), "an inner node holds 2..=M children");
    assert!(occs.iter().all(|&o| (1..=M).contains(&o)));

    // Build the leaves right-to-left so each can link to its successor.
    let mut ptrs: Vec<CountedLeafPtr> = Vec::with_capacity(n);
    let mut next = tail;
    for i in (0..n).rev() {
        let ptr = counted_leaf(base + 1_000 * i as u64, occs[i], live, next);
        ptrs.push(ptr);
        next = Some(ptr);
    }
    ptrs.reverse();

    let keys: Vec<u64> = (1..n).map(|i| base + 1_000 * i as u64).collect();
    let children: Vec<Node<u64, Counted, M>> =
        ptrs.iter().map(|p| Node::from_leaf_ptr(*p)).collect();
    (Inner::test_from_parts(keys, children), ptrs)
}

/// A height-1 inner over `n` MINIMAL leaves (ranges 1_000 apart from
/// `base`), chain ending at `tail`. Returns the inner and its first
/// leaf pointer for chain threading.
pub(crate) fn minimal_inner(
    n: usize,
    base: u64,
    live: &Arc<AtomicIsize>,
    tail: Option<CountedLeafPtr>,
) -> (Inner<u64, Counted, M>, CountedLeafPtr) {
    let (inner, ptrs) = inner_with_occupancies(&alloc::vec![LMIN; n], base, live, tail);
    (inner, ptrs[0])
}
