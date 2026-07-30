//! The differential harness: drive a [`BPlusTree`] and a
//! [`BTreeMap`](alloc::collections::BTreeMap) through the same operation
//! sequence, asserting agreement at every observable point and throwing
//! the full invariant net ([`BPlusTree::check`]) after every mutation.
//!
//! One harness, two drivers: the in-crate proptest properties
//! (`src/tree.rs`) generate [`Op`](crate::harness::Op) sequences with proptest strategies
//! and shrink failures to minimal reproductions; the fuzz target
//! (`fuzz/fuzz_targets/differential.rs`) derives them from
//! coverage-guided bytes via `arbitrary`. A failure from either is a
//! panic whose message states the violated contract.
//!
//! The invariant net lives here too (the bottom half of this file):
//! [`BPlusTree::check`] and the recursive `Node::check_invariants`
//! walk it delegates to, plus the raw key/child views they read. The
//! net reaches tree internals through the crate's test-only views
//! (`BPlusTree::test_root` and friends), not private fields.
//!
//! Only compiled for tests and under the `testutils` feature — this is
//! not public API and is exempt from semver.

use alloc::{collections::BTreeMap, vec::Vec};
use core::{ops::Bound, ptr::NonNull};

use super::Node;
use crate::{BPlusTree, Inner, Key, Leaf, NodeAllocator};

/// One step against both the tree and the model.
///
/// Keys are `u64` at this layer; [`run_differential`] widens them into
/// the tree's key type through an order-preserving `mk`, so one op
/// sequence drives trees at any fanout.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "testutils", derive(arbitrary::Arbitrary))]
pub enum Op {
    /// `insert(k, v)`: the returned prior value must agree.
    Insert(u64, u64),
    /// `remove(k)`: the returned value must agree.
    Remove(u64),
    /// `get(k)` and `contains_key(k)`: hit/miss and value must agree.
    Get(u64),
    /// `get_mut(k)`: presence must agree; a write through the borrow
    /// must stick (observed by later reads).
    GetMut(u64, u64),
    /// `range(..)` over bounds decoded from the fields: must yield
    /// exactly the model's pairs, in order. The `u8` picks each side's
    /// bound kind (Included/Excluded/Unbounded).
    Range(u64, u64, u8),
    /// `range_mut(..)`: same agreement, and writes through the borrows
    /// must stick.
    RangeMut(u64, u64, u8, u64),
    /// Full read-only sweep: `iter` (pairs, order, `ExactSizeIterator`
    /// length), `len`, `is_empty`, `first_key_value`, `last_key_value`
    /// must all agree.
    IterAll,
    /// `iter_mut` over every pair: keys and values must agree in order,
    /// and writes through the borrows must stick.
    MutateAll(u64),
    /// `clear()`: the tree must be empty afterwards.
    Clear,
}

impl Op {
    /// The same op with every key field masked into a smaller domain —
    /// how the fuzz target folds full-range fuzzer `u64`s into a domain
    /// where collisions, overwrites, and re-inserts of removed keys
    /// actually happen.
    #[must_use]
    pub fn mask_keys(self, mask: u64) -> Self {
        match self {
            Op::Insert(k, v) => Op::Insert(k & mask, v),
            Op::Remove(k) => Op::Remove(k & mask),
            Op::Get(k) => Op::Get(k & mask),
            Op::GetMut(k, v) => Op::GetMut(k & mask, v),
            Op::Range(a, b, kinds) => Op::Range(a & mask, b & mask, kinds),
            Op::RangeMut(a, b, kinds, d) => Op::RangeMut(a & mask, b & mask, kinds, d),
            Op::IterAll | Op::MutateAll(_) | Op::Clear => self,
        }
    }
}

/// Order-preserving widen of a `u64` into the fanout-3 key type:
/// big-endian bytes sort like the integers they came from.
pub fn wide(k: u64) -> [u8; 121] {
    let mut key = [0u8; 121];
    key[..8].copy_from_slice(&k.to_be_bytes());
    key
}

/// Key-derived but distinct value, so the harness catches key/value
/// drift in seeded trees.
fn v(k: u64) -> u64 {
    k.wrapping_mul(31) ^ 0xBEE7
}

/// Decode `Range`/`RangeMut` fields into bounds every `BTreeMap::range`
/// accepts: endpoints ordered, and the one `start == end` shape std
/// rejects (both bounds `Excluded`) nudged to a valid empty range.
fn decode_bounds(a: u64, b: u64, kinds: u8) -> (Bound<u64>, Bound<u64>) {
    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
    let make = |kind: u8, k: u64| match kind & 3 {
        0 => Bound::Included(k),
        1 => Bound::Excluded(k),
        _ => Bound::Unbounded,
    };
    let start = make(kinds, lo);
    let mut end = make(kinds >> 2, hi);
    if lo == hi && matches!((start, end), (Bound::Excluded(_), Bound::Excluded(_))) {
        end = Bound::Included(hi);
    }
    (start, end)
}

/// Throw the invariant net after a mutation. Under Miri every check
/// costs two orders of magnitude more, so the net is strided there;
/// native runs check after every single mutation.
fn net<K: Key + Ord, V, const N: usize, A: NodeAllocator<K, V, N>>(
    tree: &BPlusTree<K, V, N, A>,
    mutations: &mut usize,
) {
    *mutations += 1;
    if !cfg!(miri) || (*mutations).is_multiple_of(16) {
        tree.check();
    }
}

/// Bulk-load `seed` pairs through [`BPlusTree::from_sorted_iter_in`],
/// then apply `ops` to the tree and a `BTreeMap` model, asserting every
/// observable agrees step-for-step and the invariant net holds after
/// every mutation; finish with a full sweep of the survivors.
///
/// Seeded keys are the even numbers `0, 2, .., 2 * (seed - 1)`, so op
/// keys can hit them, miss between them, and extend past them. `mk`
/// widens harness keys into the tree's key type and must preserve
/// order ([`wide`] for the fanout-3 key; identity for `u64`).
pub fn run_differential<K: Key + Ord, const N: usize, A>(
    mk: impl Fn(u64) -> K,
    seed: u64,
    ops: &[Op],
) where
    // Infallible only: the harness drives the panicking surface
    // (`insert`, `from_sorted_iter_in`), which the tree type-gates to
    // allocators that cannot exhaust.
    A: NodeAllocator<K, u64, N, Exhaustion = core::convert::Infallible> + Default,
{
    let mut tree: BPlusTree<K, u64, N, A> =
        BPlusTree::from_sorted_iter_in((0..seed).map(|i| (mk(2 * i), v(2 * i))), A::default());
    let mut model: BTreeMap<u64, u64> = (0..seed).map(|i| (2 * i, v(2 * i))).collect();

    tree.check();
    assert_eq!(tree.len(), model.len(), "a bulk-loaded tree must hold every seeded pair");

    let mut mutations = 0usize;
    for (i, &op) in ops.iter().enumerate() {
        match op {
            Op::Insert(k, val) => {
                assert_eq!(
                    tree.insert(mk(k), val),
                    model.insert(k, val),
                    "insert({k}) must agree with the model (op #{i})"
                );
                net(&tree, &mut mutations);
            }
            Op::Remove(k) => {
                assert_eq!(
                    tree.remove(&mk(k)),
                    model.remove(&k),
                    "remove({k}) must agree with the model (op #{i})"
                );
                net(&tree, &mut mutations);
            }
            Op::Get(k) => {
                assert_eq!(
                    tree.get(&mk(k)),
                    model.get(&k),
                    "get({k}) must agree with the model (op #{i})"
                );
                assert_eq!(
                    tree.contains_key(&mk(k)),
                    model.contains_key(&k),
                    "contains_key({k}) must agree with the model (op #{i})"
                );
            }
            Op::GetMut(k, nv) => {
                let tv = tree.get_mut(&mk(k));
                let mv = model.get_mut(&k);
                assert_eq!(
                    tv.is_some(),
                    mv.is_some(),
                    "get_mut({k}) must hit exactly when the model hits (op #{i})"
                );
                if let (Some(tv), Some(mv)) = (tv, mv) {
                    assert_eq!(*tv, *mv, "get_mut({k}) must see the model's value (op #{i})");
                    *tv = nv;
                    *mv = nv;
                }
                net(&tree, &mut mutations);
            }
            Op::Range(a, b, kinds) => {
                let (lo, hi) = decode_bounds(a, b, kinds);
                let mut mr = model.range((lo, hi));
                for (j, (tk, tv)) in tree.range((lo.map(&mk), hi.map(&mk))).enumerate() {
                    let Some((mkey, mv)) = mr.next() else {
                        panic!(
                            "range must not yield pairs beyond the model's range \
                             (op #{i}, pair {j})"
                        );
                    };
                    assert!(
                        *tk == mk(*mkey),
                        "range must yield the model's keys in order (op #{i}, pair {j})"
                    );
                    assert_eq!(*tv, *mv, "range must yield the model's values (op #{i}, pair {j})");
                }
                assert_eq!(
                    mr.count(),
                    0,
                    "range must yield every pair the model's range yields (op #{i})"
                );
            }
            Op::RangeMut(a, b, kinds, delta) => {
                let (lo, hi) = decode_bounds(a, b, kinds);
                let mut mr = model.range_mut((lo, hi));
                for (j, (tk, tv)) in tree.range_mut((lo.map(&mk), hi.map(&mk))).enumerate() {
                    let Some((mkey, mv)) = mr.next() else {
                        panic!(
                            "range_mut must not yield pairs beyond the model's range \
                             (op #{i}, pair {j})"
                        );
                    };
                    assert!(
                        *tk == mk(*mkey),
                        "range_mut must yield the model's keys in order (op #{i}, pair {j})"
                    );
                    assert_eq!(
                        *tv, *mv,
                        "range_mut must yield the model's values (op #{i}, pair {j})"
                    );
                    *tv = tv.wrapping_add(delta);
                    *mv = mv.wrapping_add(delta);
                }
                assert_eq!(
                    mr.count(),
                    0,
                    "range_mut must yield every pair the model's range yields (op #{i})"
                );
                net(&tree, &mut mutations);
            }
            Op::IterAll => {
                assert_eq!(tree.len(), model.len(), "len must agree with the model (op #{i})");
                assert_eq!(
                    tree.is_empty(),
                    model.is_empty(),
                    "is_empty must agree with the model (op #{i})"
                );
                let it = tree.iter();
                assert_eq!(
                    it.len(),
                    model.len(),
                    "iter's ExactSizeIterator length must be the tree's len (op #{i})"
                );
                let mut mi = model.iter();
                for (j, (tk, tv)) in it.enumerate() {
                    let Some((mkey, mv)) = mi.next() else {
                        panic!("iter must not yield pairs beyond the model (op #{i}, pair {j})");
                    };
                    assert!(
                        *tk == mk(*mkey),
                        "iter must yield the model's keys in order (op #{i}, pair {j})"
                    );
                    assert_eq!(*tv, *mv, "iter must yield the model's values (op #{i}, pair {j})");
                }
                assert_eq!(mi.count(), 0, "iter must yield every pair the model yields (op #{i})");
                assert_eq!(
                    tree.first_key_value().map(|(_, v)| *v),
                    model.first_key_value().map(|(_, v)| *v),
                    "first_key_value must agree with the model (op #{i})"
                );
                assert!(
                    tree.first_key_value().map(|(k, _)| *k)
                        == model.first_key_value().map(|(k, _)| mk(*k)),
                    "first_key_value's key must agree with the model (op #{i})"
                );
                assert_eq!(
                    tree.last_key_value().map(|(_, v)| *v),
                    model.last_key_value().map(|(_, v)| *v),
                    "last_key_value must agree with the model (op #{i})"
                );
                assert!(
                    tree.last_key_value().map(|(k, _)| *k)
                        == model.last_key_value().map(|(k, _)| mk(*k)),
                    "last_key_value's key must agree with the model (op #{i})"
                );
            }
            Op::MutateAll(delta) => {
                let mut mi = model.iter_mut();
                for (j, (tk, tv)) in tree.iter_mut().enumerate() {
                    let Some((mkey, mv)) = mi.next() else {
                        panic!(
                            "iter_mut must not yield pairs beyond the model (op #{i}, pair {j})"
                        );
                    };
                    assert!(
                        *tk == mk(*mkey),
                        "iter_mut must yield the model's keys in order (op #{i}, pair {j})"
                    );
                    assert_eq!(
                        *tv, *mv,
                        "iter_mut must yield the model's values (op #{i}, pair {j})"
                    );
                    *tv = tv.wrapping_add(delta);
                    *mv = mv.wrapping_add(delta);
                }
                assert_eq!(
                    mi.count(),
                    0,
                    "iter_mut must yield every pair the model yields (op #{i})"
                );
                net(&tree, &mut mutations);
            }
            Op::Clear => {
                tree.clear();
                model.clear();
                assert!(tree.is_empty(), "a cleared tree must be empty (op #{i})");
                assert_eq!(tree.len(), 0, "a cleared tree must have len 0 (op #{i})");
                net(&tree, &mut mutations);
            }
        }
        assert_eq!(tree.len(), model.len(), "len must agree with the model (op #{i})");
    }

    // Final sweep: the net once more (unstrided), full-iteration
    // agreement, and a point-read of every surviving key.
    tree.check();
    let pairs: Vec<(K, u64)> = tree.iter().map(|(k, v)| (*k, *v)).collect();
    assert_eq!(pairs.len(), model.len(), "the final iteration must yield the model's every pair");
    for ((tk, tv), (mkey, mv)) in pairs.iter().zip(model.iter()) {
        assert!(*tk == mk(*mkey), "the final iteration must yield the model's keys in order");
        assert_eq!(tv, mv, "the final iteration must yield the model's values");
    }
    for (k, val) in &model {
        assert_eq!(tree.get(&mk(*k)), Some(val), "key {k} must match the model at the end");
    }
}

// ── the invariant net ───────────────────────────────────────────────

impl<K: Key, V, const M: usize> Leaf<K, V, M> {
    /// Test-only view of the keys, for invariant checking and fixture
    /// assertions from other modules' tests.
    pub(crate) fn test_keys(&self) -> &[K] {
        self.keys_ref()
    }
}

impl<K: Key, V, const M: usize> Inner<K, V, M> {
    /// Test-only views for invariant checking and fixture assertions
    /// from other modules' tests.
    pub(crate) fn test_keys(&self) -> &[K] {
        self.keys_ref()
    }

    pub(crate) fn test_children(&self) -> &[Node<K, V, M>] {
        self.children_ref()
    }
}

impl<K: Key, V, const M: usize> Node<K, V, M> {
    /// Test-only: recursively verify the structural invariants of the
    /// subtree rooted at this node, panicking with a description on the
    /// first violation. Per node it checks: occupancy bounds (a non-root
    /// leaf holds `Leaf::MIN_OCCUPANCY..=M` pairs, a non-root inner
    /// `Inner::MIN_OCCUPANCY..=M` children; the root is exempt down to 0
    /// pairs / 2 children), strict key ordering, separator correctness
    /// (child `i`'s keys `< keys[i] <= ` child `i + 1`'s keys), and that
    /// the leaf chain links consecutive children's leaves in order.
    ///
    /// Returns the subtree's key range (`None` only for an empty root
    /// leaf) and its first and last leaves, so a caller can check the
    /// chain across sibling subtrees and the terminal `next` itself.
    ///
    /// # Safety
    ///
    /// `height` must be the height of the subtree rooted at this node.
    #[allow(clippy::type_complexity)]
    #[track_caller]
    pub(crate) unsafe fn check_invariants(
        &self,
        height: u8,
        is_root: bool,
    ) -> (Option<(K, K)>, NonNull<Leaf<K, V, M>>, NonNull<Leaf<K, V, M>>) {
        if height == 0 {
            // SAFETY: height 0 ⇒ leaf (caller vouches for `height`).
            let leaf = unsafe { self.as_leaf() };
            let keys = leaf.test_keys();
            assert!(leaf.len() <= M, "a leaf holds at most M pairs, found {}", leaf.len());
            assert!(
                is_root || leaf.len() >= Leaf::<K, V, M>::MIN_OCCUPANCY,
                "a non-root leaf must hold at least MIN_OCCUPANCY ({}) pairs, found {}",
                Leaf::<K, V, M>::MIN_OCCUPANCY,
                leaf.len()
            );
            assert!(keys.windows(2).all(|w| w[0] < w[1]), "leaf keys must be strictly sorted");
            let range = (!keys.is_empty()).then(|| (keys[0], *keys.last().unwrap()));
            let ptr = NonNull::from(leaf);
            (range, ptr, ptr)
        } else {
            // SAFETY: height > 0 ⇒ inner (caller vouches for `height`).
            let inner = unsafe { self.as_inner() };
            let n = inner.len();
            assert!(n <= M, "an inner holds at most M children, found {n}");
            let min = if is_root { 2 } else { Inner::<K, V, M>::MIN_OCCUPANCY };
            assert!(
                n >= min,
                "an inner must hold at least {min} children (root exempt down to 2), found {n}"
            );
            let keys = inner.test_keys();
            assert!(keys.windows(2).all(|w| w[0] < w[1]), "inner keys must be strictly sorted");

            let mut first_leaf = None;
            let mut prev_last: Option<NonNull<Leaf<K, V, M>>> = None;
            let mut range: Option<(K, K)> = None;
            for (i, child) in inner.test_children().iter().enumerate() {
                // SAFETY: the children of a height-`height` inner root
                // subtrees of `height - 1`, and are never the root.
                let (child_range, first, last) =
                    unsafe { child.check_invariants(height - 1, false) };

                if let Some(prev) = prev_last {
                    // SAFETY: the previous child's last leaf is live.
                    assert_eq!(
                        unsafe { prev.as_ref() }.next(),
                        Some(first),
                        "the leaf chain must link child {} to child {i}",
                        i - 1
                    );
                }
                first_leaf.get_or_insert(first);
                prev_last = Some(last);

                let (lo, hi) =
                    child_range.expect("non-root nodes have occupancy > 0, hence a key range");
                if i > 0 {
                    assert!(
                        keys[i - 1] <= lo,
                        "child {i}'s keys must be >= the separator on its left"
                    );
                }
                if i < n - 1 {
                    assert!(hi < keys[i], "child {i}'s keys must be < the separator on its right");
                }
                range = match range {
                    None => Some((lo, hi)),
                    Some((rlo, _)) => Some((rlo, hi)),
                };
            }
            (range, first_leaf.unwrap(), prev_last.unwrap())
        }
    }
}

impl<K: Key + Ord, V, const M: usize, A: NodeAllocator<K, V, M>, const H: usize>
    BPlusTree<K, V, M, A, H>
{
    /// Test-only, for tests outside the tree module (which cannot reach
    /// the private fields) and, under `testutils`, external test
    /// drivers: the full-strength invariant net — the structural walk
    /// (`Node::check_invariants`) plus the two facts only the tree
    /// layer can vouch for: `len` equals the pairs actually on the
    /// chain, and the chain terminates at the last leaf.
    ///
    /// Panics if any invariant is violated.
    pub fn check(&self) {
        // SAFETY: `height` is the tree's impl-block invariant — exactly
        // the height of `root`'s subtree.

        if self.is_empty() {
            return;
        }

        let (_, first, last) =
            unsafe { self.test_root().check_invariants(self.test_height(), true) };

        let mut total = 0;
        let mut hops = 0;
        let mut cur = Some(first);
        while let Some(ptr) = cur {
            hops += 1;
            assert!(hops <= self.len() + 1, "the leaf chain must terminate within the tree's size");
            // SAFETY: every leaf on a valid tree's chain is live.
            let leaf = unsafe { ptr.as_ref() };
            total += leaf.len();
            cur = leaf.next();
        }
        assert_eq!(total, self.len(), "tree.len must equal the pairs actually on the chain");
        // SAFETY: `last` is the walk's final live leaf.
        assert_eq!(
            unsafe { last.as_ref() }.next(),
            None,
            "the tree's last leaf must terminate the chain"
        );
    }
}
