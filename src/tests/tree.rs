//! Contract tests for the tree layer, plus the height-0 teardown
//! contract for `Node` (heights 1 and 2 are pinned alongside the
//! `Inner` tests).
//!
//! The `BPlusTree` tests come in two flavors. Hand-built-fixture tests
//! assemble trees directly through the private fields (this module can)
//! and pin single behaviors: routing, `len` bookkeeping, root shrink.
//! Public-API tests drive everything through `new`/`insert`/`remove`/
//! `get` and are RED until the whole stack under them lands — they are
//! the specification for `insert`'s grow path and `remove`'s shrink
//! path, capped by the model-mirroring churn test. `check_tree` is the
//! full-strength net: the structural walk (`Node::check_invariants`)
//! plus the two facts only this layer can vouch for — `len` equals the
//! pairs on the chain, and the chain terminates at the last leaf.

#[cfg(test)]
use alloc::{boxed::Box, sync::Arc, vec, vec::Vec};
#[cfg(test)]
use core::sync::atomic::{AtomicIsize, Ordering::Relaxed};

use super::*;
#[cfg(test)]
use crate::test_util::{Counted, IMIN, LMIN, M, counted_leaf, minimal_inner, v, xorshift};

/// Test-only views into the tree's private fields, for the invariant
/// net (`crate::harness`), which lives outside the `tree` module.
impl<K: Key, V, const N: usize, A: NodeAllocator<K, V, N>, const H: usize>
    BPlusTree<K, V, N, A, H>
{
    pub(crate) fn test_root(&self) -> &Node<K, V, N> {
        &self.root
    }

    pub(crate) fn test_height(&self) -> u8 {
        self.height
    }
}

/// Delegate methods on [`Node`] to the leaf that owns the key, threading the
/// subtree height down the descent: at `height == 0` the node is cast to
/// [`Leaf`] and the call terminates there, so each listed method must exist on
/// `Leaf` with the written signature; at `height > 0` the node is cast to
/// [`Inner`], routed to the child owning the key, and the call recurses on
/// `Node` at `height - 1`. Doc comments and other attributes carry over.
///
/// The first argument must be the routing key (`key: &K`) — it is what the
/// inner arm routes on.
///
/// The generated method is `unsafe`: `height` must be the true height of the
/// subtree rooted at this node, per the type's safety contract. A wrong
/// height reinterprets one pointee type as the other.
///
/// Methods whose signature mentions the pointee's own kind (e.g. `merge`,
/// which consumes a same-kind sibling) can't be delegated this way — the
/// same-kind argument or result needs erasing into a `Node`. Those are
/// written out by hand below.
#[cfg(test)]
macro_rules! delegate {
    () => {};
    (
        $(#[$attr:meta])*
        fn $name:ident(&self, $key:ident: &K $(, $arg:ident: $ty:ty)*) -> $ret:ty;
        $($rest:tt)*
    ) => {
        $(#[$attr])*
        ///
        /// # Safety
        ///
        /// `height` must be the height of the subtree rooted at this node.
        pub(crate) unsafe fn $name(&self, height: u8, $key: &K $(, $arg: $ty)*) -> $ret {
            if height == 0 {
                // SAFETY: height 0 ⇒ this node is a leaf (caller vouches
                // for `height`).
                unsafe { self.as_leaf().$name($key $(, $arg)*) }
            } else {
                // SAFETY: height > 0 ⇒ this node is inner (caller vouches
                // for `height`), and the child it routes to roots a subtree
                // of height - 1.
                unsafe { self.as_inner().$name(height, $key $(, $arg)*) }
            }
        }
        delegate! { $($rest)* }
    };
    (
        $(#[$attr:meta])*
        fn $name:ident(&mut self, $key:ident: &K $(, $arg:ident: $ty:ty)*) -> $ret:ty;
        $($rest:tt)*
    ) => {
        $(#[$attr])*
        ///
        /// # Safety
        ///
        /// `height` must be the height of the subtree rooted at this node.
        pub(crate) unsafe fn $name(&mut self, height: u8, $key: &K $(, $arg: $ty)*) -> $ret {
            if height == 0 {
                // SAFETY: height 0 ⇒ this node is a leaf (caller vouches
                // for `height`).
                unsafe { self.as_leaf_mut().$name($key $(, $arg)*) }
            } else {
                // SAFETY: height > 0 ⇒ this node is inner (caller vouches
                // for `height`), and the child it routes to roots a subtree
                // of height - 1.
                unsafe {
                    self.as_inner_mut().$name(height, $key $(, $arg)*)
                }
            }
        }
        delegate! { $($rest)* }
    };
}

#[cfg(test)]
impl<K: Key, V, const M: usize> Node<K, V, M> {
    delegate! {
        /// Get a reference to a value in the subtree rooted at this node, if
        /// it is present.
        ///
        /// Test-only — the production descent is iterative; fixtures in
        /// the node-layer tests read subtrees through it.
        fn get(&self, key: &K) -> Option<&V>;
    }

    /// Remove a key from the subtree rooted at this node, if it exists.
    ///
    /// Test-only — the production descent is iterative; the
    /// node-layer tests drive `rebalance` through it.
    ///
    /// # Safety
    ///
    /// `height` must be the height of the subtree rooted at this node.
    pub(crate) unsafe fn remove<A: NodeAllocator<K, V, M>>(
        &mut self,
        height: u8,
        key: &K,
        alloc: &mut A,
    ) -> Option<V> {
        if height == 0 {
            // SAFETY: height 0 ⇒ this node is a leaf (caller vouches
            // for `height`).
            unsafe { self.as_leaf_mut().remove(key) }
        } else {
            // SAFETY: height > 0 ⇒ this node is inner (caller vouches
            // for `height`), and the child it routes to roots a subtree
            // of height - 1.
            unsafe { self.as_inner_mut().remove(height, key, alloc) }
        }
    }
}

/// Tearing down a height-0 node must drop the leaf it owns — observed
/// through the leaf's values dropping exactly once.
#[test]
fn drop_subtree_at_height_zero_drops_the_leafs_values_exactly_once() {
    let live = Arc::new(AtomicIsize::new(0));

    let mut leaf: Leaf<u64, Counted, M> = Leaf::new(None);
    for k in 0..3 {
        leaf.raw_append(k, Counted::new(k, &live));
    }
    let node: Node<u64, Counted, M> = Node::from_leaf_ptr(Global.alloc_leaf(leaf));
    assert_eq!(live.load(Relaxed), 3, "one live value per stored key before teardown");

    // SAFETY: `node` roots a single leaf (height 0) and owns it.
    unsafe { node.drop_subtree(0, &mut Global) };
    assert_eq!(
        live.load(Relaxed),
        0,
        "drop_subtree(0) must drop the leaf's every value exactly once \
         (positive = leak, negative = double-drop)"
    );
}

// ── tree-layer fixtures and the full-strength invariant net ────────

/// Full-tree invariant check: delegates to [`BPlusTree::check`], the
/// cross-module net (the structural walk plus the tree-layer
/// bookkeeping only `BPlusTree` can vouch for). Kept as a free
/// function so this module's many call sites read unchanged.
#[cfg(test)]
fn check_tree<K: Key + Ord, V, const N: usize, A: NodeAllocator<K, V, N>>(
    tree: &BPlusTree<K, V, N, A>,
) {
    tree.check()
}

// ── BPlusTree: hand-built fixtures ─────────────────────────────────

/// A fresh tree is empty: zero pairs, misses on every read, and a
/// clean teardown.
#[test]
fn new_tree_is_empty_and_reads_miss() {
    let tree: BPlusTree<u64, u64, M> = BPlusTree::new();
    assert_eq!(tree.len(), 0, "a new tree holds no pairs");
    assert!(tree.is_empty());
    assert!(tree.get(&0).is_none(), "a new tree must miss on any key");
    assert!(!tree.contains_key(&0));
    check_tree(&tree);
}

/// Reads route through a hand-built height-2 tree: hits return the
/// right values, misses return None, and nothing is disturbed.
#[test]
fn get_routes_through_a_hand_built_height_two_tree() {
    let live = Arc::new(AtomicIsize::new(0));
    let (b, b_first) = minimal_inner(IMIN, 1_000_000, &live, None);
    let (a, _) = minimal_inner(IMIN, 0, &live, Some(b_first));
    let len = 2 * IMIN * LMIN;
    let root = Inner::test_from_parts(
        vec![1_000_000],
        vec![Node::from_inner(a, &mut Global), Node::from_inner(b, &mut Global)],
    );
    let tree =
        BPlusTree { root: Node::from_inner(root, &mut Global), height: 2, len, allocator: Global };

    check_tree(&tree);
    for k in [0u64, 10, 1_000, 1_000_000, 1_000_000 + 1_000 * (IMIN as u64 - 1)] {
        let got = tree.get(&k);
        assert!(got.is_some_and(|c| c.0 == k), "key {k} must be reachable through the tree");
        assert!(tree.contains_key(&k));
    }
    for k in [5u64, 999, 2_000_000] {
        assert!(tree.get(&k).is_none(), "absent key {k} must miss");
    }

    drop(tree);
    assert_eq!(
        live.load(Relaxed),
        0,
        "dropping the tree must drop every value exactly once \
         (positive = leak, negative = double-drop)"
    );
}

/// `get_mut` exposes a value for in-place mutation, observable through
/// a subsequent `get`.
#[test]
fn get_mut_mutates_in_place() {
    let live = Arc::new(AtomicIsize::new(0));
    let (inner, _) = minimal_inner(2, 0, &live, None);
    let len = 2 * LMIN;
    let mut tree =
        BPlusTree { root: Node::from_inner(inner, &mut Global), height: 1, len, allocator: Global };

    tree.get_mut(&10).expect("present key must be gettable").0 = 424_242;
    assert_eq!(
        tree.get(&10).map(|c| c.0),
        Some(424_242),
        "a get_mut write must be visible to the next get"
    );
    assert_eq!(tree.len(), len, "get_mut must not change the pair count");
    check_tree(&tree);
}

/// Tree-level remove: the value comes back, `len` ticks down, and the
/// invariants hold afterwards.
#[test]
fn remove_returns_the_value_and_updates_len() {
    let live = Arc::new(AtomicIsize::new(0));
    let (inner, _) = minimal_inner(2, 0, &live, None);
    let len = 2 * LMIN;
    let mut tree =
        BPlusTree { root: Node::from_inner(inner, &mut Global), height: 1, len, allocator: Global };

    let got = tree.remove(&0);
    assert!(got.is_some_and(|c| c.0 == 0), "removing a present key must return its value");
    assert_eq!(tree.len(), len - 1, "a hit must decrement len");
    assert!(tree.remove(&0).is_none(), "a second remove of the same key must miss");
    assert_eq!(tree.len(), len - 1, "a miss must not change len");
    check_tree(&tree);

    drop(tree);
    assert_eq!(live.load(Relaxed), 0, "teardown must drop every survivor exactly once");
}

/// A cascade merge that leaves the root with one child must hoist it:
/// the tree gets one level shorter and stays fully valid.
#[test]
fn remove_hoists_a_single_child_root() {
    let live = Arc::new(AtomicIsize::new(0));
    let (b, b_first) = minimal_inner(IMIN, 1_000_000, &live, None);
    let (a, _) = minimal_inner(IMIN, 0, &live, Some(b_first));
    let len = 2 * IMIN * LMIN;
    let root = Inner::test_from_parts(
        vec![1_000_000],
        vec![Node::from_inner(a, &mut Global), Node::from_inner(b, &mut Global)],
    );
    let mut tree =
        BPlusTree { root: Node::from_inner(root, &mut Global), height: 2, len, allocator: Global };

    // One remove: leaf merge inside `a` → `a` deficient → both inners
    // minimal → they merge → the root has one child → hoist.
    let got = tree.remove(&0);
    assert!(got.is_some_and(|c| c.0 == 0), "removing a present key must return its value");
    assert_eq!(tree.len(), len - 1);
    assert_eq!(tree.height, 1, "hoisting the merged child must shorten the tree by one level");
    check_tree(&tree);
    assert!(tree.get(&1_000_000).is_some(), "keys from the absorbed side must remain reachable");

    drop(tree);
    assert_eq!(
        live.load(Relaxed),
        0,
        "teardown after a hoist must drop every survivor exactly once \
         (positive = leak, negative = double-drop)"
    );
}

/// `clear` empties the tree in one call and drops every value exactly
/// once, leaving a usable empty tree.
#[test]
fn clear_resets_to_an_empty_tree() {
    let live = Arc::new(AtomicIsize::new(0));
    let (inner, _) = minimal_inner(2, 0, &live, None);
    let len = 2 * LMIN;
    let mut tree =
        BPlusTree { root: Node::from_inner(inner, &mut Global), height: 1, len, allocator: Global };

    tree.clear();
    assert_eq!(live.load(Relaxed), 0, "clear must drop every value exactly once");
    assert_eq!(tree.len(), 0);
    assert!(tree.is_empty());
    assert!(tree.get(&0).is_none());
    check_tree(&tree);
}

// ── BPlusTree: the public-API specification (insert-driven) ────────

/// Inserting distinct keys grows the tree through leaf and root
/// splits; every pair stays reachable, `len` is exact, and the
/// invariants hold at every size.
#[test]
fn insert_get_roundtrip_grows_through_splits() {
    const N: u64 = 2_000;
    let mut tree: BPlusTree<u64, u64, M> = BPlusTree::new();

    for i in 0..N {
        let k = (i * 37) % N; // coprime stride: a shuffled bijection
        assert_eq!(tree.insert(k, v(k)), None, "first insert of key {k} must return None");
    }
    assert_eq!(tree.len(), N as usize, "len must count every distinct insert");
    assert!(tree.height >= 2, "{N} pairs at fanout {M} must have grown the root at least twice");
    check_tree(&tree);

    for k in 0..N {
        assert_eq!(tree.get(&k), Some(&v(k)), "key {k} must round-trip");
    }
    assert!(tree.get(&N).is_none(), "an absent key must miss");
}

/// Re-inserting a key replaces its value in place: the old value comes
/// back, and the pair count does not change.
#[test]
fn insert_replaces_and_returns_the_previous_value() {
    let mut tree: BPlusTree<u64, u64, M> = BPlusTree::new();
    assert_eq!(tree.insert(7, 1), None);
    assert_eq!(tree.insert(7, 2), Some(1), "a replace must return the previous value");
    assert_eq!(tree.len(), 1, "a replace must not change the pair count");
    assert_eq!(tree.get(&7), Some(&2));
    check_tree(&tree);
}

#[test]
fn from_iterator_collects_every_pair() {
    let tree: BPlusTree<u64, u64, M> = (0..300u64).map(|k| (k, v(k))).collect();
    assert_eq!(tree.len(), 300);
    check_tree(&tree);
    for k in 0..300 {
        assert_eq!(tree.get(&k), Some(&v(k)), "collected key {k} must round-trip");
    }
}

/// `from_sorted_iter` must build a structurally valid tree at every
/// awkward size: empty, single pair, exactly one leaf, one-past (the
/// worst ragged tail), exact multiples (whose final level is one
/// exactly-full node), and multi-level trees whose tails cascade
/// repairs at both the leaf and inner levels (`M² + 1`, `M³ + 1`).
/// `check_tree` is the net: occupancy invariants, separator routing,
/// leaf-chain integrity, and `len` bookkeeping.
///
/// Under Miri the `m³` sizes are skipped: interpreting them costs
/// tens of minutes, and the deep build path is Miri-covered by
/// `bulk::tests::deep_cascade_loads_own_every_value_exactly_once`.
/// Regular CI runs the full list.
#[test]
fn from_sorted_iter_builds_valid_trees_at_awkward_sizes() {
    let m = M as u64;
    #[rustfmt::skip]
    let sizes = [
        0, 1, 2, m - 1, m, m + 1, 2 * m, 2 * m + 1,
        m * m, m * m + 1, m * m + m + 3,
        m * m * m, m * m * m + 1,
    ];
    for n in sizes {
        if cfg!(miri) && n >= m * m * m {
            continue;
        }
        let tree: BPlusTree<u64, u64, M> = BPlusTree::from_sorted_iter((0..n).map(|k| (k, v(k))));

        assert_eq!(tree.len(), n as usize, "len must count the drained pairs (n={n})");
        let want_height = if n <= m {
            0
        } else if n <= m * m {
            1
        } else if n <= m * m * m {
            2
        } else {
            3
        };
        assert_eq!(tree.height, want_height, "height must match the level count (n={n})");
        check_tree(&tree);

        for k in 0..n {
            assert_eq!(tree.get(&k), Some(&v(k)), "key {k} lost in a bulk load of {n}");
        }
        assert_eq!(tree.get(&n), None, "unknown keys must miss (n={n})");
    }
}

/// A bulk-loaded tree owns its values end to end: the load itself
/// drops nothing, lookups see every value, and dropping the tree
/// drops each exactly once.
#[test]
fn from_sorted_iter_drops_values_exactly_once() {
    // One-past-two-full-levels: leaf and inner tail repairs both fire.
    let n = M * M + 1;
    let live = Arc::new(AtomicIsize::new(0));
    {
        let tree: BPlusTree<u64, Counted, M> =
            BPlusTree::from_sorted_iter((0..n as u64).map(|k| (k, Counted::new(k, &live))));
        assert_eq!(live.load(Relaxed), n as isize, "the load itself must not drop anything");
        check_tree(&tree);
    }
    assert_eq!(
        live.load(Relaxed),
        0,
        "dropping the tree must drop every bulk-loaded value exactly once \
         (positive = leak, negative = double-drop)"
    );
}

/// Draining a grown tree through the public API shrinks it back to an
/// empty root leaf: every removal returns its value, and at the end
/// the tree is empty at height 0.
#[test]
fn remove_drains_the_tree_back_to_height_zero() {
    const N: u64 = 600;
    let mut tree: BPlusTree<u64, u64, M> = BPlusTree::new();
    for k in 0..N {
        tree.insert(k, v(k));
    }
    assert!(tree.height >= 1, "{N} pairs must not fit a single leaf");
    check_tree(&tree);

    for i in 0..N {
        let k = (i * 7) % N; // coprime stride
        assert_eq!(tree.remove(&k), Some(v(k)), "removing present key {k} must return its value");
        assert_eq!(tree.len(), (N - 1 - i) as usize, "len must tick down on every hit");
        if i % 100 == 0 {
            check_tree(&tree);
        }
    }
    assert!(tree.is_empty(), "a drained tree is empty");
    assert_eq!(tree.height, 0, "a drained tree must have shrunk back to a root leaf");
    check_tree(&tree);
}

/// The payoff test: a deterministic insert/remove churn mirrored
/// against `alloc::collections::BTreeMap`. Every operation must agree
/// with the model — return values, lengths, and final contents — with
/// the invariants checked throughout.
#[test]
fn churn_mirrors_btreemap() {
    use alloc::collections::BTreeMap;

    let mut tree: BPlusTree<u64, u64, M> = BPlusTree::new();
    let mut model: BTreeMap<u64, u64> = BTreeMap::new();
    let mut state: u64 = 0x5EED_CAFE_F00D_D00D;

    for step in 0..1_500u64 {
        let r = xorshift(&mut state);
        let key = r % 200;
        if (r >> 32) % 5 < 3 {
            assert_eq!(
                tree.insert(key, step),
                model.insert(key, step),
                "insert({key}) must agree with the model at step {step}"
            );
        } else {
            assert_eq!(
                tree.remove(&key),
                model.remove(&key),
                "remove({key}) must agree with the model at step {step}"
            );
        }
        assert_eq!(tree.len(), model.len(), "len must agree with the model at step {step}");
        if step % 128 == 0 {
            check_tree(&tree);
        }
    }

    check_tree(&tree);
    for (k, val) in &model {
        assert_eq!(tree.get(k), Some(val), "key {k} must match the model at the end");
    }
    assert!(tree.get(&10_000).is_none());
}

/// The whole stack drops values exactly once when the tree does —
/// grown through the public API, torn down through `Drop`.
#[test]
fn tree_drop_drops_every_value_exactly_once() {
    let live = Arc::new(AtomicIsize::new(0));
    {
        let mut tree: BPlusTree<u64, Counted, M> = BPlusTree::new();
        for k in 0..(3 * M as u64) {
            tree.insert(k, Counted::new(k, &live));
        }
        assert_eq!(live.load(Relaxed), 3 * M as isize, "one live value per inserted key");
        check_tree(&tree);
    }
    assert_eq!(
        live.load(Relaxed),
        0,
        "dropping the tree must drop every value exactly once \
         (positive = leak, negative = double-drop)"
    );
}

// ── remove: rebalance-path pins ─────────────────────────────────────
//
// One contract per test, so a red test names the violated behavior on
// its own: a miss leaves the tree untouched, a deficient node borrows
// before it merges, and the upward repair stops at the first healthy
// level. (The hit/hoist/drain/churn contracts are pinned with the
// other public-API tests above.)

/// A probe for an absent key returns `None` and leaves the tree
/// untouched: same `len`, same reachable pairs, invariants intact.
#[test]
fn remove_miss_leaves_the_tree_untouched() {
    let live = Arc::new(AtomicIsize::new(0));
    let (inner, _) = minimal_inner(2, 0, &live, None);
    let len = 2 * LMIN;
    let mut tree =
        BPlusTree { root: Node::from_inner(inner, &mut Global), height: 1, len, allocator: Global };

    for k in [5u64, 999, u64::MAX] {
        assert!(tree.remove(&k).is_none(), "absent key {k} must miss");
    }
    assert_eq!(tree.len(), len, "misses must not change len");
    check_tree(&tree);
    let seeds = (0..2).flat_map(|c| (0..LMIN as u64).map(move |j| 1_000 * c + 10 * j));
    for k in seeds {
        assert!(tree.get(&k).is_some_and(|c| c.0 == k), "key {k} must remain reachable");
    }
}

/// A deficient leaf with a sibling strictly above its minimum is
/// repaired by borrowing, not merging: the root keeps both children
/// and every survivor stays reachable.
#[test]
fn remove_repairs_a_deficient_leaf_by_borrowing() {
    let live = Arc::new(AtomicIsize::new(0));
    // Child 0 minimal, child 1 one above minimum: removing from
    // child 0 must borrow from child 1.
    let right = counted_leaf(1_000, LMIN + 1, &live, None);
    let left = counted_leaf(0, LMIN, &live, Some(right));
    let root = Inner::test_from_parts(
        vec![1_000],
        vec![Node::from_leaf_ptr(left), Node::from_leaf_ptr(right)],
    );
    let len = 2 * LMIN + 1;
    let mut tree =
        BPlusTree { root: Node::from_inner(root, &mut Global), height: 1, len, allocator: Global };

    let got = tree.remove(&0);
    assert!(got.is_some_and(|c| c.0 == 0), "removing a present key must return its value");
    assert_eq!(tree.height, 1, "borrowing must not change the tree's shape");
    // SAFETY: height-1 subtree, judged as a root.
    let _ = unsafe { tree.root.check_invariants(1, true) };
    assert_eq!(
        unsafe { tree.root.len(1) },
        2,
        "a repair by borrowing must keep both children alive"
    );
    check_tree(&tree);

    drop(tree);
    assert_eq!(live.load(Relaxed), 0, "teardown must drop every survivor exactly once");
}

/// Two minimal leaf siblings merge when one dips below minimum — and
/// under a root with children to spare, the merge is absorbed: no
/// hoist, chain intact, all survivors reachable.
#[test]
fn remove_merges_minimal_leaf_siblings() {
    let live = Arc::new(AtomicIsize::new(0));
    let (inner, _) = minimal_inner(3, 0, &live, None);
    let len = 3 * LMIN;
    let mut tree =
        BPlusTree { root: Node::from_inner(inner, &mut Global), height: 1, len, allocator: Global };

    let got = tree.remove(&0);
    assert!(got.is_some_and(|c| c.0 == 0), "removing a present key must return its value");
    assert_eq!(tree.height, 1, "a merge absorbed by the root must not shrink the tree");
    assert_eq!(
        unsafe { tree.root.len(1) },
        2,
        "merging one minimal pair under a 3-child root must leave 2 children"
    );
    check_tree(&tree);

    drop(tree);
    assert_eq!(live.load(Relaxed), 0, "teardown must drop every survivor exactly once");
}

/// A leaf repair below a HEALTHY parent must not climb further: with
/// the parent above its minimum, one removal repairs the leaf level
/// and stops, leaving every level valid. (Height 2, so there is a
/// level above the repair to get this wrong in.)
#[test]
fn remove_stops_rebalancing_at_the_first_healthy_level() {
    let live = Arc::new(AtomicIsize::new(0));
    // Right subtree: IMIN minimal leaves. Left subtree: IMIN + 1, so
    // the leaf merge inside it leaves it at IMIN — still legal, and
    // the cascade must stop there.
    let (b, b_first) = minimal_inner(IMIN, 1_000_000, &live, None);
    let (a, _) = minimal_inner(IMIN + 1, 0, &live, Some(b_first));
    let len = (2 * IMIN + 1) * LMIN;
    let root = Inner::test_from_parts(
        vec![1_000_000],
        vec![Node::from_inner(a, &mut Global), Node::from_inner(b, &mut Global)],
    );
    let mut tree =
        BPlusTree { root: Node::from_inner(root, &mut Global), height: 2, len, allocator: Global };

    let got = tree.remove(&0);
    assert!(got.is_some_and(|c| c.0 == 0), "removing a present key must return its value");
    assert_eq!(tree.len(), len - 1);
    assert_eq!(tree.height, 2, "a repair absorbed at the leaf level must not shrink the tree");
    check_tree(&tree);
    for k in [10u64, 1_000, 1_000_000] {
        assert!(tree.get(&k).is_some_and(|c| c.0 == k), "key {k} must remain reachable");
    }

    drop(tree);
    assert_eq!(
        live.load(Relaxed),
        0,
        "teardown must drop every survivor exactly once \
         (positive = leak, negative = double-drop)"
    );
}

// ── beyond the default fanout ──────────────────────────────────────

/// The fanout each byte-array key size is chosen to produce
/// (`M = 512 / (S + 8)`): the key size selects the fanout.
const _: () = assert!(<[u8; 121] as Key>::FANOUT == 3);
const _: () = assert!(<[u8; 120] as Key>::FANOUT == 4);
const _: () = assert!(<[u8; 94] as Key>::FANOUT == 5);

#[cfg(test)]
fn bkey<const S: usize>(k: u8) -> [u8; S] {
    [k; S]
}

/// The full public-API lifecycle at fanout `N`: shuffled inserts
/// growing the tree to at least `min_height` (small fanouts make
/// depth cheap — this covers splits and remove cascades through
/// multiple inner levels, which the u64 tests never reach), every
/// key served, then a differently-shuffled drain back to the empty
/// root leaf, with the invariant net thrown repeatedly and values
/// dropping exactly once.
#[cfg(test)]
fn lifecycle_at_fanout<K: Key + Ord, const N: usize>(mk: impl Fn(u8) -> K, min_height: u8) {
    const KEYS: usize = 120;
    let live = Arc::new(AtomicIsize::new(0));
    {
        let mut tree: BPlusTree<K, Counted, N> = BPlusTree::new();

        let mut keys: Vec<u8> = (0..KEYS as u8).collect();
        keys.sort_by_key(|k| (*k as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        for (i, &k) in keys.iter().enumerate() {
            assert!(
                tree.insert(mk(k), Counted::new(k as u64, &live)).is_none(),
                "key {k} is fresh — there is no value to replace (M={N}, insert #{i})"
            );
            assert_eq!(
                tree.len(),
                i + 1,
                "len must count every inserted pair (M={N}, insert #{i})"
            );
            if i % 16 == 0 {
                check_tree(&tree);
            }
        }
        check_tree(&tree);
        assert!(
            tree.height >= min_height,
            "at fanout {N}, {KEYS} keys must build a tree of height >= {min_height} (got {})",
            tree.height
        );
        assert_eq!(live.load(Relaxed), KEYS as isize, "one live value per inserted key (M={N})");
        for k in 0..KEYS as u8 {
            assert!(
                tree.get(&mk(k)).is_some_and(|v| v.0 == k as u64),
                "key {k} must be served from the deep tree (M={N})"
            );
        }

        keys.sort_by_key(|k| (*k as u64).wrapping_mul(0xD1B5_4A32_D192_ED03));
        for (i, &k) in keys.iter().enumerate() {
            let got = tree.remove(&mk(k));
            assert!(
                got.is_some_and(|v| v.0 == k as u64),
                "removing stored key {k} must return its value (M={N}, removal #{i})"
            );
            assert_eq!(tree.len(), KEYS - i - 1, "len must shrink by one (M={N}, removal #{i})");
            if i % 16 == 0 {
                check_tree(&tree);
            }
        }
        check_tree(&tree);
        assert!(tree.is_empty(), "a full drain must empty the tree (M={N})");
        assert_eq!(tree.height, 0, "a drained tree must shrink back to a root leaf (M={N})");
        assert_eq!(
            live.load(Relaxed),
            0,
            "every removed value must have dropped exactly once via the returned \
             handle (M={N})"
        );
    }
    assert_eq!(
        live.load(Relaxed),
        0,
        "dropping the drained tree must drop nothing further \
         (positive = leak, negative = double-drop; M={N})"
    );
}

/// M == 3, the minimum: `⌈M/2⌉` arithmetic at its tightest.
#[test]
fn full_lifecycle_at_minimum_fanout() {
    lifecycle_at_fanout::<[u8; 121], 3>(bkey::<121>, 3);
}

/// M == 4, the smallest EVEN fanout: the split midpoint and
/// min-occupancy land on the other parity from 3 and 5.
#[test]
fn full_lifecycle_at_even_fanout() {
    lifecycle_at_fanout::<[u8; 120], 4>(bkey::<120>, 3);
}

/// M == 5: with 3, 4, and the u64 default 32, both parities are
/// covered on each side of the `div_ceil` midpoint.
#[test]
fn full_lifecycle_at_odd_fanout() {
    lifecycle_at_fanout::<[u8; 94], 5>(bkey::<94>, 2);
}

/// The churn contract must hold regardless of seed: several seeds,
/// each mirrored against `BTreeMap` from a fresh tree. Skipped under
/// Miri — the single-seed churn above already runs there; this one
/// buys breadth, not memory-safety coverage.
#[test]
#[cfg_attr(miri, ignore)]
fn churn_mirrors_btreemap_across_seeds() {
    use alloc::collections::BTreeMap;

    for seed in [1u64, 42, 0xDEAD_BEEF, 0xABCD_EF01_2345_6789, u64::MAX / 7] {
        let mut tree: BPlusTree<u64, u64, M> = BPlusTree::new();
        let mut model: BTreeMap<u64, u64> = BTreeMap::new();
        let mut state = seed;

        for step in 0..600u64 {
            let r = xorshift(&mut state);
            let key = r % 300;
            if (r >> 32) % 5 < 3 {
                assert_eq!(
                    tree.insert(key, step),
                    model.insert(key, step),
                    "insert({key}) must agree with the model (seed {seed:#x}, step {step})"
                );
            } else {
                assert_eq!(
                    tree.remove(&key),
                    model.remove(&key),
                    "remove({key}) must agree with the model (seed {seed:#x}, step {step})"
                );
            }
            assert_eq!(
                tree.len(),
                model.len(),
                "len must agree with the model (seed {seed:#x}, step {step})"
            );
            if step % 128 == 0 {
                check_tree(&tree);
            }
        }

        check_tree(&tree);
        for (k, val) in &model {
            assert_eq!(
                tree.get(k),
                Some(val),
                "key {k} must match the model at the end (seed {seed:#x})"
            );
        }
    }
}

// ── property-based differential testing ────────────────────────────
//
// proptest generates random op sequences and, on failure, SHRINKS
// the sequence to a minimal reproduction, persisted under
// proptest-regressions/ — COMMIT those files; each is a permanent
// regression test. The property is the differential harness
// (crate::harness, shared with the fuzz targets): every operation
// agrees with `BTreeMap`, and the invariant net holds after every
// mutation.

#[cfg(test)]
use proptest::prelude::*;
#[cfg(test)]
use proptest::test_runner::FileFailurePersistence;

#[cfg(test)]
use crate::harness::{Op, run_differential, wide};

/// Keys mostly from a small domain, so collisions, replacements, and
/// re-inserts of removed keys actually happen.
#[cfg(test)]
fn key_strategy() -> impl Strategy<Value = u64> + Clone {
    prop_oneof![3 => 0u64..64, 1 => any::<u64>()]
}

/// Weighted toward inserts so trees grow deep enough to split, with
/// the full observable surface sprinkled in: point reads, mutable
/// reads, bounded and full iteration (shared and mutable), and the
/// occasional clear.
#[cfg(test)]
fn op_strategy() -> impl Strategy<Value = Op> + Clone {
    prop_oneof![
        40 => (key_strategy(), any::<u64>()).prop_map(|(k, v)| Op::Insert(k, v)),
        25 => key_strategy().prop_map(Op::Remove),
        10 => key_strategy().prop_map(Op::Get),
        8 => (key_strategy(), any::<u64>()).prop_map(|(k, v)| Op::GetMut(k, v)),
        6 => (key_strategy(), key_strategy(), any::<u8>())
            .prop_map(|(a, b, kinds)| Op::Range(a, b, kinds)),
        5 => (key_strategy(), key_strategy(), any::<u8>(), any::<u64>())
            .prop_map(|(a, b, kinds, d)| Op::RangeMut(a, b, kinds, d)),
        3 => Just(Op::IterAll),
        2 => any::<u64>().prop_map(Op::MutateAll),
        1 => Just(Op::Clear),
    ]
}

/// Bulk-load seed sizes: deep enough to start above height 0, small
/// enough that the per-mutation invariant net stays affordable
/// (especially under Miri).
#[cfg(test)]
fn seed_strategy() -> impl Strategy<Value = u64> + Clone {
    0u64..if cfg!(miri) { 48 } else { 768 }
}

/// `Debug` renders the tree exactly like the reference map renders
/// the same pairs: `debug_map` shape, ascending key order.
#[test]
fn debug_formats_like_btreemap() {
    use alloc::{collections::BTreeMap, format};

    let n = 2 * M as u64 + 3;
    let tree: BPlusTree<u64, u64, M> = (0..n).map(|k| (k, v(k))).collect();
    let model: BTreeMap<u64, u64> = (0..n).map(|k| (k, v(k))).collect();
    assert_eq!(
        format!("{tree:?}"),
        format!("{model:?}"),
        "Debug must render the pairs in ascending key order, debug_map-shaped"
    );

    let empty: BPlusTree<u64, u64, M> = BPlusTree::new();
    assert_eq!(format!("{empty:?}"), "{}", "an empty tree must render as an empty map");
}

/// `first_key_value`/`last_key_value`: `None` on the empty tree, the
/// extreme pairs otherwise — tracking growth and removal.
#[test]
fn first_and_last_key_value_return_the_extreme_pairs() {
    let mut tree: BPlusTree<u64, u64, M> = BPlusTree::new();
    assert_eq!(tree.first_key_value(), None, "an empty tree has no first pair");
    assert_eq!(tree.last_key_value(), None, "an empty tree has no last pair");

    tree.insert(5, v(5));
    assert_eq!(tree.first_key_value(), Some((&5, &v(5))), "a lone pair is both extremes");
    assert_eq!(tree.last_key_value(), Some((&5, &v(5))), "a lone pair is both extremes");

    let n = (M * M + 1) as u64;
    let mut tree: BPlusTree<u64, u64, M> = BPlusTree::from_sorted_iter((0..n).map(|k| (k, v(k))));
    assert_eq!(tree.first_key_value(), Some((&0, &v(0))), "first must be the minimum pair");
    assert_eq!(tree.last_key_value(), Some((&(n - 1), &v(n - 1))), "last must be the maximum pair");

    tree.remove(&0);
    tree.remove(&(n - 1));
    assert_eq!(
        tree.first_key_value(),
        Some((&1, &v(1))),
        "first must track removal of the minimum"
    );
    assert_eq!(
        tree.last_key_value(),
        Some((&(n - 2), &v(n - 2))),
        "last must track removal of the maximum"
    );
}

// ── pop_first / pop_last ────────────────────────────────────────
//
// The `BTreeMap` contract: `pop_first` removes and returns the
// minimum pair, `pop_last` the maximum, and both return `None` on
// the empty tree. A pop is a removal at an end — the same rebalance
// obligations as `remove` — pinned here through full drains and a
// model-mirroring churn.

/// `pop_first` on an empty tree returns `None` — whether the tree is
/// freshly built or was emptied by removal — leaving it valid and
/// usable.
#[test]
fn pop_first_on_an_empty_tree_returns_none() {
    let mut tree: BPlusTree<u64, u64, M> = BPlusTree::new();
    assert_eq!(tree.pop_first(), None, "a fresh tree has no first pair to pop");
    assert_eq!(tree.len(), 0, "a pop miss must not change len");
    check_tree(&tree);

    tree.insert(1, v(1));
    assert_eq!(tree.remove(&1), Some(v(1)));
    assert_eq!(tree.pop_first(), None, "an emptied tree has no first pair to pop");
    check_tree(&tree);

    tree.insert(2, v(2));
    assert_eq!(tree.get(&2), Some(&v(2)), "the tree must remain usable after pop misses");
    assert_eq!(tree.len(), 1);
}

/// `pop_last` on an empty tree returns `None` — whether the tree is
/// freshly built or was emptied by removal — leaving it valid and
/// usable.
#[test]
fn pop_last_on_an_empty_tree_returns_none() {
    let mut tree: BPlusTree<u64, u64, M> = BPlusTree::new();
    assert_eq!(tree.pop_last(), None, "a fresh tree has no last pair to pop");
    assert_eq!(tree.len(), 0, "a pop miss must not change len");
    check_tree(&tree);

    tree.insert(1, v(1));
    assert_eq!(tree.remove(&1), Some(v(1)));
    assert_eq!(tree.pop_last(), None, "an emptied tree has no last pair to pop");
    check_tree(&tree);

    tree.insert(2, v(2));
    assert_eq!(tree.get(&2), Some(&v(2)), "the tree must remain usable after pop misses");
    assert_eq!(tree.len(), 1);
}

/// A lone pair is both extremes: either pop returns it and leaves an
/// empty, valid tree at height 0.
#[test]
fn popping_the_lone_pair_empties_the_tree() {
    for pop_last in [false, true] {
        let mut tree: BPlusTree<u64, u64, M> = BPlusTree::new();
        tree.insert(7, v(7));

        let got = if pop_last { tree.pop_last() } else { tree.pop_first() };
        assert_eq!(got, Some((7, v(7))), "the lone pair is both extremes (pop_last={pop_last})");
        assert!(tree.is_empty(), "popping the lone pair must empty the tree (pop_last={pop_last})");
        check_tree(&tree);
    }
}

/// `pop_first` removes and returns the minimum pair — the pair
/// `first_key_value` reports — decrementing `len` and promoting the
/// next key to minimum.
#[test]
fn pop_first_returns_the_minimum_pair() {
    let n = (M * M + 1) as u64;
    let mut tree: BPlusTree<u64, u64, M> = BPlusTree::from_sorted_iter((0..n).map(|k| (k, v(k))));

    assert_eq!(tree.pop_first(), Some((0, v(0))), "pop_first must return the minimum pair");
    assert_eq!(tree.len(), (n - 1) as usize, "a pop hit must decrement len");
    assert_eq!(tree.get(&0), None, "the popped pair must be gone from the tree");
    assert_eq!(
        tree.first_key_value(),
        Some((&1, &v(1))),
        "the next key up must become the minimum"
    );
    check_tree(&tree);
}

/// `pop_last` removes and returns the maximum pair — the pair
/// `last_key_value` reports — decrementing `len` and demoting the
/// next key down to maximum.
#[test]
fn pop_last_returns_the_maximum_pair() {
    let n = (M * M + 1) as u64;
    let mut tree: BPlusTree<u64, u64, M> = BPlusTree::from_sorted_iter((0..n).map(|k| (k, v(k))));

    assert_eq!(tree.pop_last(), Some((n - 1, v(n - 1))), "pop_last must return the maximum pair");
    assert_eq!(tree.len(), (n - 1) as usize, "a pop hit must decrement len");
    assert_eq!(tree.get(&(n - 1)), None, "the popped pair must be gone from the tree");
    assert_eq!(
        tree.last_key_value(),
        Some((&(n - 2), &v(n - 2))),
        "the next key down must become the maximum"
    );
    check_tree(&tree);
}

/// Draining a multi-level tree through `pop_first` alone yields every
/// pair in ascending key order — through leaf borrows, merges, and
/// root shrinks — back to an empty root leaf.
#[test]
fn pop_first_drains_in_ascending_order() {
    let n = (M * M + 1) as u64;
    let mut tree: BPlusTree<u64, u64, M> = BPlusTree::from_sorted_iter((0..n).map(|k| (k, v(k))));

    for expect in 0..n {
        assert_eq!(
            tree.pop_first(),
            Some((expect, v(expect))),
            "pop_first must yield pair {expect} of the ascending order"
        );
        assert_eq!(tree.len(), (n - 1 - expect) as usize, "len must tick down on every pop");
        if expect % 128 == 0 {
            check_tree(&tree);
        }
    }
    assert!(tree.is_empty(), "a full drain must empty the tree");
    assert_eq!(tree.height, 0, "a drained tree must shrink back to a root leaf");
    check_tree(&tree);
}

/// Draining through `pop_last` alone yields every pair in descending
/// key order — through leaf borrows, merges, and root shrinks — back
/// to an empty root leaf.
#[test]
fn pop_last_drains_in_descending_order() {
    let n = (M * M + 1) as u64;
    let mut tree: BPlusTree<u64, u64, M> = BPlusTree::from_sorted_iter((0..n).map(|k| (k, v(k))));

    for expect in (0..n).rev() {
        assert_eq!(
            tree.pop_last(),
            Some((expect, v(expect))),
            "pop_last must yield pair {expect} of the descending order"
        );
        assert_eq!(tree.len(), expect as usize, "len must tick down on every pop");
        if expect % 128 == 0 {
            check_tree(&tree);
        }
    }
    assert!(tree.is_empty(), "a full drain must empty the tree");
    assert_eq!(tree.height, 0, "a drained tree must shrink back to a root leaf");
    check_tree(&tree);
}

/// Alternating pops from both ends drain inward in agreement with
/// `BTreeMap`, every pop a hit until the tree is empty.
#[test]
fn alternating_pops_mirror_btreemap() {
    use alloc::collections::BTreeMap;

    let n = (M * M + 1) as u64;
    let mut tree: BPlusTree<u64, u64, M> = BPlusTree::from_sorted_iter((0..n).map(|k| (k, v(k))));
    let mut model: BTreeMap<u64, u64> = (0..n).map(|k| (k, v(k))).collect();

    for step in 0..n {
        let (got, want) = if step % 2 == 0 {
            (tree.pop_first(), model.pop_first())
        } else {
            (tree.pop_last(), model.pop_last())
        };
        assert!(want.is_some(), "fixture arithmetic: {n} pops of {n} pairs all hit");
        assert_eq!(got, want, "pop must agree with the model at step {step}");
        assert_eq!(tree.len(), model.len(), "len must agree with the model at step {step}");
        if step % 128 == 0 {
            check_tree(&tree);
        }
    }
    assert!(tree.is_empty(), "the model drained, so the tree must have too");
    check_tree(&tree);
}

/// Pops interleaved with inserts and removes: the whole mutation mix
/// must keep agreeing with the model — return values, lengths, and
/// final contents — with the invariants checked throughout.
#[test]
fn pop_churn_mirrors_btreemap() {
    use alloc::collections::BTreeMap;

    // Seeded non-empty so the churn spends its steps on populated
    // trees; the empty-tree pop contract has its own pins above.
    let mut tree: BPlusTree<u64, u64, M> = BPlusTree::from_sorted_iter((0..64).map(|k| (k, v(k))));
    let mut model: BTreeMap<u64, u64> = (0..64).map(|k| (k, v(k))).collect();
    let mut state: u64 = 0xB0B5_1ED5_0DD5_EED5;

    for step in 0..1_500u64 {
        let r = xorshift(&mut state);
        let key = r % 200;
        match (r >> 32) % 8 {
            0..=4 => assert_eq!(
                tree.insert(key, step),
                model.insert(key, step),
                "insert({key}) must agree with the model at step {step}"
            ),
            5 => assert_eq!(
                tree.remove(&key),
                model.remove(&key),
                "remove({key}) must agree with the model at step {step}"
            ),
            6 => assert_eq!(
                tree.pop_first(),
                model.pop_first(),
                "pop_first must agree with the model at step {step}"
            ),
            _ => assert_eq!(
                tree.pop_last(),
                model.pop_last(),
                "pop_last must agree with the model at step {step}"
            ),
        }
        assert_eq!(tree.len(), model.len(), "len must agree with the model at step {step}");
        if step % 128 == 0 {
            check_tree(&tree);
        }
    }

    check_tree(&tree);
    for (k, val) in &model {
        assert_eq!(tree.get(k), Some(val), "key {k} must match the model at the end");
    }
}

/// A pop transfers ownership of the pair to the caller: the value
/// drops when the caller drops it, not inside the tree, and teardown
/// drops only the survivors.
#[test]
fn pop_moves_the_pair_out_without_dropping_it() {
    let live = Arc::new(AtomicIsize::new(0));
    let n = 3 * M as u64;
    {
        let mut tree: BPlusTree<u64, Counted, M> = BPlusTree::new();
        for k in 0..n {
            tree.insert(k, Counted::new(k, &live));
        }
        assert_eq!(live.load(Relaxed), n as isize, "one live value per inserted key");

        let (fk, fv) = tree.pop_first().expect("a populated tree must pop a first pair");
        let (lk, lv) = tree.pop_last().expect("a populated tree must pop a last pair");
        assert_eq!(fk, 0, "pop_first must hand back the minimum key");
        assert_eq!(fv.0, 0, "pop_first must hand back the minimum key's value");
        assert_eq!(lk, n - 1, "pop_last must hand back the maximum key");
        assert_eq!(lv.0, n - 1, "pop_last must hand back the maximum key's value");
        assert_eq!(live.load(Relaxed), n as isize, "a pop must move the values, not drop them");

        drop(fv);
        drop(lv);
        assert_eq!(
            live.load(Relaxed),
            n as isize - 2,
            "dropping the popped values must drop each exactly once"
        );
        check_tree(&tree);
    }
    assert_eq!(
        live.load(Relaxed),
        0,
        "teardown must drop every survivor exactly once \
         (positive = leak, negative = double-drop)"
    );
}

// ── first_leaf / last_leaf ──────────────────────────────────────

/// A height-0 tree is its own chain: `first_leaf` and `last_leaf`
/// are the root leaf itself — empty, and at every fill short of a
/// split.
#[test]
fn first_and_last_leaf_of_a_root_leaf_tree_are_the_root() {
    let mut tree: BPlusTree<u64, u64, M> = BPlusTree::new();
    assert!(
        core::ptr::eq(tree.first_leaf(), tree.last_leaf()),
        "an empty tree's one leaf must be both ends"
    );
    assert_eq!(tree.first_leaf().len(), 0, "the empty root leaf holds nothing");

    // Fill the root leaf to capacity without splitting it.
    for k in 0..M as u64 {
        tree.insert(k, v(k));
    }
    let first = tree.first_leaf();
    let last = tree.last_leaf();
    assert!(core::ptr::eq(first, last), "a lone full root leaf must still be both ends");
    assert_eq!(*first.first_key(), 0, "first_leaf must hold the minimum key");
    assert_eq!(
        first.test_keys().last(),
        Some(&(M as u64 - 1)),
        "last_leaf must hold the maximum key"
    );
}

/// `first_leaf` must hold the tree's minimum key and `last_leaf` its
/// maximum — the chain's terminal leaf — on multi-level trees from
/// both construction paths (insert-grown and bulk-loaded).
#[test]
fn first_and_last_leaf_hold_the_extreme_keys() {
    let n = (M * M + M) as u64;
    let mut grown: BPlusTree<u64, u64, M> = BPlusTree::new();
    for i in 0..n {
        // A stride-permuted insert order, so growth splits all over.
        let k = (i * 7919) % n;
        grown.insert(k, v(k));
    }
    let loaded: BPlusTree<u64, u64, M> = BPlusTree::from_sorted_iter((0..n).map(|k| (k, v(k))));

    for (tree, how) in [(&grown, "insert-grown"), (&loaded, "bulk-loaded")] {
        let first = tree.first_leaf();
        let last = tree.last_leaf();
        assert_eq!(*first.first_key(), 0, "first_leaf must hold the minimum key ({how})");
        assert_eq!(
            last.test_keys().last(),
            Some(&(n - 1)),
            "last_leaf must hold the maximum key ({how})"
        );
        assert!(last.next().is_none(), "last_leaf must be the end of the leaf chain ({how})");
    }
}

/// Walking the sibling links from `first_leaf` must visit every key
/// in ascending order and arrive exactly at `last_leaf`.
#[test]
fn the_leaf_chain_runs_from_first_leaf_to_last_leaf() {
    let n = (M * M + 1) as u64;
    let tree: BPlusTree<u64, u64, M> = BPlusTree::from_sorted_iter((0..n).map(|k| (k, v(k))));

    let mut leaf = tree.first_leaf();
    let mut expect = 0u64;
    loop {
        for k in leaf.test_keys() {
            assert_eq!(*k, expect, "the chain must visit every key in ascending order");
            expect += 1;
        }
        let Some(next) = leaf.next() else { break };
        // SAFETY: chain links point at live leaves the tree owns,
        // borrowed here at the tree's lifetime.
        leaf = unsafe { next.as_ref() };
    }
    assert_eq!(expect, n, "the chain must visit every pair");
    assert!(
        core::ptr::eq(leaf, tree.last_leaf()),
        "the chain from first_leaf must terminate at last_leaf"
    );
}

/// The ends must track mutation: draining inward from both ends —
/// through leaf borrows, merges, and root shrinks — `first_leaf` and
/// `last_leaf` must always hold the current minimum and maximum,
/// down to a lone root leaf again.
#[test]
fn first_and_last_leaf_track_removes() {
    let n = (M * M + 1) as u64;
    let mut tree: BPlusTree<u64, u64, M> = BPlusTree::from_sorted_iter((0..n).map(|k| (k, v(k))));

    let (mut lo, mut hi) = (0u64, n - 1);
    while lo < hi {
        assert_eq!(
            *tree.first_leaf().first_key(),
            lo,
            "first_leaf must hold the minimum after draining below {lo}"
        );
        assert_eq!(
            tree.last_leaf().test_keys().last(),
            Some(&hi),
            "last_leaf must hold the maximum after draining above {hi}"
        );
        tree.remove(&lo);
        tree.remove(&hi);
        lo += 1;
        hi -= 1;
    }

    // n is odd, so exactly one pair remains.
    assert_eq!(tree.len(), 1, "one pair must remain (fixture arithmetic otherwise)");
    let first = tree.first_leaf();
    let last = tree.last_leaf();
    assert!(core::ptr::eq(first, last), "the last pair must live in a lone root leaf");
    assert_eq!(*first.first_key(), lo, "the survivor must be the middle key");
}

#[cfg(test)]
proptest! {
    #![proptest_config(ProptestConfig {
        cases: if cfg!(miri) { 2 } else { 256 },
        // Persistence resolves paths through `getcwd`, which Miri's
        // isolation forbids; regression files are a native-run luxury.
        //
        // Pinned to a fixed path rather than the `SourceParallel` default:
        // this file is attached to the module tree via
        // `#[path = "../tests/tree.rs"]` in `tree/mod.rs`, so `file!()`
        // reports the un-normalized `src/tree/../tests/tree.rs`. proptest's
        // default resolution walks that path looking for a sibling
        // `lib.rs`/`main.rs` and gets thrown off by the literal `..`,
        // landing one directory too shallow (`src/tree/proptest-regressions/`
        // instead of the crate-root `proptest-regressions/`).
        failure_persistence: if cfg!(miri) {
            None
        } else {
            Some(Box::new(FileFailurePersistence::Direct(
                "proptest-regressions/tests/tree.txt",
            )))
        },
        ..ProptestConfig::default()
    })]

    /// Any op sequence, applied to any bulk-loaded starting tree,
    /// must agree with `BTreeMap` at every observable point — with
    /// the invariant net thrown after every mutation — at the
    /// default fanout (M == 32)...
    #[test]
    fn differential_vs_btreemap_at_default_fanout(
        seed in seed_strategy(),
        ops in proptest::collection::vec(op_strategy(), 0..512)
    ) {
        run_differential::<u64, M, Slabs<u64, u64, M>>(|k| k, seed, &ops);
    }

    /// ...and at the minimum fanout (M == 3), where the same
    /// sequences build deep trees and cascade through multiple inner
    /// levels.
    #[test]
    fn differential_vs_btreemap_at_minimum_fanout(
        seed in seed_strategy(),
        ops in proptest::collection::vec(op_strategy(), 0..256)
    ) {
        run_differential::<[u8; 121], 3, Slabs<[u8; 121], u64, 3>>(wide, seed, &ops);
    }
}

/// The level-cap (`H`) contract: `H` level slots must fit any tree the
/// cap admits (a height-`h` tree occupies `h + 1` levels), the scratch
/// they size must cost exactly one slot per level, and a tree grown
/// past its cap must refuse the next descent rather than write out of
/// bounds.
#[cfg(test)]
mod height_cap {
    use super::*;

    /// `Descent` is `H` path slots plus a fixed handful of words — the
    /// per-op stack cost that tuning `H` down exists to shrink.
    #[test]
    fn descent_scratch_scales_with_h() {
        type D4 = crate::tree::descent::Descent<u64, u64, M, Slabs<u64, u64, M>, 4>;
        type D32 = crate::tree::descent::Descent<u64, u64, M, Slabs<u64, u64, M>, 32>;
        let word = size_of::<usize>();
        assert_eq!(
            size_of::<D4>() + 28 * 2 * word,
            size_of::<D32>(),
            "each level of H must cost exactly one (node ptr, child idx) path slot"
        );
        assert!(
            size_of::<D4>() <= 13 * word,
            "the descent's fixed overhead beyond the path must stay a handful of words"
        );
    }

    /// The tightest always-safe cap — `max_levels(M)` — must support
    /// the full lifecycle exactly like the default cap does.
    #[test]
    fn max_levels_cap_supports_the_full_lifecycle() {
        let n: u64 = if cfg!(miri) { 200 } else { 4096 };
        let mut tree: BPlusTree<u64, u64, M, Slabs<u64, u64, M>, { crate::max_levels(M) }> =
            BPlusTree::new();
        for k in 0..n {
            tree.insert(k, v(k));
        }
        tree.check();
        for k in 0..n {
            assert_eq!(tree.get(&k), Some(&v(k)), "every inserted pair must be readable");
        }
        for k in 0..n {
            assert_eq!(tree.remove(&k), Some(v(k)), "every inserted pair must be removable");
        }
        assert!(tree.is_empty(), "the emptied tree must report empty");
    }

    /// `H = 1` admits a tree that never outgrows its root leaf: up to
    /// `M` pairs living entirely at height 0, with every op available.
    #[test]
    fn min_cap_fits_a_root_leaf_tree() {
        let mut tree: BPlusTree<u64, u64, M, Slabs<u64, u64, M>, 1> = BPlusTree::new();
        for k in 0..(M as u64 - 1) {
            tree.insert(k, v(k));
        }
        tree.check();
        for k in 0..(M as u64 - 1) {
            assert_eq!(tree.remove(&k), Some(v(k)), "a height-0 tree must support every op");
        }
        assert!(tree.is_empty(), "the emptied tree must report empty");
    }

    /// `H = 2` admits heights 0 and 1; the insert that must descend a
    /// height-2 tree panics instead of writing past the path. At the
    /// minimum fanout (3), 100 sorted inserts overshoot that height
    /// many times over.
    #[test]
    #[should_panic]
    fn outgrowing_the_cap_panics() {
        let mut tree: BPlusTree<[u8; 121], u64, 3, Slabs<[u8; 121], u64, 3>, 2> = BPlusTree::new();
        for k in 0..100 {
            tree.insert(crate::harness::wide(k), k);
        }
    }

    /// `H = 1` admits only a lone root leaf at height 0. `M` pairs fill
    /// that leaf without splitting it; the very next insert is the one
    /// that must grow the tree to height 1 — a second level `H = 1`
    /// has no room for. That specific insert must panic right there,
    /// not on some other call before or after it.
    #[test]
    #[should_panic]
    fn the_insert_that_outgrows_the_cap_is_the_one_that_panics() {
        let mut tree: BPlusTree<u64, u64, M, Slabs<u64, u64, M>, 1> = BPlusTree::new();
        for k in 0..M as u64 {
            tree.insert(k, v(k));
        }
        tree.insert(M as u64, v(M as u64));
    }
}
