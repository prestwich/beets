//! Contract tests for subtrees rooted at an `Inner`: teardown, sibling
//! merges, and the remove/underflow path.
//!
//! Contract pinned: `Node::drop_subtree`, called at the subtree's true
//! height, drops every node and every value in the subtree exactly
//! once — nothing leaked, nothing dropped twice, nothing outside the
//! initialized prefix touched. `Inner` itself has no drop glue BY
//! DESIGN: it cannot know its children's types, so teardown is driven
//! from above with the height in hand.
//!
//! Instrumentation: `Key` requires `Copy`, so keys can't observe their own
//! drops; the children carry the counters instead. Each child is a
//! heap-allocated leaf holding one `Counted` value, so a child dropped
//! exactly once moves the live counter by exactly one. Run these under
//! `cargo miri test` as well as plain `cargo test` — Miri turns any
//! violation of the memory-safety half of the contract into a clean
//! report instead of whatever a plain run degenerates into, and its leak
//! checker is the backstop for the "exactly once" half.

use alloc::sync::Arc;
use core::sync::atomic::{AtomicIsize, Ordering::Relaxed};

use super::*;
use crate::test_util::{Counted, IMIN, LMIN, M, counted_leaf, inner_with_occupancies, leak};
use crate::{Global, Leaf};

impl<K: Key, V, const M: usize> Inner<K, V, M> {
    /// Test-only constructor: assemble an inner from `children` and the
    /// `children.len() - 1` separators between them, so fixtures in other
    /// modules' tests (the fields are private to this one) can build
    /// multi-level trees.
    pub(crate) fn test_from_parts(keys: Vec<K>, children: Vec<Node<K, V, M>>) -> Self {
        assert!((2..=M).contains(&children.len()), "an inner node holds 2..=M children");
        assert_eq!(keys.len() + 1, children.len(), "n children need n - 1 separators");

        let mut node = Self::new();
        node.child_count = children.len();
        for (i, child) in children.into_iter().enumerate() {
            node.children[i].write(child);
        }
        for (i, k) in keys.into_iter().enumerate() {
            node.keys[i].write(k);
        }
        node
    }

    /// Get a reference to a value in the subtree rooted at this inner node,
    /// if it is present: route to the child owning `key` and recurse at
    /// `height - 1`.
    ///
    /// Test-only — the production descent is iterative (`tree.rs`);
    /// this module's tests assert reachability through it.
    ///
    /// # Safety
    ///
    /// `height` must be the true height of the subtree rooted at this node
    /// (necessarily > 0: this node is an `Inner`; the child it routes to
    /// roots a subtree of `height - 1`).
    pub(crate) unsafe fn get(&self, height: u8, key: &K) -> Option<&V> {
        let child = self.child_for_key(key);
        // SAFETY: height propagation.
        unsafe { child.get(height - 1, key) }
    }

    /// Remove `key` from the subtree rooted at this inner node, if present:
    /// route to the child owning `key` and recurse at `height - 1`.
    /// Recursive remove: route to the child, then rebalance it if the
    /// removal left it deficient.
    ///
    /// Test-only — the production descent is iterative (`tree.rs`);
    /// the node-layer tests drive `rebalance` through it.
    ///
    /// # Safety
    ///
    /// `height` must be the true height of the subtree rooted at this node
    /// (necessarily > 0: this node is an `Inner`; the child it routes to
    /// roots a subtree of `height - 1`).
    #[track_caller]
    pub(crate) unsafe fn remove<A: NodeAllocator<K, V, M>>(
        &mut self,
        height: u8,
        key: &K,
        alloc: &mut A,
    ) -> Option<V> {
        debug_assert!(height > 0);
        // Route once and reuse the index — `child_for_key_mut` would
        // repeat the identical search.
        let child_idx = self.child_idx_for_key(key);
        let child = &mut self.children_mut()[child_idx];

        // SAFETY:
        // Same safety assumptions documented on the function.
        let val = unsafe { child.remove(height - 1, key, alloc) };

        // SAFETY: height propagation.
        unsafe {
            if child.is_deficient(height - 1) {
                self.rebalance(height, child_idx, alloc);
            }
        }

        val
    }
}

/// A leaf-child holding a single counted value, boxed and erased the
/// same way the tree hands leaves to their parents.
fn counted_leaf_child(k: u64, live: &Arc<AtomicIsize>) -> Node<u64, Counted, M> {
    Node::from_leaf_ptr(counted_leaf(k, 1, live, None))
}

/// An `Inner` with `n` children and the matching `n - 1` separator
/// keys. Child `i` holds the key `base + 10 * i`; separator `i` is
/// the min key of child `i + 1`, per the crate's separator
/// convention.
fn inner_with_children(n: usize, base: u64, live: &Arc<AtomicIsize>) -> Inner<u64, Counted, M> {
    let keys: Vec<u64> = (1..n as u64).map(|i| base + 10 * i).collect();
    let children: Vec<Node<u64, Counted, M>> =
        (0..n as u64).map(|i| counted_leaf_child(base + 10 * i, live)).collect();
    Inner::test_from_parts(keys, children)
}

/// Tearing down a height-1 subtree must drop each of the inner node's
/// children exactly once, whatever its occupancy: swept from the
/// 2-child minimum to a full `M` children.
#[test]
fn drop_subtree_at_height_one_drops_every_child_exactly_once() {
    for n in 2..=M {
        let live = Arc::new(AtomicIsize::new(0));

        let node = Node::from_inner(inner_with_children(n, 0, &live), &mut Global);
        assert_eq!(live.load(Relaxed), n as isize, "one live value per child before teardown");

        // SAFETY: `node` roots an inner with leaf children (height 1)
        // and owns the subtree.
        unsafe { node.drop_subtree(1, &mut Global) };
        assert_eq!(
            live.load(Relaxed),
            0,
            "drop_subtree(1) over {n} children must drop each child exactly once \
             (positive = leak, negative = double-drop)"
        );
    }
}

/// Tearing down a height-2 subtree (inner of inners of leaves) must
/// recurse through the middle layer and drop every value exactly once.
/// Pins that teardown is genuinely recursive — the union design has
/// no typed `Drop` to recurse for it; `drop_subtree` must do the
/// walk itself.
#[test]
fn drop_subtree_at_height_two_drops_the_whole_subtree_exactly_once() {
    let live = Arc::new(AtomicIsize::new(0));

    // Two height-1 subtrees over disjoint key ranges (0.. and 100..),
    // separated by the right subtree's min key.
    let left = Node::from_inner(inner_with_children(2, 0, &live), &mut Global);
    let right = Node::from_inner(inner_with_children(2, 100, &live), &mut Global);

    let root = Node::from_inner(Inner::test_from_parts(vec![100], vec![left, right]), &mut Global);
    assert_eq!(live.load(Relaxed), 4, "one live value per grandchild before teardown");

    // SAFETY: `root` roots an inner of inners of leaves (height 2) and
    // owns the subtree.
    unsafe { root.drop_subtree(2, &mut Global) };
    assert_eq!(
        live.load(Relaxed),
        0,
        "drop_subtree(2) must recurse through the middle layer and drop every \
         value exactly once (positive = leak, negative = double-drop)"
    );
}

// ── merge / remove fixtures ────────────────────────────────────────

/// Like [`inner_with_children`], but with the leaf sibling chain wired:
/// child `i`'s leaf links to child `i + 1`'s, and the last child's leaf
/// links to `tail`. Accepts `n == 1` (a keyless single-child node).
/// Returns the node plus the leaf pointers in key order, so assertions
/// can walk the chain afterwards.
#[allow(clippy::type_complexity)]
fn linked_inner_with_children(
    n: usize,
    base: u64,
    live: &Arc<AtomicIsize>,
    tail: Option<NonNull<Leaf<u64, Counted, M>>>,
) -> (Inner<u64, Counted, M>, Vec<NonNull<Leaf<u64, Counted, M>>>) {
    assert!((1..=M).contains(&n), "1..=M children");

    // Build the leaves right-to-left so each can link to its successor.
    let mut ptrs: Vec<NonNull<Leaf<u64, Counted, M>>> = Vec::with_capacity(n);
    let mut next = tail;
    for i in (0..n).rev() {
        let ptr = counted_leaf(base + 10 * i as u64, 1, live, next);
        ptrs.push(ptr);
        next = Some(ptr);
    }
    ptrs.reverse();

    // Assembled by hand rather than through `test_from_parts`, whose
    // 2-child floor would reject the keyless n == 1 node the merge
    // minimum-occupancy test needs.
    let mut keys = [MaybeUninit::uninit(); M];
    let mut children: [MaybeUninit<Node<u64, Counted, M>>; M] =
        core::array::from_fn(|_| MaybeUninit::uninit());
    for (i, ptr) in ptrs.iter().enumerate() {
        children[i].write(Node::from_leaf_ptr(*ptr));
        if i > 0 {
            keys[i - 1].write(base + 10 * i as u64);
        }
    }

    (
        Inner {
            #[cfg(debug_assertions)]
            kind: NodeKind::Inner,
            child_count: n,
            keys,
            children,
        },
        ptrs,
    )
}

/// Walk the leaf chain from `head`, collecting each visited leaf's
/// `(len, first_key)` (`None` first key for an empty leaf). `max_hops`
/// is a cycle guard.
///
/// # Safety
///
/// Every leaf reachable from `head` along `next` must be live.
unsafe fn walk_chain(
    head: NonNull<Leaf<u64, Counted, M>>,
    max_hops: usize,
) -> Vec<(usize, Option<u64>)> {
    let mut out = Vec::new();
    let mut cur = Some(head);
    while let Some(ptr) = cur {
        assert!(out.len() < max_hops, "the leaf chain must terminate (cycle suspected)");
        // SAFETY: the caller vouches every chained leaf is live.
        let leaf = unsafe { ptr.as_ref() };
        let first = (leaf.len() != 0).then(|| *leaf.first_key());
        out.push((leaf.len(), first));
        cur = leaf.next();
    }
    out
}

// ── Inner::merge ───────────────────────────────────────────────────

/// Merging two siblings must concatenate their children in key order,
/// with the separator that sat between them in the parent becoming a
/// key of the merged node — afterwards every key reaches its value
/// through the merged node alone, and teardown stays exactly-once.
#[test]
fn merge_concatenates_children_and_demotes_the_separator() {
    let live = Arc::new(AtomicIsize::new(0));
    let (right, rptrs) = linked_inner_with_children(2, 100, &live, None);
    let (mut left, _) = linked_inner_with_children(2, 0, &live, Some(rptrs[0]));

    // SAFETY: `right` is `left`'s immediate right sibling by
    // construction — adjacent disjoint ranges separated by 100 (the min
    // key under `right`), same height (1), 2 + 2 <= M, and `right`
    // owns its subtree.
    unsafe { left.merge(100, right) };

    assert_eq!(left.len(), 4, "the merged node must hold both sides' children");
    assert_eq!(
        left.keys_ref(),
        &[10, 100, 110],
        "the merged node's keys must be left's keys, the demoted separator, then right's"
    );
    for k in [0u64, 10, 100, 110] {
        // SAFETY: the merged node roots a height-1 subtree.
        let got = unsafe { left.get(1, &k) };
        assert!(got.is_some_and(|v| v.0 == k), "key {k} must remain reachable after the merge");
    }

    assert_eq!(live.load(Relaxed), 4, "a merge must not drop any value");
    // SAFETY: the merged node owns all four leaves (height 1).
    unsafe { Node::from_inner(left, &mut Global).drop_subtree(1, &mut Global) };
    assert_eq!(
        live.load(Relaxed),
        0,
        "teardown after a merge must drop every value exactly once \
         (positive = leak, negative = double-drop)"
    );
}

/// A single-child node brings no keys of its own to a merge; merging
/// two of them must still succeed, producing a two-child node whose
/// only key is the demoted separator.
#[test]
fn merge_succeeds_at_the_one_child_minimum() {
    let live = Arc::new(AtomicIsize::new(0));
    let (right, rptrs) = linked_inner_with_children(1, 100, &live, None);
    let (mut left, _) = linked_inner_with_children(1, 0, &live, Some(rptrs[0]));

    // SAFETY: adjacent single-child siblings over disjoint ranges
    // separated by 100, same height (1), 1 + 1 <= M, `right` owns its
    // subtree. Both members hold at least one child, as the contract
    // requires — no stronger occupancy is demanded of a merge input.
    unsafe { left.merge(100, right) };

    assert_eq!(left.len(), 2, "the merged node must hold both children");
    assert_eq!(left.keys_ref(), &[100], "the demoted separator must be the merged node's only key");
    for k in [0u64, 100] {
        // SAFETY: the merged node roots a height-1 subtree.
        let got = unsafe { left.get(1, &k) };
        assert!(got.is_some_and(|v| v.0 == k), "key {k} must remain reachable after the merge");
    }

    // SAFETY: the merged node owns both leaves (height 1).
    unsafe { Node::from_inner(left, &mut Global).drop_subtree(1, &mut Global) };
    assert_eq!(
        live.load(Relaxed),
        0,
        "teardown after a merge must drop every value exactly once \
         (positive = leak, negative = double-drop)"
    );
}

/// The fit bound is inclusive: a merge may fill the surviving node to
/// exactly `M` children (and so `M - 1` keys), with every key still
/// reachable and teardown exactly-once.
#[test]
fn merge_fills_a_node_to_exact_capacity() {
    let live = Arc::new(AtomicIsize::new(0));
    let a = M / 2;
    let b = M - a;
    let (right, rptrs) = linked_inner_with_children(b, 1000, &live, None);
    let (mut left, _) = linked_inner_with_children(a, 0, &live, Some(rptrs[0]));

    // SAFETY: adjacent siblings over disjoint ranges separated by 1000,
    // same height (1), a + b == M <= M, `right` owns its subtree.
    unsafe { left.merge(1000, right) };

    assert_eq!(left.len(), M, "the merged node must hold exactly M children");
    assert_eq!(left.keys_ref().len(), M - 1, "M children need exactly M - 1 keys");
    for k in [0, 10 * (a as u64 - 1), 1000, 1000 + 10 * (b as u64 - 1)] {
        // SAFETY: the merged node roots a height-1 subtree.
        let got = unsafe { left.get(1, &k) };
        assert!(got.is_some_and(|v| v.0 == k), "key {k} must remain reachable after the merge");
    }

    // SAFETY: the merged node owns all M leaves (height 1).
    unsafe { Node::from_inner(left, &mut Global).drop_subtree(1, &mut Global) };
    assert_eq!(
        live.load(Relaxed),
        0,
        "teardown after a capacity-filling merge must drop every value exactly once \
         (positive = leak, negative = double-drop)"
    );
}

// ── Inner::rotate_from_{right,left} ────────────────────────────────

/// Rotating in from the right sibling must: demote the parent
/// separator to the receiver's LAST key, move the donor's FIRST child
/// across, and promote the donor's first key out as the replacement
/// separator — with every key still routing to its value on the
/// correct side, and nothing dropped.
#[test]
fn rotate_from_right_demotes_sep_takes_first_child_promotes_first_key() {
    let live = Arc::new(AtomicIsize::new(0));
    let (mut donor, dptrs) = linked_inner_with_children(3, 100, &live, None);
    let (mut recv, _) = linked_inner_with_children(2, 0, &live, Some(dptrs[0]));

    // SAFETY: adjacent same-parent siblings over disjoint ranges
    // separated by 100 (the donor subtree's min); both height 1; the
    // receiver has room and the donor keeps >= 1 child.
    let new_sep = unsafe { recv.rotate_from_right(100, &mut donor) };

    assert_eq!(new_sep, 110, "the donor's first key must promote out as the new separator");
    assert_eq!(recv.len(), 3, "the receiver must gain exactly one child");
    assert_eq!(donor.len(), 2, "the donor must lose exactly one child");
    assert_eq!(
        recv.test_keys(),
        &[10, 100],
        "the old separator must demote to the receiver's LAST key"
    );
    assert_eq!(donor.test_keys(), &[120], "the donor's keys must close over the promoted one");
    for (node, k) in [(&recv, 0u64), (&recv, 10), (&recv, 100), (&donor, 110), (&donor, 120)] {
        // SAFETY: each node roots a height-1 subtree.
        let got = unsafe { node.get(1, &k) };
        assert!(got.is_some_and(|v| v.0 == k), "key {k} must route to its value");
    }
    assert_eq!(live.load(Relaxed), 5, "a rotation must not drop any value");

    // SAFETY: each node owns its (rearranged) height-1 subtree.
    unsafe { Node::from_inner(recv, &mut Global).drop_subtree(1, &mut Global) };
    unsafe { Node::from_inner(donor, &mut Global).drop_subtree(1, &mut Global) };
    assert_eq!(
        live.load(Relaxed),
        0,
        "teardown after a rotation must drop every value exactly once \
         (positive = leak, negative = double-drop)"
    );
}

/// Rotating in from the left sibling must: demote the parent separator
/// to the receiver's FIRST key, move the donor's LAST child across,
/// and promote the donor's last key out as the replacement separator.
#[test]
fn rotate_from_left_demotes_sep_takes_last_child_promotes_last_key() {
    let live = Arc::new(AtomicIsize::new(0));
    let (mut recv, rptrs) = linked_inner_with_children(2, 100, &live, None);
    let (mut donor, _) = linked_inner_with_children(3, 0, &live, Some(rptrs[0]));

    // SAFETY: adjacent same-parent siblings over disjoint ranges
    // separated by 100 (the receiver subtree's min); both height 1;
    // the receiver has room and the donor keeps >= 1 child.
    let new_sep = unsafe { recv.rotate_from_left(100, &mut donor) };

    assert_eq!(new_sep, 20, "the donor's last key must promote out as the new separator");
    assert_eq!(recv.len(), 3, "the receiver must gain exactly one child");
    assert_eq!(donor.len(), 2, "the donor must lose exactly one child");
    assert_eq!(
        recv.test_keys(),
        &[100, 110],
        "the old separator must demote to the receiver's FIRST key"
    );
    assert_eq!(donor.test_keys(), &[10], "the donor's keys must close over the promoted one");
    for (node, k) in [(&donor, 0u64), (&donor, 10), (&recv, 20), (&recv, 100), (&recv, 110)] {
        // SAFETY: each node roots a height-1 subtree.
        let got = unsafe { node.get(1, &k) };
        assert!(got.is_some_and(|v| v.0 == k), "key {k} must route to its value");
    }
    assert_eq!(live.load(Relaxed), 5, "a rotation must not drop any value");

    // SAFETY: each node owns its (rearranged) height-1 subtree.
    unsafe { Node::from_inner(recv, &mut Global).drop_subtree(1, &mut Global) };
    unsafe { Node::from_inner(donor, &mut Global).drop_subtree(1, &mut Global) };
    assert_eq!(
        live.load(Relaxed),
        0,
        "teardown after a rotation must drop every value exactly once \
         (positive = leak, negative = double-drop)"
    );
}

// ── remove under the min-occupancy invariant ───────────────────────
//
// These tests pin the C (classical rebalancing) contract for
// `Inner::remove`: after any remove, every node in the subtree
// satisfies its minimum occupancy (checked by
// `Node::check_invariants`), a deficient child is repaired by
// borrowing from a sibling above its minimum — merging only when both
// members sit at the minimum — and the leaf chain, reachability, and
// drop-exactly-once accounting all survive the repair.

/// A root over `left` and `right` height-1 subtrees, separated by
/// `sep` (the right subtree's min key).
fn root_over(
    left: Inner<u64, Counted, M>,
    right: Inner<u64, Counted, M>,
    sep: u64,
) -> Inner<u64, Counted, M> {
    Inner::test_from_parts(
        vec![sep],
        vec![Node::from_inner(left, &mut Global), Node::from_inner(right, &mut Global)],
    )
}

/// Removing below the minimum from a leaf whose RIGHT sibling is above
/// its own minimum must repair by borrowing: no leaf is freed, both
/// leaves end at or above the minimum, and the invariants hold.
#[test]
fn remove_from_a_minimal_leaf_borrows_from_a_richer_right_sibling() {
    let live = Arc::new(AtomicIsize::new(0));
    let (inner, ptrs) = inner_with_occupancies(&[LMIN, LMIN + 1], 0, &live, None);
    let total = (2 * LMIN + 1) as isize;
    let mut node = Node::from_inner(inner, &mut Global);

    // SAFETY: `node` roots a height-1 subtree; 1 is its true height.
    let got = unsafe { node.remove(1, &0, &mut Global) };
    assert!(got.is_some_and(|v| v.0 == 0), "removing present key 0 must return its value");
    assert_eq!(live.load(Relaxed), total - 1, "exactly the removed value must drop");

    // Borrow, not merge: both leaves survive, both at or above MIN.
    // SAFETY: a borrow frees no leaf, so the fixture pointers are live.
    let hops = unsafe { walk_chain(ptrs[0], 3) };
    assert_eq!(hops.len(), 2, "a borrow must not free either leaf: {hops:?}");
    assert!(
        hops.iter().all(|(len, _)| *len >= LMIN),
        "both leaves must end at or above MIN_OCCUPANCY: {hops:?}"
    );

    // SAFETY: height-1 subtree, judged as a root (2 children).
    unsafe { node.check_invariants(1, true) };

    // The boundary that moved: the donor's old first key must now
    // route LEFT of the updated separator.
    for k in [10u64, 1_000, 1_010] {
        // SAFETY: height-1 subtree.
        let got = unsafe { node.get(1, &k) };
        assert!(got.is_some_and(|v| v.0 == k), "surviving key {k} must remain reachable");
    }
    // SAFETY: height-1 subtree.
    assert!(unsafe { node.get(1, &0) }.is_none(), "removed key 0 must be absent");

    // SAFETY: `node` owns the subtree.
    unsafe { node.drop_subtree(1, &mut Global) };
    assert_eq!(
        live.load(Relaxed),
        0,
        "teardown must drop every survivor exactly once \
         (positive = leak, negative = double-drop)"
    );
}

/// Removing below the minimum from a leaf with no right sibling must
/// borrow from its richer LEFT sibling.
#[test]
fn remove_from_a_minimal_leaf_borrows_from_a_richer_left_sibling() {
    let live = Arc::new(AtomicIsize::new(0));
    let (inner, ptrs) = inner_with_occupancies(&[LMIN + 1, LMIN], 0, &live, None);
    let total = (2 * LMIN + 1) as isize;
    let mut node = Node::from_inner(inner, &mut Global);

    // SAFETY: `node` roots a height-1 subtree; 1 is its true height.
    let got = unsafe { node.remove(1, &1_000, &mut Global) };
    assert!(got.is_some_and(|v| v.0 == 1_000), "removing present key 1000 must return its value");
    assert_eq!(live.load(Relaxed), total - 1, "exactly the removed value must drop");

    // SAFETY: a borrow frees no leaf, so the fixture pointers are live.
    let hops = unsafe { walk_chain(ptrs[0], 3) };
    assert_eq!(hops.len(), 2, "a borrow must not free either leaf: {hops:?}");
    assert!(
        hops.iter().all(|(len, _)| *len >= LMIN),
        "both leaves must end at or above MIN_OCCUPANCY: {hops:?}"
    );

    // SAFETY: height-1 subtree, judged as a root (2 children).
    unsafe { node.check_invariants(1, true) };

    // The donor's old last key crossed the boundary rightward.
    for k in [0u64, 10 * LMIN as u64, 1_010] {
        // SAFETY: height-1 subtree.
        let got = unsafe { node.get(1, &k) };
        assert!(got.is_some_and(|v| v.0 == k), "surviving key {k} must remain reachable");
    }
    // SAFETY: height-1 subtree.
    assert!(unsafe { node.get(1, &1_000) }.is_none(), "removed key 1000 must be absent");

    // SAFETY: `node` owns the subtree.
    unsafe { node.drop_subtree(1, &mut Global) };
    assert_eq!(
        live.load(Relaxed),
        0,
        "teardown must drop every survivor exactly once \
         (positive = leak, negative = double-drop)"
    );
}

/// A borrow rewrites exactly ONE separator: the boundary between the
/// receiver and its donor. With a third child in the node, the
/// separator above the rewritten one is live — and must keep its
/// value, so keys under it stay reachable and the key order holds.
#[test]
fn borrowing_from_the_right_leaves_the_separator_above_intact() {
    let live = Arc::new(AtomicIsize::new(0));
    // Child 0 will go deficient; child 1 is the richer right donor;
    // child 2 exists only to keep a live separator (2000) above the
    // rewritten boundary.
    let (mut inner, _ptrs) = inner_with_occupancies(&[LMIN, LMIN + 1, LMIN], 0, &live, None);

    // SAFETY: `inner` roots a height-1 subtree; 1 is its true height.
    let got = unsafe { inner.remove(1, &0, &mut Global) };
    assert!(got.is_some_and(|v| v.0 == 0), "removing present key 0 must return its value");

    assert_eq!(
        inner.test_keys(),
        &[1_010, 2_000],
        "only the receiver/donor boundary separator may change: the donor's new \
         first key replaces 1000, and the separator 2000 above it must survive"
    );

    let node = Node::from_inner(inner, &mut Global);
    // SAFETY: height-1 subtree, judged as a root.
    unsafe { node.check_invariants(1, true) };

    // The keys of the child above the borrow must remain reachable.
    for k in [2_000u64, 2_010] {
        // SAFETY: height-1 subtree.
        let got = unsafe { node.get(1, &k) };
        assert!(got.is_some_and(|v| v.0 == k), "surviving key {k} must remain reachable");
    }

    // SAFETY: `node` owns the subtree.
    unsafe { node.drop_subtree(1, &mut Global) };
    assert_eq!(
        live.load(Relaxed),
        0,
        "teardown must drop every survivor exactly once \
         (positive = leak, negative = double-drop)"
    );
}

/// Mirror of the right-borrow pin for a LEFT borrow away from the
/// node's edge: the deficient middle child borrows from its richer
/// left sibling (its right sibling is at minimum and cannot donate),
/// rewriting the boundary below it — and the separator above the
/// receiver must keep its value.
#[test]
fn borrowing_from_the_left_leaves_the_separator_above_intact() {
    let live = Arc::new(AtomicIsize::new(0));
    // Child 1 will go deficient; child 0 is the richer left donor;
    // child 2 (at minimum, no donor) keeps a live separator (2000)
    // above the rewritten boundary.
    let (mut inner, _ptrs) = inner_with_occupancies(&[LMIN + 1, LMIN, LMIN], 0, &live, None);

    // SAFETY: `inner` roots a height-1 subtree; 1 is its true height.
    let got = unsafe { inner.remove(1, &1_000, &mut Global) };
    assert!(got.is_some_and(|v| v.0 == 1_000), "removing present key 1000 must return its value");

    assert_eq!(
        inner.test_keys(),
        &[10 * LMIN as u64, 2_000],
        "only the donor/receiver boundary separator may change: the donor's moved \
         last key replaces 1000, and the separator 2000 above it must survive"
    );

    let node = Node::from_inner(inner, &mut Global);
    // SAFETY: height-1 subtree, judged as a root.
    unsafe { node.check_invariants(1, true) };

    // The keys of the child above the borrow must remain reachable.
    for k in [2_000u64, 2_010] {
        // SAFETY: height-1 subtree.
        let got = unsafe { node.get(1, &k) };
        assert!(got.is_some_and(|v| v.0 == k), "surviving key {k} must remain reachable");
    }

    // SAFETY: `node` owns the subtree.
    unsafe { node.drop_subtree(1, &mut Global) };
    assert_eq!(
        live.load(Relaxed),
        0,
        "teardown must drop every survivor exactly once \
         (positive = leak, negative = double-drop)"
    );
}

/// When the deficient leaf and its sibling BOTH sit at the minimum,
/// they merge. At the head position this exercises the chain-critical
/// direction: the head leaf's address must survive (an outside
/// predecessor could link into it), and walking from it must find
/// every surviving pair, in order.
#[test]
fn remove_merges_minimal_leaves_and_preserves_the_chain() {
    let live = Arc::new(AtomicIsize::new(0));
    let (inner, ptrs) = inner_with_occupancies(&[LMIN, LMIN, LMIN], 0, &live, None);
    let total = (3 * LMIN) as isize;
    let mut node = Node::from_inner(inner, &mut Global);

    // SAFETY: `node` roots a height-1 subtree; 1 is its true height.
    let got = unsafe { node.remove(1, &0, &mut Global) };
    assert!(got.is_some_and(|v| v.0 == 0), "removing present key 0 must return its value");
    assert_eq!(live.load(Relaxed), total - 1, "exactly the removed value must drop");

    // Merge, not borrow: one leaf slot closes; the head survives.
    // SAFETY: the head leaf's address must outlive the merge; the rest
    // of the chain is whatever remove left reachable from it.
    let hops = unsafe { walk_chain(ptrs[0], 4) };
    assert_eq!(hops.len(), 2, "minimal siblings must merge into one leaf: {hops:?}");
    let pairs: usize = hops.iter().map(|(len, _)| len).sum();
    assert_eq!(pairs, 3 * LMIN - 1, "the chain must hold every surviving pair exactly once");
    let firsts: Vec<u64> = hops.iter().filter_map(|(_, first)| *first).collect();
    assert!(firsts.windows(2).all(|w| w[0] < w[1]), "chain order must hold: {firsts:?}");

    // SAFETY: height-1 subtree, judged as a root (2 children).
    unsafe { node.check_invariants(1, true) };

    for k in [10u64, 1_000, 2_000] {
        // SAFETY: height-1 subtree.
        let got = unsafe { node.get(1, &k) };
        assert!(got.is_some_and(|v| v.0 == k), "surviving key {k} must remain reachable");
    }

    // SAFETY: `node` owns the subtree.
    unsafe { node.drop_subtree(1, &mut Global) };
    assert_eq!(
        live.load(Relaxed),
        0,
        "teardown must drop every survivor exactly once \
         (positive = leak, negative = double-drop)"
    );
}

/// The mirror merge at the TAIL position: the deficient last leaf
/// folds into its left sibling.
#[test]
fn remove_merges_minimal_leaves_at_the_tail() {
    let live = Arc::new(AtomicIsize::new(0));
    let (inner, ptrs) = inner_with_occupancies(&[LMIN, LMIN, LMIN], 0, &live, None);
    let total = (3 * LMIN) as isize;
    let mut node = Node::from_inner(inner, &mut Global);

    // SAFETY: `node` roots a height-1 subtree; 1 is its true height.
    let got = unsafe { node.remove(1, &2_000, &mut Global) };
    assert!(got.is_some_and(|v| v.0 == 2_000), "removing present key 2000 must return its value");
    assert_eq!(live.load(Relaxed), total - 1, "exactly the removed value must drop");

    // SAFETY: the head leaf is untouched by a tail-side merge.
    let hops = unsafe { walk_chain(ptrs[0], 4) };
    assert_eq!(hops.len(), 2, "minimal siblings must merge into one leaf: {hops:?}");
    let pairs: usize = hops.iter().map(|(len, _)| len).sum();
    assert_eq!(pairs, 3 * LMIN - 1, "the chain must hold every surviving pair exactly once");

    // SAFETY: height-1 subtree, judged as a root (2 children).
    unsafe { node.check_invariants(1, true) };

    for k in [0u64, 1_000, 2_010] {
        // SAFETY: height-1 subtree.
        let got = unsafe { node.get(1, &k) };
        assert!(got.is_some_and(|v| v.0 == k), "surviving key {k} must remain reachable");
    }
    // SAFETY: height-1 subtree.
    assert!(unsafe { node.get(1, &2_000) }.is_none(), "removed key 2000 must be absent");

    // SAFETY: `node` owns the subtree.
    unsafe { node.drop_subtree(1, &mut Global) };
    assert_eq!(
        live.load(Relaxed),
        0,
        "teardown must drop every survivor exactly once \
         (positive = leak, negative = double-drop)"
    );
}

/// A leaf merge that leaves an inner node deficient must cascade: the
/// inner borrows a child from its richer sibling (a rotation through
/// the root separator), leaving every node at or above its minimum.
#[test]
fn remove_cascades_a_borrow_across_inner_nodes() {
    let live = Arc::new(AtomicIsize::new(0));
    // A: IMIN minimal leaves — one leaf merge inside makes A itself
    // deficient. B: IMIN + 1 minimal leaves, the donor.
    let (b, bptrs) = inner_with_occupancies(&vec![LMIN; IMIN + 1], 1_000_000, &live, None);
    let (a, _) = inner_with_occupancies(&[LMIN; IMIN], 0, &live, Some(bptrs[0]));
    let total = ((2 * IMIN + 1) * LMIN) as isize;
    let mut root = root_over(a, b, 1_000_000);

    // SAFETY: `root` roots a height-2 subtree; 2 is its true height.
    let got = unsafe { root.remove(2, &0, &mut Global) };
    assert!(got.is_some_and(|v| v.0 == 0), "removing present key 0 must return its value");
    assert_eq!(live.load(Relaxed), total - 1, "exactly the removed value must drop");

    assert_eq!(root.len(), 2, "the root must keep both inner children (borrow, not merge)");

    let root = Node::from_inner(root, &mut Global);
    // SAFETY: height-2 subtree, judged as a root.
    unsafe { root.check_invariants(2, true) };

    // The child that crossed sides and the donor's remainder both
    // stay reachable through the root.
    for k in [10u64, 1_000_000, 1_001_000] {
        // SAFETY: height-2 subtree.
        let got = unsafe { root.get(2, &k) };
        assert!(got.is_some_and(|v| v.0 == k), "surviving key {k} must remain reachable");
    }
    // SAFETY: height-2 subtree.
    assert!(unsafe { root.get(2, &0) }.is_none(), "removed key 0 must be absent");

    // SAFETY: `root` owns the subtree.
    unsafe { root.drop_subtree(2, &mut Global) };
    assert_eq!(
        live.load(Relaxed),
        0,
        "teardown must drop every survivor exactly once \
         (positive = leak, negative = double-drop)"
    );
}

/// When both inner siblings sit at the minimum, the cascade merges
/// them, leaving the root a single child — whose subtree must be fully
/// valid. Hoisting that child (root shrink) is the tree layer's job
/// and is out of scope here.
#[test]
fn remove_cascades_a_merge_across_minimal_inner_nodes() {
    let live = Arc::new(AtomicIsize::new(0));
    let (b, bptrs) = inner_with_occupancies(&[LMIN; IMIN], 1_000_000, &live, None);
    let (a, _) = inner_with_occupancies(&[LMIN; IMIN], 0, &live, Some(bptrs[0]));
    let total = (2 * IMIN * LMIN) as isize;
    let mut root = root_over(a, b, 1_000_000);

    // SAFETY: `root` roots a height-2 subtree; 2 is its true height.
    let got = unsafe { root.remove(2, &0, &mut Global) };
    assert!(got.is_some_and(|v| v.0 == 0), "removing present key 0 must return its value");
    assert_eq!(live.load(Relaxed), total - 1, "exactly the removed value must drop");

    assert_eq!(
        root.len(),
        1,
        "minimal inner siblings must merge, leaving the root a single child \
         for the tree layer to hoist"
    );
    // The merged child is a fully valid NON-root inner.
    // SAFETY: the root's child roots a height-1 subtree.
    unsafe { root.test_children()[0].check_invariants(1, false) };

    for k in [10u64, 1_000_000, 1_000_000 + 1_000 * (IMIN as u64 - 1)] {
        // SAFETY: height-2 subtree.
        let got = unsafe { root.get(2, &k) };
        assert!(got.is_some_and(|v| v.0 == k), "surviving key {k} must remain reachable");
    }

    // SAFETY: `root` owns the height-2 subtree (a 1-child root still
    // owns everything below it).
    unsafe { Node::from_inner(root, &mut Global).drop_subtree(2, &mut Global) };
    assert_eq!(
        live.load(Relaxed),
        0,
        "teardown must drop every survivor exactly once \
         (positive = leak, negative = double-drop)"
    );
}

// ── the write path: insert_unchecked / insert_child /
//    splitting_insert_child ─────────────────────────────────────────
//
// The scene is always the same: a child leaf of this node has just
// split — the new right leaf is already spliced into the LEAF CHAIN
// (that is `Leaf::splitting_insert`'s job) — and this node must now
// adopt the (separator, new child) pair: in place when it has room,
// splitting itself when full. Contract pinned, whichever path runs:
// the node counts the new child, every key in the subtree routes to
// its leaf, tree child order agrees with the leaf chain
// (`check_invariants`), and teardown drops every value exactly once.
// Run these under `cargo miri test` too — the memory-safety half of
// the contract (no reads or writes outside the initialized prefixes)
// is Miri's department.

/// A leaf holding the `LMIN` pairs `base, base + 1, ..`, boxed and
/// leaked, linked to `next` — occupancy-legal for a non-root leaf, so
/// `check_invariants` can judge the whole subtree.
fn scene_leaf(
    base: u64,
    live: &Arc<AtomicIsize>,
    next: Option<NonNull<Leaf<u64, Counted, M>>>,
) -> NonNull<Leaf<u64, Counted, M>> {
    let mut leaf: Leaf<u64, Counted, M> = Leaf::new(next);
    for j in 0..LMIN as u64 {
        leaf.raw_append(base + j, Counted::new(base + j, live));
    }
    leak(leaf)
}

/// The canonical adoption scene: a parent over `n` leaves (leaf `i`
/// holds the keys `100·i ..`, separators `100·i` per the crate
/// convention), with a FRESH leaf (keys `100·split_idx + 50 ..`)
/// already spliced into the leaf chain immediately after child
/// `split_idx` — exactly the state a leaf split leaves behind before
/// the parent has adopted the new sibling. Returns the parent, the
/// separator (the new leaf's min key), and the new leaf as an erased
/// child, ready to adopt at `partition == split_idx`.
fn adoption_scene(
    n: usize,
    split_idx: usize,
    live: &Arc<AtomicIsize>,
) -> (Inner<u64, Counted, M>, u64, Node<u64, Counted, M>) {
    assert!((2..=M).contains(&n), "an inner node holds 2..=M children");
    assert!(split_idx < n);

    let sep = 100 * split_idx as u64 + 50;

    // Build right-to-left so each leaf links to its chain successor.
    let mut next = None;
    let mut new_leaf = None;
    let mut ptrs: Vec<NonNull<Leaf<u64, Counted, M>>> = Vec::with_capacity(n);
    for i in (0..n).rev() {
        if i == split_idx {
            let ptr = scene_leaf(sep, live, next);
            new_leaf = Some(ptr);
            next = Some(ptr);
        }
        let ptr = scene_leaf(100 * i as u64, live, next);
        ptrs.push(ptr);
        next = Some(ptr);
    }
    ptrs.reverse();

    let keys: Vec<u64> = (1..n as u64).map(|i| 100 * i).collect();
    let children: Vec<Node<u64, Counted, M>> =
        ptrs.iter().map(|p| Node::from_leaf_ptr(*p)).collect();

    (Inner::test_from_parts(keys, children), sep, Node::from_leaf_ptr(new_leaf.unwrap()))
}

/// Every key the scene's `n + 1` leaves hold, for routing sweeps.
fn scene_keys(n: usize, split_idx: usize) -> Vec<u64> {
    (0..n as u64)
        .map(|i| 100 * i)
        .chain([100 * split_idx as u64 + 50])
        .flat_map(|base| (0..LMIN as u64).map(move |j| base + j))
        .collect()
}

/// The separators the parent must hold after adopting the scene's
/// split without splitting itself: the old separators plus the new
/// one, in sorted order.
fn adopted_keys(n: usize, split_idx: usize) -> Vec<u64> {
    let mut keys: Vec<u64> =
        (1..n as u64).map(|i| 100 * i).chain([100 * split_idx as u64 + 50]).collect();
    keys.sort_unstable();
    keys
}

/// Adopting a split into a node with room must: count the new child,
/// slot the separator into sorted key order, keep tree child order in
/// agreement with the leaf chain, leave every key routable, and drop
/// every value exactly once on teardown. Swept over every split
/// position, at a small occupancy and at the fullest occupancy that
/// still has room (`M - 1` children).
#[test]
fn insert_unchecked_adopts_a_split_at_every_position() {
    for n in [3, M - 1] {
        for split_idx in 0..n {
            let live = Arc::new(AtomicIsize::new(0));
            let (mut parent, sep, new_child) = adoption_scene(n, split_idx, &live);

            // SAFETY: the node has room (`n < M`); the new child's
            // slot `split_idx + 1` is in `1..=child_count` because
            // `split_idx == child_idx_for_key(&sep) < n` by the
            // scene's construction; and the pair is ordered — `sep`
            // is the new leaf's min key, strictly between its
            // neighbors' ranges.
            unsafe { parent.insert_child_unchecked(split_idx + 1, sep, new_child) };

            assert_eq!(
                parent.len(),
                n + 1,
                "adopting a split must grow the child count by exactly one \
                 (n={n}, split_idx={split_idx})"
            );
            assert_eq!(
                parent.test_keys(),
                &adopted_keys(n, split_idx)[..],
                "after adoption the keys must be the old separators plus the new one, \
                 in sorted order (n={n}, split_idx={split_idx})"
            );

            let node = Node::from_inner(parent, &mut Global);
            // SAFETY: height-1 subtree, judged as a root.
            unsafe { node.check_invariants(1, true) };

            for k in scene_keys(n, split_idx) {
                // SAFETY: height-1 subtree.
                let got = unsafe { node.get(1, &k) };
                assert!(
                    got.is_some_and(|v| v.0 == k),
                    "key {k} must route to its leaf after the adoption \
                     (n={n}, split_idx={split_idx})"
                );
            }

            // SAFETY: `node` owns the subtree.
            unsafe { node.drop_subtree(1, &mut Global) };
            assert_eq!(
                live.load(Relaxed),
                0,
                "teardown after an adoption must drop every value exactly once \
                 (positive = leak, negative = double-drop; n={n}, split_idx={split_idx})"
            );
        }
    }
}

/// `insert_child` on a node with room must adopt in place and report
/// no split — same contract as the direct `insert_unchecked` test,
/// through the dispatch.
#[test]
fn insert_child_with_room_adopts_in_place() {
    let live = Arc::new(AtomicIsize::new(0));
    let (mut parent, sep, new_child) = adoption_scene(3, 1, &live);

    let split = parent.insert_child(1, sep, new_child, &mut Global);
    assert!(split.is_none(), "a node with room must adopt without splitting");
    assert_eq!(parent.len(), 4, "adopting a split must grow the child count by exactly one");
    assert_eq!(
        parent.test_keys(),
        &adopted_keys(3, 1)[..],
        "after adoption the keys must be the old separators plus the new one, in order"
    );

    let node = Node::from_inner(parent, &mut Global);
    // SAFETY: height-1 subtree, judged as a root.
    unsafe { node.check_invariants(1, true) };
    for k in scene_keys(3, 1) {
        // SAFETY: height-1 subtree.
        let got = unsafe { node.get(1, &k) };
        assert!(got.is_some_and(|v| v.0 == k), "key {k} must route to its leaf");
    }

    // SAFETY: `node` owns the subtree.
    unsafe { node.drop_subtree(1, &mut Global) };
    assert_eq!(
        live.load(Relaxed),
        0,
        "teardown after an adoption must drop every value exactly once \
         (positive = leak, negative = double-drop)"
    );
}

/// `insert_child` on a FULL node must split it and hand back the
/// promoted separator and new right sibling for the caller to adopt
/// in turn — after which the reassembled two-level tree is fully
/// valid and every key still routes.
#[test]
fn insert_child_when_full_splits() {
    let live = Arc::new(AtomicIsize::new(0));
    let (mut parent, sep, new_child) = adoption_scene(M, M / 2, &live);

    let (promoted, right) = parent
        .insert_child(M / 2, sep, new_child, &mut Global)
        .expect("a full node must split to adopt");

    // Adopt the split the way the parent's parent (or the tree root)
    // would.
    let root = Node::from_inner(
        Inner::from_pair(
            promoted,
            Node::from_inner(parent, &mut Global),
            Node::from_inner_ptr(right),
        ),
        &mut Global,
    );
    // SAFETY: height-2 subtree, judged as a root.
    unsafe { root.check_invariants(2, true) };
    for k in scene_keys(M, M / 2) {
        // SAFETY: height-2 subtree.
        let got = unsafe { root.get(2, &k) };
        assert!(got.is_some_and(|v| v.0 == k), "key {k} must route to its leaf after the split");
    }

    // SAFETY: `root` owns the subtree.
    unsafe { root.drop_subtree(2, &mut Global) };
    assert_eq!(
        live.load(Relaxed),
        0,
        "teardown after a split must drop every value exactly once \
         (positive = leak, negative = double-drop)"
    );
}

/// The heart of the split contract, swept over every insertion point
/// `0..M`: the two halves hold all `M + 1` children between them,
/// near-balanced and both at or above `MIN_OCCUPANCY`; the left keys,
/// then the promoted separator, then the right keys are exactly the
/// old separators plus the new one, in order (every key lands in
/// exactly one of the three destinations); the reassembled tree is
/// fully valid with every key routable; and teardown drops every
/// value exactly once — so every child handle ended up owned by
/// exactly one node.
#[test]
fn splitting_insert_child_covers_every_insertion_point() {
    for split_idx in 0..M {
        let live = Arc::new(AtomicIsize::new(0));
        let (mut parent, sep, new_child) = adoption_scene(M, split_idx, &live);

        let (promoted, right_ptr) =
            parent.splitting_insert_child(split_idx, sep, new_child, &mut Global);

        {
            // SAFETY: the split hands back a live, exclusively-owned
            // right sibling; this borrow ends before the move below.
            let right = unsafe { right_ptr.as_ref() };

            assert_eq!(
                parent.len() + right.len(),
                M + 1,
                "the halves must hold all M + 1 children between them \
                 (inserting at {split_idx})"
            );
            assert!(
                parent.len().abs_diff(right.len()) <= 1,
                "the split must be near-balanced: left={}, right={} \
                 (inserting at {split_idx})",
                parent.len(),
                right.len()
            );
            assert!(
                !parent.is_deficient() && !right.is_deficient(),
                "both halves must satisfy MIN_OCCUPANCY: left={}, right={} \
                 (inserting at {split_idx})",
                parent.len(),
                right.len()
            );

            let mut combined: Vec<u64> = parent.test_keys().to_vec();
            combined.push(promoted);
            combined.extend_from_slice(right.test_keys());
            assert_eq!(
                combined,
                adopted_keys(M, split_idx),
                "left keys, then the promoted separator, then right keys must be \
                 exactly the old separators plus the new one, in order \
                 (inserting at {split_idx})"
            );
        }

        let root = Node::from_inner(
            Inner::from_pair(
                promoted,
                Node::from_inner(parent, &mut Global),
                Node::from_inner_ptr(right_ptr),
            ),
            &mut Global,
        );
        // SAFETY: height-2 subtree, judged as a root.
        unsafe { root.check_invariants(2, true) };
        for k in scene_keys(M, split_idx) {
            // SAFETY: height-2 subtree.
            let got = unsafe { root.get(2, &k) };
            assert!(
                got.is_some_and(|v| v.0 == k),
                "key {k} must route to its leaf after the split (inserting at {split_idx})"
            );
        }

        // SAFETY: `root` owns the subtree.
        unsafe { root.drop_subtree(2, &mut Global) };
        assert_eq!(
            live.load(Relaxed),
            0,
            "teardown after a split must drop every value exactly once — each child \
             handle must have ended up owned by exactly one node \
             (positive = leak, negative = double-drop; inserting at {split_idx})"
        );
    }
}

/// `from_pair` — the root-grow constructor — must produce a fully
/// valid two-child node: both children counted, the separator its
/// only key, every key routable to the correct side, teardown
/// exactly-once.
#[test]
fn from_pair_builds_a_valid_two_child_node() {
    let live = Arc::new(AtomicIsize::new(0));
    let right_leaf = scene_leaf(100, &live, None);
    let left_leaf = scene_leaf(0, &live, Some(right_leaf));

    let node = Node::from_inner(
        Inner::from_pair(100, Node::from_leaf_ptr(left_leaf), Node::from_leaf_ptr(right_leaf)),
        &mut Global,
    );

    // SAFETY: height-1 subtree, judged as a root.
    unsafe { node.check_invariants(1, true) };
    for base in [0u64, 100] {
        for j in 0..LMIN as u64 {
            let k = base + j;
            // SAFETY: height-1 subtree.
            let got = unsafe { node.get(1, &k) };
            assert!(
                got.is_some_and(|v| v.0 == k),
                "key {k} must route through the separator to its side"
            );
        }
    }

    // SAFETY: `node` owns the subtree.
    unsafe { node.drop_subtree(1, &mut Global) };
    assert_eq!(
        live.load(Relaxed),
        0,
        "teardown must drop both children's values exactly once \
         (positive = leak, negative = double-drop)"
    );
}

/// `child_idx_for_key` must route by the separator convention — a
/// separator IS its right child's minimum key. Probes below every
/// separator, at each separator exactly, inside the gaps, and above
/// everything. The equality probes are the load-bearing ones: a key
/// EQUAL to a separator lives under the separator's RIGHT child.
#[test]
fn child_idx_for_key_routes_hits_gaps_and_extremes() {
    let live = Arc::new(AtomicIsize::new(0));
    // Three children over the ranges 0.., 100.., and 200.. — so the
    // separators are 100 and 200, each its right child's min key.
    let right = scene_leaf(200, &live, None);
    let mid = scene_leaf(100, &live, Some(right));
    let left = scene_leaf(0, &live, Some(mid));
    let node = Inner::test_from_parts(
        vec![100, 200],
        vec![Node::from_leaf_ptr(left), Node::from_leaf_ptr(mid), Node::from_leaf_ptr(right)],
    );

    assert_eq!(node.child_idx_for_key(&0), 0, "a key below every separator routes to child 0");
    assert_eq!(node.child_idx_for_key(&99), 0, "a gap key routes to the child covering it");
    assert_eq!(
        node.child_idx_for_key(&100),
        1,
        "a key EQUAL to a separator must route to the separator's RIGHT child — \
         the separator is that child's minimum key"
    );
    assert_eq!(node.child_idx_for_key(&150), 1, "a gap key routes to the child covering it");
    assert_eq!(
        node.child_idx_for_key(&199),
        1,
        "a key just below a separator routes to the separator's LEFT child"
    );
    assert_eq!(
        node.child_idx_for_key(&200),
        2,
        "a key equal to the LAST separator must route to the last child"
    );
    assert_eq!(
        node.child_idx_for_key(&u64::MAX),
        2,
        "a key above every separator routes to the last child"
    );

    // SAFETY: `node` owns its three leaves (height 1).
    unsafe { Node::from_inner(node, &mut Global).drop_subtree(1, &mut Global) };
    assert_eq!(
        live.load(Relaxed),
        0,
        "teardown must drop every value exactly once \
         (positive = leak, negative = double-drop)"
    );
}
