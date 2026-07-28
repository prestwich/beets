//! Unit tests for the treepath (`TreeProgress::push`) in isolation —
//! the chunk/hold-back/carry mechanics `from_sorted_iter` is built
//! on — plus behavioral pins for the whole load that the structural
//! tests in `tree.rs` don't cover: mutating a bulk-loaded tree, and
//! value ownership through the deepest carry cascade.
//!
//! Chunk-shape contract (the level-0 half lives in `leaf.rs`'s drain
//! tests): a chunk's up-key is its first child's min key, and its
//! separators are the later children's min keys, in stream order.

use alloc::{collections::BTreeMap, sync::Arc, vec::Vec};
use core::sync::atomic::{AtomicIsize, Ordering::Relaxed};

use super::*;
use crate::Global;
use crate::test_util::{Counted, M, counted_leaf};

impl<K: Key, V, const M: usize, A: SlotAllocator<Leaf<K, V, M>>> Unyielded<'_, K, V, M, A> {
    /// Test-only raw view of the held leaf, for chain-link assertions
    /// (the field itself stays private).
    pub(crate) fn as_ptr(&self) -> NonNull<Leaf<K, V, M>> {
        self.0
    }

    /// Test-only ownership transfer: move the held leaf out of its slot,
    /// defusing the guard so the caller is its sole owner.
    pub(crate) fn into_leaf(self) -> Leaf<K, V, M> {
        let this = core::mem::ManuallyDrop::new(self);
        // SAFETY: the guard held the only handle to the leaf, and
        // `ManuallyDrop` keeps it from re-retiring the slot `deallocate`
        // reclaims here.
        unsafe { this.1.borrow_mut().deallocate(this.0) }
    }
}

/// A `(min_key, node)` input pair: a one-pair leaf for key `k` —
/// enough structure to be owned, counted, and torn down. The treepath
/// never looks inside its children.
fn pair(k: u64, live: &Arc<AtomicIsize>) -> (u64, Node<u64, Counted, M>) {
    (k, Node::from_leaf_ptr(counted_leaf(k, 1, live, None)))
}

/// Feed `n` pairs (keys `0, 10, 20, ..`) into a fresh treepath, all at
/// level 0, drawing chunks from `alloc` (caller-owned so the
/// treepath's shared handle has somewhere to point).
fn path_of<'a>(
    n: usize,
    live: &Arc<AtomicIsize>,
    alloc: &'a RefCell<Global>,
) -> TreeProgress<'a, u64, Counted, M, Global> {
    let mut progress = TreeProgress::new(alloc);
    for i in 0..n as u64 {
        let (key, node) = pair(10 * i, live);
        progress.push(0, key, node);
    }
    progress
}

/// Assert a level-0 slot holds a chunk of `count` children keyed
/// `first, first + 10, ..` in stream order: its up-key must be the
/// first child's key and its separators the later children's keys.
#[track_caller]
fn check_chunk(slot: &Option<(u64, NonNull<Inner<u64, Counted, M>>)>, first: u64, count: usize) {
    let (up_key, ptr) = slot.as_ref().expect("the slot must hold a chunk");
    // SAFETY: the treepath holds the only handle to the chunk; the test
    // only reads through it.
    let inner = unsafe { ptr.as_ref() };
    assert_eq!(*up_key, first, "a chunk's up-key must be its first child's min key");
    assert_eq!(inner.len(), count, "chunk child count");
    for (i, sep) in inner.test_keys().iter().enumerate() {
        assert_eq!(
            *sep,
            first + 10 * (i as u64 + 1),
            "separators must be the later children's min keys, in stream order"
        );
    }
}

#[track_caller]
fn assert_untouched(lvl: &LevelState<u64, Counted, M>, msg: &str) {
    assert!(lvl.pending.is_none() && lvl.current.is_none(), "{msg}");
}

/// Pairs short of a full chunk collect in the level's filling slot,
/// in stream order — nothing is held back, nothing carries upward.
#[test]
fn push_fills_the_current_chunk_in_stream_order() {
    let live = Arc::new(AtomicIsize::new(0));
    let alloc = RefCell::new(Global);
    let progress = path_of(M - 1, &live, &alloc);

    check_chunk(&progress[0].current, 0, M - 1);
    assert!(progress[0].pending.is_none(), "an incomplete chunk must not be held back");
    assert_untouched(&progress[1], "an incomplete chunk must not carry upward");

    assert_eq!(live.load(Relaxed), (M - 1) as isize, "pushes must not drop anything");
    drop(progress);
    assert_eq!(
        live.load(Relaxed),
        0,
        "teardown must drop every value exactly once \
         (positive = leak, negative = double-drop)"
    );
}

/// The hold-back timing rule: a chunk that completes with no
/// successor in sight is parked as the level's complete node, not
/// emitted — the stream may end here, and the final short chunk may
/// still need to borrow from it.
#[test]
fn a_completed_chunk_is_held_back_not_emitted() {
    let live = Arc::new(AtomicIsize::new(0));
    let alloc = RefCell::new(Global);
    let progress = path_of(M, &live, &alloc);

    check_chunk(&progress[0].pending, 0, M);
    assert!(
        progress[0].current.is_none(),
        "the filling slot must restart empty after a chunk completes"
    );
    assert_untouched(&progress[1], "a chunk with no successor must not climb");

    assert_eq!(live.load(Relaxed), M as isize, "pushes must not drop anything");
    drop(progress);
    assert_eq!(live.load(Relaxed), 0, "teardown must drop every value exactly once");
}

/// A chunk completing behind a full predecessor displaces it: the
/// displaced chunk — and only it — climbs to the next level, the new
/// chunk takes its place as the complete node, and the filling slot
/// restarts fresh for the pairs that follow.
#[test]
fn a_full_successor_displaces_the_held_chunk_upward() {
    let live = Arc::new(AtomicIsize::new(0));
    let alloc = RefCell::new(Global);
    let n = 2 * M + 3;
    let progress = path_of(n, &live, &alloc);

    // Level 0: the second full chunk is now the held-back node, and
    // the three trailing pairs restarted the filling slot.
    check_chunk(&progress[0].pending, 10 * M as u64, M);
    check_chunk(&progress[0].current, 10 * (2 * M) as u64, 3);

    // Level 1: exactly the displaced first chunk arrived, keyed by
    // its up-key; nothing was held back and nothing climbed further.
    let (up_key, ptr) =
        progress[1].current.as_ref().expect("the displaced chunk must sit at level 1");
    assert_eq!(*up_key, 0, "the carry must keep the displaced chunk's up-key");
    // SAFETY: the treepath holds the only handle; the test only reads.
    assert_eq!(unsafe { ptr.as_ref() }.len(), 1, "only the displaced chunk may climb");
    assert!(progress[1].pending.is_none(), "one carry cannot complete a chunk");
    assert_untouched(&progress[2], "one carry must not cascade further");

    assert_eq!(live.load(Relaxed), n as isize, "pushes must not drop anything");
    drop(progress);
    assert_eq!(live.load(Relaxed), 0, "teardown must drop every value exactly once");
}

/// A bulk-loaded tree is a first-class tree: the ragged `M² + 1`
/// load (tail repairs at both levels) must take interleaved inserts
/// (splitting its packed nodes) and then a scattered full drain of
/// removes (rebalancing down to empty), agreeing with `BTreeMap` at
/// every step.
#[test]
fn bulk_loaded_trees_mutate_like_btreemap() {
    let n = (M * M + 1) as u64;
    let mut tree: BPlusTree<u64, u64, M> = BPlusTree::from_sorted_iter((0..n).map(|k| (10 * k, k)));
    let mut model: BTreeMap<u64, u64> = (0..n).map(|k| (10 * k, k)).collect();

    for k in 0..n {
        assert_eq!(
            tree.insert(10 * k + 5, k),
            model.insert(10 * k + 5, k),
            "insert into the loaded tree must agree with the model (k={k})"
        );
    }
    assert_eq!(tree.len(), model.len(), "len must track the interleaved inserts");
    for (k, v) in &model {
        assert_eq!(tree.get(k), Some(v), "key {k} must survive the insert phase");
    }

    // Drain in a scattered (stride-permuted) order, so removes hit
    // borrows and merges all over the loaded structure.
    let keys: Vec<u64> = model.keys().copied().collect();
    let count = keys.len();
    for i in 0..count {
        let k = keys[(i * 7919) % count];
        assert_eq!(
            tree.remove(&k),
            model.remove(&k),
            "remove from the loaded tree must agree with the model (key {k})"
        );
    }
    assert!(model.is_empty(), "the stride must be a permutation (fixture bug otherwise)");
    assert!(tree.is_empty(), "draining every key must empty the tree");
}

/// The deepest load path — a finish-time carry cascading into a
/// fourth level (`M³ + 1`) — owns every value end to end: the load
/// itself drops nothing, and teardown drops each exactly once.
#[test]
fn deep_cascade_loads_own_every_value_exactly_once() {
    let n = M * M * M + 1;
    let live = Arc::new(AtomicIsize::new(0));
    {
        let tree: BPlusTree<u64, Counted, M> =
            BPlusTree::from_sorted_iter((0..n as u64).map(|k| (k, Counted::new(k, &live))));
        assert_eq!(live.load(Relaxed), n as isize, "the load itself must not drop anything");
        assert_eq!(tree.len(), n, "len must count every drained pair");
    }
    assert_eq!(
        live.load(Relaxed),
        0,
        "teardown must drop every value exactly once \
         (positive = leak, negative = double-drop)"
    );
}
