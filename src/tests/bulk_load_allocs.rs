//! Integration pin for the bulk loader's allocation contract: a bulk
//! load allocates exactly the tree's own nodes and frees nothing along
//! the way — no scaffolding boxes, no built-then-discarded nodes.
//!
//! Measured over the per-node [`Global`] backing, where one node is one
//! heap allocation, so the loader's discipline is legible in the raw
//! alloc count. (The default `Slabs` arena draws in slab chunks — its
//! backing contract is pinned in `global_alloc_allocators.rs`.)
//!
//! This module registers the test binary's `#[global_allocator]`: a
//! counting pass-through that only tallies while the measuring thread
//! has switched it on, so the exact counts hold even with parallel test
//! threads allocating in the same process.

use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    sync::atomic::{AtomicUsize, Ordering::Relaxed},
};

use crate::common;

use crate::{BPlusTree, Global};
use common::M;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static FREES: AtomicUsize = AtomicUsize::new(0);

std::thread_local! {
    /// Whether THIS thread's traffic is being tallied. Off everywhere by
    /// default, switched on only inside the measured region, so other
    /// test threads sharing the process never muddy the counters.
    static TALLYING: Cell<bool> = const { Cell::new(false) };
}

/// Pass-through to [`System`] that counts this thread's traffic while
/// [`TALLYING`] is on.
struct ThreadTallied;

impl ThreadTallied {
    fn on(counter: &AtomicUsize) {
        // `try_with` because the allocator can be called during thread
        // teardown, after the TLS slot is gone — those aren't measured
        // traffic, so they fall through uncounted.
        if TALLYING.try_with(Cell::get).unwrap_or(false) {
            counter.fetch_add(1, Relaxed);
        }
    }
}

unsafe impl GlobalAlloc for ThreadTallied {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        Self::on(&ALLOCS);
        // SAFETY: forwards this method's own contract.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        Self::on(&FREES);
        // SAFETY: forwards this method's own contract; `ptr` came from
        // the matching `alloc` above (same pass-through).
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static COUNTING: ThreadTallied = ThreadTallied;

/// The nodes a bulk-loaded tree of `n` pairs is made of: `⌈n/M⌉` leaves
/// (the tail borrow moves pairs between nodes, never adds one), then one
/// inner per `M`-chunk of each level until a level is a single node —
/// the root. `n == 0` is the empty tree, a lone root leaf.
fn expected_nodes(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let mut nodes = n.div_ceil(M);
    let mut level = nodes;
    while level > 1 {
        level = level.div_ceil(M);
        nodes += level;
    }
    nodes
}

#[test]
fn bulk_load_allocates_exactly_the_trees_nodes() {
    // Sizes straddling every shape: empty, lone root leaf, ragged tails,
    // exactly-full levels, and a three-level tree. Under Miri the
    // three-level size is skipped: it costs minutes to interpret, and
    // the deep load path is Miri-covered by the bulk loader's own
    // tests; the alloc-count contract is scale-independent.
    for n in [0, 1, M - 1, M, M + 1, M * M, M * M + 1, M * M * M + 7] {
        if cfg!(miri) && n >= M * M * M {
            continue;
        }
        let allocs_before = ALLOCS.load(Relaxed);
        let frees_before = FREES.load(Relaxed);
        TALLYING.set(true);
        let tree: BPlusTree<u64, u64, M, Global> =
            BPlusTree::from_sorted_iter_in((0..n as u64).map(|k| (k, k)), Global);
        TALLYING.set(false);
        let allocs = ALLOCS.load(Relaxed) - allocs_before;
        let frees = FREES.load(Relaxed) - frees_before;

        assert_eq!(tree.len(), n, "the load must keep every pair (n={n})");
        assert_eq!(
            allocs,
            expected_nodes(n),
            "a bulk load must allocate exactly the tree's nodes (n={n})"
        );
        assert_eq!(frees, 0, "a bulk load must not free anything it built (n={n})");
    }
}
