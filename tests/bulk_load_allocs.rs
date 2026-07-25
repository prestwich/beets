//! Integration pin for the bulk loader's allocation contract: a bulk
//! load allocates exactly the tree's own nodes and frees nothing along
//! the way — no scaffolding boxes, no built-then-discarded nodes.
//!
//! Measured over the per-node [`Global`] backing, where one node is one
//! heap allocation, so the loader's discipline is legible in the raw
//! alloc count. (The default `Slabs` arena draws in slab chunks — its
//! backing contract is pinned in `global_alloc_allocators.rs`.)
//!
//! This lives in its own integration binary because it swaps in a
//! counting `#[global_allocator]`, and must stay this binary's only test
//! so no parallel test thread muddies the counters.

mod common;

use std::sync::atomic::AtomicUsize;

use beets::{BPlusTree, Global};
use common::{Counting, M};

static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static FREES: AtomicUsize = AtomicUsize::new(0);

#[global_allocator]
static COUNTING: Counting = Counting::new(&ALLOCS, &FREES);

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
    // exactly-full levels, and a three-level tree.
    for n in [0, 1, M - 1, M, M + 1, M * M, M * M + 1, M * M * M + 7] {
        let allocs_before = COUNTING.allocs();
        let frees_before = COUNTING.frees();
        let tree: BPlusTree<u64, u64, M, Global> =
            BPlusTree::from_sorted_iter_in((0..n as u64).map(|k| (k, k)), Global);
        let allocs = COUNTING.allocs() - allocs_before;
        let frees = COUNTING.frees() - frees_before;

        assert_eq!(tree.len(), n, "the load must keep every pair (n={n})");
        assert_eq!(
            allocs,
            expected_nodes(n),
            "a bulk load must allocate exactly the tree's nodes (n={n})"
        );
        assert_eq!(frees, 0, "a bulk load must not free anything it built (n={n})");
    }
}
