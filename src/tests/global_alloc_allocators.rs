//! Integration pin for the allocator facade's blanket contract: a
//! global allocator — any `GlobalAlloc` implementor — can be plugged in
//! as a tree's node allocator, no per-type glue required, and node
//! traffic actually flows through the supplied allocator's own
//! `alloc`/`dealloc`.

use crate::{common, common::counting};

use std::alloc::System;

use crate::{BPlusTree, NodeAllocator, Slabs};
use common::{Counting, M, fill, v};

/// Compile-time half of the pin: `System` (a plain `GlobalAlloc`) must
/// satisfy the tree's allocator bound for an arbitrary key/value pair.
fn assert_is_node_allocator<A: NodeAllocator<u64, u64, M>>() {}

/// A `GlobalAlloc` implementor satisfies `NodeAllocator` — the bound the
/// tree requires of its allocator parameter — without any hand-written
/// per-node-type glue.
#[test]
fn a_global_allocator_satisfies_the_node_allocator_bound() {
    assert_is_node_allocator::<System>();
}

/// A tree built over a `GlobalAlloc` allocator supports the full
/// mutation cycle — insert, probe, remove — exactly like the default
/// tree.
#[test]
fn a_tree_runs_on_a_global_allocator() {
    let mut tree: BPlusTree<u64, u64, M, System> = BPlusTree::new_in(System);

    fill(&mut tree, 1_000);
    assert_eq!(tree.len(), 1_000, "every inserted pair must be counted");
    assert_eq!(tree.get(&700), Some(&v(700)), "a stored value must be readable back");

    for k in 0..500u64 {
        assert_eq!(tree.remove(&k), Some(v(k)), "key {k} must remove exactly once");
    }
    assert_eq!(tree.len(), 500, "removals must be discounted");
}

/// The blanket is not just a capability marker: a tree built over a
/// custom `GlobalAlloc` sends every node allocation through THAT
/// allocator's `alloc`, and every node free through its `dealloc` —
/// balancing exactly by the time the tree is dropped.
#[test]
fn node_traffic_flows_through_the_supplied_allocator() {
    let counting = counting!();
    let mut tree: BPlusTree<u64, u64, M, Counting> = BPlusTree::new_in(counting);

    fill(&mut tree, 1_000);
    let after_fill = counting.allocs();
    assert!(
        after_fill > 1,
        "growing past one leaf must allocate nodes through the supplied allocator \
         (counted {after_fill} allocations)"
    );

    // Churn the low half back out: merges/borrows must free through the
    // same allocator.
    for k in 0..500u64 {
        tree.remove(&k);
    }
    drop(tree);

    counting.assert_balanced("node");
}

/// The slab arena's backing contract: slab memory is drawn from the
/// supplied `GlobalAlloc` in CHUNKS (far fewer backing allocations than
/// nodes), none of it is returned before the arena drops (the
/// high-water-mark behavior the module docs promise), and all of it is
/// returned — through the same backing — when the arena drops.
#[test]
fn slab_memory_comes_from_and_returns_to_the_supplied_backing() {
    const N: u64 = 2_000;

    let counting = counting!();
    let mut tree: BPlusTree<u64, u64, M, Slabs<u64, u64, M, Counting>> =
        BPlusTree::new_in(Slabs::new_in(counting));

    fill(&mut tree, N);
    let after_fill = counting.allocs();
    assert!(after_fill > 0, "slab memory must come from the supplied backing");
    assert!(
        after_fill < 20,
        "slabs are chunked: {N} pairs must need far fewer backing allocations \
         than nodes (counted {after_fill})"
    );

    // Drain the tree completely: retired slots are recycled in place —
    // no slab goes back to the backing before the arena drops.
    for k in 0..N {
        tree.remove(&k);
    }
    assert!(tree.is_empty(), "every pair must have been removed");
    assert_eq!(
        counting.frees(),
        0,
        "an emptied arena must hold its high-water mark — nothing returns \
         to the backing until the arena drops"
    );

    drop(tree);
    counting.assert_balanced("slab");
}
