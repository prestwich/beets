//! An in-memory B+tree.
//!
//! # Example
//!
//! ```
//! use beets::{BPlusTree, Key};
//!
//! // The fanout `M` is always `K::FANOUT` for the key type — a
//! // mismatch is a compile error where nodes are born.
//! let mut tree = BPlusTree::<u64, &str, { <u64 as Key>::FANOUT }>::new();
//!
//! tree.insert(2, "two");
//! tree.insert(1, "one");
//! tree.insert(3, "three");
//!
//! assert_eq!(tree.get(&2), Some(&"two"));
//! assert_eq!(tree.len(), 3);
//!
//! // Iteration is in ascending key order.
//! assert!(tree.keys().copied().eq(1..=3));
//!
//! assert_eq!(tree.remove(&2), Some("two"));
//! assert_eq!(tree.get(&2), None);
//! ```
//!
//! Sorted input bulk-loads densely ([`from_sorted_iter`](BPlusTree::from_sorted_iter)),
//! and the slab arena ([`Slabs`]) drops in as the allocator:
//!
//! ```
//! use beets::{BPlusTree, Key, Slabs};
//!
//! const M: usize = <u64 as Key>::FANOUT;
//! let tree: BPlusTree<u64, u64, M, Slabs<u64, u64, M>> =
//!     BPlusTree::from_sorted_iter_in((0..1000).map(|k| (k, k * 2)), Slabs::new());
//!
//! assert_eq!(tree.get(&700), Some(&1400));
//! ```

// TODO (crate-wide):
// - API ergonomics: callers must write `BPlusTree<K, V, { K::FANOUT }>` and
//   keep `M` in sync with `K` by hand. Explore hiding `M` (wrapper type,
//   macro, or restructuring so `M` isn't user-facing).
// - perf: tune NODE_BUDGET empirically (benches/vs_btreemap.rs is the
//   scoreboard); 512 is a guess. Sweep 256/512/1024/4096.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

mod allocator;
pub use allocator::{Global, NodeAllocator, Slabs};

mod key;
pub use key::Key;

mod tree;
pub use tree::BPlusTree;
pub(crate) use tree::{FullIterator, Inner, IntoIter, Leaf, Node, Range};

#[cfg(debug_assertions)]
pub(crate) use tree::NodeKind;

mod set;
pub use set::BPlusSet;

/// Test utils. Primarily a harness, and methods for asserting invariants about
/// tree structure.
#[cfg(any(test, feature = "testutils"))]
#[path = "tests/harness.rs"]
pub mod harness;

/// The target size of a node's key allocation, in bytes.
const NODE_BUDGET: usize = 512;

/// The heuristic number of keys that fit in a node.
const fn fanout(key_size: usize) -> usize {
    NODE_BUDGET / (key_size + size_of::<u64>())
}

/// The max height of a tree with fanout `m`.
pub const fn max_height(m: usize) -> usize {
    assert_fanout_floor(m);
    ((usize::BITS - 2) / m.div_ceil(2).ilog(2)) as usize
}

/// The maximum number of levels in a tree path. Used to specify the `H`
/// parameter.
pub const fn max_levels(m: usize) -> usize {
    assert_fanout_floor(m);
    max_height(m) + 1
}

/// Hard floor on the fanout: `M >= 3` makes `MIN_OCCUPANCY >= 2`, so a
/// deficient node (`MIN_OCCUPANCY - 1`) still has an entry, and a donor
/// strictly above minimum (`MIN_OCCUPANCY + 1 <= M`) can exist. The node
/// constructors evaluate this in a `const` block — every node is born
/// there, so a too-small `M` is a compile error at monomorphization.
pub(crate) const fn assert_fanout_floor(m: usize) {
    assert!(m >= MIN_FANOUT, "fanout M must be at least 3");
}

/// Trees MUST have a fanout capacity of at least 3.
pub(crate) const MIN_FANOUT: usize = 3;

/// One slot per possible tree level, for the fixed per-level scratch
/// arrays ([`insert`](BPlusTree::insert)/[`remove`](BPlusTree::remove)'s
/// descent paths, the bulk loader's `TreeProgress`). `usize::BITS` is
/// unreachable: the minimum fanout of 3 caps the
/// height of even a [`usize::MAX`]-pair tree well below it (see
/// [`BPlusTree::MAX_HEIGHT`]).
pub(crate) const DEFAULT_MAX_LEVELS: usize = usize::BITS as usize;

#[cfg(test)]
#[path = "tests/test_util.rs"]
pub(crate) mod test_util;

// The integration pins: cross-module contract tests driving the tree
// through its outermost surface, plus their shared fixtures.
#[cfg(test)]
#[path = "tests/common.rs"]
pub(crate) mod common;

#[cfg(test)]
#[path = "tests/auto_traits.rs"]
mod auto_traits;

#[cfg(test)]
#[path = "tests/bulk_load_allocs.rs"]
mod bulk_load_allocs;

#[cfg(test)]
#[path = "tests/bulk_load_panic_safety.rs"]
mod bulk_load_panic_safety;

#[cfg(test)]
#[path = "tests/global_alloc_allocators.rs"]
mod global_alloc_allocators;

#[cfg(test)]
#[path = "tests/wholesale_reclaim.rs"]
mod wholesale_reclaim;
