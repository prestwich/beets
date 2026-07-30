//! Node allocation: the trait node storage goes through
//! ([`NodeAllocator`]), the slab arena ([`Slabs`]) as its default
//! implementation, and the global heap as the boxed-per-node
//! alternative.
//!
//! A [`NodeAllocator`] hands out node slots, one leaf or inner at a
//! time, at STABLE ADDRESSES: a slot never moves between its
//! allocation and its retirement. That stability is the whole contract
//! — every node pointer in the tree ([`Node`](crate::Node)'s union
//! fields, the leaf chain, recorded descents) relies on it, which is
//! why the trait deals in `NonNull` and not indices or lifetime-bound
//! references.
//!
//! The allocation PRIMITIVE is uninitialized ([`MaybeUninit`] slots):
//! callers acquire storage separately from initializing it.
//! Value-moving convenience methods are provided on top.
//!
//! Receivers are `&mut self`: allocation is a mutation, and exclusive
//! receivers let implementations keep their state in plain fields — no
//! interior mutability, nothing that would cost an allocator (or the
//! tree over it) `Sync`. The one caller that must SHARE an allocator —
//! the bulk loader, whose unwind guards hold it to free on panic while
//! the loader goes on allocating — reconciles that sharing with a
//! loader-local `RefCell`, paying for the borrow flag only there (see
//! `bulk.rs`).
//!
//! [`MaybeUninit`]: core::mem::MaybeUninit

mod slab;
pub use slab::Slabs;

mod r#trait;
pub use r#trait::{Global, NodeAllocator};

/// One slot's storage. Which field is live is positional state (see the
/// module invariants), mirroring the crate's untagged-[`Node`] discipline:
/// LIVE slots hold `value`, FREE slots hold `next_free`, VIRGIN slots
/// hold nothing.
///
/// [`Node`]: crate::Node
pub(crate) union Slot<T> {
    _value: core::mem::ManuallyDrop<T>,
    next_free: Option<core::ptr::NonNull<Slot<T>>>,
}
