//! Slot allocation: the trait node storage goes through, the global
//! heap as its default implementation, and the slab arena ([`Slabs`]).
//!
//! A [`SlotAllocator<T>`] hands out fixed-size slots, one `T` at a time, at
//! STABLE ADDRESSES: a slot never moves between its
//! [`allocate`](SlotAllocator::allocate) and its
//! [`deallocate`](SlotAllocator::deallocate). That stability is the
//! whole contract — every node
//! pointer in the tree ([`Node`](crate::Node)'s union fields, the leaf chain, recorded
//! descents) relies on it, which is why the trait deals in `NonNull<T>`
//! and not indices or lifetime-bound references.
//!
//! Receivers are `&mut self`: allocation is a mutation, and exclusive
//! receivers let implementations keep their state in plain fields — no
//! interior mutability, nothing that would cost an allocator (or the
//! tree over it) `Sync`. The one caller that must SHARE an allocator —
//! the bulk loader, whose unwind guards hold it to free on panic while
//! the loader goes on allocating — reconciles that sharing with a
//! loader-local `RefCell`, paying for the borrow flag only there (see
//! `bulk.rs`).

mod slab;
pub use slab::Slabs;

mod r#trait;
pub use r#trait::{Global, NodeAllocator, SlotAllocator};

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
