//! The [`SlotAllocator`] trait and the global heap ([`Global`]) as its
//! default implementation; the contract lives on the parent module.

use core::alloc::Layout;
use core::ptr::NonNull;

use crate::{Inner, Key, Leaf};

/// An allocator of single `T`-slots at stable addresses.
///
/// The tree is generic over one of these (two bounds, one per node
/// type, spelled once as the [`NodeAllocator`] alias). [`Global`] is
/// the default and boxes each node as its own heap allocation;
/// `slab.rs`'s [`Slabs`] is the packed alternative.
///
/// # Contract (for implementors)
///
/// - The pointer returned by [`allocate`](Self::allocate) refers to an
///   initialized `T` holding exactly the passed value, and remains
///   valid — same address, exclusively the caller's — until it is
///   passed to [`deallocate`](Self::deallocate) or the allocator is
///   dropped. Allocation is infallible: on heap exhaustion, abort via
///   [`alloc::alloc::handle_alloc_error`] (matching [`Box`]).
/// - Dropping the allocator must not read or drop any still-live `T` —
///   teardown order (values first, then allocator) is the caller's job,
///   and the tree's [`Drop`] upholds it. Whether dropping also reclaims
///   outstanding slots' MEMORY is [`OWNS_ALL`](Self::OWNS_ALL)'s call;
///   under an `OWNS_ALL = false` allocator, a slot never
///   [`deallocate`](Self::deallocate)d is leaked.
///
/// [`Slabs`]: crate::Slabs
pub trait SlotAllocator<T> {
    /// Whether this allocator owns every slot's memory WHOLESALE:
    /// `true` promises that dropping the allocator — or calling
    /// [`clear_all`](Self::clear_all) — reclaims all outstanding slot
    /// memory with no per-slot [`deallocate`](Self::deallocate) needed.
    /// `false` (the default, and the truth for the blanket
    /// [`GlobalAlloc`](alloc::alloc::GlobalAlloc) impl, where every
    /// slot is its own heap allocation) means slots not individually
    /// deallocated are leaked.
    ///
    /// An associated const so teardown code can branch on it in const
    /// context: each monomorphization keeps exactly one path and the
    /// untaken block is never codegenned.
    ///
    /// Implementations declaring `true` MUST override
    /// [`clear_all`](Self::clear_all).
    const OWNS_ALL: bool = false;

    /// Move `value` into a fresh slot and return its address.
    fn allocate(&mut self, value: T) -> NonNull<T>;

    /// Move the value out of `ptr`'s slot and retire the slot. The
    /// returned `T` is the caller's; the slot may be reused by a later
    /// [`allocate`](Self::allocate) immediately.
    ///
    /// (Returns the value — rather than expecting the caller to have
    /// moved it out — to mirror `*Box::from_raw`, so the `into_leaf`/
    /// `into_inner` accessors port 1:1. Callers that only want the
    /// memory back just [`drop`] the result.)
    ///
    /// # Safety
    ///
    /// - `ptr` must have come from [`allocate`](Self::allocate) on THIS allocator, not yet
    ///   deallocated (each slot retires exactly once).
    /// - The slot must hold an initialized `T`, and no other pointer to
    ///   it may be used after this call.
    unsafe fn deallocate(&mut self, ptr: NonNull<T>) -> T;

    /// Forget every outstanding slot at once, leaving the allocator
    /// empty and immediately reusable — the wholesale counterpart of
    /// retiring slots one [`deallocate`](Self::deallocate) at a time,
    /// for a caller abandoning its whole structure in one stroke.
    ///
    /// Reclaims MEMORY only, like the drop clause of the trait
    /// contract: values still resident in slots are forgotten, never
    /// read or dropped.
    ///
    /// (The `&mut self` receiver carries extra weight here: this call
    /// invalidates every outstanding slot pointer, so exclusivity is
    /// the point, not just the trait's uniform calling convention — no
    /// held borrow of the allocator can witness the reset.)
    ///
    /// # Safety
    ///
    /// - Callable only when [`OWNS_ALL`](Self::OWNS_ALL) is `true`;
    ///   the default body panics.
    /// - Every pointer previously returned by
    ///   [`allocate`](Self::allocate) is invalidated — the caller must
    ///   never use any of them again.
    /// - Still-resident values are forgotten: the caller must have
    ///   already dropped them, or know that forgetting them has no
    ///   observable effect (no drop glue that matters).
    unsafe fn clear_all(&mut self) {
        assert!(
            Self::OWNS_ALL,
            "attempted to call clear_all on an allocator that does not declare that it owns its memory"
        );
        unimplemented!("clear_all is only callable on allocators declaring OWNS_ALL")
    }
}

/// Alias for the tree's allocator bound: one `A` must serve slots for
/// both node types. Blanket-implemented, so it is never implemented by
/// hand — write `impl SlotAllocator<Leaf<..>> for X` and
/// `impl SlotAllocator<Inner<..>> for X`, and `X: NodeAllocator` follows.
pub trait NodeAllocator<K: Key, V, const M: usize>:
    SlotAllocator<Leaf<K, V, M>> + SlotAllocator<Inner<K, V, M>>
{
}

impl<K: Key, V, const M: usize, A> NodeAllocator<K, V, M> for A where
    A: SlotAllocator<Leaf<K, V, M>> + SlotAllocator<Inner<K, V, M>>
{
}

impl<T, G> SlotAllocator<T> for G
where
    G: alloc::alloc::GlobalAlloc,
{
    fn allocate(&mut self, value: T) -> NonNull<T> {
        if const { core::mem::size_of::<T>() == 0 } {
            // NB: this is not a memory leak, as T is a ZST
            core::mem::forget(value);
            return NonNull::dangling();
        }

        // SAFETY:
        // The layout is valid, the ptr is checked before non-null is
        // constructed, We shortcut earllier on ZSTs.
        unsafe {
            let layout = alloc::alloc::Layout::new::<T>();
            let ptr = self.alloc(layout);
            if ptr.is_null() {
                ::alloc::alloc::handle_alloc_error(layout)
            }

            let ptr = NonNull::new_unchecked(ptr).cast();
            ptr.write(value);
            ptr
        }
    }

    unsafe fn deallocate(&mut self, ptr: NonNull<T>) -> T {
        // SAFETY:
        // as trait fn docs. ptr is non-null, value must be valid.
        // caller must be 18 years or older, not located in Nebraska
        unsafe {
            // NB: if T is a ZST this will be a dangling pointer, but it's fine.
            let val = ptr.read();

            // SAFETY: ptr is non-null, layout is accurate, we check for ZST.
            if const { core::mem::size_of::<T>() != 0 } {
                self.dealloc(ptr.as_ptr().cast(), Layout::new::<T>());
            }

            val
        }
    }
}

/// Use the rust `Global` allocator.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Global;

// SAFETY: a pure passthrough — both methods forward verbatim to the
// registered global allocator, which upholds `GlobalAlloc`'s contract
// (live, layout-matched blocks; no unwinding).
unsafe impl ::alloc::alloc::GlobalAlloc for Global {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: passthrough. same requirements
        unsafe { ::alloc::alloc::alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: passthrough. same requirements
        unsafe {
            ::alloc::alloc::dealloc(ptr, layout);
        }
    }
}
