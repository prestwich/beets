//! The [`NodeAllocator`] trait and the global heap ([`Global`]) as its
//! default implementation; the contract lives on the parent module.

use core::{mem::MaybeUninit, ptr::NonNull};

use crate::{Inner, Key, Leaf};

/// An allocator of node slots — `Leaf`s and `Inner`s — at stable
/// addresses.
///
/// The tree is generic over one of these. [`Slabs`] is the default and
/// packs each node kind into its own slab pool; [`Global`] boxes each
/// node as its own heap allocation.
///
/// The trait is deliberately NOT generic over what it allocates: it has
/// exactly one consumer (the tree) and exactly two slot types, so each
/// concept appears as a leaf/inner method pair instead of a type
/// parameter. The primitive is UNINITIALIZED slot acquisition —
/// [`alloc_leaf_uninit`](Self::alloc_leaf_uninit) and kin — separating
/// storage acquisition from initialization; the value-moving methods
/// are provided on top.
///
/// # Contract (for implementors)
///
/// - A slot pointer returned by [`alloc_leaf_uninit`](Self::alloc_leaf_uninit)/
///   [`alloc_inner_uninit`](Self::alloc_inner_uninit) refers to
///   storage valid for a node of that kind, and remains valid — same
///   address, exclusively the caller's — until it is retired through the
///   matching `dealloc_*` method or the allocator is dropped. The
///   allocator never reads, writes, or moves a slot it has handed out;
///   initialization is entirely the caller's.
/// - Allocation is infallible: on exhaustion, abort via
///   [`handle_alloc_error`] (matching [`Box`]).
/// - The `dealloc_*_uninit` methods retire STORAGE only: they must not
///   read the slot's contents (it may never have been initialized).
/// - Dropping the allocator must not read or drop any still-live node —
///   teardown order (values first, then allocator) is the caller's job,
///   and the tree's [`Drop`] upholds it. Whether dropping also reclaims
///   outstanding slots' MEMORY is [`reclaim_all`](Self::reclaim_all)'s
///   story: under an allocator whose `reclaim_all` returns `false`, a
///   slot never deallocated is leaked.
///
/// [`Slabs`]: crate::Slabs
/// [`handle_alloc_error`]: alloc::alloc::handle_alloc_error
pub trait NodeAllocator<K: Key, V, const M: usize> {
    // ------------------------- required --------------------------

    /// Hand out one uninitialized leaf slot at a stable address. The
    /// slot is the caller's — the allocator never touches its contents
    /// — until retired via
    /// [`dealloc_leaf_uninit`](Self::dealloc_leaf_uninit) (never
    /// initialized, or value already moved out) or
    /// [`dealloc_leaf`](Self::dealloc_leaf) (initialized).
    fn alloc_leaf_uninit(&mut self) -> NonNull<MaybeUninit<Leaf<K, V, M>>>;

    /// As [`alloc_leaf_uninit`](Self::alloc_leaf_uninit), for an
    /// inner-node slot.
    fn alloc_inner_uninit(&mut self) -> NonNull<MaybeUninit<Inner<K, V, M>>>;

    /// Retire a leaf slot WITHOUT reading it — the return path for
    /// acquired-but-never-initialized slots, and the storage-reclaim
    /// half of [`dealloc_leaf`](Self::dealloc_leaf). The slot may be
    /// reused by a later allocation immediately.
    ///
    /// # Safety
    ///
    /// - `ptr` must have come from
    ///   [`alloc_leaf_uninit`](Self::alloc_leaf_uninit) on THIS
    ///   allocator, not yet retired (each slot retires exactly once).
    /// - If the slot was initialized, the value must already have been
    ///   moved out or dropped — this method reclaims storage only.
    /// - No pointer to the slot may be used after this call.
    unsafe fn dealloc_leaf_uninit(&mut self, ptr: NonNull<MaybeUninit<Leaf<K, V, M>>>);

    /// As [`dealloc_leaf_uninit`](Self::dealloc_leaf_uninit), for an
    /// inner-node slot.
    ///
    /// # Safety
    ///
    /// As [`dealloc_leaf_uninit`](Self::dealloc_leaf_uninit).
    unsafe fn dealloc_inner_uninit(&mut self, ptr: NonNull<MaybeUninit<Inner<K, V, M>>>);

    // ------------------------- provided --------------------------

    /// Move `leaf` into a fresh slot and return its (stable) address.
    /// Uninit acquisition ([`alloc_leaf_uninit`](Self::alloc_leaf_uninit))
    /// plus the initializing write.
    fn alloc_leaf(&mut self, leaf: Leaf<K, V, M>) -> NonNull<Leaf<K, V, M>> {
        let slot = self.alloc_leaf_uninit().cast();

        unsafe {
            slot.write(leaf);
        }
        slot
    }

    /// As [`alloc_leaf`](Self::alloc_leaf), for an inner node.
    fn alloc_inner(&mut self, inner: Inner<K, V, M>) -> NonNull<Inner<K, V, M>> {
        let slot = self.alloc_inner_uninit().cast();

        unsafe {
            slot.write(inner);
        }
        slot
    }

    /// Move the value out of `ptr`'s slot and retire the slot. The
    /// returned `Leaf` is the caller's; the slot may be reused by a
    /// later allocation immediately.
    ///
    /// (Returns the value — rather than expecting the caller to have
    /// moved it out — to mirror `*Box::from_raw`, so the `into_leaf`/
    /// `into_inner` accessors port 1:1. Callers that only want the
    /// memory back just [`drop`] the result.)
    ///
    /// # Safety
    ///
    /// - `ptr` must have come from a leaf allocation on THIS allocator,
    ///   not yet retired.
    /// - The slot must hold an initialized `Leaf`, and no other
    ///   pointer to it may be used after this call.
    unsafe fn dealloc_leaf(&mut self, ptr: NonNull<Leaf<K, V, M>>) -> Leaf<K, V, M> {
        // SAFETY:
        // as trait function contract.
        // - `ptr` must have come from a leaf allocation on THIS allocator,
        //   not yet retired.
        // - The slot must hold an initialized [`Leaf`], and no other
        //   pointer to it may be used after this call.
        unsafe {
            let val = ptr.read();
            self.dealloc_leaf_uninit(ptr.cast());
            val
        }
    }

    /// As [`dealloc_leaf`](Self::dealloc_leaf), for an inner node.
    ///
    /// # Safety
    ///
    /// As [`dealloc_leaf`](Self::dealloc_leaf).
    unsafe fn dealloc_inner(&mut self, ptr: NonNull<Inner<K, V, M>>) -> Inner<K, V, M> {
        // SAFETY:
        // as trait function contract.
        // - `ptr` must have come from a inner allocation on THIS allocator,
        //   not yet retired.
        // - The slot must hold an initialized [`Inner`], and no other
        //   pointer to it may be used after this call.
        unsafe {
            let val = ptr.read();
            self.dealloc_inner_uninit(ptr.cast());
            val
        }
    }

    /// Wholesale reclaim: forget every outstanding slot of BOTH node
    /// kinds at once, leaving the allocator empty and immediately
    /// reusable — or do nothing at all. All-or-nothing: `true` means
    /// every slot's memory was reclaimed with no per-slot retirement;
    /// `false` (the default, and the truth for [`Global`], where every
    /// slot is its own heap allocation) means nothing happened and the
    /// caller must retire slots individually or accept the leak.
    ///
    /// Reclaims MEMORY only: values still resident in slots are
    /// forgotten, never read or dropped.
    ///
    /// (The `&mut self` receiver carries extra weight here: a `true`
    /// return invalidates every outstanding slot pointer, so exclusivity
    /// is the point, not just the trait's uniform calling convention —
    /// no held borrow of the allocator can witness the reset.)
    ///
    /// # Safety
    ///
    /// On a `true` return:
    ///
    /// - every pointer previously handed out by this allocator is
    ///   invalidated — the caller must never use any of them again.
    /// - still-resident values are forgotten: the caller must have
    ///   already dropped them, or know that forgetting them has no
    ///   observable effect (no drop glue that matters).
    ///
    /// On a `false` return nothing has changed and no obligation arises.
    unsafe fn reclaim_all(&mut self) -> bool {
        false
    }
}

/// Every [`GlobalAlloc`](core::alloc::GlobalAlloc) is a node allocator:
/// each slot is its own heap allocation, acquired and released
/// per-node. Exhaustion is handled by aborting
/// ([`handle_alloc_error`](alloc::alloc::handle_alloc_error), the
/// [`Box`] posture).
impl<K: Key, V, const M: usize, G> NodeAllocator<K, V, M> for G
where
    G: core::alloc::GlobalAlloc,
{
    // Design notes pinned during the redesign, for all four bodies:
    // - Exhaustion (a null from `GlobalAlloc::alloc`) aborts via
    //   `handle_alloc_error`, the `Box` posture — the trait contract's
    //   infallibility clause.
    // - No ZST shortcut needed, unlike the old generic-over-T blanket:
    //   the only slot types are `Leaf`/`Inner`, never zero-sized
    //   (`M >= 3`), so `Layout::new::<..>()` is always non-empty.

    fn alloc_leaf_uninit(&mut self) -> NonNull<MaybeUninit<Leaf<K, V, M>>> {
        let layout = core::alloc::Layout::new::<MaybeUninit<Leaf<K, V, M>>>();
        // SAFETY:
        // Leaf is never a ZST.
        // NonNull::new_unchecked is checked by if ptr.is_null divergence.
        unsafe {
            let ptr = self.alloc(layout);
            if ptr.is_null() {
                ::alloc::alloc::handle_alloc_error(layout)
            }
            NonNull::new_unchecked(ptr).cast()
        }
    }

    fn alloc_inner_uninit(&mut self) -> NonNull<MaybeUninit<Inner<K, V, M>>> {
        let layout = core::alloc::Layout::new::<MaybeUninit<Inner<K, V, M>>>();
        // SAFETY:
        // Inner is never a ZST.
        // NonNull::new_unchecked is checked by if ptr.is_null divergence.
        // Cast is safe, as MaybeUninit has the same
        unsafe {
            let ptr = self.alloc(layout);
            if ptr.is_null() {
                ::alloc::alloc::handle_alloc_error(layout)
            }
            NonNull::new_unchecked(ptr).cast()
        }
    }

    unsafe fn dealloc_leaf_uninit(&mut self, ptr: NonNull<MaybeUninit<Leaf<K, V, M>>>) {
        let layout = core::alloc::Layout::new::<Leaf<K, V, M>>();
        // Safety:
        // as in trait safety contract. Caller must ensure that ptr was
        // allocated by this allocator.
        unsafe {
            self.dealloc(ptr.as_ptr().cast(), layout);
        }
    }

    unsafe fn dealloc_inner_uninit(&mut self, ptr: NonNull<MaybeUninit<Inner<K, V, M>>>) {
        let layout = core::alloc::Layout::new::<Inner<K, V, M>>();
        // Safety:
        // as in trait safety contract. Caller must ensure that ptr was
        // allocated by this allocator.
        unsafe {
            self.dealloc(ptr.as_ptr().cast(), layout);
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
    unsafe fn alloc(&self, layout: core::alloc::Layout) -> *mut u8 {
        // SAFETY: passthrough. same requirements
        unsafe { ::alloc::alloc::alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: core::alloc::Layout) {
        // SAFETY: passthrough. same requirements
        unsafe {
            ::alloc::alloc::dealloc(ptr, layout);
        }
    }
}
