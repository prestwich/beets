//! The [`NodeAllocator`] trait and the global heap ([`Global`]) as its
//! default implementation; the contract lives on the parent module.

use core::{convert::Infallible, mem::MaybeUninit, ptr::NonNull};

use crate::{Inner, Key, Leaf, allocator::Reservation};

/// The allocator cannot produce a slot: the fixed region is full, or the
/// backing heap refused to grow.
///
/// Deliberately unit — exhaustion is the only thing a slot allocator can
/// fail at, so the type carries no further diagnosis. Allocators that
/// never report failure (they abort instead) use [`Infallible`] rather
/// than this type; see [`NodeAllocator::Error`].
///
/// [`Infallible`]: core::convert::Infallible
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AllocError;

impl core::fmt::Display for AllocError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("node allocator exhausted")
    }
}

impl core::error::Error for AllocError {}

/// An allocator of node slots — [`Leaf`]s and [`Inner`]s — at stable
/// addresses.
///
/// The tree is generic over one of these. [`Slabs`] is the default and
/// packs each node kind into its own slab pool; [`Global`] boxes each
/// node as its own heap allocation; [`FixedNodes`] serves slots from a
/// caller-provided fixed region and is the one allocator whose
/// exhaustion is an honest [`Err`].
///
/// The trait is deliberately NOT generic over what it allocates: it has
/// exactly one consumer (the tree) and exactly two slot types, so each
/// concept appears as a leaf/inner method pair instead of a type
/// parameter. The primitive is UNINITIALIZED slot acquisition —
/// [`try_alloc_leaf_uninit`](Self::try_alloc_leaf_uninit) and kin — so a
/// caller can reserve slots fallibly BEFORE committing values to them
/// (`try_insert`'s reserve-then-commit discipline); the value-moving
/// methods are provided on top.
///
/// # `Exhaustion`
///
/// Fallibility is per-implementation. An allocator that handles
/// exhaustion itself — aborting via [`handle_alloc_error`], the [`Box`]
/// posture — declares `Exhaustion = Infallible`, and every `Result` in this
/// trait (and every tree code path plumbing it) loses its `Err` variant
/// at monomorphization: that is a layout guarantee, not an optimizer
/// favor. An allocator with a real out-of-slots answer declares
/// `Exhaustion = AllocError` and returns it.
///
/// # Contract (for implementors)
///
/// - A slot pointer returned by [`try_alloc_leaf_uninit`](Self::try_alloc_leaf_uninit)/
///   [`try_alloc_inner_uninit`](Self::try_alloc_inner_uninit) refers to
///   storage valid for a node of that kind, and remains valid — same
///   address, exclusively the caller's — until it is retired through the
///   matching `dealloc_*` method or the allocator is dropped. The
///   allocator never reads, writes, or moves a slot it has handed out;
///   initialization is entirely the caller's.
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
/// [`FixedNodes`]: crate::FixedNodes
/// [`handle_alloc_error`]: alloc::alloc::handle_alloc_error
pub trait NodeAllocator<K: Key, V, const M: usize> {
    /// The exhaustion error: [`AllocError`] for allocators that report
    /// running out, [`Infallible`](core::convert::Infallible) for
    /// allocators that abort instead (see the trait docs).
    type Exhaustion: core::error::Error;

    // ------------------------- required --------------------------

    /// Hand out one uninitialized leaf slot at a stable address, or
    /// report exhaustion. The slot is the caller's — the allocator never
    /// touches its contents — until retired via
    /// [`dealloc_leaf_uninit`](Self::dealloc_leaf_uninit) (never
    /// initialized, or value already moved out) or
    /// [`dealloc_leaf`](Self::dealloc_leaf) (initialized).
    fn try_alloc_leaf_uninit(
        &mut self,
    ) -> Result<NonNull<MaybeUninit<Leaf<K, V, M>>>, Self::Exhaustion>;

    /// As [`try_alloc_leaf_uninit`](Self::try_alloc_leaf_uninit), for an
    /// inner-node slot.
    fn try_alloc_inner_uninit(
        &mut self,
    ) -> Result<NonNull<MaybeUninit<Inner<K, V, M>>>, Self::Exhaustion>;

    /// Retire a leaf slot WITHOUT reading it — the return path for
    /// reserved-but-never-initialized slots, and the storage-reclaim
    /// half of [`dealloc_leaf`](Self::dealloc_leaf). The slot may be
    /// reused by a later allocation immediately.
    ///
    /// # Safety
    ///
    /// - `ptr` must have come from
    ///   [`try_alloc_leaf_uninit`](Self::try_alloc_leaf_uninit) on THIS
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

    /// The hard ceiling on simultaneously-live leaf slots, or `None` if
    /// there is none (growth bounded only by the backing heap).
    fn leaf_capacity(&self) -> Option<usize>;

    /// As [`leaf_capacity`](Self::leaf_capacity), for inner-node slots.
    fn inner_capacity(&self) -> Option<usize>;

    /// Leaf slots servable RIGHT NOW without acquiring new memory from
    /// the backing source: retired slots awaiting reuse plus never-used
    /// slots in already-acquired storage. [`Global`] answers 0 — every
    /// slot is a fresh heap allocation.
    fn leaf_available(&self) -> usize;

    /// As [`leaf_available`](Self::leaf_available), for inner-node slots.
    fn inner_available(&self) -> usize;

    /// True if we can allocate `count` leaves.
    fn leaves_available(&mut self, count: usize) -> bool {
        self.leaf_available() >= count || self.leaf_capacity().is_none()
    }

    /// True if we can allocate `count` inners
    fn inners_available(&mut self, count: usize) -> bool {
        self.inner_available() >= count || self.inner_capacity().is_none()
    }

    fn reserve(&mut self, inner_count: usize) -> Option<Reservation<K, V, M>> {
        if !self.inners_available(count) || !(self.leaf_available() > 0) {
            return None;
        }

        for i in inner_count {
            self.try_alloc_inner_uninit().unwrap()
        }

        Reservation { inner_count }
    }

    // ------------------------- provided --------------------------

    /// Move `leaf` into a fresh slot and return its (stable) address, or
    /// hand `leaf` back to the caller on exhaustion. Uninit acquisition
    /// ([`try_alloc_leaf_uninit`](Self::try_alloc_leaf_uninit)) plus the
    /// initializing write.
    fn try_alloc_leaf(
        &mut self,
        leaf: Leaf<K, V, M>,
    ) -> Result<NonNull<Leaf<K, V, M>>, Leaf<K, V, M>> {
        let Ok(slot) = self.try_alloc_leaf_uninit() else {
            return Err(leaf);
        };

        let slot = slot.cast();

        unsafe {
            slot.write(leaf);
        }
        Ok(slot)
    }

    /// As [`try_alloc_leaf`](Self::try_alloc_leaf), for an inner node.
    fn try_alloc_inner(
        &mut self,
        inner: Inner<K, V, M>,
    ) -> Result<NonNull<Inner<K, V, M>>, Inner<K, V, M>> {
        let Ok(slot) = self.try_alloc_inner_uninit() else {
            return Err(inner);
        };

        let slot = slot.cast();

        unsafe {
            slot.write(inner);
        }
        Ok(slot)
    }

    /// Infallible convenience over [`Self::try_alloc_leaf`], available
    /// only where the type system proves exhaustion impossible.
    #[track_caller]
    fn alloc_leaf(&mut self, leaf: Leaf<K, V, M>) -> NonNull<Leaf<K, V, M>>
    where
        Self: NodeAllocator<K, V, M, Exhaustion = Infallible>,
    {
        self.try_alloc_leaf(leaf).unwrap()
    }

    /// Infallible convenience over [`Self::try_alloc_leaf`].
    ///
    /// # Safety
    ///
    /// Caller must pre-flight allocation with [`Self::leaf_capacity`] and/or
    /// [`Self::leaf_available`].
    unsafe fn alloc_leaf_unchecked(&mut self, leaf: Leaf<K, V, M>) -> NonNull<Leaf<K, V, M>> {
        self.try_alloc_leaf(leaf).unwrap()
    }

    /// Infallible convenience over [`Self::try_alloc_inner`].
    ///
    /// # Panics
    ///
    /// On exhaustion. Unreachable — and compiled out — when
    /// [`Exhaustion`](Self::Exhaustion) is uninhabited.
    #[track_caller]
    fn alloc_inner(&mut self, inner: Inner<K, V, M>) -> NonNull<Inner<K, V, M>>
    where
        Self: NodeAllocator<K, V, M, Exhaustion = Infallible>,
    {
        self.try_alloc_inner(inner).unwrap()
    }

    /// Infallible convenience over [`Self::try_alloc_inner`].
    ///
    /// # Safety
    ///
    /// Caller must pre-flight allocation with [`Self::inner_capacity`] and/or
    /// [`Self::inner_available`].
    unsafe fn alloc_inner_unchecked(&mut self, inner: Inner<K, V, M>) -> NonNull<Inner<K, V, M>> {
        self.try_alloc_inner(inner).unwrap()
    }

    /// Move the value out of `ptr`'s slot and retire the slot. The
    /// returned [`Leaf`] is the caller's; the slot may be reused by a
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
    /// - The slot must hold an initialized [`Leaf`], and no other
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
            self.dealloc_leaf_uninit(ptr.cast());
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
/// [`Box`] posture), so `Error = Infallible` and the fallible surface
/// costs nothing.
#[cfg(feature = "alloc")]
impl<K: Key, V, const M: usize, G> NodeAllocator<K, V, M> for G
where
    G: core::alloc::GlobalAlloc,
{
    type Exhaustion = Infallible;

    // Design notes pinned during the redesign, for all four bodies:
    // - Exhaustion (a null from `GlobalAlloc::alloc`) aborts via
    //   `handle_alloc_error`, the `Box` posture — that abort is what
    //   `Error = Infallible` signs for.
    // - No ZST shortcut needed, unlike the old generic-over-T blanket:
    //   the only slot types are `Leaf`/`Inner`, never zero-sized
    //   (`M >= 3`), so `Layout::new::<..>()` is always non-empty.

    fn try_alloc_leaf_uninit(
        &mut self,
    ) -> Result<NonNull<MaybeUninit<Leaf<K, V, M>>>, Self::Exhaustion> {
        let layout = core::alloc::Layout::new::<MaybeUninit<Leaf<K, V, M>>>();
        // SAFETY:
        // Leaf is never a ZST.
        // NonNull::new_unchecked is checked by if ptr.is_null divergence.
        unsafe {
            let ptr = self.alloc(layout);
            if ptr.is_null() {
                ::alloc::alloc::handle_alloc_error(layout)
            }
            Ok(NonNull::new_unchecked(ptr).cast())
        }
    }

    fn try_alloc_inner_uninit(
        &mut self,
    ) -> Result<NonNull<MaybeUninit<Inner<K, V, M>>>, Self::Exhaustion> {
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
            Ok(NonNull::new_unchecked(ptr).cast())
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

    fn leaf_capacity(&self) -> Option<usize> {
        None
    }

    fn inner_capacity(&self) -> Option<usize> {
        None
    }

    fn leaf_available(&self) -> usize {
        0
    }

    fn inner_available(&self) -> usize {
        0
    }
}

/// Use the rust `Global` allocator.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Global;

// SAFETY: a pure passthrough — both methods forward verbatim to the
// registered global allocator, which upholds `GlobalAlloc`'s contract
// (live, layout-matched blocks; no unwinding).
#[cfg(feature = "alloc")]
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
