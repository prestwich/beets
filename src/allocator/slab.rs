//! The slab arena: [`SlotAllocator`] backed by chunked, stable-address slabs.
//!
//! # Design
//!
//! A [`SlabAlloc<T>`] owns a chain of SLABS — each one heap allocation
//! holding a small header and `slab_capacity` slots. Slabs are born when
//! allocation outruns capacity, are NEVER moved or shrunk, and are freed
//! only when the allocator drops: a slot's address is stable for its
//! whole allocate→deallocate life, which is the [`SlotAllocator`] contract
//! the tree's [`NonNull`]s ride on. (Consequence, accepted: an emptied
//! slab is not returned to the OS — memory holds its high-water mark
//! until the allocator drops.)
//!
//! Retired slots form an INTRUSIVE free list: a freed slot's own storage
//! holds the link to the next free slot (the [`Slot<T>`](Slot) union below), so
//! deallocation is pointer-only — no index derivation, no per-slot
//! metadata, no search for the owning slab. Allocation pops the free
//! list if it can, else bumps the newest slab's never-used tail, else
//! grows by one slab.
//!
//! The pools' bookkeeping (slab chain, free list, bump window) lives in
//! plain fields behind the trait's `&mut self` receivers — no interior
//! mutability anywhere. What still suppresses the auto traits is the
//! `NonNull`s that bookkeeping is made of, so thread affordances are
//! manual impls with contracts: [`Slabs`] is [`Send`] and [`Sync`] by
//! the impls below, and the tree's own impls ride on them.
//!
//! # Safety invariants
//!
//! - Every slot pointer handed out derives from its slab's single
//!   allocation; slab headers and their slot arrays live in that one
//!   allocation (header first, slots trailing — see
//!   [`slab_layout`](SlabAlloc::slab_layout)).
//!   No pointer is ever offset across slabs (`offset_from` between
//!   slabs is UB; nothing here needs it).
//! - A slot is in exactly one state: LIVE (holds a `T` the caller owns),
//!   FREE (holds a `next_free` link, on the free list), or VIRGIN (in
//!   the newest slab's untouched tail, no bytes initialized). State
//!   changes only at [`allocate`](SlotAllocator::allocate) (free/virgin → live) and
//!   [`deallocate`](SlotAllocator::deallocate) (live → free).
//! - Drop frees slab memory only. It must not read slots: any still-live
//!   `T` is the caller's teardown bug (the tree drops values via
//!   [`drop_subtree`](crate::Node::drop_subtree) before its allocator field drops).
//!
//! # Test contracts to pin (under `cargo test` AND `cargo miri test`)
//!
//! - allocate/deallocate round-trips the value; addresses stay stable
//!   and distinct across growth (fill several slabs, then revisit).
//! - a freed slot is reused before any virgin slot (free list first).
//! - teardown after mixed alloc/dealloc traffic frees every slab
//!   exactly once and touches no live `T` (Counted + miri leak check).
//! - `contains` accepts every live pointer and rejects foreign ones.
//!
//! # Provenance
//!
//! I really liked reading [`slabbin`] while writing this.
//!
//! [`slabbin`]: https://docs.rs/slabbin/latest/slabbin/

use alloc::alloc::{GlobalAlloc, Layout, handle_alloc_error};
use core::{marker::PhantomData, mem::ManuallyDrop, ptr::NonNull};

use crate::allocator::SlotAllocator;
use crate::{Global, Inner, Key, Leaf};

/// One slot's storage. Which field is live is positional state (see the
/// module invariants), mirroring the crate's untagged-[`Node`] discipline:
/// LIVE slots hold `value`, FREE slots hold `next_free`, VIRGIN slots
/// hold nothing.
///
/// [`Node`]: crate::Node
union Slot<T> {
    _value: ManuallyDrop<T>,
    next_free: Option<NonNull<Slot<T>>>,
}

/// A slab's header. The `slab_capacity` slots trail it in the SAME
/// allocation (layout computed by [`SlabAlloc::slab_layout`]); only the
/// header is a named field — the slots are reached by pointer math from
/// the header's end, never through a Rust array (capacity is a runtime
/// value).
pub(crate) struct SlabHeader<T> {
    /// Next slab in the teardown chain (newest first).
    next: Option<NonNull<Self>>,
    /// Rust does not permit generic types that only appear in recursive
    /// type construction. So we add the phantomdata here.
    marker: PhantomData<Slot<T>>,
}

/// A stable-address slab allocator for `T`-slots. See the module docs
/// for the design; see [`SlotAllocator`] for the contract it implements.
pub(crate) struct SlabAlloc<T, A: GlobalAlloc = Global> {
    /// Head of the slab chain (newest first) — the teardown walk.
    slabs: Option<NonNull<SlabHeader<T>>>,
    /// Head of the intrusive free list of retired slots.
    free_list: Option<NonNull<Slot<T>>>,
    /// The newest slab's virgin tail: next never-used slot, and the
    /// one-past-the-end fence. `bump_next == bump_end` means no virgin
    /// slots remain (also the empty-allocator state).
    bump_next: Option<NonNull<Slot<T>>>,
    bump_end: Option<NonNull<Slot<T>>>,
    /// Slots per slab, fixed at construction.
    slab_capacity: usize,

    /// Allocator.
    // DO NOT REORDER THIS FIELD.
    alloc: A,
}

impl<T, A: GlobalAlloc> SlabAlloc<T, A> {
    /// An empty allocator that will grow in slabs of `slab_capacity`
    /// slots. Allocates nothing until the first [`allocate`](SlotAllocator::allocate).
    ///
    /// Capacity guidance: size slabs to a byte budget (a few pages,
    /// e.g. 64 KiB) rather than a slot count — [`Slabs`] below does
    /// this for both node types.
    ///
    /// # Panics
    ///
    /// If `slab_capacity` is 0.
    pub const fn new_in(slab_capacity: usize, alloc: A) -> Self {
        assert!(slab_capacity > 0);

        Self { slabs: None, free_list: None, bump_next: None, bump_end: None, slab_capacity, alloc }
    }

    fn iter_slabs(&self) -> impl Iterator<Item = NonNull<SlabHeader<T>>> {
        // SAFETY:
        // Slab list always contains only valid slabs.
        core::iter::successors(self.slabs, |f| unsafe { f.as_ref() }.next)
    }

    /// Get a free slot. Caller is responsible for ensuring the slot does not
    /// get leaked.
    ///
    /// SAFETY:
    ///
    /// Caller must ensure that they are allowed to mutate the slab state.
    unsafe fn take_next_free(&mut self) -> Option<NonNull<T>> {
        let next = self.free_list?;

        // SAFETY:
        // free list always contains only valid free nodes. Never contains data.
        self.free_list = unsafe { next.as_ref().next_free };

        Some(next.cast())
    }

    /// Get a free bump slot. Caller is responsible for ensuring the slot does
    /// not get leaked.
    unsafe fn take_bump(&mut self) -> Option<NonNull<T>> {
        self.bump_next.map(|available| {
            // unwrap is okay, as we never have start without end.
            let end = self.bump_end.unwrap();
            // SAFETY: `bump_end` is one-past-the-end of the newest slab's
            // slot array and `slab_capacity >= 1`, so one step back stays
            // within the same allocation.
            let last = unsafe { end.sub(1) };
            if available == last {
                self.bump_end = None;
                self.bump_next = None;
            } else {
                // SAFETY: if statement above checks that add(1) is in the
                // Slab allocation
                self.bump_next = Some(unsafe { available.add(1) });
            }
            available.cast()
        })
    }

    /// Pop the free list; else take the next virgin slot; else
    /// [`grow()`](Self::grow) and take. Write `value` into the slot and hand out
    /// the (stable) pointer.
    unsafe fn take_next_slot(&mut self) -> NonNull<T> {
        // SAFETY: forwards this fn's own contract to the two sources.
        // The final unwrap holds: `grow` always installs a fresh bump
        // window of `slab_capacity >= 1` virgin slots.
        unsafe {
            self.take_next_free().or_else(|| self.take_bump()).unwrap_or_else(|| {
                self.grow();
                self.take_bump().unwrap()
            })
        }
    }

    /// Pre-pend a slot to the free list.
    unsafe fn return_slot(&mut self, mut slot: NonNull<Slot<T>>) {
        // The link write is unconditional: a slot returned to an EMPTY
        // list must hold `None`, not its moved-out value's leftover
        // bytes — `take_next_free` installs whatever sits here as the
        // new head when this slot is popped.
        // SAFETY: `slot` is a slot of this allocator whose value has
        // already been moved out (deallocate's contract), so the
        // storage is exclusively ours to repurpose as the list link.
        unsafe { slot.as_mut() }.next_free = self.free_list;
        self.free_list = Some(slot);
    }

    /// The layout of one slab allocation — header, then `slab_capacity`
    /// slots — and the byte offset from the slab base to slot 0.
    const fn slab_layout(&self) -> (core::alloc::Layout, usize) {
        let layout = Layout::new::<SlabHeader<T>>();

        let extend = match Layout::array::<Slot<T>>(self.slab_capacity) {
            Ok(layout) => layout,
            Err(_) => panic!(),
        };

        match layout.extend(extend) {
            Ok(layout) => layout,
            Err(_) => panic!(),
        }
    }

    /// The size of the array of slabs.
    const fn slot_array_size(&self) -> usize {
        self.slab_capacity * core::mem::size_of::<Slot<T>>()
    }

    /// Allocate and chain a fresh slab, resetting the bump window to its
    /// virgin slots. Called by [`allocate`](SlotAllocator::allocate) when both the free list and
    /// the bump window are empty. Aborts via [`handle_alloc_error`] on
    /// heap exhaustion (the infallibility half of the trait contract).
    fn grow(&mut self) {
        let (layout, slot0) = self.slab_layout();
        let last_slot = slot0 + self.slot_array_size();
        // SAFETY: the layout has nonzero size (a header plus at least one
        // slot), and every offset written below stays inside the fresh
        // slab allocation.
        unsafe {
            let new_slab = self.alloc.alloc(layout);
            if new_slab.is_null() {
                handle_alloc_error(layout)
            }

            // We just checked above that it is non-null
            let mut new_slab: NonNull<SlabHeader<T>> = NonNull::new_unchecked(new_slab.cast());

            // The pointer is to `u8` so these adds are byte offsets.
            let first_slot = new_slab.cast::<u8>().add(slot0).cast();
            let last_slot = new_slab.cast::<u8>().add(last_slot).cast();

            self.bump_next = Some(first_slot);
            self.bump_end = Some(last_slot);

            // Chain the new slab at the head of the teardown list. The
            // link write is unconditional: the very first slab must both
            // initialize its header's `next` (to `None`, the empty list)
            // and become the chain head, or teardown never sees it.
            new_slab.as_mut().next = self.slabs;
            self.slabs = Some(new_slab);
        }
    }
}

impl<T, A: GlobalAlloc> SlotAllocator<T> for SlabAlloc<T, A> {
    /// The pool draws slot memory in whole slabs and returns it in
    /// whole slabs: its `Drop` (and `clear_all`) reclaim everything
    /// without per-slot retirement.
    const OWNS_ALL: bool = true;

    fn allocate(&mut self, value: T) -> NonNull<T> {
        // SAFETY: nothing else mutates the slab state (`&mut self` receivers,
        // `!Sync`); the slot handed back is vacant, exclusively ours, and
        // at a stable address — writing `value` initializes it before the
        // pointer escapes.
        unsafe {
            let slot = self.take_next_slot();
            slot.write(value);
            slot
        }
    }

    unsafe fn deallocate(&mut self, ptr: NonNull<T>) -> T {
        // Read the value out, then overwrite the slot with the current
        // free-list head and make the slot the new head. Pointer-only:
        // the slot's own storage is the list node.
        // SAFETY: per the trait contract `ptr` is a live slot of this
        // allocator, retired exactly once — the read moves the value out,
        // and `return_slot` then owns the vacated storage.
        unsafe {
            let val = ptr.read();
            self.return_slot(ptr.cast());
            val
        }
    }

    unsafe fn clear_all(&mut self) {
        self.free_list = None;
        self.bump_next = None;
        self.bump_end = None;

        unsafe {
            // deallocate all slabs except the first.
            self.iter_slabs()
                .skip(1)
                .for_each(|slab| self.alloc.dealloc(slab.as_ptr().cast(), self.slab_layout().0));

            // If there is a first slab, then we set our bump
            self.slabs = self.slabs.map(|mut slab| {
                let (_, slot0) = self.slab_layout();
                let last_slot = slot0 + self.slot_array_size();

                slab.as_mut().next = None;

                // Cast to u8, apply offsets, set bumps.
                let first_slot = slab.cast::<u8>().add(slot0).cast();
                let last_slot = slab.cast::<u8>().add(last_slot).cast();

                self.bump_next = Some(first_slot);
                self.bump_end = Some(last_slot);

                slab
            })
        }
    }
}

impl<T, A: GlobalAlloc> Drop for SlabAlloc<T, A> {
    /// Walk the slab chain and free each slab allocation. Memory only:
    /// slots are NOT read, and still-live `T`s are NOT dropped — per the
    /// trait contract, value teardown happened before this (the tree's
    /// `drop_subtree` walk).
    fn drop(&mut self) {
        let layout = self.slab_layout().0;
        // SAFETY: every header in the chain is the base of one live slab
        // allocated in `grow` with this same layout; the chain visits
        // each exactly once, and drop ends all use of the allocator.
        self.iter_slabs().for_each(|slab| unsafe {
            self.alloc.dealloc(slab.as_ptr().cast(), layout);
        });
    }
}

/// The tree's arena: a leaf pool and an inner pool, so one value
/// satisfies both of `BPlusTree`'s allocator bounds
/// (the `NodeAllocator` alias). Leaves outnumber
/// inners by roughly the fanout, so the pools' slab capacities are sized
/// independently (both from a shared per-slab byte budget).
pub struct Slabs<K: Key, V, const M: usize, A: GlobalAlloc = Global> {
    leaves: SlabAlloc<Leaf<K, V, M>, A>,
    inners: SlabAlloc<Inner<K, V, M>, A>,
}

// SAFETY: a `Slabs` exclusively owns its two pools' slab memory, and
// every pointer in the pools' bookkeeping (slab chain, free list,
// bump window) targets that owned memory — nothing foreign, nothing
// shared, no interior mutability. What a move actually transfers is
// the slabs' contents — live slots holding `Leaf`/`Inner` values
// (payloads of `K`s and `V`s, plus intra-arena node pointers) and
// backing memory obtained from `A` — which is what the three `Send`
// bounds sign for. Outstanding slot pointers a caller still holds are
// already governed by [`SlotAllocator`]'s contract (the slot is
// exclusively that caller's, and every use is `unsafe`): keeping them
// coherent with a moved arena is that caller's obligation, discharged
// for the tree by `BPlusTree` moving arena and node graph as one
// value.
unsafe impl<K, V, const M: usize, A> Send for Slabs<K, V, M, A>
where
    K: Key + Send,
    V: Send,
    A: GlobalAlloc + Send,
{
}

// SAFETY: sharing `&Slabs` shares a read-only view of plain data —
// the arena has no interior mutability, and every mutation of the
// pools' bookkeeping sits behind the trait's `&mut self` receivers,
// so while shared borrows exist no thread can write anything a reader
// reaches. What a reader can (in principle) reach through the arena —
// resident `K`s and `V`s and the backing `A` — is what the three
// `Sync` bounds sign for.
unsafe impl<K, V, const M: usize, A> Sync for Slabs<K, V, M, A>
where
    K: Key + Sync,
    V: Sync,
    A: GlobalAlloc + Sync,
{
}

impl<K: Key, V, const M: usize, A: GlobalAlloc + Default + Clone> Default for Slabs<K, V, M, A> {
    fn default() -> Self {
        Self::new_in(Default::default())
    }
}

impl<K: Key, V, const M: usize> Slabs<K, V, M> {
    /// An empty arena: one pool per node type. Allocates nothing until
    /// the tree's first node; each pool then grows in slabs of
    /// `SLAB_BUDGET` bytes.
    pub fn new() -> Self {
        Self::new_in(Global)
    }
}

impl<K: Key, V, const M: usize, A: GlobalAlloc> Slabs<K, V, M, A> {
    /// Bytes per slab, for deriving each pool's slot capacity from its
    /// slot size. A few pages: big enough to amortize slab overhead,
    /// small enough that a near-empty tree isn't sitting on much.
    const SLAB_BUDGET: usize = 64 * 1024;

    /// An empty arena: one pool per node type. Allocates nothing until
    /// the tree's first node; each pool then grows in slabs of
    /// `SLAB_BUDGET` bytes.
    pub fn new_in(alloc: A) -> Self
    where
        A: Clone,
    {
        Self {
            leaves: SlabAlloc::new_in(
                Self::SLAB_BUDGET / core::mem::size_of::<Leaf<K, V, M>>(),
                alloc.clone(),
            ),
            inners: SlabAlloc::new_in(
                Self::SLAB_BUDGET / core::mem::size_of::<Inner<K, V, M>>(),
                alloc,
            ),
        }
    }
}

impl<K: Key, V, const M: usize, A: GlobalAlloc> SlotAllocator<Leaf<K, V, M>> for Slabs<K, V, M, A> {
    // Slab allocators own their slots wholesale
    const OWNS_ALL: bool = true;

    fn allocate(&mut self, value: Leaf<K, V, M>) -> NonNull<Leaf<K, V, M>> {
        self.leaves.allocate(value)
    }

    unsafe fn deallocate(&mut self, ptr: NonNull<Leaf<K, V, M>>) -> Leaf<K, V, M> {
        // SAFETY: forwarded — the caller's obligations are this pool's.
        unsafe { self.leaves.deallocate(ptr) }
    }

    unsafe fn clear_all(&mut self) {
        // SAFETY: forwarded — the caller's obligations are this pool's.
        unsafe { self.leaves.clear_all() }
    }
}

impl<K: Key, V, const M: usize, A: GlobalAlloc> SlotAllocator<Inner<K, V, M>>
    for Slabs<K, V, M, A>
{
    // Slab allocators own their slots wholesale
    const OWNS_ALL: bool = true;

    fn allocate(&mut self, value: Inner<K, V, M>) -> NonNull<Inner<K, V, M>> {
        self.inners.allocate(value)
    }

    unsafe fn deallocate(&mut self, ptr: NonNull<Inner<K, V, M>>) -> Inner<K, V, M> {
        // SAFETY: forwarded — the caller's obligations are this pool's.
        unsafe { self.inners.deallocate(ptr) }
    }

    unsafe fn clear_all(&mut self) {
        // SAFETY: forwarded — the caller's obligations are this pool's.
        unsafe { self.inners.clear_all() }
    }
}

#[cfg(test)]
#[path = "../tests/slab.rs"]
mod tests;
