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

impl<T> SlabAlloc<T> {
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
    #[cfg(test)]
    pub(crate) const fn new(slab_capacity: usize) -> Self {
        Self::new_in(slab_capacity, Global)
    }
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

    /// The byte offset from the slab base to slot 0.
    #[cfg(test)]
    const fn slot_offset(&self) -> usize {
        self.slab_layout().1
    }

    #[inline(always)]
    #[cfg(test)]
    fn slab_contains(&self, slab: NonNull<SlabHeader<T>>, ptr: NonNull<T>) -> bool {
        let p = ptr.addr().get();
        let slot0 = self.slot_offset();
        let slab = slab.addr().get();

        let range = slab + slot0..slab + slot0 + self.slot_array_size();
        range.contains(&p)
    }

    /// Debug aid: whether `ptr` points into any of this allocator's
    /// slabs' slot ranges. Address-range checks only (raw address
    /// comparison across allocations is well-defined; this never offsets
    /// a pointer). Intended for `debug_assert!`s at the tree's free
    /// sites, not for correctness.
    #[cfg(test)]
    pub fn contains(&self, ptr: NonNull<T>) -> bool {
        self.iter_slabs().any(|slab| self.slab_contains(slab, ptr))
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
mod tests {
    //! Contract tests for the slab allocator, pinning the module header's
    //! four test contracts (round-trip + stability, free-before-virgin,
    //! teardown, `contains`) plus the construction contracts and the
    //! arena-backed tree integration. Every test here must pass under
    //! `cargo test` AND `cargo miri test` — several of the contracts
    //! (in-bounds slot writes, slabs freed exactly once, no reads of
    //! still-live values during drop) are only fully checked by miri.

    use alloc::boxed::Box;
    use alloc::sync::Arc;
    use alloc::vec::Vec;
    use core::sync::atomic::{AtomicIsize, Ordering::Relaxed};

    use super::*;
    use crate::BPlusTree;
    use crate::test_util::{Counted, M, shuffled, v};

    /// An allocator that never allocated must drop cleanly, freeing
    /// nothing.
    #[test]
    fn an_unused_allocator_drops_cleanly() {
        drop(SlabAlloc::<u64>::new(4));
    }

    /// Construction contract, per `new`'s docs: a zero-slot slab is
    /// refused with a panic.
    #[test]
    #[should_panic]
    fn a_zero_capacity_allocator_is_refused_at_construction() {
        let _ = SlabAlloc::<u64>::new(0);
    }

    /// `allocate` moves the value in; `deallocate` moves exactly that
    /// value back out.
    #[test]
    fn allocate_then_deallocate_round_trips_the_value() {
        let mut alloc = SlabAlloc::<u64>::new(4);
        let p = alloc.allocate(0xBEE7);
        assert!(alloc.contains(p), "a just-allocated pointer must test as contained");
        // SAFETY: `p` came from this allocator and is retired only here.
        let got = unsafe { alloc.deallocate(p) };
        assert_eq!(got, 0xBEE7, "deallocate must return exactly the value allocate moved in");
    }

    /// The stable-address contract across growth: fill several slabs,
    /// then revisit every slot — each must still sit at its original
    /// address (checked by reading its own value back through the
    /// original pointer) and no two slots may share an address.
    #[test]
    fn addresses_stay_stable_and_distinct_across_slab_growth() {
        const CAP: usize = 4;
        const N: u64 = 3 * CAP as u64 + 1;

        let mut alloc = SlabAlloc::<u64>::new(CAP);
        let ptrs: Vec<_> = (0..N).map(|k| alloc.allocate(v(k))).collect();

        for (i, a) in ptrs.iter().enumerate() {
            for b in &ptrs[i + 1..] {
                assert_ne!(a, b, "every live slot must have its own address");
            }
        }
        for (k, p) in ptrs.iter().enumerate() {
            // SAFETY: the slot is live and exclusively ours.
            let got = unsafe { *p.as_ref() };
            assert_eq!(got, v(k as u64), "slot {k} must still hold its value after growth");
            assert!(alloc.contains(*p), "live pointer {k} must test as contained");
        }

        for p in ptrs {
            // SAFETY: each pointer is live and retired exactly once.
            unsafe { alloc.deallocate(p) };
        }
    }

    /// Values smaller than a pointer must satisfy the same contracts —
    /// distinct stable addresses and full round-trip across several
    /// slabs of odd capacity. (Under miri this additionally checks that
    /// every slot write stays inside its slab's allocation.)
    #[test]
    fn small_values_round_trip_across_slab_growth() {
        const CAP: usize = 5;
        const N: u32 = 3 * CAP as u32 + 1;

        let mut alloc = SlabAlloc::<u32>::new(CAP);
        let ptrs: Vec<_> = (0..N).map(|k| alloc.allocate(k)).collect();

        for (i, a) in ptrs.iter().enumerate() {
            for b in &ptrs[i + 1..] {
                assert_ne!(a, b, "every live slot must have its own address");
            }
        }

        for (k, p) in ptrs.iter().enumerate() {
            // SAFETY: the slot is live and exclusively ours.
            let got = unsafe { *p.as_ref() };
            assert_eq!(got, k as u32, "slot {k} must still hold its value after growth");
        }
        for p in ptrs {
            // SAFETY: each pointer is live and retired exactly once.
            unsafe { alloc.deallocate(p) };
        }
    }

    /// The free list is consulted before the virgin tail: a freed slot's
    /// address comes back on the very next allocation, even though
    /// never-used slots remain.
    #[test]
    fn a_freed_slot_is_reused_before_any_virgin_slot() {
        let mut alloc = SlabAlloc::<u64>::new(8);
        let a = alloc.allocate(1);
        let b = alloc.allocate(2);

        // SAFETY: `b` is live and retired exactly once (reborn as `c`).
        unsafe { alloc.deallocate(b) };
        let c = alloc.allocate(3);
        assert_eq!(c, b, "a freed slot must be reused before any virgin slot");

        // SAFETY: both remaining slots are live and retired exactly once.
        unsafe {
            alloc.deallocate(a);
            alloc.deallocate(c);
        }
    }

    /// Freed slots come back newest-first: the free list is a stack, so
    /// two frees replay in reverse order.
    #[test]
    fn freed_slots_are_reused_most_recent_first() {
        let mut alloc = SlabAlloc::<u64>::new(8);
        let a = alloc.allocate(1);
        let b = alloc.allocate(2);

        // SAFETY: both are live and each is retired exactly once
        // (reborn below).
        unsafe {
            alloc.deallocate(a);
            alloc.deallocate(b);
        }
        assert_eq!(alloc.allocate(3), b, "the most recently freed slot must come back first");
        assert_eq!(alloc.allocate(4), a, "the earlier freed slot must come back second");

        // SAFETY: live, retired exactly once.
        unsafe {
            alloc.deallocate(a);
            alloc.deallocate(b);
        }
    }

    /// Draining the free list must end cleanly: after the last freed
    /// slot is popped, the next allocation falls through to the virgin
    /// tail. Exercises the pop-past-the-tail step the reuse tests above
    /// stop short of — the tail slot is the one that was freed into an
    /// EMPTY list, and popping it must leave a well-formed (empty) list
    /// behind, whatever bytes the slot's moved-out value left in its
    /// storage.
    #[test]
    fn a_drained_free_list_falls_through_to_virgin_slots() {
        let mut alloc = SlabAlloc::<u64>::new(8);
        let a = alloc.allocate(0xDEAD_BEEF);
        // SAFETY: `a` is live and retired exactly once (reborn as `b`).
        unsafe { alloc.deallocate(a) };

        // Pop the lone freed slot; the free list is now drained.
        let b = alloc.allocate(2);
        assert_eq!(b, a, "the freed slot must be reused before any virgin slot");

        // The step past the tail: this allocation must come from the
        // virgin bump window, at a fresh address.
        let c = alloc.allocate(3);
        assert_ne!(c, b, "both slots are live — they must not share an address");

        // SAFETY: live, each retired exactly once.
        unsafe {
            alloc.deallocate(b);
            alloc.deallocate(c);
        }
    }

    /// Teardown after mixed alloc/dealloc traffic: every value drops
    /// exactly once, and (under miri) every slab is freed exactly once
    /// with nothing leaked.
    #[test]
    fn mixed_traffic_teardown_drops_every_value_exactly_once() {
        const N: usize = 20;
        let live = Arc::new(AtomicIsize::new(0));
        let mut alloc = SlabAlloc::<Counted>::new(4);

        let mut ptrs = Vec::new();
        for k in 0..N {
            ptrs.push(alloc.allocate(Counted::new(k as u64, &live)));
        }
        // Punch holes, then refill some of them.
        for i in (0..N).step_by(2) {
            // SAFETY: live, retired exactly once.
            drop(unsafe { alloc.deallocate(ptrs[i]) });
        }
        for k in N..N + 5 {
            ptrs.push(alloc.allocate(Counted::new(k as u64, &live)));
        }

        // Values first, then the allocator — the teardown order the
        // trait contract demands.
        for i in (1..N).step_by(2).chain(N..N + 5) {
            // SAFETY: live, retired exactly once.
            drop(unsafe { alloc.deallocate(ptrs[i]) });
        }
        drop(alloc);

        assert_eq!(
            live.load(Relaxed),
            0,
            "every value must drop exactly once (positive = leak, negative = double-drop)"
        );
    }

    /// Dropping the allocator reclaims slot memory only: a value the
    /// caller never retired must NOT be dropped by the allocator.
    ///
    /// The probe here is deliberately heap-free (unlike `Counted`, whose
    /// `Arc` clone would itself leak): this test intentionally abandons
    /// the value — that leak is the caller's teardown bug by contract —
    /// and it must not trip miri's exit leak check while doing so.
    #[test]
    fn dropping_the_allocator_never_drops_still_live_values() {
        static DROPS: AtomicIsize = AtomicIsize::new(0);
        struct Probe(#[allow(dead_code)] u64);
        impl Drop for Probe {
            fn drop(&mut self) {
                DROPS.fetch_add(1, Relaxed);
            }
        }

        let mut alloc = SlabAlloc::<Probe>::new(4);
        let _p = alloc.allocate(Probe(7));

        drop(alloc);
        assert_eq!(
            DROPS.load(Relaxed),
            0,
            "drop reclaims slot memory only — an unretired value must not be dropped \
             (its leak is the caller's teardown bug, not the allocator's)"
        );
    }

    /// `contains` accepts every live pointer — across several slabs —
    /// and rejects pointers from other allocators and from the plain
    /// heap.
    #[test]
    fn contains_accepts_live_pointers_and_rejects_foreign_ones() {
        const CAP: usize = 4;
        let mut alloc = SlabAlloc::<u64>::new(CAP);
        let mut other = SlabAlloc::<u64>::new(CAP);

        let ptrs: Vec<_> = (0..3 * CAP as u64).map(|k| alloc.allocate(k)).collect();
        let foreign_slab = other.allocate(99);
        let foreign_heap = Box::new(7u64);

        for (i, p) in ptrs.iter().enumerate() {
            assert!(alloc.contains(*p), "live pointer {i} must be accepted");
        }
        assert!(!alloc.contains(foreign_slab), "another allocator's pointer must be rejected");
        assert!(
            !alloc.contains(NonNull::from(&*foreign_heap)),
            "a plain heap pointer must be rejected"
        );

        // SAFETY: each pointer is live, retired exactly once, on the
        // allocator it came from.
        unsafe {
            for p in ptrs {
                alloc.deallocate(p);
            }
            other.deallocate(foreign_slab);
        }
    }

    // ── the arena, end to end ───────────────────────────────────────────

    /// An arena-backed tree supports the full mutation cycle — insert,
    /// probe, remove, invariant check, drop — exactly like the
    /// heap-backed tree.
    #[test]
    fn an_arena_backed_tree_supports_the_full_mutation_cycle() {
        let mut tree: BPlusTree<u64, u64, M, Slabs<u64, u64, M>> = BPlusTree::new_in(Slabs::new());

        let keys = shuffled(512);
        for &k in &keys {
            tree.insert(k, v(k));
        }
        tree.check();
        for &k in &keys {
            assert_eq!(tree.get(&k), Some(&v(k)), "key {k} must be present after insert");
        }

        for &k in keys.iter().take(256) {
            assert_eq!(tree.remove(&k), Some(v(k)), "key {k} must remove exactly once");
        }
        tree.check();
        assert_eq!(tree.len(), 256);
    }

    /// An arena-backed tree bulk-loads through `from_sorted_iter_in` and
    /// satisfies the structural invariants.
    #[test]
    fn an_arena_backed_tree_bulk_loads() {
        const N: u64 = 1_000;
        let tree: BPlusTree<u64, u64, M, Slabs<u64, u64, M>> =
            BPlusTree::from_sorted_iter_in((0..N).map(|k| (k, v(k))), Slabs::new());

        tree.check();
        assert_eq!(tree.len(), N as usize);
        assert!(tree.iter().map(|(k, _)| *k).eq(0..N), "iteration must replay the loaded keys");
    }

    /// Arena-backed values drop exactly once through the whole tree
    /// lifecycle, including `clear` and the final drop.
    #[test]
    fn an_arena_backed_tree_owns_every_value_exactly_once() {
        let live = Arc::new(AtomicIsize::new(0));
        let mut tree: BPlusTree<u64, Counted, M, Slabs<u64, Counted, M>> =
            BPlusTree::new_in(Slabs::new());

        for k in shuffled(300) {
            tree.insert(k, Counted::new(k, &live));
        }
        assert_eq!(live.load(Relaxed), 300, "one live value per inserted key");

        for k in 0..150 {
            drop(tree.remove(&k));
        }
        assert_eq!(live.load(Relaxed), 150, "each removed value must drop exactly once");

        tree.clear();
        assert_eq!(live.load(Relaxed), 0, "clear must drop every remaining value exactly once");

        for k in shuffled(100) {
            tree.insert(k, Counted::new(k, &live));
        }
        drop(tree);
        assert_eq!(
            live.load(Relaxed),
            0,
            "the tree's drop must drop every value exactly once \
             (positive = leak, negative = double-drop)"
        );
    }

    /// `clear_all` forgets every outstanding slot at once: the pool is
    /// immediately reusable, and its eventual drop is clean. (Miri
    /// checks the other half of the contract — no slab leaked, none
    /// freed twice.)
    #[test]
    fn clear_all_resets_a_grown_pool_for_reuse() {
        const CAP: usize = 4;
        let mut alloc = SlabAlloc::<u64>::new(CAP);
        for k in 0..(3 * CAP as u64 + 1) {
            alloc.allocate(v(k));
        }

        // SAFETY: `SlabAlloc` declares `OWNS_ALL`; every outstanding
        // pointer is abandoned here, and forgetting `u64`s has no
        // observable effect.
        unsafe { alloc.clear_all() };

        let p = alloc.allocate(0xBEE7);
        assert!(alloc.contains(p), "a cleared pool must hand out contained slots again");
        // SAFETY: fresh from `allocate`, retired exactly once.
        let got = unsafe { alloc.deallocate(p) };
        assert_eq!(got, 0xBEE7, "a cleared pool must round-trip values like a new one");
    }

    /// `clear_all` supersedes all prior slot state: slots on the free
    /// list and slots still live at the call are indistinguishable
    /// afterward — fresh allocations are distinct, hold their values,
    /// and retire cleanly. (Under miri this additionally checks that no
    /// slot is handed out twice.)
    #[test]
    fn clear_all_supersedes_the_free_list() {
        let mut alloc = SlabAlloc::<u64>::new(4);
        let first: Vec<_> = (0..6u64).map(|k| alloc.allocate(k)).collect();
        // SAFETY: both slots are live and retired exactly once, here —
        // seeding the free list before the reset.
        unsafe {
            alloc.deallocate(first[1]);
            alloc.deallocate(first[4]);
        }

        // SAFETY: every remaining pointer is abandoned here, and
        // forgetting `u64`s has no observable effect.
        unsafe { alloc.clear_all() };

        let fresh: Vec<_> = (0..6u64).map(|k| alloc.allocate(100 + k)).collect();
        for (i, a) in fresh.iter().enumerate() {
            for b in &fresh[i + 1..] {
                assert_ne!(a, b, "every post-clear slot must have its own address");
            }
        }
        for (i, p) in fresh.iter().enumerate() {
            // SAFETY: the slot is live and exclusively ours.
            let got = unsafe { *p.as_ref() };
            assert_eq!(got, 100 + i as u64, "post-clear slot {i} must hold its fresh value");
        }
        for p in fresh {
            // SAFETY: each pointer is live and retired exactly once.
            unsafe { alloc.deallocate(p) };
        }
    }

    /// `clear_all` reclaims memory ONLY: values still resident in slots
    /// are forgotten, never read or dropped — value teardown is the
    /// caller's job, before the call. Neither the reset nor the pool's
    /// own drop may run a resident value's drop glue.
    #[test]
    fn clear_all_never_drops_resident_values() {
        use core::sync::atomic::AtomicUsize;
        static DROPS: AtomicUsize = AtomicUsize::new(0);
        struct Loud {
            _x: u64,
        }
        impl Drop for Loud {
            fn drop(&mut self) {
                DROPS.fetch_add(1, Relaxed);
            }
        }

        let mut alloc = SlabAlloc::<Loud>::new(4);
        for k in 0..10 {
            alloc.allocate(Loud { _x: k });
        }

        // SAFETY: every outstanding pointer is abandoned here, and
        // forgetting a `Loud` is the very behavior under test — it owns
        // nothing beyond its drop-side effect.
        unsafe { alloc.clear_all() };
        assert_eq!(DROPS.load(Relaxed), 0, "clear_all must not drop resident values");

        drop(alloc);
        assert_eq!(DROPS.load(Relaxed), 0, "the pool's drop must not drop them either");
    }

    /// `clear_all` on a never-used pool is a harmless no-op: nothing to
    /// forget, and the pool allocates normally afterward.
    #[test]
    fn clear_all_on_a_virgin_pool_is_harmless() {
        let mut alloc = SlabAlloc::<u64>::new(4);
        // SAFETY: no outstanding slots exist to invalidate.
        unsafe { alloc.clear_all() };

        let p = alloc.allocate(7);
        // SAFETY: fresh from `allocate`, retired exactly once.
        let got = unsafe { alloc.deallocate(p) };
        assert_eq!(got, 7, "a cleared virgin pool must allocate like a new one");
    }
}
