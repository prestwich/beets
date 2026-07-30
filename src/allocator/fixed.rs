//! The fixed-region arena: [`NodeAllocator`] over caller-provided
//! storage, for environments where allocation is impossible or
//! unwelcome. Exhaustion here is an honest [`Err`]: this is the
//! allocator the tree's `try_` surface exists for.
//!
//! # Design
//!
//! [`NodeStorage`] is the memory: a const-constructible, fully
//! UNINITIALIZED block holding `LEAVES` leaf slots and `INNERS` inner
//! slots. The caller declares it where the memory should live — on
//! embedded targets, a `static`, where being uninitialized means it
//! lands in `.bss` (no flash image, no init cost).
//!
//! [`FixedNodes`] is the allocator: it exclusively borrows a
//! [`NodeStorage`] for `'a` and serves slots from it with the same
//! discipline as the slab pools — an intrusive free list of retired
//! slots plus a bump cursor over the never-used tail — minus growth.
//! Where [`Slabs`](crate::Slabs) would allocate a new slab,
//! [`FixedNodes`] returns [`AllocError`].
//!
//! # The embedded pattern
//!
//! Getting an exclusive `&'static mut` borrow of a `static` is the
//! caller's side of the deal. The ecosystem idiom is
//! [`static_cell::ConstStaticCell`], which is const-evaluated (the
//! storage never transits the stack — the classic fixed-collection
//! init hazard) and hands out the borrow exactly once:
//!
//! ```text
//! static STORAGE: ConstStaticCell<NodeStorage<u64, Reading, M, 1024, 128>> =
//!     ConstStaticCell::new(NodeStorage::new());
//!
//! let tree = BPlusTree::try_new_in(FixedNodes::new(STORAGE.take()))
//!     .unwrap_or_else(|_| unreachable!("fresh storage serves the root leaf"));
//! ```
//!
//! Users with linker-scripted regions (CCM/TCM/DMA RAM) conjure the
//! `&'static mut NodeStorage` themselves from the section symbols;
//! everything after the borrow is identical.
//!
//! # Safety invariants
//!
//! - Every slot pointer handed out derives from the borrowed storage;
//!   the exclusive borrow makes the region the pool's alone for `'a`.
//! - A slot is in exactly one state: LIVE (holds a node the caller
//!   owns), FREE (holds a `next_free` link, on the free list), or
//!   NEVER-USED (in the never-used tail past the bump cursor, no bytes
//!   initialized).
//! - The pools never read a LIVE slot; teardown order (values first,
//!   then allocator) is the tree's job, exactly as for the slab arena.
//! - Dropping a [`FixedNodes`] releases only the borrow. The storage
//!   outlives it as "uninitialized" memory: any values not torn down
//!   were forgotten, never to be read through it again.

use core::{mem::MaybeUninit, ptr::NonNull};

use crate::{
    Inner, Key, Leaf,
    allocator::{AllocError, NodeAllocator, Slot},
};

/// One fixed slot's storage: a [`Slot`] (LIVE value / FREE link union)
/// that additionally remembers it may be never-used — never written at all —
/// which is what lets [`NodeStorage`] be const-constructed with no
/// initialization whatsoever.
#[repr(transparent)]
pub(crate) struct SlotStorage<T>(MaybeUninit<Slot<T>>);

impl<T> SlotStorage<T> {
    /// A never-used slot: no bytes initialized.
    pub(crate) const fn new() -> Self {
        Self(MaybeUninit::uninit())
    }
}

/// Backing storage for a fixed-capacity node arena ([`FixedNodes`]):
/// `LEAVES` leaf slots and `INNERS` inner slots, wholly uninitialized.
///
/// Declare it where the memory should live — typically a `static` (see
/// the module docs for the `ConstStaticCell` idiom and why
/// const-construction matters there), a stack array in tests. The
/// storage is inert on its own; [`FixedNodes::new`] borrows it
/// exclusively and does all the bookkeeping.
///
/// `LEAVES` must be at least 1 (the tree's root leaf always exists),
/// checked at compile time where the storage is born.
///
/// Sizing guidance: `LEAVES` bounds the tree's pairs (`LEAVES * M` at
/// perfect packing, roughly half that at steady-state minimum
/// occupancy); derive `INNERS` from `LEAVES` with
/// [`worst_case_inners`](Self::worst_case_inners) rather than guessing —
/// running out of inner slots first is never the intended limit.
pub struct NodeStorage<K: Key, V, const M: usize, const LEAVES: usize, const INNERS: usize> {
    leaves: [SlotStorage<Leaf<K, V, M>>; LEAVES],
    inners: [SlotStorage<Inner<K, V, M>>; INNERS],
}

// SAFETY: the storage is logically uninitialized whenever it is not
// exclusively borrowed by a pool: live values exist only under a
// `FixedNodes` borrow (and values abandoned past a borrow's end are
// forgotten — inert bytes no code will read again). Free-list links
// point only into this same storage and are followed only by the
// borrowing pool. Moving or sharing the unborrowed storage therefore
// transfers no reachable `K`, `V`, or pointer target — there is nothing
// for `K`/`V` bounds to sign for.
unsafe impl<K: Key, V, const M: usize, const LEAVES: usize, const INNERS: usize> Send
    for NodeStorage<K, V, M, LEAVES, INNERS>
{
}

// SAFETY: `&NodeStorage` exposes no method at all — a shared reference
// to inert, logically-uninitialized storage. (Required for a `static`
// declaration, which is the type's whole purpose.)
unsafe impl<K: Key, V, const M: usize, const LEAVES: usize, const INNERS: usize> Sync
    for NodeStorage<K, V, M, LEAVES, INNERS>
{
}

impl<K: Key, V, const M: usize, const LEAVES: usize, const INNERS: usize> Default
    for NodeStorage<K, V, M, LEAVES, INNERS>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Key, V, const M: usize, const LEAVES: usize, const INNERS: usize>
    NodeStorage<K, V, M, LEAVES, INNERS>
{
    /// Wholly uninitialized storage. `const`, and deliberately so: a
    /// `static` initialized by this lands in `.bss` and never transits
    /// the stack.
    pub const fn new() -> Self {
        const {
            assert!(LEAVES >= 1, "the tree's root leaf always exists: LEAVES must be >= 1");
        }
        Self {
            leaves: [const { SlotStorage::new() }; LEAVES],
            inners: [const { SlotStorage::new() }; INNERS],
        }
    }

    /// The number of inner slots that suffices for ANY tree drawing at
    /// most `leaves` leaf slots — the exact worst case, so `INNERS` can
    /// be declared once and never thought about again.
    ///
    /// The worst case is the thinnest legal tree: every non-root inner
    /// at minimum occupancy (`ceil(M / 2)` children, the tree's
    /// `MIN_OCCUPANCY`), so the level above `n` nodes holds at most
    /// `n / ceil(M / 2)` inners — summed level by level. Once a level
    /// cannot host two minimum-occupancy inners, the single node above
    /// it is the root (which may hold as few as two children) and the
    /// chain ends. A lone leaf IS the root: no inner exists at all.
    ///
    /// (The closed-form geometric bound `leaves / (ceil(M / 2) - 1)`
    /// UNDER-counts — it forgets the root chain: two leaves already
    /// need one inner it doesn't budget — and under-provisioning is the
    /// one failure a sizing helper must never have.)
    pub const fn worst_case_inners(leaves: usize) -> usize {
        // `M >= 3` puts minimum occupancy at >= 2, so the per-level
        // counts below strictly shrink and the loop terminates.
        crate::assert_fanout_floor(M);

        if leaves < 2 {
            return 0;
        }

        // The fewest children a NON-ROOT inner may hold.
        let min_children = M.div_ceil(2);

        // Nodes on the level currently being covered, and inners
        // counted so far.
        let mut below = leaves;
        let mut total = 0;
        loop {
            let thinnest = below / min_children;
            if thinnest < 2 {
                // No level of two-or-more minimum-occupancy inners fits
                // over `below` nodes: the one node above is the root.
                return total + 1;
            }
            total += thinnest;
            below = thinnest;
        }
    }
}

/// One node kind's half of a [`FixedNodes`]: free-list + bump-cursor
/// slot service over a borrowed region, mirroring the slab pool minus
/// growth.

pub(crate) struct FixedPool<'a, T> {
    /// The backing region, exclusively ours for `'a`.
    storage: &'a mut [SlotStorage<T>],

    /// Head of the intrusive free list of retired slots.
    free_list: Option<NonNull<Slot<T>>>,

    /// Index of the first never-used slot: `storage[bump..]` is the
    /// never-used tail.
    bump: usize,

    /// Retired slots currently on the free list, so
    /// [`available`](Self::available) is O(1).
    availability: usize,
}

impl<'a, T> FixedPool<'a, T> {
    /// An empty pool over `storage`: everything never-used, nothing on the
    /// free list.
    pub(crate) fn new(storage: &'a mut [SlotStorage<T>]) -> Self {
        let availability = storage.len();
        Self { storage, free_list: None, bump: 0, availability }
    }

    /// Get a free slot. Caller is responsible for ensuring the slot does not
    /// get leaked.
    fn take_next_free(&mut self) -> Option<NonNull<MaybeUninit<T>>> {
        let next = self.free_list?;

        // SAFETY:
        // free list always contains only valid free node pointers. Never
        // contains data.
        self.free_list = unsafe { next.as_ref().next_free };

        self.availability -= 1;
        Some(next.cast())
    }

    /// Get a free bump slot. Caller is responsible for ensuring the slot does
    /// not get leaked.
    fn take_bump(&mut self) -> Option<NonNull<MaybeUninit<T>>> {
        if self.bump == self.storage.len() {
            return None;
        };

        let available = &mut self.storage[self.bump];

        self.bump += 1;
        self.availability -= 1;
        // SAFETY:
        // The pointer is an owned slot.
        // Cast is safe as MaybeUninit<Slot<T>> and MaybeUninit<T> have the
        // same layout
        Some(unsafe { NonNull::new_unchecked(available.0.as_mut_ptr()).cast() })
    }

    /// Hand out a vacant slot at a stable address: pop the free list,
    /// else take the next never-used slot, else report exhaustion.
    pub(crate) fn try_take_slot(&mut self) -> Result<NonNull<MaybeUninit<T>>, AllocError> {
        self.take_next_free().or_else(|| self.take_bump()).ok_or(AllocError)
    }

    /// Pre-pend a slot to the free list.
    ///
    /// # Safety
    /// - `slot` MUST be a slot of this allocator, that has already had
    ///   its value moved out.
    unsafe fn return_slot(&mut self, mut slot: NonNull<Slot<T>>) {
        // SAFETY: `slot` is a slot of this allocator whose value has
        // already been moved out (deallocate's contract), so the
        // storage is exclusively ours to repurpose as the list link.
        unsafe { slot.as_mut() }.next_free = self.free_list;
        self.free_list = Some(slot);
        self.availability += 1;
    }

    /// Forget every outstanding slot: empty free list, bump cursor back
    /// to zero — the pool is as fresh as at construction. Memory only;
    /// values still resident are forgotten, never read or dropped.
    ///
    /// # Safety
    ///
    /// Every pointer previously handed out by this pool is invalidated;
    /// the caller must never use any of them again.
    pub(crate) unsafe fn reclaim(&mut self) {
        self.free_list = None;
        self.bump = 0;
        self.availability = self.storage.len();
    }

    /// The pool's hard slot ceiling.
    pub(crate) fn capacity(&self) -> usize {
        self.storage.len()
    }
}

/// A fixed-capacity [`NodeAllocator`] over an exclusively borrowed
/// [`NodeStorage`]: one leaf pool, one inner pool, no growth, no
/// allocation — ever. The one allocator whose exhaustion is an honest
/// [`Err`], which is what makes the tree's `try_` surface meaningful.
///
/// Construction is [`FixedNodes::new`]; the storage's `LEAVES`/`INNERS`
/// erase into slice lengths, so the arena type carries only the node
/// types and the borrow.
pub struct FixedNodes<'a, K: Key, V, const M: usize> {
    leaves: FixedPool<'a, Leaf<K, V, M>>,
    inners: FixedPool<'a, Inner<K, V, M>>,
}

// SAFETY: a `FixedNodes` exclusively borrows its storage, and every
// pointer in its bookkeeping (free lists, bump cursors) targets that
// borrowed region — nothing foreign, no interior mutability. What a
// move transfers is live slots holding `Leaf`/`Inner` values (payloads
// of `K`s and `V`s, plus intra-arena node pointers), which is what the
// two bounds sign for. Outstanding slot pointers a caller still holds
// are governed by [`NodeAllocator`]'s contract (the slot is exclusively
// that caller's, every use `unsafe`); keeping them coherent with a
// moved arena is that caller's obligation, discharged for the tree by
// `BPlusTree` moving arena and node graph as one value.
unsafe impl<K, V, const M: usize> Send for FixedNodes<'_, K, V, M>
where
    K: Key + Send,
    V: Send,
{
}

// SAFETY: sharing `&FixedNodes` shares a read-only view of plain data —
// no interior mutability, and every mutation of the pools' bookkeeping
// sits behind the trait's `&mut self` receivers, so while shared
// borrows exist no thread can write anything a reader reaches. What a
// reader can (in principle) reach — resident `K`s and `V`s — is what
// the two bounds sign for.
unsafe impl<K, V, const M: usize> Sync for FixedNodes<'_, K, V, M>
where
    K: Key + Sync,
    V: Sync,
{
}

impl<'a, K: Key, V, const M: usize> FixedNodes<'a, K, V, M> {
    /// An empty arena over `storage`, borrowing it exclusively for the
    /// arena's whole life. Everything is never-used: capacity exactly
    /// `LEAVES`/`INNERS`, nothing allocated until the tree's first node.
    pub fn new<const LEAVES: usize, const INNERS: usize>(
        storage: &'a mut NodeStorage<K, V, M, LEAVES, INNERS>,
    ) -> Self {
        let NodeStorage { leaves, inners } = storage;
        Self { leaves: FixedPool::new(leaves), inners: FixedPool::new(inners) }
    }
}

impl<K: Key, V, const M: usize> NodeAllocator<K, V, M> for FixedNodes<'_, K, V, M> {
    /// Exhaustion is real here and the caller hears about it.
    type Exhaustion = AllocError;

    fn try_alloc_leaf_uninit(
        &mut self,
    ) -> Result<NonNull<MaybeUninit<Leaf<K, V, M>>>, Self::Exhaustion> {
        self.leaves.try_take_slot()
    }

    fn try_alloc_inner_uninit(
        &mut self,
    ) -> Result<NonNull<MaybeUninit<Inner<K, V, M>>>, Self::Exhaustion> {
        self.inners.try_take_slot()
    }

    unsafe fn dealloc_leaf_uninit(&mut self, ptr: NonNull<MaybeUninit<Leaf<K, V, M>>>) {
        // SAFETY:
        // as trait contract. caller must ensure this is a valid allocation from
        // this arena
        // Cast is safe, as MaybeUninit<T> and Slot<T> have same layout as T
        unsafe { self.leaves.return_slot(ptr.cast()) }
    }

    unsafe fn dealloc_inner_uninit(&mut self, ptr: NonNull<MaybeUninit<Inner<K, V, M>>>) {
        // SAFETY:
        // as trait contract. caller must ensure this is a valid allocation from
        // this arena
        // Cast is safe, as MaybeUninit<T> and Slot<T> have same layout as T
        unsafe { self.inners.return_slot(ptr.cast()) }
    }

    fn leaf_capacity(&self) -> Option<usize> {
        Some(self.leaves.capacity())
    }

    fn inner_capacity(&self) -> Option<usize> {
        Some(self.inners.capacity())
    }

    fn leaf_available(&self) -> usize {
        self.leaves.availability
    }

    fn inner_available(&self) -> usize {
        self.inners.availability
    }

    unsafe fn reclaim_all(&mut self) -> bool {
        // SAFETY:
        // as trait contract. caller must ensure that no pointers are held.
        unsafe {
            self.leaves.reclaim();
            self.inners.reclaim();
        }
        true
    }
}

#[cfg(test)]
#[path = "../tests/fixed.rs"]
mod tests;
