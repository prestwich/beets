//! Contract tests for the fixed-region arena: the pool's slot service
//! over borrowed storage (drain-to-exhaustion, free-before-virgin,
//! reclaim), the arena's `NodeAllocator` surface (honest `AllocError`,
//! per-pool capacity/available bookkeeping, wholesale reclaim), the
//! storage's static-declarability, and the sizing helper. Every test
//! here must pass under `cargo test` AND `cargo miri test` — in-bounds
//! slot service and never-reading-virgin-slots are only fully checked
//! by miri.

use alloc::vec::Vec;

use super::*;
use crate::test_util::M;

/// Stack-borrowed pool storage for the unit tests.
fn pool_storage<const N: usize>() -> [SlotStorage<u64>; N] {
    core::array::from_fn(|_| SlotStorage::new())
}

// ── FixedPool: the slot service over one borrowed region ───────────

/// A fresh pool serves exactly its region's slot count, at distinct
/// stable addresses, and then reports exhaustion — there is no growth.
#[test]
fn a_pool_drains_to_exactly_its_capacity_then_errs() {
    let mut storage = pool_storage::<8>();
    let mut pool = FixedPool::new(&mut storage);
    assert_eq!(pool.capacity(), 8, "capacity is the region's slot count");
    assert_eq!(pool.availability, 8, "everything is servable at birth");

    let slots: Vec<_> = (0..8)
        .map(|i| pool.try_take_slot().unwrap_or_else(|_| panic!("slot {i} must be servable")))
        .collect();
    for (i, a) in slots.iter().enumerate() {
        for b in &slots[i + 1..] {
            assert_ne!(a, b, "every live slot must have its own address");
        }
    }
    assert_eq!(pool.availability, 0, "a drained pool has nothing on hand");
    assert!(pool.try_take_slot().is_err(), "a drained pool must report exhaustion, not grow");

    // Slots are real, writable storage: fill and read every one back
    // through its original pointer.
    for (i, slot) in slots.iter().enumerate() {
        // SAFETY: each slot is live and exclusively ours.
        unsafe { slot.cast::<u64>().write(i as u64) };
    }
    for (i, slot) in slots.iter().enumerate() {
        // SAFETY: written just above, address stable since `try_take`.
        let got = unsafe { slot.cast::<u64>().read() };
        assert_eq!(got, i as u64, "slot {i} must still hold its value after the full drain");
    }

    for slot in slots {
        // SAFETY: live, its value read out (u64: no drop glue), retired
        // exactly once. The cast is the pool's own storage convention
        // (`MaybeUninit<T>` and `Slot<T>` share layout).
        unsafe { pool.return_slot(slot.cast()) };
    }
    assert_eq!(pool.availability, 8, "returning every slot restores full availability");
}

/// The free list is consulted before the virgin tail: a returned
/// slot's address comes back on the very next take, even though
/// never-used slots remain.
#[test]
fn a_returned_slot_is_reused_before_any_virgin_slot() {
    let mut storage = pool_storage::<8>();
    let mut pool = FixedPool::new(&mut storage);

    let a = pool.try_take_slot().expect("first slot");
    let b = pool.try_take_slot().expect("second slot");

    // SAFETY: `b` was never written — return-uninit is exactly the
    // reservation-rollback path.
    unsafe { pool.return_slot(b.cast()) };
    let c = pool.try_take_slot().expect("a slot is on the free list");
    assert_eq!(c, b, "a returned slot must be reused before any virgin slot");

    // SAFETY: both live, retired exactly once, never written.
    unsafe {
        pool.return_slot(a.cast());
        pool.return_slot(c.cast());
    }
}

/// `reclaim` forgets every outstanding slot at once: full availability,
/// and the region is served over again from scratch.
#[test]
fn reclaim_resets_the_pool_for_reuse() {
    let mut storage = pool_storage::<4>();
    let mut pool = FixedPool::new(&mut storage);

    while pool.try_take_slot().is_ok() {}
    assert_eq!(pool.availability, 0);

    // SAFETY: every outstanding pointer is abandoned here; nothing was
    // written (no drop glue to lose either way).
    unsafe { pool.reclaim() };
    assert_eq!(pool.availability, 4, "a reclaimed pool is as fresh as at construction");
    assert_eq!(pool.capacity(), 4, "reclaim never changes the ceiling");
    assert!(pool.try_take_slot().is_ok(), "a reclaimed pool serves slots again");
}

/// A zero-length region is a valid (if useless) pool: exhausted from
/// birth, never UB.
#[test]
fn an_empty_region_is_exhausted_from_birth() {
    let mut storage = pool_storage::<0>();
    let mut pool = FixedPool::new(&mut storage);
    assert_eq!(pool.capacity(), 0);
    assert_eq!(pool.availability, 0);
    assert!(pool.try_take_slot().is_err(), "no storage, no slots — an honest Err, not UB");
}

// ── FixedNodes: the NodeAllocator surface ───────────────────────────

/// Capacity is the declared ceiling per node kind; availability starts
/// there and tracks slot traffic per pool, independently.
#[test]
fn capacity_and_available_are_per_pool_and_honest() {
    let mut storage = NodeStorage::<u64, u64, M, 4, 2>::new();
    let mut arena = FixedNodes::new(&mut storage);

    assert_eq!(arena.leaf_capacity(), Some(4), "leaf ceiling is the declared LEAVES");
    assert_eq!(arena.inner_capacity(), Some(2), "inner ceiling is the declared INNERS");
    assert_eq!(arena.leaf_available(), 4);
    assert_eq!(arena.inner_available(), 2);

    let leaf = arena.try_alloc_leaf_uninit().expect("a leaf slot");
    assert_eq!(arena.leaf_available(), 3, "taking a leaf slot is debited to the leaf pool");
    assert_eq!(arena.inner_available(), 2, "…and never to the inner pool");

    // SAFETY: never initialized — the reservation-rollback path.
    unsafe { arena.dealloc_leaf_uninit(leaf) };
    assert_eq!(arena.leaf_available(), 4, "returning the slot restores availability");
}

/// Exhaustion is an honest `AllocError` per pool: draining the leaf
/// pool must not impair the inner pool, and vice versa.
#[test]
fn exhaustion_is_per_pool_and_reports_alloc_error() {
    let mut storage = NodeStorage::<u64, u64, M, 2, 3>::new();
    let mut arena = FixedNodes::new(&mut storage);

    let _a = arena.try_alloc_leaf_uninit().expect("leaf slot 1");
    let _b = arena.try_alloc_leaf_uninit().expect("leaf slot 2");
    assert_eq!(
        arena.try_alloc_leaf_uninit().unwrap_err(),
        AllocError,
        "a drained leaf pool must report AllocError"
    );

    assert!(
        arena.try_alloc_inner_uninit().is_ok(),
        "leaf exhaustion must leave the inner pool serving"
    );
}

/// The arena reclaims wholesale: one call, both pools reset, `true`
/// reported — a fixed region owns its slots by construction.
#[test]
fn reclaim_all_resets_both_pools_and_reports_true() {
    let mut storage = NodeStorage::<u64, u64, M, 2, 2>::new();
    let mut arena = FixedNodes::new(&mut storage);

    while arena.try_alloc_leaf_uninit().is_ok() {}
    while arena.try_alloc_inner_uninit().is_ok() {}

    // SAFETY: every outstanding pointer is abandoned here; none was
    // initialized.
    let reclaimed = unsafe { arena.reclaim_all() };
    assert!(reclaimed, "a fixed arena owns its slots wholesale and must report the reset");
    assert_eq!(arena.leaf_available(), 2, "the leaf pool must be fresh again");
    assert_eq!(arena.inner_available(), 2, "the inner pool must be fresh again");
    assert!(arena.try_alloc_leaf_uninit().is_ok(), "a reclaimed arena serves slots again");
}

/// The value-level provided methods ride the primitives: a leaf moved
/// in comes back out exactly once, and the slot is reusable after.
#[test]
fn a_leaf_value_round_trips_through_the_arena() {
    let mut storage = NodeStorage::<u64, u64, M, 1, 1>::new();
    let mut arena = FixedNodes::new(&mut storage);

    let mut leaf = Leaf::<u64, u64, M>::new(None);
    leaf.raw_append(7, 42);
    let Ok(ptr) = arena.try_alloc_leaf(leaf) else {
        panic!("one leaf slot exists — the arena must accept the leaf")
    };

    // SAFETY: live, initialized by try_alloc_leaf, retired exactly once.
    let back = unsafe { arena.dealloc_leaf(ptr) };
    assert_eq!(back.len(), 1, "the value that comes back is the value that went in");
    assert_eq!(back.get(&7), Some(&42), "…contents included");
    assert_eq!(arena.leaf_available(), 1, "the slot is servable again");
}

// ── NodeStorage: declarability and sizing ───────────────────────────

/// The storage type is fit for its whole purpose: const-constructible
/// (a `static` initializer, no stack transit) and `Sync` (a `static`
/// requires it). This item compiling IS the test.
static STORAGE_IS_STATIC_DECLARABLE: NodeStorage<u64, u64, M, 2, 1> = NodeStorage::new();

/// …and movable/shareable by the auto-trait story the borrow design
/// needs.
#[test]
fn storage_and_arena_satisfy_the_auto_traits() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<NodeStorage<u64, u64, M, 2, 1>>();
    assert_send_sync::<FixedNodes<'static, u64, u64, M>>();
    // Touch the static so it is not dead code.
    let _ = &STORAGE_IS_STATIC_DECLARABLE;
}

/// The sizing helper answers "how many inner slots could `leaves`
/// leaves ever need": zero for a lone root leaf, exactly the root for
/// two, never less for more leaves than for fewer, and — at this
/// fanout — never a silly over-ask.
#[test]
fn worst_case_inners_matches_the_structural_bounds() {
    type Storage = NodeStorage<u64, u64, M, 2, 1>;

    assert_eq!(
        Storage::worst_case_inners(1),
        0,
        "one leaf slot can only ever host the root leaf — no inner exists"
    );
    assert_eq!(
        Storage::worst_case_inners(2),
        1,
        "two leaves demand exactly one inner: the root above them"
    );

    let mut prev = 0;
    for leaves in 1..=256 {
        let inners = Storage::worst_case_inners(leaves);
        assert!(inners >= prev, "the bound must be monotone in the leaf count");
        prev = inners;
    }

    for leaves in 2..=256 {
        let inners = Storage::worst_case_inners(leaves);
        assert!(
            inners <= leaves,
            "at fanout {M}, {leaves} leaves must never ask for more than {leaves} \
             inners (got {inners})"
        );
    }
}
