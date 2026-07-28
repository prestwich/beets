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

impl<T> SlabAlloc<T> {
    /// An empty allocator that will grow in slabs of `slab_capacity`
    /// slots. Allocates nothing until the first [`allocate`](SlotAllocator::allocate).
    ///
    /// Capacity guidance: size slabs to a byte budget (a few pages,
    /// e.g. 64 KiB) rather than a slot count — [`Slabs`] does
    /// this for both node types.
    ///
    /// # Panics
    ///
    /// If `slab_capacity` is 0.
    pub(crate) const fn new(slab_capacity: usize) -> Self {
        Self::new_in(slab_capacity, Global)
    }
}

impl<T, A: GlobalAlloc> SlabAlloc<T, A> {
    /// The byte offset from the slab base to slot 0.
    const fn slot_offset(&self) -> usize {
        self.slab_layout().1
    }

    #[inline(always)]
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
    pub(crate) fn contains(&self, ptr: NonNull<T>) -> bool {
        self.iter_slabs().any(|slab| self.slab_contains(slab, ptr))
    }
}

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
