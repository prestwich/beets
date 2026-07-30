//! Integration pins for the fallible tree surface (`try_new_in`,
//! `try_insert`, `try_from_sorted_iter_in`): failure returns the
//! caller's property — the pair, the allocator — and the tree is left
//! EXACTLY as it was. Over an infallible allocator the same surface
//! never errs. The fixed-region arena is the allocator these contracts
//! exist for, so it stars in every exhaustion test.

use core::{mem::MaybeUninit, ptr::NonNull};
use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering::Relaxed},
    },
};

use proptest::prelude::*;

use crate::{
    AllocError, BPlusTree, FixedNodes, Inner, Leaf, NodeAllocator, NodeStorage, Slabs,
    test_util::{M, v},
};

/// A fixed-pool tree over the given storage. Panics never: fresh
/// storage always serves the root leaf (`LEAVES >= 1` is const-checked).
fn fixed_tree<'a, const L: usize, const I: usize>(
    storage: &'a mut NodeStorage<u64, u64, M, L, I>,
) -> BPlusTree<u64, u64, M, FixedNodes<'a, u64, u64, M>> {
    match BPlusTree::try_new_in(FixedNodes::new(storage)) {
        Ok(tree) => tree,
        Err(_) => unreachable!("fresh storage always serves the root leaf"),
    }
}

// ── try_new_in ──────────────────────────────────────────────────────

/// `try_new_in` over fresh fixed storage succeeds, and the tree it
/// returns is fully serviceable.
#[test]
fn try_new_in_builds_a_serviceable_fixed_tree() {
    let mut storage = NodeStorage::<u64, u64, M, 2, 1>::new();
    let mut tree = fixed_tree(&mut storage);

    assert!(tree.is_empty(), "a new tree holds nothing");
    assert_eq!(tree.try_insert(7, v(7)), Ok(None));
    assert_eq!(tree.get(&7), Some(&v(7)));
    assert_eq!(tree.remove(&7), Some(v(7)));
    tree.check();
}

/// `try_new_in` on an exhausted arena reports failure by RETURNING THE
/// ALLOCATOR — losing it would forfeit the borrowed storage forever.
#[test]
fn try_new_in_hands_the_allocator_back_on_exhaustion() {
    let mut storage = NodeStorage::<u64, u64, M, 1, 1>::new();
    let mut arena = FixedNodes::new(&mut storage);
    let squatter = arena.try_alloc_leaf_uninit().expect("the lone leaf slot");

    let result: Result<BPlusTree<u64, u64, M, _>, _> = BPlusTree::try_new_in(arena);
    let Err(mut arena) = result else {
        panic!("an arena with no leaf slot cannot host the root leaf")
    };

    // The returned arena is the same, live arena: give the squatted
    // slot back and it serves again.
    // SAFETY: `squatter` came from this arena (moves of the arena value
    // do not move the borrowed storage its slots live in), was never
    // initialized, and is retired exactly once, here.
    unsafe { arena.dealloc_leaf_uninit(squatter) };
    assert_eq!(arena.leaf_available(), 1, "the returned arena must be fully usable");
}

// ── try_insert: exhaustion honesty and atomicity ────────────────────

/// Filling a fixed tree ends in an `Err` that returns exactly the
/// rejected pair — and the tree is untouched: same length, same pairs,
/// same order, still invariant-clean.
#[test]
fn try_insert_returns_the_pair_and_leaves_the_tree_untouched() {
    let mut storage = NodeStorage::<u64, u64, M, 3, 2>::new();
    let mut tree = fixed_tree(&mut storage);

    let mut accepted = alloc::vec::Vec::new();
    let mut k = 0u64;
    let rejected = loop {
        match tree.try_insert(k, v(k)) {
            Ok(prev) => {
                assert!(prev.is_none(), "keys are fresh — nothing to replace");
                accepted.push(k);
                k += 1;
            }
            Err(pair) => break pair,
        }
        assert!(k < 10_000, "a 3-leaf pool must exhaust well before 10k pairs");
    };

    assert_eq!(rejected, (k, v(k)), "the rejected pair comes back to the caller, intact");
    assert_eq!(tree.len(), accepted.len(), "a failed insert must not change the count");
    assert_eq!(tree.get(&k), None, "the rejected key must not be half-present");
    assert!(
        tree.iter().map(|(kk, vv)| (*kk, *vv)).eq(accepted.iter().map(|&kk| (kk, v(kk)))),
        "a failed insert must not disturb the resident pairs"
    );
    tree.check();

    // Replacement stores no new node: it must succeed on a pool with
    // nothing left to give.
    let existing = accepted[accepted.len() / 2];
    assert_eq!(
        tree.try_insert(existing, 0xDEAD),
        Ok(Some(v(existing))),
        "replacing in place allocates nothing and must succeed on a full pool"
    );
}

/// Failed inserts are repeatable and leak-free: every retry errs
/// identically, and after `clear` the pool accepts exactly the same
/// fill again — a leaked reservation would shrink it.
#[test]
fn failed_inserts_leak_no_slots() {
    let mut storage = NodeStorage::<u64, u64, M, 3, 2>::new();
    let mut tree = fixed_tree(&mut storage);

    let mut first_fill = 0u64;
    while tree.try_insert(first_fill, v(first_fill)).is_ok() {
        first_fill += 1;
    }

    for round in 0..100u64 {
        let key = 1_000_000 + round;
        assert_eq!(
            tree.try_insert(key, 1),
            Err((key, 1)),
            "every insert on a full pool must err, identically, forever"
        );
    }
    assert_eq!(tree.len(), first_fill as usize, "failed inserts must not change the tree");

    tree.clear();
    let mut refilled = 0u64;
    while tree.try_insert(refilled, v(refilled)).is_ok() {
        refilled += 1;
    }
    assert_eq!(
        refilled, first_fill,
        "a cleared pool must accept exactly the same fill again — \
         a reservation leaked by a failed insert would shrink it"
    );
}

/// A split that can reserve only PART of its bill (the right leaf, but
/// no inner for the root) must roll the partial reservation back
/// completely.
#[test]
fn a_partially_reservable_split_rolls_back_completely() {
    // Leaves to spare, ZERO inners: the first split can reserve its
    // right leaf but never the root inner above it.
    let mut storage = NodeStorage::<u64, u64, M, 4, 0>::new();
    let mut tree = fixed_tree(&mut storage);

    let mut accepted = 0u64;
    while tree.try_insert(accepted, v(accepted)).is_ok() {
        accepted += 1;
    }
    tree.check();
    let first_fill = tree.len();

    for round in 0..100u64 {
        assert!(
            tree.try_insert(10_000 + round, 1).is_err(),
            "the split stays impossible with zero inner slots"
        );
    }
    assert_eq!(tree.len(), first_fill, "failed splits must not change the tree");

    tree.clear();
    let mut refilled = 0u64;
    while tree.try_insert(refilled, v(refilled)).is_ok() {
        refilled += 1;
    }
    assert_eq!(
        refilled, accepted,
        "a cleared pool must accept the same fill again — a reservation \
         abandoned mid-rollback would shrink it"
    );
}

/// An insert with room in its leaf touches no allocator state at all:
/// it must succeed even when every pool is exhausted.
#[test]
fn room_in_the_leaf_needs_no_allocator_at_all() {
    // One leaf slot (the root takes it), zero inners.
    let mut storage = NodeStorage::<u64, u64, M, 1, 0>::new();
    let mut tree = fixed_tree(&mut storage);

    for k in 0..M as u64 {
        assert_eq!(
            tree.try_insert(k, v(k)),
            Ok(None),
            "an in-leaf insert (key {k}) allocates nothing and must succeed"
        );
    }
    assert!(
        tree.try_insert(M as u64, 1).is_err(),
        "pair M+1 needs a split, and there is nothing to split into"
    );
    tree.check();
}

// ── try_from_sorted_iter_in ─────────────────────────────────────────

/// A bulk load within capacity succeeds and is structurally clean.
#[test]
fn try_from_sorted_iter_in_loads_within_capacity() {
    const N: u64 = 1_000;
    let mut storage = NodeStorage::<u64, u64, M, 64, 8>::new();

    let result: Result<BPlusTree<u64, u64, M, _>, _> =
        BPlusTree::try_from_sorted_iter_in((0..N).map(|k| (k, v(k))), FixedNodes::new(&mut storage));
    let Ok(tree) = result else { panic!("{N} pairs fit comfortably in 64 leaves") };

    tree.check();
    assert_eq!(tree.len(), N as usize);
    assert!(tree.iter().map(|(k, _)| *k).eq(0..N), "iteration must replay the loaded keys");
}

/// A bulk load past capacity reports failure by returning the
/// allocator, EMPTIED: the torn-down load must have given every slot
/// back, leaving the arena immediately reusable.
#[test]
fn try_from_sorted_iter_in_returns_an_emptied_allocator_on_exhaustion() {
    let mut storage = NodeStorage::<u64, u64, M, 2, 1>::new();

    let result: Result<BPlusTree<u64, u64, M, _>, _> = BPlusTree::try_from_sorted_iter_in(
        (0..10_000u64).map(|k| (k, v(k))),
        FixedNodes::new(&mut storage),
    );
    let Err(arena) = result else { panic!("10k pairs cannot fit in two leaves") };

    assert_eq!(arena.leaf_available(), 2, "the failed load must return every leaf slot");
    assert_eq!(arena.inner_available(), 1, "…and every inner slot");

    // Immediately reusable: host a small tree in the same storage.
    // (Through the try_ door — the panicking constructors are
    // type-gated to infallible allocators.)
    let mut tree: BPlusTree<u64, u64, M, _> = match BPlusTree::try_new_in(arena) {
        Ok(tree) => tree,
        Err(_) => panic!("an emptied arena must serve the root leaf"),
    };
    assert_eq!(tree.try_insert(1, v(1)), Ok(None));
    assert_eq!(tree.get(&1), Some(&v(1)));
    tree.check();
}

// ── the same surface over an infallible allocator ───────────────────

/// Over `Slabs` the try surface never errs and agrees with the
/// infallible one: fresh keys report `Ok(None)`, replacements report
/// the old value.
#[test]
fn the_try_surface_is_infallible_over_slabs() {
    let mut tree: BPlusTree<u64, u64, M, Slabs<u64, u64, M>> =
        BPlusTree::try_new_in(Slabs::new()).unwrap_or_else(|_| unreachable!());

    for k in 0..5_000 {
        assert_eq!(tree.try_insert(k, v(k)), Ok(None), "an infallible allocator never rejects");
    }
    assert_eq!(tree.try_insert(7, 42), Ok(Some(v(7))), "replacement reports the old value");
    assert_eq!(tree.get(&7), Some(&42));
    tree.check();
}

// ── reservation accounting on the failure path ──────────────────────

/// Forwards to a fixed arena, tallying the uninit primitives — the
/// reservation traffic — so the failure path's accounting is
/// observable from outside the tree.
struct Audited<'a> {
    arena: FixedNodes<'a, u64, u64, M>,
    acquires: Arc<AtomicUsize>,
    releases: Arc<AtomicUsize>,
}

impl NodeAllocator<u64, u64, M> for Audited<'_> {
    type Exhaustion = AllocError;

    fn try_alloc_leaf_uninit(
        &mut self,
    ) -> Result<NonNull<MaybeUninit<Leaf<u64, u64, M>>>, Self::Exhaustion> {
        let slot = self.arena.try_alloc_leaf_uninit()?;
        self.acquires.fetch_add(1, Relaxed);
        Ok(slot)
    }

    fn try_alloc_inner_uninit(
        &mut self,
    ) -> Result<NonNull<MaybeUninit<Inner<u64, u64, M>>>, Self::Exhaustion> {
        let slot = self.arena.try_alloc_inner_uninit()?;
        self.acquires.fetch_add(1, Relaxed);
        Ok(slot)
    }

    unsafe fn dealloc_leaf_uninit(&mut self, ptr: NonNull<MaybeUninit<Leaf<u64, u64, M>>>) {
        self.releases.fetch_add(1, Relaxed);
        // SAFETY: forwarded — the caller's obligations are the arena's.
        unsafe { self.arena.dealloc_leaf_uninit(ptr) }
    }

    unsafe fn dealloc_inner_uninit(&mut self, ptr: NonNull<MaybeUninit<Inner<u64, u64, M>>>) {
        self.releases.fetch_add(1, Relaxed);
        // SAFETY: forwarded — the caller's obligations are the arena's.
        unsafe { self.arena.dealloc_inner_uninit(ptr) }
    }

    fn leaf_capacity(&self) -> Option<usize> {
        self.arena.leaf_capacity()
    }

    fn inner_capacity(&self) -> Option<usize> {
        self.arena.inner_capacity()
    }

    fn leaf_available(&self) -> usize {
        self.arena.leaf_available()
    }

    fn inner_available(&self) -> usize {
        self.arena.inner_available()
    }
}

/// A failed `try_insert` must release exactly what it reserved: across
/// any number of failures, acquires and releases move in lockstep —
/// the leak-freedom of the rollback path, observed at the allocator
/// boundary rather than inferred from refill behavior.
#[test]
fn failed_inserts_balance_their_reservation_traffic() {
    // Leaves to spare, ZERO inners: every post-fill insert reserves
    // partially, then must roll that partial reservation back.
    let mut storage = NodeStorage::<u64, u64, M, 4, 0>::new();
    let acquires = Arc::new(AtomicUsize::new(0));
    let releases = Arc::new(AtomicUsize::new(0));
    let audited = Audited {
        arena: FixedNodes::new(&mut storage),
        acquires: Arc::clone(&acquires),
        releases: Arc::clone(&releases),
    };

    let mut tree: BPlusTree<u64, u64, M, Audited<'_>> = match BPlusTree::try_new_in(audited) {
        Ok(tree) => tree,
        Err(_) => unreachable!("fresh storage serves the root leaf"),
    };

    let mut k = 0u64;
    while tree.try_insert(k, v(k)).is_ok() {
        k += 1;
    }

    let acquired_before = acquires.load(Relaxed);
    let released_before = releases.load(Relaxed);

    for round in 0..50u64 {
        assert!(tree.try_insert(100_000 + round, 1).is_err(), "the pool stays full");
    }

    let acquired = acquires.load(Relaxed) - acquired_before;
    let released = releases.load(Relaxed) - released_before;
    assert_eq!(
        acquired, released,
        "every slot a failed insert reserves must be released by that same failure \
         ({acquired} acquired, {released} released across 50 failures)"
    );
}

// ── differential: honest exhaustion against a model ─────────────────

proptest! {
    /// A fixed-pool tree agrees with `BTreeMap` on every operation it
    /// ACCEPTS; an `Err` from `try_insert` is a legal outcome of the
    /// deliberately tiny pool, and must leave tree and model still in
    /// agreement.
    #[test]
    fn a_fixed_tree_tracks_the_model_modulo_honest_exhaustion(
        ops in proptest::collection::vec((any::<bool>(), 0u64..64), 0..256)
    ) {
        let mut storage = NodeStorage::<u64, u64, M, 3, 2>::new();
        let mut tree = fixed_tree(&mut storage);
        let mut model = BTreeMap::new();

        for (is_insert, k) in ops {
            if is_insert {
                match tree.try_insert(k, v(k)) {
                    Ok(prev) => {
                        prop_assert_eq!(prev, model.insert(k, v(k)), "insert outcomes must agree");
                    }
                    Err(pair) => {
                        prop_assert_eq!(pair, (k, v(k)), "the rejected pair comes back intact");
                        // The model deliberately skips the op: the tree
                        // promised it changed nothing.
                    }
                }
            } else {
                prop_assert_eq!(tree.remove(&k), model.remove(&k), "remove outcomes must agree");
            }
            prop_assert_eq!(tree.len(), model.len(), "sizes must agree after every op");
        }

        tree.check();
        prop_assert!(
            tree.iter().map(|(k, vv)| (*k, *vv)).eq(model.iter().map(|(k, vv)| (*k, *vv))),
            "after the run, tree and model must hold identical pairs"
        );
    }
}
