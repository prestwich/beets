//! Integration pins for wholesale teardown: the `OWNS_ALL` switch and
//! the tree fast paths it licenses. When the allocator owns all slot
//! memory wholesale AND the values have no drop glue, the tree's `Drop`
//! and `clear` must skip the per-node walk — teardown retires zero
//! individual slots. When either condition fails, the walk (and every
//! value drop it owes) must still happen.

use crate::{common, common::counting};

use std::{
    alloc::System,
    ptr::NonNull,
    sync::{
        Arc,
        atomic::{AtomicIsize, AtomicUsize, Ordering::Relaxed},
    },
};

use crate::{BPlusTree, Inner, Leaf, Slabs, SlotAllocator};
use common::{Counted, M, fill, v};

/// The switch is an associated const — evaluable in const context, so
/// the teardown branch folds away at monomorphization — and it is set
/// honestly per allocator: the slab arena owns its slots wholesale, a
/// box-per-node global allocator does not.
#[test]
fn owns_all_is_const_and_honest() {
    const {
        assert!(
            <Slabs<u64, u64, M> as SlotAllocator<Leaf<u64, u64, M>>>::OWNS_ALL
                && <Slabs<u64, u64, M> as SlotAllocator<Inner<u64, u64, M>>>::OWNS_ALL,
            "the slab arena owns both pools wholesale"
        );
        assert!(
            !<System as SlotAllocator<Leaf<u64, u64, M>>>::OWNS_ALL,
            "a global allocator boxes each node separately and must not claim wholesale ownership"
        );
    }
}

/// Forwards both node pools to a wrapped slab arena, tallying every
/// per-slot `deallocate` and every `clear_all`, so tests can observe
/// which teardown path the tree took. `OWNS_ALL` is inherited
/// truthfully: the wrapped arena owns the memory, so the spy does too.
struct Spy<V> {
    arena: Slabs<u64, V, M>,
    deallocs: Arc<AtomicUsize>,
    leaf_clears: Arc<AtomicUsize>,
    inner_clears: Arc<AtomicUsize>,
}

impl<V> Spy<V> {
    fn new() -> Self {
        Self {
            arena: Slabs::new(),
            deallocs: Arc::default(),
            leaf_clears: Arc::default(),
            inner_clears: Arc::default(),
        }
    }
}

impl<V> SlotAllocator<Leaf<u64, V, M>> for Spy<V> {
    const OWNS_ALL: bool = true;

    fn allocate(&mut self, value: Leaf<u64, V, M>) -> NonNull<Leaf<u64, V, M>> {
        self.arena.allocate(value)
    }

    unsafe fn deallocate(&mut self, ptr: NonNull<Leaf<u64, V, M>>) -> Leaf<u64, V, M> {
        self.deallocs.fetch_add(1, Relaxed);
        // SAFETY: forwarded — the caller's obligations are the arena's.
        unsafe { self.arena.deallocate(ptr) }
    }

    unsafe fn clear_all(&mut self) {
        self.leaf_clears.fetch_add(1, Relaxed);
        // SAFETY: forwarded — the caller's obligations are the arena's.
        unsafe { SlotAllocator::<Leaf<u64, V, M>>::clear_all(&mut self.arena) }
    }
}

impl<V> SlotAllocator<Inner<u64, V, M>> for Spy<V> {
    const OWNS_ALL: bool = true;

    fn allocate(&mut self, value: Inner<u64, V, M>) -> NonNull<Inner<u64, V, M>> {
        self.arena.allocate(value)
    }

    unsafe fn deallocate(&mut self, ptr: NonNull<Inner<u64, V, M>>) -> Inner<u64, V, M> {
        self.deallocs.fetch_add(1, Relaxed);
        // SAFETY: forwarded — the caller's obligations are the arena's.
        unsafe { self.arena.deallocate(ptr) }
    }

    unsafe fn clear_all(&mut self) {
        self.inner_clears.fetch_add(1, Relaxed);
        // SAFETY: forwarded — the caller's obligations are the arena's.
        unsafe { SlotAllocator::<Inner<u64, V, M>>::clear_all(&mut self.arena) }
    }
}

/// An allocator that is honest per pool but ASYMMETRIC: leaf slots come
/// from a slab arena that owns them wholesale (`OWNS_ALL = true` on the
/// leaf impl, truthfully — the arena's drop reclaims them), while every
/// inner node is its own boxed allocation through a counting backing
/// (`OWNS_ALL = false` on the inner impl, equally truthfully — an inner
/// never passed to `deallocate` is leaked memory).
struct Lopsided {
    leaves: Slabs<u64, u64, M>,
    inners: common::Counting,
}

impl SlotAllocator<Leaf<u64, u64, M>> for Lopsided {
    const OWNS_ALL: bool = true;

    fn allocate(&mut self, value: Leaf<u64, u64, M>) -> NonNull<Leaf<u64, u64, M>> {
        self.leaves.allocate(value)
    }

    unsafe fn deallocate(&mut self, ptr: NonNull<Leaf<u64, u64, M>>) -> Leaf<u64, u64, M> {
        // SAFETY: forwarded — the caller's obligations are the arena's.
        unsafe { self.leaves.deallocate(ptr) }
    }

    unsafe fn clear_all(&mut self) {
        // SAFETY: forwarded — the caller's obligations are the arena's.
        unsafe { SlotAllocator::<Leaf<u64, u64, M>>::clear_all(&mut self.leaves) }
    }
}

impl SlotAllocator<Inner<u64, u64, M>> for Lopsided {
    const OWNS_ALL: bool = false;

    fn allocate(&mut self, value: Inner<u64, u64, M>) -> NonNull<Inner<u64, u64, M>> {
        self.inners.allocate(value)
    }

    unsafe fn deallocate(&mut self, ptr: NonNull<Inner<u64, u64, M>>) -> Inner<u64, u64, M> {
        // SAFETY: forwarded — the caller's obligations are the backing's.
        unsafe { self.inners.deallocate(ptr) }
    }

    // No `clear_all`: with `OWNS_ALL = false` it must never be called,
    // and the trait default enforces that with a panic.
}

/// The teardown shortcut is licensed by ALL of the tree's node memory
/// being wholesale-owned, not some of it: under an allocator whose
/// inner nodes are individually boxed, dropping the tree must return
/// every boxed inner through `deallocate` — whatever it does about the
/// leaves.
#[test]
fn drop_retires_every_boxed_inner_under_an_asymmetric_allocator() {
    let counting = counting!();
    let mut tree: BPlusTree<u64, u64, M, Lopsided> =
        BPlusTree::new_in(Lopsided { leaves: Slabs::new(), inners: counting });

    fill(&mut tree, 2_000);
    assert!(
        counting.allocs() > 0,
        "2k pairs must build a tree tall enough to box inner nodes through the backing"
    );

    drop(tree);
    counting.assert_balanced("boxed inner node");
}

/// With drop-free values under a whole-owning allocator, dropping the
/// tree must not walk the nodes: zero per-slot retirements — the
/// allocator's own drop reclaims all slot memory wholesale.
#[test]
fn drop_skips_the_per_node_walk_for_plain_values() {
    let spy = Spy::<u64>::new();
    let deallocs = Arc::clone(&spy.deallocs);

    let mut tree: BPlusTree<u64, u64, M, Spy<u64>> = BPlusTree::new_in(spy);
    fill(&mut tree, 2_000);
    drop(tree);

    assert_eq!(
        deallocs.load(Relaxed),
        0,
        "dropping a tree of drop-free values under a wholesale-owning allocator \
         must retire zero individual slots"
    );
}

/// With drop-free values under a whole-owning allocator, `clear` must
/// release wholesale: zero per-slot retirements, each node pool reset
/// exactly once, and the tree fresh and serviceable afterward.
#[test]
fn clear_releases_wholesale_for_plain_values() {
    let spy = Spy::<u64>::new();
    let deallocs = Arc::clone(&spy.deallocs);
    let leaf_clears = Arc::clone(&spy.leaf_clears);
    let inner_clears = Arc::clone(&spy.inner_clears);

    let mut tree: BPlusTree<u64, u64, M, Spy<u64>> = BPlusTree::new_in(spy);
    fill(&mut tree, 2_000);
    tree.clear();

    assert_eq!(
        deallocs.load(Relaxed),
        0,
        "clearing a tree of drop-free values under a wholesale-owning allocator \
         must retire zero individual slots"
    );
    assert_eq!(leaf_clears.load(Relaxed), 1, "clear must reset the leaf pool exactly once");
    assert_eq!(inner_clears.load(Relaxed), 1, "clear must reset the inner pool exactly once");

    assert!(tree.is_empty(), "a cleared tree holds no pairs");
    tree.insert(7, v(7));
    assert_eq!(tree.get(&7), Some(&v(7)), "a cleared tree must serve fresh inserts");
    assert_eq!(tree.len(), 1, "a cleared tree counts only fresh inserts");
}

/// Value drop glue disables the shortcut: dropping the tree must still
/// drop every live value exactly once.
#[test]
fn drop_still_drops_every_value_when_values_need_it() {
    let live = Arc::new(AtomicIsize::new(0));
    let mut tree: BPlusTree<u64, Counted, M> = BPlusTree::new();
    for k in 0..2_000 {
        tree.insert(k, Counted::new(k, &live));
    }
    assert_eq!(live.load(Relaxed), 2_000, "one live value per inserted key");

    drop(tree);
    assert_eq!(
        live.load(Relaxed),
        0,
        "the tree's drop must drop every value exactly once \
         (positive = leak, negative = double-drop)"
    );
}

/// Value drop glue disables the shortcut for `clear` too: every live
/// value drops exactly once, and the tree stays serviceable.
#[test]
fn clear_still_drops_every_value_when_values_need_it() {
    let live = Arc::new(AtomicIsize::new(0));
    let mut tree: BPlusTree<u64, Counted, M> = BPlusTree::new();
    for k in 0..2_000 {
        tree.insert(k, Counted::new(k, &live));
    }

    tree.clear();
    assert_eq!(
        live.load(Relaxed),
        0,
        "clear must drop every value exactly once (positive = leak, negative = double-drop)"
    );

    tree.insert(7, Counted::new(7, &live));
    assert_eq!(live.load(Relaxed), 1, "a cleared tree must own fresh values normally");
    drop(tree);
    assert_eq!(live.load(Relaxed), 0, "and drop them exactly once at the end");
}

/// Whichever teardown path runs, every slab drawn from the backing is
/// returned to it by the time the tree drops — the fast path may skip
/// the walk, never the reclamation.
#[test]
fn teardown_returns_all_slab_memory_to_the_backing() {
    let counting = counting!();
    let mut tree: BPlusTree<u64, u64, M, Slabs<u64, u64, M, common::Counting>> =
        BPlusTree::new_in(Slabs::new_in(counting));
    fill(&mut tree, 2_000);
    drop(tree);

    counting.assert_balanced("slab");
}

/// A clear-then-refill cycle is leak-free end to end: the arena is
/// reusable after `clear`, the refilled tree reads back correctly, and
/// all backing memory balances at drop — whatever reset strategy
/// `clear` uses underneath.
#[test]
fn clear_then_refill_reuses_the_arena_without_leaking() {
    let counting = counting!();
    let mut tree: BPlusTree<u64, u64, M, Slabs<u64, u64, M, common::Counting>> =
        BPlusTree::new_in(Slabs::new_in(counting));

    fill(&mut tree, 2_000);
    tree.clear();
    assert!(tree.is_empty(), "clear must empty the tree");

    fill(&mut tree, 2_000);
    assert_eq!(tree.len(), 2_000, "a refilled tree counts every fresh pair");
    assert_eq!(tree.get(&1_234), Some(&v(1_234)), "a refilled tree must read back its pairs");

    drop(tree);
    counting.assert_balanced("slab");
}
