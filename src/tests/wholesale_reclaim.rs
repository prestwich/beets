//! Integration pins for wholesale teardown: `reclaim_all` and the tree
//! fast paths it licenses. When the allocator reclaims every slot's
//! memory wholesale (`reclaim_all` → `true`) AND the values have no
//! drop glue, the tree's `Drop` and `clear` must skip the per-node walk
//! — teardown retires zero individual slots. When either condition
//! fails, the walk (and every value drop it owes) must still happen.

use crate::{common, common::counting};

use std::{
    alloc::System,
    mem::MaybeUninit,
    ptr::NonNull,
    sync::{
        Arc,
        atomic::{AtomicIsize, AtomicUsize, Ordering::Relaxed},
    },
};

use crate::{BPlusTree, Inner, Leaf, NodeAllocator, Slabs};
use common::{Counted, M, fill, v};

/// `reclaim_all` is all-or-nothing and honest per allocator: the slab
/// arena owns both pools wholesale, resets them, and reports `true`; a
/// box-per-node global allocator cannot reclaim wholesale, so it must
/// do nothing and report `false` — the trait default.
#[test]
fn reclaim_all_is_honest_per_allocator() {
    let mut arena: Slabs<u64, u64, M> = Slabs::new();
    // SAFETY: no outstanding slots exist to invalidate.
    let reclaimed = unsafe { arena.reclaim_all() };
    assert!(reclaimed, "the slab arena owns both pools wholesale and must report the reset");

    // SAFETY: no outstanding slots exist, and a `false` return
    // obligates nothing anyway.
    let reclaimed = unsafe { NodeAllocator::<u64, u64, M>::reclaim_all(&mut System) };
    assert!(
        !reclaimed,
        "a global allocator boxes each node separately and must decline wholesale reclaim"
    );
}

/// Forwards to a wrapped slab arena, tallying every per-slot
/// retirement (through the uninit primitives, which the provided
/// value-level `dealloc_*` must route through) and every wholesale
/// reclaim, so tests can observe which teardown path the tree took.
struct Spy<V> {
    arena: Slabs<u64, V, M>,
    deallocs: Arc<AtomicUsize>,
    reclaims: Arc<AtomicUsize>,
}

impl<V> Spy<V> {
    fn new() -> Self {
        Self { arena: Slabs::new(), deallocs: Arc::default(), reclaims: Arc::default() }
    }
}

impl<V> NodeAllocator<u64, V, M> for Spy<V> {
    type Exhaustion = core::convert::Infallible;

    fn try_alloc_leaf_uninit(
        &mut self,
    ) -> Result<NonNull<MaybeUninit<Leaf<u64, V, M>>>, Self::Exhaustion> {
        self.arena.try_alloc_leaf_uninit()
    }

    fn try_alloc_inner_uninit(
        &mut self,
    ) -> Result<NonNull<MaybeUninit<Inner<u64, V, M>>>, Self::Exhaustion> {
        self.arena.try_alloc_inner_uninit()
    }

    unsafe fn dealloc_leaf_uninit(&mut self, ptr: NonNull<MaybeUninit<Leaf<u64, V, M>>>) {
        self.deallocs.fetch_add(1, Relaxed);
        // SAFETY: forwarded — the caller's obligations are the arena's.
        unsafe { self.arena.dealloc_leaf_uninit(ptr) }
    }

    unsafe fn dealloc_inner_uninit(&mut self, ptr: NonNull<MaybeUninit<Inner<u64, V, M>>>) {
        self.deallocs.fetch_add(1, Relaxed);
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

    unsafe fn reclaim_all(&mut self) -> bool {
        self.reclaims.fetch_add(1, Relaxed);
        // SAFETY: forwarded — the caller's obligations are the arena's.
        unsafe { self.arena.reclaim_all() }
    }
}

/// An allocator that is honest but ASYMMETRIC: leaf slots come from a
/// slab arena, while every inner node is its own boxed allocation
/// through a counting backing. Wholesale reclaim is therefore
/// impossible — the boxed inners can only be retired one at a time —
/// so `reclaim_all` declines, which is exactly the trait default (no
/// override below).
struct Lopsided {
    leaves: Slabs<u64, u64, M>,
    inners: common::Counting,
}

impl NodeAllocator<u64, u64, M> for Lopsided {
    type Exhaustion = core::convert::Infallible;

    fn try_alloc_leaf_uninit(
        &mut self,
    ) -> Result<NonNull<MaybeUninit<Leaf<u64, u64, M>>>, Self::Exhaustion> {
        self.leaves.try_alloc_leaf_uninit()
    }

    fn try_alloc_inner_uninit(
        &mut self,
    ) -> Result<NonNull<MaybeUninit<Inner<u64, u64, M>>>, Self::Exhaustion> {
        self.inners.try_alloc_inner_uninit()
    }

    unsafe fn dealloc_leaf_uninit(&mut self, ptr: NonNull<MaybeUninit<Leaf<u64, u64, M>>>) {
        // SAFETY: forwarded — the caller's obligations are the arena's.
        unsafe { self.leaves.dealloc_leaf_uninit(ptr) }
    }

    unsafe fn dealloc_inner_uninit(&mut self, ptr: NonNull<MaybeUninit<Inner<u64, u64, M>>>) {
        // SAFETY: forwarded — the caller's obligations are the backing's.
        unsafe { self.inners.dealloc_inner_uninit(ptr) }
    }

    fn leaf_capacity(&self) -> Option<usize> {
        self.leaves.leaf_capacity()
    }

    fn inner_capacity(&self) -> Option<usize> {
        NodeAllocator::<u64, u64, M>::inner_capacity(&self.inners)
    }

    fn leaf_available(&self) -> usize {
        self.leaves.leaf_available()
    }

    fn inner_available(&self) -> usize {
        NodeAllocator::<u64, u64, M>::inner_available(&self.inners)
    }

    // No `reclaim_all` override: declining is the default, and declining
    // is the truth here.
}

/// The teardown shortcut is licensed by ALL of the tree's node memory
/// being wholesale-reclaimable, not some of it: under an allocator
/// whose inner nodes are individually boxed, dropping the tree must
/// return every boxed inner through per-slot retirement — whatever it
/// does about the leaves.
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

/// With drop-free values under a wholesale-reclaiming allocator,
/// dropping the tree must not walk the nodes: zero per-slot
/// retirements — the allocator's own drop reclaims all slot memory.
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
        "dropping a tree of drop-free values under a wholesale-reclaiming allocator \
         must retire zero individual slots"
    );
}

/// With drop-free values under a wholesale-reclaiming allocator,
/// `clear` must release wholesale: zero per-slot retirements, exactly
/// one `reclaim_all` covering both pools, and the tree fresh and
/// serviceable afterward.
#[test]
fn clear_releases_wholesale_for_plain_values() {
    let spy = Spy::<u64>::new();
    let deallocs = Arc::clone(&spy.deallocs);
    let reclaims = Arc::clone(&spy.reclaims);

    let mut tree: BPlusTree<u64, u64, M, Spy<u64>> = BPlusTree::new_in(spy);
    fill(&mut tree, 2_000);
    tree.clear();

    assert_eq!(
        deallocs.load(Relaxed),
        0,
        "clearing a tree of drop-free values under a wholesale-reclaiming allocator \
         must retire zero individual slots"
    );
    assert_eq!(
        reclaims.load(Relaxed),
        1,
        "clear must reset the allocator through exactly one reclaim_all call \
         (it covers both pools)"
    );

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
