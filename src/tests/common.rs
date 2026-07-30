//! Shared fixtures for the integration pins: the u64 fanout, the
//! counting pass-through allocator, the live-instance counter, and the
//! standard fill traffic. Mounted once from `lib.rs`; each
//! integration-pin module uses its own subset.
#![allow(dead_code)]

use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::{
        Arc,
        atomic::{AtomicIsize, AtomicUsize, Ordering::Relaxed},
    },
};

use crate::{BPlusTree, Key, NodeAllocator};

/// The u64 fanout. The crate const-asserts `M == K::FANOUT` at node
/// construction, so a drifted value fails to build.
pub const M: usize = <u64 as Key>::FANOUT;

/// The key→value mapping every fill uses. Key-derived but distinct, so
/// probes catch key/value drift.
pub fn v(k: u64) -> u64 {
    k * 2
}

/// Insert `0..n` with [`v`]-mapped values. (Infallible only: `insert`
/// is type-gated to allocators that cannot exhaust.)
pub fn fill<A: NodeAllocator<u64, u64, M, Exhaustion = core::convert::Infallible>>(
    tree: &mut BPlusTree<u64, u64, M, A>,
    n: u64,
) {
    for k in 0..n {
        tree.insert(k, v(k));
    }
}

/// Pass-through allocator that tallies every `alloc`/`dealloc` it
/// receives in the two counters it is built over.
///
/// The counters are `&'static` so one type serves every role: point a
/// `const`-constructed instance at `static` counters to register it as
/// the `#[global_allocator]`, or give each test its own pair of local
/// `static`s for isolated per-instance counting (no `Box::leak`, which
/// would trip miri's exit leak check).
#[derive(Clone, Copy)]
pub struct Counting {
    allocs: &'static AtomicUsize,
    frees: &'static AtomicUsize,
}

impl Counting {
    pub const fn new(allocs: &'static AtomicUsize, frees: &'static AtomicUsize) -> Self {
        Self { allocs, frees }
    }

    /// Allocations tallied so far.
    pub fn allocs(&self) -> usize {
        self.allocs.load(Relaxed)
    }

    /// Frees tallied so far.
    pub fn frees(&self) -> usize {
        self.frees.load(Relaxed)
    }

    /// Every allocation drawn through this allocator has been returned
    /// through it. `what` names the traffic being balanced.
    #[track_caller]
    pub fn assert_balanced(&self, what: &str) {
        assert_eq!(
            self.allocs(),
            self.frees(),
            "every {what} allocated through the supplied allocator must be freed \
             through it (positive difference = leak or a free that bypassed it)"
        );
    }
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        self.allocs.fetch_add(1, Relaxed);
        // SAFETY: forwards this method's own contract.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        self.frees.fetch_add(1, Relaxed);
        // SAFETY: forwards this method's own contract; `ptr` came from
        // the matching `alloc` above (same pass-through).
        unsafe { System.dealloc(ptr, layout) }
    }
}

/// Declare a pair of test-local counters and build a [`Counting`] over
/// them. One macro call per test keeps parallel tests isolated.
macro_rules! counting {
    () => {{
        static ALLOCS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        static FREES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        $crate::common::Counting::new(&ALLOCS, &FREES)
    }};
}
pub(crate) use counting;

/// A value that counts live instances: a leak leaves the counter
/// positive, a double-drop drives it negative.
pub struct Counted(#[allow(dead_code)] u64, Arc<AtomicIsize>);

impl Counted {
    pub fn new(x: u64, live: &Arc<AtomicIsize>) -> Self {
        live.fetch_add(1, Relaxed);
        Counted(x, Arc::clone(live))
    }
}

impl Drop for Counted {
    fn drop(&mut self) {
        self.1.fetch_sub(1, Relaxed);
    }
}
