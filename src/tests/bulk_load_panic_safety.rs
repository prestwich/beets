//! Integration pins for `from_sorted_iter`'s panic-safety contract: the
//! source iterator is caller code and may panic between any two pairs.
//! Whatever the load is holding when that happens, unwinding must drop
//! every already-drawn value exactly once — a bulk load must never leak
//! or double-drop on a mid-stream panic.
//!
//! Each test panics the source at one chosen depth and asserts the live
//! counter returns to zero (positive = leak, negative = double-drop).
//! This lives in `tests/` because `catch_unwind` needs `std`.

use crate::common;

use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc,
        atomic::{AtomicIsize, Ordering::Relaxed},
    },
};

use crate::{BPlusTree, Slabs, harness::wide};
use common::{Counted, M};

/// Run a bulk load whose source panics just before yielding pair
/// `panic_at`, catch the unwind, and return the live-value counter —
/// which the contract says must be back at zero.
fn live_after_panic_at(panic_at: u64) -> isize {
    let live = Arc::new(AtomicIsize::new(0));
    let result = catch_unwind(AssertUnwindSafe(|| {
        let _tree: BPlusTree<u64, Counted, M> =
            BPlusTree::from_sorted_iter((0..u64::MAX).map(|k| {
                assert!(k < panic_at, "source iterator panics at pair {k}");
                (k, Counted::new(k, &live))
            }));
    }));
    assert!(result.is_err(), "the fixture must actually panic mid-load");
    live.load(Relaxed)
}

#[track_caller]
fn assert_no_leak(panic_at: u64) {
    assert_eq!(
        live_after_panic_at(panic_at),
        0,
        "a source panic at pair {panic_at} must drop every already-drawn \
         value exactly once (positive = leak, negative = double-drop)"
    );
}

/// A panic while the very first `M` pairs are being drawn must leak
/// nothing.
#[test]
fn a_panic_in_the_first_chunk_of_pairs_leaks_nothing() {
    assert_no_leak(2);
    assert_no_leak(M as u64 - 1);
}

/// A panic while the second `M` pairs are being drawn must leak nothing
/// — including the full chunk drawn before it.
#[test]
fn a_panic_in_the_second_chunk_of_pairs_leaks_nothing() {
    assert_no_leak(M as u64 + 2);
    assert_no_leak(2 * M as u64 - 1);
}

/// A panic while the third `M` pairs are being drawn must leak nothing
/// — including both full chunks drawn before it.
#[test]
fn a_panic_in_the_third_chunk_of_pairs_leaks_nothing() {
    assert_no_leak(2 * M as u64 + 2);
    assert_no_leak(3 * M as u64 - 1);
}

/// A panic deep in a load big enough to be several levels tall must
/// leak nothing: every value drawn across the whole stream — however
/// far the build has progressed — must drop on unwind.
#[test]
fn a_panic_deep_in_a_multi_level_load_leaks_nothing() {
    let (m, m2, m3) = (M as u64, (M * M) as u64, (M * M * M) as u64);
    // Straddle the interesting depths: just past a two-level tree's
    // capacity, mid third level, and just past a full three-level tree.
    assert_no_leak(m2 + m + 2);
    assert_no_leak(2 * m2 + 3 * m + 5);
    assert_no_leak(m3 + m2 + 2);
}

/// Run a bulk load of `n` pairs into a level-capped tree (fanout 3,
/// `H = 2` — two levels, so at most three packed leaves under one
/// root), catch the loader's over-cap refusal, and return the
/// live-value counter — which the contract says must be back at zero.
fn live_after_overcap_load(n: u64) -> isize {
    let live = Arc::new(AtomicIsize::new(0));
    let result = catch_unwind(AssertUnwindSafe(|| {
        let _tree: BPlusTree<[u8; 121], Counted, 3, Slabs<[u8; 121], Counted, 3>, 2> =
            BPlusTree::from_sorted_iter((0..n).map(|k| (wide(k), Counted::new(k, &live))));
    }));
    assert!(result.is_err(), "an over-cap load must panic");
    live.load(Relaxed)
}

/// The loader's own level-cap refusal is a mid-load panic like any
/// other: unwinding must drop every already-drawn value exactly once.
/// Sized so the refusal fires while pairs are still streaming — the
/// climb out of a completed second chunk is what exceeds the cap.
#[test]
fn an_overcap_refusal_mid_stream_leaks_nothing() {
    assert_eq!(
        live_after_overcap_load(54),
        0,
        "an over-cap load must drop every already-drawn value exactly once \
         on unwind (positive = leak, negative = double-drop)"
    );
}

/// As above, but sized so the stream itself fits the treepath and only
/// the finish-time fold — emitting the held-back chunk and its short
/// tail upward — is what would exceed the cap.
#[test]
fn an_overcap_refusal_in_the_finishing_fold_leaks_nothing() {
    assert_eq!(
        live_after_overcap_load(12),
        0,
        "an over-cap load must drop every already-drawn value exactly once \
         on unwind (positive = leak, negative = double-drop)"
    );
}
