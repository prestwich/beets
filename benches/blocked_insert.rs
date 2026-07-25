//! Criterion bench for the target write workload: keys arrive in blocks
//! of `B` sorted inputs (each block internally ascending, block order
//! and placement varying). Two block shapes bracket the design space:
//!
//! - `local`: a block's keys are consecutive — a run walks through one
//!   leaf, then the next. The friendliest case for a last-written-leaf
//!   finger.
//! - `strided`: a block's keys are spread evenly across the whole
//!   keyspace — sorted within the block, but consecutive keys land far
//!   apart. The case where only batch-aware insertion (or level links)
//!   could help.
//!
//! `insert_sequential` (one block of everything) and `insert_shuffled`
//! (blocks of one) on the vs_btreemap board are this workload's two
//! edges.
//!
//! Run with `cargo bench --bench blocked_insert`; smoke-check with
//! `cargo bench --bench blocked_insert -- --test`.

use std::collections::BTreeMap;

use beets::BPlusTree;
use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use sweep_bptree::BPlusTreeMap;

/// The u64 fanout, `512 / (8 + 8)`. The crate const-asserts `M ==
/// K::FANOUT` at node construction, so a drifted literal fails to build.
const M: usize = 32;

/// Total keys per workload. Every block size below divides it, so the
/// generators cover `0..N` exactly once with no ragged tail.
const N: usize = 100_000;

/// Block lengths: how many consecutive inputs arrive already sorted.
const BLOCKS: &[usize] = &[10, 100, 1_000];

/// Deterministic shuffle of `0..n` (no rand dep), matching the other
/// boards' idiom.
fn shuffled(n: usize) -> Vec<u64> {
    let mut ks: Vec<u64> = (0..n as u64).collect();
    ks.sort_by_key(|k| k.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    ks
}

/// Blocks of `block` CONSECUTIVE keys: chop `0..N` into `N / block`
/// ranges, emit the ranges in shuffled order, each range ascending.
fn blocked_local(block: usize) -> Vec<u64> {
    let mut keys = Vec::with_capacity(N);
    for start in shuffled(N / block) {
        let base = start * block as u64;
        keys.extend((0..block as u64).map(|i| base + i));
    }
    assert_workload(&keys, block);
    keys
}

/// Blocks of `block` keys STRIDED across the whole keyspace: block `b`
/// holds `b, b + stride, b + 2*stride, ..` (ascending, `stride = N /
/// block` apart), blocks emitted in shuffled order.
fn blocked_strided(block: usize) -> Vec<u64> {
    let stride = (N / block) as u64;
    let mut keys = Vec::with_capacity(N);
    for b in shuffled(N / block) {
        keys.extend((0..block as u64).map(|i| b + i * stride));
    }
    assert_workload(&keys, block);
    keys
}

/// Generator self-check: exactly the keys `0..N`, each once, and every
/// block internally ascending. Runs once per generation — noise-free.
fn assert_workload(keys: &[u64], block: usize) {
    assert_eq!(keys.len(), N, "workload must hold exactly N keys");
    assert!(
        keys.chunks(block).all(|c| c.windows(2).all(|w| w[0] < w[1])),
        "every block must be internally ascending"
    );
    let mut sorted = keys.to_vec();
    sorted.sort_unstable();
    assert!(sorted.iter().copied().eq(0..N as u64), "workload must cover 0..N exactly once");
}

/// Bench one blocked workload across the three contenders.
fn blocked_group(c: &mut Criterion, name: &str, keys_for: impl Fn(usize) -> Vec<u64>) {
    let mut group = c.benchmark_group(name);
    for &block in BLOCKS {
        let keys = keys_for(block);
        group.throughput(Throughput::Elements(N as u64));
        group.bench_with_input(BenchmarkId::new("beets", block), &keys, |b, keys| {
            b.iter_batched_ref(
                BPlusTree::<u64, u64, M>::new,
                |tree| {
                    for &k in keys {
                        tree.insert(k, k);
                    }
                },
                BatchSize::LargeInput,
            )
        });
        group.bench_with_input(BenchmarkId::new("btreemap", block), &keys, |b, keys| {
            b.iter_batched_ref(
                BTreeMap::<u64, u64>::new,
                |map| {
                    for &k in keys {
                        map.insert(k, k);
                    }
                },
                BatchSize::LargeInput,
            )
        });
        group.bench_with_input(BenchmarkId::new("sweep", block), &keys, |b, keys| {
            b.iter_batched_ref(
                BPlusTreeMap::<u64, u64>::new,
                |map| {
                    for &k in keys {
                        map.insert(k, k);
                    }
                },
                BatchSize::LargeInput,
            )
        });
    }
    group.finish();
}

fn benches(c: &mut Criterion) {
    blocked_group(c, "insert_blocked_local", blocked_local);
    blocked_group(c, "insert_blocked_strided", blocked_strided);
}

criterion_group!(blocked, benches);
criterion_main!(blocked);
