//! Criterion benches: `beets::BPlusTree` against two references —
//! `std::collections::BTreeMap` (the same-semantics baseline everyone
//! holds an ordered map to) and `sweep_bptree::BPlusTreeMap` (another
//! single-threaded in-memory B+tree, the closest design twin).
//!
//! Run with `cargo bench`; smoke-check quickly with `cargo bench -- --test`.
//! HTML reports land in `target/criterion/`.

use core::hint::black_box;
use std::collections::BTreeMap;

use beets::BPlusTree;
use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use sweep_bptree::BPlusTreeMap;

/// The u64 fanout, `512 / (8 + 8)`. The crate const-asserts `M ==
/// K::FANOUT` at node construction, so a drifted literal fails to build.
const M: usize = 32;

const SIZES: &[usize] = &[1_000, 100_000];

/// `n` distinct keys in deterministic shuffled order (no rand dep).
fn shuffled_keys(n: usize) -> Vec<u64> {
    let mut ks: Vec<u64> = (0..n as u64).collect();
    ks.sort_by_key(|k| k.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    ks
}

fn tree_of(keys: &[u64]) -> BPlusTree<u64, u64, M> {
    keys.iter().map(|&k| (k, k)).collect()
}

fn map_of(keys: &[u64]) -> BTreeMap<u64, u64> {
    keys.iter().map(|&k| (k, k)).collect()
}

fn sweep_of(keys: &[u64]) -> BPlusTreeMap<u64, u64> {
    keys.iter().map(|&k| (k, k)).collect()
}

/// Bench one insert workload across all three contenders.
fn insert_group(c: &mut Criterion, name: &str, keys_for: impl Fn(usize) -> Vec<u64>) {
    let mut group = c.benchmark_group(name);
    for &n in SIZES {
        let keys = keys_for(n);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("beets", n), &keys, |b, keys| {
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
        group.bench_with_input(BenchmarkId::new("btreemap", n), &keys, |b, keys| {
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
        group.bench_with_input(BenchmarkId::new("sweep_bptree", n), &keys, |b, keys| {
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

fn bench_insert_sequential(c: &mut Criterion) {
    insert_group(c, "insert_sequential", |n| (0..n as u64).collect());
}

fn bench_insert_shuffled(c: &mut Criterion) {
    insert_group(c, "insert_shuffled", shuffled_keys);
}

/// Bench point lookups across all three contenders. `keys` builds the
/// maps; `probes` is what gets looked up.
fn get_group(c: &mut Criterion, name: &str, keys: &[u64], probes: &[u64]) {
    let n = keys.len();
    let tree = tree_of(keys);
    let map = map_of(keys);
    let sweep = sweep_of(keys);

    let mut group = c.benchmark_group(name);
    group.throughput(Throughput::Elements(n as u64));
    group.bench_with_input(BenchmarkId::new("beets", n), &probes, |b, probes| {
        b.iter(|| {
            for k in *probes {
                black_box(tree.get(black_box(k)));
            }
        })
    });
    group.bench_with_input(BenchmarkId::new("btreemap", n), &probes, |b, probes| {
        b.iter(|| {
            for k in *probes {
                black_box(map.get(black_box(k)));
            }
        })
    });
    group.bench_with_input(BenchmarkId::new("sweep_bptree", n), &probes, |b, probes| {
        b.iter(|| {
            for k in *probes {
                black_box(sweep.get(black_box(k)));
            }
        })
    });
    group.finish();
}

fn bench_get_hit(c: &mut Criterion) {
    for &n in SIZES {
        let keys = shuffled_keys(n);
        get_group(c, "get_hit", &keys, &keys);
    }
}

fn bench_get_miss(c: &mut Criterion) {
    for &n in SIZES {
        // Store the EVEN keys and probe the ODD ones, shuffled: every
        // probe is a miss that lands BETWEEN stored keys, so each
        // descent takes a realistic path instead of walking one cached
        // tree path (which is what probes above the whole range would
        // measure).
        let keys: Vec<u64> = shuffled_keys(n).iter().map(|k| 2 * k).collect();
        let probes: Vec<u64> = shuffled_keys(n).iter().map(|k| 2 * k + 1).collect();
        get_group(c, "get_miss", &keys, &probes);
    }
}

fn bench_remove_shuffled(c: &mut Criterion) {
    let mut group = c.benchmark_group("remove_shuffled");
    for &n in SIZES {
        let keys = shuffled_keys(n);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("beets", n), &keys, |b, keys| {
            b.iter_batched_ref(
                || tree_of(keys),
                |tree| {
                    for k in keys {
                        black_box(tree.remove(k));
                    }
                },
                BatchSize::LargeInput,
            )
        });
        group.bench_with_input(BenchmarkId::new("btreemap", n), &keys, |b, keys| {
            b.iter_batched_ref(
                || map_of(keys),
                |map| {
                    for k in keys {
                        black_box(map.remove(k));
                    }
                },
                BatchSize::LargeInput,
            )
        });
        group.bench_with_input(BenchmarkId::new("sweep_bptree", n), &keys, |b, keys| {
            b.iter_batched_ref(
                || sweep_of(keys),
                |map| {
                    for k in keys {
                        black_box(map.remove(k));
                    }
                },
                BatchSize::LargeInput,
            )
        });
    }
    group.finish();
}

/// The churn test's op mix (60% insert / 40% remove over a small key
/// domain), as a bench: sustained mixed mutation at a steady size.
fn bench_churn(c: &mut Criterion) {
    fn ops(n: usize) -> Vec<(bool, u64)> {
        let mut state: u64 = 0x5EED_CAFE_F00D_D00D;
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 32) % 5 < 3, state % (n as u64 / 4).max(64))
            })
            .collect()
    }

    let mut group = c.benchmark_group("churn");
    for &n in SIZES {
        let ops = ops(n);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("beets", n), &ops, |b, ops| {
            b.iter_batched_ref(
                BPlusTree::<u64, u64, M>::new,
                |tree| {
                    for &(ins, k) in ops {
                        if ins {
                            black_box(tree.insert(k, k));
                        } else {
                            black_box(tree.remove(&k));
                        }
                    }
                },
                BatchSize::LargeInput,
            )
        });
        group.bench_with_input(BenchmarkId::new("btreemap", n), &ops, |b, ops| {
            b.iter_batched_ref(
                BTreeMap::<u64, u64>::new,
                |map| {
                    for &(ins, k) in ops {
                        if ins {
                            black_box(map.insert(k, k));
                        } else {
                            black_box(map.remove(&k));
                        }
                    }
                },
                BatchSize::LargeInput,
            )
        });
        group.bench_with_input(BenchmarkId::new("sweep_bptree", n), &ops, |b, ops| {
            b.iter_batched_ref(
                BPlusTreeMap::<u64, u64>::new,
                |map| {
                    for &(ins, k) in ops {
                        if ins {
                            black_box(map.insert(k, k));
                        } else {
                            black_box(map.remove(&k));
                        }
                    }
                },
                BatchSize::LargeInput,
            )
        });
    }
    group.finish();
}

/// Bench teardown across all three contenders: build from shuffled
/// keys (untimed setup), time only the drop.
fn bench_drop(c: &mut Criterion) {
    let mut group = c.benchmark_group("drop");
    for &n in SIZES {
        let keys = shuffled_keys(n);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("beets", n), &keys, |b, keys| {
            b.iter_batched(|| tree_of(keys), drop, BatchSize::LargeInput)
        });
        group.bench_with_input(BenchmarkId::new("btreemap", n), &keys, |b, keys| {
            b.iter_batched(|| map_of(keys), drop, BatchSize::LargeInput)
        });
        group.bench_with_input(BenchmarkId::new("sweep_bptree", n), &keys, |b, keys| {
            b.iter_batched(|| sweep_of(keys), drop, BatchSize::LargeInput)
        });
    }
    group.finish();
}

/// How many pairs each range scan reads.
const SCAN_LEN: usize = 100;

/// Bench full in-order iteration: sum every key and value. Trees are
/// built from shuffled keys, so nodes sit at realistic occupancy
/// rather than bulk-load-packed.
fn bench_iterate_all(c: &mut Criterion) {
    let mut group = c.benchmark_group("iterate_all");
    for &n in SIZES {
        let keys = shuffled_keys(n);
        let tree = tree_of(&keys);
        let map = map_of(&keys);
        let sweep = sweep_of(&keys);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_function(BenchmarkId::new("beets", n), |b| {
            b.iter(|| {
                let mut acc = 0u64;
                for (k, v) in tree.iter() {
                    acc = acc.wrapping_add(*k).wrapping_add(*v);
                }
                black_box(acc)
            })
        });
        group.bench_function(BenchmarkId::new("btreemap", n), |b| {
            b.iter(|| {
                let mut acc = 0u64;
                for (k, v) in map.iter() {
                    acc = acc.wrapping_add(*k).wrapping_add(*v);
                }
                black_box(acc)
            })
        });
        group.bench_function(BenchmarkId::new("sweep_bptree", n), |b| {
            b.iter(|| {
                let mut acc = 0u64;
                for (k, v) in sweep.iter() {
                    acc = acc.wrapping_add(*k).wrapping_add(*v);
                }
                black_box(acc)
            })
        });
    }
    group.finish();
}

/// Bench short range scans (the index-scan shape): seek a random
/// present key, read the next `SCAN_LEN` pairs. Each iteration runs
/// `n / SCAN_LEN` scans, touching ~`n` pairs total (scans seeded near
/// the top of the keyspace run short; every contender sees the same
/// starts). `sweep_bptree` has no range/seek API, so it sits this
/// group out.
fn bench_range_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("range_scan");
    for &n in SIZES {
        let keys = shuffled_keys(n);
        let starts = &keys[..n / SCAN_LEN];
        let tree = tree_of(&keys);
        let map = map_of(&keys);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("beets", n), &starts, |b, starts| {
            b.iter(|| {
                let mut acc = 0u64;
                for &s in *starts {
                    for (k, v) in tree.range(s..).take(SCAN_LEN) {
                        acc = acc.wrapping_add(*k).wrapping_add(*v);
                    }
                }
                black_box(acc)
            })
        });
        group.bench_with_input(BenchmarkId::new("btreemap", n), &starts, |b, starts| {
            b.iter(|| {
                let mut acc = 0u64;
                for &s in *starts {
                    for (k, v) in map.range(s..).take(SCAN_LEN) {
                        acc = acc.wrapping_add(*k).wrapping_add(*v);
                    }
                }
                black_box(acc)
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_insert_sequential,
    bench_insert_shuffled,
    bench_get_hit,
    bench_get_miss,
    bench_remove_shuffled,
    bench_churn,
    bench_drop,
    bench_iterate_all,
    bench_range_scan,
);
criterion_main!(benches);
