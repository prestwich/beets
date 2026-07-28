//! Coverage-guided differential fuzzing: one arbitrary case — a
//! bulk-load seed size, a key-domain width, and an op sequence — is run
//! through the crate's differential harness (`beets::harness`, the same
//! one the proptest properties drive) at the default fanout on both
//! allocators, and at the minimum fanout where trees run deep.
//!
//! The harness asserts agreement with `BTreeMap` at every observable
//! point and throws the full invariant net after every mutation, so any
//! divergence or structural violation is a crash for libFuzzer to
//! minimize.

#![no_main]

use arbitrary::Arbitrary;
use beets::{
    Global, Key, Slabs,
    harness::{Op, run_differential, wide},
};
use libfuzzer_sys::fuzz_target;

const M: usize = <u64 as Key>::FANOUT;

/// One fuzz case. The fuzzer chooses the starting tree (bulk-loaded
/// `seed % 2048` pairs), the key domain (`1..=64` mask bits, so it can
/// explore both collision-dense small domains and sparse full-range
/// keys), and the op sequence.
#[derive(Arbitrary, Debug)]
struct Case {
    seed: u16,
    mask_bits: u8,
    ops: Vec<Op>,
}

fuzz_target!(|case: Case| {
    let seed = u64::from(case.seed % 2048);
    let bits = case.mask_bits % 64 + 1;
    let mask = if bits == 64 { u64::MAX } else { (1u64 << bits) - 1 };
    let ops: Vec<Op> = case.ops.iter().map(|op| op.mask_keys(mask)).collect();

    // Default fanout (M == 32), on both allocators: the slab arena the
    // tree defaults to, and the plain global-allocation path.
    run_differential::<u64, M, Slabs<u64, u64, M>>(|k| k, seed, &ops);
    run_differential::<u64, M, Global>(|k| k, seed, &ops);

    // Minimum fanout (M == 3): the same ops build deep trees whose
    // splits, merges, and borrows cascade through many inner levels.
    run_differential::<[u8; 121], 3, Slabs<[u8; 121], u64, 3>>(wide, seed.min(256), &ops);
});
