# perf notes

This file is unstructured claude notes.

## what we optimize for

Writes arrive as blocks of N sorted inputs (100 sorted keys, then the
next 100). Reads are uniform random point lookups.

- the write-side design hinges on how wide a key range one block
  spans:
  - keys close together: a last-written-leaf finger (check
    `first_key <= key < next leaf's first_key`, insert in place on a
    hit, maybe chase the sibling once) captures nearly the whole win;
    only block boundaries (1 in N) pay a full descent. Fancier caches
    (path cache, level links) buy ~nothing: the boundary jump is
    arbitrary, so they degenerate exactly when the finger does.
  - block strides the whole keyspace: the finger misses constantly.
    Tools: level links (inner-node sibling pointers; resume rightward
    in O(log stride)) or, better, a sorted-batch insert API — one
    rightward merge-like pass over the leaf chain, exploiting that
    blocks are known-sorted; beats any per-key cache.
  - measure the real block span before building either.
- split policy is the top structural lever: block inserts are
  ascending, so half-splits leave those leaves ~50% full. Splitting at
  the insertion point keeps run-built regions ~full, degrades to
  half-split on random input. Denser leaves = shallower tree; the read
  gap is depth-driven — helps both halves.
- the fanout/NODE_BUDGET re-sweep matters more (depth again).
- last-touched-leaf READ cache stays "probably never": uniform-random
  reads are its worst case.
- blocked-sorted board: benches/blocked_insert.rs — blocks of
  10/100/1000 at 100k keys, `local` (consecutive keys) and `strided`
  (samples the whole keyspace); insert_sequential and insert_shuffled
  are the edge guards.
- characterization (rough, noisy day):
  - local: beets FLAT across block sizes (~4.05ms); btreemap improves
    31% (5.8 -> 4.0), ties us at 1000 — that flatness is the finger's
    opportunity, quantified.
  - strided: beets wins at every size (5.6/5.8/4.4 vs 6.7/6.6/5.6).
    Level links / batch insert would defend a lead, not chase a
    deficit — weak case.
- headline metrics: insert_blocked_local and get_hit/get_miss.

## scoreboard

us vs `std::collections::BTreeMap` vs `sweep_bptree` vs the C++
incumbents (tlx `btree_map`, ex-STX; absl `btree_map`), u64 keys, 100k
elements, as of 2026-07-25. UPDATE THIS TABLE on every rerun.

Rust columns are criterion mid estimates. C++ columns come from
`benches/cpp` (bit-identical key sequences, hand-rolled median-of-25
with criterion-style untimed setup, Apple clang -O3, both containers
at their default 256-byte nodes; build/run one-liners in its
CMakeLists). Same machine, same day, but a different harness —
cross-language deltas under ~5% are not signal.

| bench (100k)                   | beets       | std  | sweep | tlx      | absl     |
| ------------------------------ | ----------- | ---- | ----- | -------- | -------- |
| get_hit                        | **1.56 ms** | 5.94 | 2.13  | 4.43     | 5.29     |
| get_miss                       | **1.63 ms** | 6.02 | 2.11  | 4.38     | 5.24     |
| insert_sequential              | 3.90 ms     | 3.89 | 2.89  | 3.81     | **2.77** |
| insert_shuffled                | **5.55 ms** | 6.80 | 8.65  | 8.10     | 6.76     |
| insert_blocked_local (B=100)   | 4.10 ms     | 4.30 | 4.57  | 3.91     | **3.61** |
| insert_blocked_strided (B=100) | 5.73 ms     | 6.52 | 7.11  | **5.55** | 6.81     |
| remove_shuffled                | **5.23 ms** | 7.05 | 8.84  | 9.94     | 6.88     |
| churn                          | **4.52 ms** | 5.56 | 6.02  | 6.65     | 5.02     |
| drop (µs, shuffled build)      | **8.6 µs**  | 143  | 48.1  | 106      | 75       |
| iterate_all (µs)               | **51.1 µs** | 90.8 | 58.3  | 96       | 129      |
| range_scan (µs, len=100)       | **104 µs**  | 132  | —     | 172      | 139      |

- random point reads (half the target workload): a blowout over the
  whole field — ~3.8x vs std, ~1.4x vs sweep, ~2.8x vs tlx (the
  fastest C++). Vs sweep the lead holds at EVERY size: get_hit 1M
  33.4 ms vs 51.1 (1.5x), 10M 1.05 s vs 1.25 (1.2x); misses match.
  Lead narrows out of cache, never inverts.
- the write side is where the field leads: insert_sequential (absl
  1.4x faster, sweep 1.35x, std a coin flip), insert_blocked_local
  B=100 (absl 12% faster), insert_blocked_strided B=100 (tlx edges us,
  ~3% — inside cross-harness noise). The target workload's other half
  — why the finger + split policy are next.
- shuffled mutation is ours: insert_shuffled, remove_shuffled, and
  churn beat every contender (absl is the nearest on all three).
- iteration is ours on both rows: iterate_all sums the whole tree
  (leaf chain + arena locality; absl, which climbs parent pointers
  instead of a leaf chain, runs 2.5x slower); range_scan seeks a
  random present key and reads 100 pairs, n/100 scans per iteration.
  sweep has no range/seek API and sits range_scan out.
- at 1k everything ties except gets (we win outright) and sweep's
  shuffled insert/remove (bad for them).
- `drop` numbers are for value types **WITHOUT** drop glue, using the
  `Slabs` allocator. This allows freeing an entire slab at a time,
  rather than traversing the tree to free each node individually.

## history

Criterion runs on my machine; only tree-level benches count — slice
microbenches kept contradicting the real tree.

## kept

### branchless linear scan (the big one)

Count the keys below the target; the count is the index. No early
exit, no branch to mispredict; the loop vectorizes.

- gets: 26% faster at u16, 40% u32, 64% u64, 25% u128
- removes: ~18% faster at 100k
- inserts/churn: same
- also tried: u8-accumulator (faster narrow, worse u128 — dropped; one
  impl for all widths), chunked early-exit (won u128
  gets, cost u64 insert/churn 20–30% — dropped), branchless binary
  search (lost everywhere)

### arena allocator (slab)

One arena per node type instead of a Box per node. A/B (2026-07-21,
opt-in arena vs Box): drop 3.5x faster; bulk load 16% faster at 100k;
gets ~4% faster at 1M (nodes closer together); removes 5–8% slower
(free-list bookkeeping).

Made DEFAULT 2026-07-24 (`BPlusTree<K, V, M>` now means `A = Slabs`;
slab memory comes from any `GlobalAlloc`). Switch cost at 100k: gets
+2–3%, insert_sequential −3.5%, churn −2.8%, remove_shuffled +1.7% —
milder on removes than predicted; every win/loss vs std/sweep
unchanged.

BENCH TRAP (2026-07-24 evening): arena_ab's `HeapTree` alias left `A`
off — silently an arena tree, so post-switch runs compared the arena
to itself. Fixed by pinning `Global`; earlier post-switch
numbers invalid. (arena_ab and scan_ab since deleted — settled,
recorded here.)

### wholesale teardown (`OWNS_ALL` fast path)

`SlotAllocator` grew const `OWNS_ALL` (reclaims all slot memory
wholesale on drop) plus a `clear_all` wholesale reset. `Drop`/`clear`
skip the per-node walk when `OWNS_ALL && !needs_drop::<V>()` — an
`if const {}` branch, one codegen path per instantiation; hot paths
untouched (all scoreboard rows within noise).

Drop timings, 2026-07-24 evening (bulk-loaded u64 trees; "walk" = gate
forced off; heap rows moved ≤4% between legs, so the deltas are real):

| n    | boxed walk (Global) | arena walk | arena fast path |
| ---- | ------------------- | ---------- | --------------- |
| 1k   | **367 ns**          | 670 ns     | 645 ns          |
| 100k | 44.5 µs             | 14.8 µs    | **8.6 µs**      |
| 1M   | 570 µs              | 210 µs     | **84 µs**       |
| 10M  | 4.94 ms             | 2.65 ms    | **0.89 ms**     |

- vs the walk: 1.7x at 100k, 2.5x at 1M, 3.0x at 10M — grows with
  depth; the walk chases pointers, the fast path frees O(slabs)
  chunks.
- vs the old boxed drop: 5.2x/6.8x/5.5x at 100k/1M/10M.
- the 1k reversal is real and accepted: ~2 slab frees (64 KiB, ~320 ns
  each) vs ~33 small box frees. Boxed wins under ~a few thousand keys;
  the fast path shaves only ~4% there.
- `clear`: same fast path plus per-pool `clear_all` (keeps the head
  slab); no clear bench — drop numbers are the proxy.
- large sizes: 1k/100k from `arena_drop` (arena_ab.rs, deleted);
  1M/10M from a temporary `arena_drop_large` group (PerIteration,
  sample_size 20), also removed.
- vs_btreemap's permanent `drop` group (scoreboard row): 100k 16.5x vs
  std, 5.6x vs sweep; 1k sweep edges us (537 vs 654 ns) — small-tree
  slab-free cost; 10M ~17x vs std, ~25x vs sweep.

## tried and thrown away

### fused probe (one unconditional load for the hit check)

Clamp the index and always load instead of branching on "real hit".
get_miss 1k ~10% faster; every hit ~2% slower. Misses got cheaper by
making hits pay. Dropped.

### unchecked child indexing (skip bounds checks in the descent)

Reads ~3.5% faster at 100k, but 1k reads and churn 1–3% slower. A
trade that flips sign with depth smells like code-layout luck.
Dropped.

### fixed-shape scans / padded key arrays

Pad unused key slots with the biggest key so the scan reads all M
slots with a constant loop bound — should vectorize better.

What happened:

- reads ~2x SLOWER at u32/u64 (+106–113%), +50% u16, +21% u128
- nearly all in the inner-node scan (3–4 inner scans per get vs 1 leaf
  scan)
- `#[inline(always)]` didn't help
- standalone codegen is beautiful unrolled branchless NEON — it only
  loses in the real tree. Best guess: extra cache lines per node;
  never proved.
- padding was nearly free on writes (+1–4%); the loss was purely scan
  shape
- it dragged in a lazy root (can't pad an empty tree): empty union
  variant, Option accessors, empty-tree checks everywhere; Miri caught
  a leaked root leaf in the drain-to-empty path it created

Everything reverted: prefix scans over live keys, plain MaybeUninit
arrays, eager root leaf. Back to pre-pad numbers.

Recurring lessons:

1. trust only the tree-level readout, vs a fresh baseline with a null
   run (same code twice) for the noise band — ±1–2% quiet, useless if
   anything (Steam) chews a core
2. pretty codegen on the isolated function proves nothing about the
   tree
3. every "make one path cheaper" idea made another path pay — the
   boring branchless prefix scan is undefeated

## still open

- NEXT UP: the last-written-leaf finger (design above). Hits also
  require leaf room; any remove clears the finger; misses take the
  normal descent. Judged on insert_blocked_local; insert_shuffled
  guards miss overhead. Fresh baseline + null run on a quiet machine
  first.
- the fanout/NODE_BUDGET re-sweep — the depth lever for both halves of
  the workload; gets lead sweep at every size (scoreboard), so it's
  defending a lead.
- last-touched-leaf cache for point reads (TODO atop tree.rs, cf.
  sweep's `try_cache`): remember the last-get leaf + key range, skip
  the descent on a hit. Constraints: written under `&self` (sweep uses
  a relaxed atomic node id; our handle is NonNull, a Cell costs
  `Sync`); every structural mutation invalidates; win is
  workload-shaped — big for sequential/skewed probes, ~nothing
  uniform-random. Needs a locality-heavy get workload first, or it
  measures as a flat regression.
- why the fixed scan loses in-tree despite better codegen is
  unexplained; answer that before any retry
