# beets

An in-memory B+tree implementation in Rust.

This project is an experiment in handwriting code, while using Claude for
scaffolding, testing, and benching. I wrote the algorithms, found the bugs,
and made the design decisions.

## Top-line performance

Check `perf.md` for more information. `beets` is comparable-to or
better-than `std::collections::BTreeMap`. This is not exactly apples-to-apples,
as the `std` structure is a B-tree rather than a B+tree.

| bench (100k)                   | beets       | std  |
| ------------------------------ | ----------- | ---- |
| get_hit                        | **1.56 ms** | 5.94 |
| get_miss                       | **1.63 ms** | 6.02 |
| insert_sequential              | 3.90 ms     | 3.89 |
| insert_shuffled                | **5.55 ms** | 6.80 |
| insert_blocked_local (B=100)   | **4.10 ms** | 4.30 |
| insert_blocked_strided (B=100) | **5.73 ms** | 6.52 |
| remove_shuffled                | **5.23 ms** | 7.05 |
| churn                          | **4.52 ms** | 5.56 |
| drop (µs, shuffled build)      | **8.6 µs**  | 143  |
| iterate_all (µs)               | **51.1 µs** | 90.8 |
| range_scan (µs, len=100)       | **104 µs**  | 132  |

(this table collected 7/25/26, on my M4 Max Macbook)

## Goals:

- No reachable UB (miri should pass cleanly)
- `no_std` support — with `alloc` for the heap-backed allocators, or
  fully core-only over a fixed memory region
- I hand write the type system and business logic.
- Claude assists by writing
  - tests
  - safety comments
  - docs
  - benches
- Claude **DOES NOT** spoil bugs for me.
  - Claude MAY write very specific tests tho :)

## Design Notes:

### `M` -- The Fanout Const

The fanout for a tree is a const generic. This is a bit annoying, as it leaks
into tree type definitions. I would suggest using a type alias in your library.
The correct `M` is derived from the `K` type of the tree, with a heuristic
assumption about cache line sizes. I'm still benching this and it is not really
stable.

`pub type MyTree<V> = BPlusTree<MyKey, V, { MyKey::FANOUT }>;`

### Allocators

By default, the `BPlusTree` uses a simple arena with slab-allocated space for
inner nodes and leaves. I used this as a fun way to learn how to write a slab
allocator. By default, the `BPlusTree` holds a `Slabs` that wraps the global
allocator. Any `::std::alloc::GlobalAlloc` implementer also works directly as
a node allocator (each node becomes its own heap allocation); custom
allocators implement the `NodeAllocator` trait.

For allocation-free environments there is `FixedNodes`, an arena over a
caller-declared `NodeStorage` region (typically a `static`). It is the one
allocator whose exhaustion is a reportable error rather than an abort, which
is what the tree's `try_insert` / `try_new_in` / `try_from_sorted_iter_in`
surface exists for: on failure they hand back exactly what you gave them —
the key-value pair, or the allocator — and the tree is untouched. Building
with `--no-default-features` (no `alloc`) leaves the fixed arena as the only
allocator and the crate fully core-only.

### Nodes

Nodes are reached through `Node` — an untagged 8-byte union of pointers to
`Inner` and `Leaf`. There is no runtime tag: which pointee a handle has is
inferred from position. A subtree of height `h > 0` is rooted by an `Inner`; `h
== 0` by a `Leaf`. This is sound because a B+tree is perfectly height-balanced
and its height changes only at the root, so `BPlusTree` stores the one true
height and every descent threads it downward, decrementing per level. Debug
builds additionally tag every node with a kind byte and assert it at
each cast.
