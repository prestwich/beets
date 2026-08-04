# beets

An in-memory B+tree implementation in Rust, with set support.

This project is an experiment in handwriting code, while using Claude for
scaffolding, testing, and benching. I wrote the algorithms, found the bugs,
and made the design decisions.

## Top-line performance

`beets` is optimized primarily for **large in-memory trees**. If your workload
is measured in hundreds or thousands of keys, it is not a good fit, use `std`
instead.

Check `perf.md` for more information. `beets` is comparable-to or
better-than `std::collections::BTreeMap`. This is not exactly apples-to-apples,
as the `std` structure is a B-tree rather than a B+tree.

| bench (100k entries)           | beets       | std  |
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
- `no_std` support (with `alloc`)
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

### `H` -- The Height Const

The **maximum** height for a tree is also a const generic, with a conservative
default provided. During operations that potentially mutate the tree (`insert`,
`remove`) `beets` records the path taken to reach the leaf (or leaves) affected.
This is stored as a stack-local array of length `H`. The default `H` is chosen
based on the machine word size, however this is significantly larger than
necessary for most users, and results in some perf degradation. If you can
statically guarantee that your tree will never exceed some depth, you can set
`H` to that depth. This is particularly useful for trees built using sorted
iterators which are guaranteed to be dense in-memory.

### Allocators

By default, the `BPlusTree` uses a simple arena with slab-allocated space for
inner nodes and leaves. I used this as a fun way to learn how to write a slab
allocator. By default. the `BPlusTree` holds a `Slabs` that wraps the global
allocator. Any `::std::alloc::GlobalAlloc` implementer will work, and custom
allocators can be written via the `SlotAllocator` trait.

The `Slabs` allocator is intended for large trees, and pre-allocates a
significant region of memory. A small tree may prefer the `Global` allocator,
which keeps nodes in individual `Box`es.

### Nodes

Nodes are reached through `Node` — an untagged 8-byte union of pointers to
`Inner` and `Leaf`. There is no runtime tag: which pointee a handle has is
inferred from position. A subtree of height `h > 0` is rooted by an `Inner`; `h
== 0` by a `Leaf`. This is sound because a B+tree is perfectly height-balanced
and its height changes only at the root, so `BPlusTree` stores the one true
height and every descent threads it downward, decrementing per level. Debug
builds additionally tag every node with a kind byte and assert it at
each cast.
