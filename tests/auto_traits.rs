//! Pins the tree's `Send` and `Sync` contracts: a tree whose key,
//! value, and allocator types all carry the trait carries it too — for
//! the default `Slabs` arena and for the boxed-per-node `Global`
//! allocator alike. A sent tree is fully usable (read, mutate, drop)
//! on the receiving thread; a shared tree serves concurrent readers.
//!
//! The negative directions (a non-`Send` constituent denies `Send`; a
//! `Send`-but-not-`Sync` constituent denies `Sync`) are pinned as
//! `compile_fail` doctests on `BPlusTree` itself.

mod common;

use beets::{BPlusTree, Global, Slabs};
use common::{M, fill, v};

/// Compile-time pin: both stock allocator configurations yield a
/// `Send` tree over `Send` keys and values.
#[test]
fn send_holds_for_stock_allocators() {
    fn require_send<T: Send>() {}

    require_send::<BPlusTree<u64, u64, M>>();
    require_send::<BPlusTree<u64, u64, M, Slabs<u64, u64, M>>>();
    require_send::<BPlusTree<u64, u64, M, Global>>();
    require_send::<BPlusTree<u64, String, M>>();
}

/// Compile-time pin: both stock allocator configurations yield a
/// `Sync` tree over `Sync` keys and values.
#[test]
fn sync_holds_for_stock_allocators() {
    fn require_sync<T: Sync>() {}

    require_sync::<BPlusTree<u64, u64, M>>();
    require_sync::<BPlusTree<u64, u64, M, Slabs<u64, u64, M>>>();
    require_sync::<BPlusTree<u64, u64, M, Global>>();
    require_sync::<BPlusTree<u64, String, M>>();
}

/// A shared tree must serve concurrent readers: several threads
/// holding the same `&BPlusTree` at once each see the complete,
/// correct contents through point reads and full scans.
#[test]
fn shared_tree_serves_concurrent_readers() {
    // Enough traffic for a multi-level tree; trimmed under miri, where
    // the point is the aliasing/race check, not volume.
    const N: u64 = if cfg!(miri) { 300 } else { 10_000 };
    const READERS: usize = 4;

    let mut tree: BPlusTree<u64, u64, M> = BPlusTree::new();
    fill(&mut tree, N);
    let tree = &tree;

    std::thread::scope(|s| {
        for r in 0..READERS as u64 {
            s.spawn(move || {
                // Stagger each reader's probe order so the threads
                // overlap on different nodes at any given moment.
                for i in 0..N {
                    let k = (i + r * (N / READERS as u64)) % N;
                    assert_eq!(
                        tree.get(&k),
                        Some(&v(k)),
                        "a shared tree must serve every reader's point reads (key {k})"
                    );
                }
                assert_eq!(
                    tree.iter().count(),
                    N as usize,
                    "a shared tree must serve full scans to every reader"
                );
            });
        }
    });
}

/// A tree built on one thread must be fully usable on another: reads
/// see everything inserted before the move, mutation works, and
/// teardown happens on the receiving thread.
#[test]
fn sent_tree_is_usable_and_droppable_on_receiving_thread() {
    // Enough traffic to grow several slabs and a few tree levels;
    // trimmed under miri, where the point is the aliasing check, not
    // volume.
    const N: u64 = if cfg!(miri) { 500 } else { 10_000 };

    let mut tree: BPlusTree<u64, u64, M> = BPlusTree::new();
    fill(&mut tree, N);

    std::thread::spawn(move || {
        for k in 0..N {
            assert_eq!(
                tree.get(&k),
                Some(&v(k)),
                "sent tree must serve reads for key {k} on the receiving thread"
            );
        }

        tree.insert(N, v(N));
        assert_eq!(
            tree.get(&N),
            Some(&v(N)),
            "sent tree must accept inserts on the receiving thread"
        );

        drop(tree);
    })
    .join()
    .expect("receiving thread must complete without panicking");
}
