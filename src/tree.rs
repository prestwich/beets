use core::{mem::MaybeUninit, ptr::NonNull};

use crate::allocator::{Global, NodeAllocator, SlotAllocator};
use crate::{Inner, Key, Leaf, MAX_LEVELS, Slabs, iter};

// TODO:
// - perf: last-touched-leaf cache for point reads (cf. sweep_bptree's
//   `try_cache`): remember the leaf the previous get landed at together
//   with its key-range bounds, and let a probe that falls inside the
//   range skip the descent entirely. Design questions to settle first:
//   the cache must be written under `&self` (sweep uses a relaxed
//   AtomicUsize node id; our handle is a NonNull, and a Cell would cost
//   `Sync`); every structural mutation must invalidate or re-validate
//   it; and the payoff is workload-shaped — big for sequential/skewed
//   key streams, ~nil for uniform-random probes (benches would need a
//   locality-heavy get workload to see it at all).

/// Debug-only node-kind tag. In debug builds it is the first field of both
/// [`Leaf`] and [`Inner`] (which are `repr(C)` there), so the cast accessors on
/// [`Node`] can soundly read it through an erased pointer — before knowing
/// which type they point at — and assert the height-inferred kind against
/// reality. In release builds the tag, the asserts, and their cost do not
/// exist.
#[cfg(debug_assertions)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NodeKind {
    Leaf,
    Inner,
}

/// An untyped handle to a heap-allocated tree node.
///
/// # The depth-type invariant
///
/// There is no tag: which field is live is inferred from position in the
/// tree. A `Node` rooting a subtree of height `h` points at an [`Inner`] if
/// `h > 0` and a [`Leaf`] if `h == 0`. This is sound because the tree is
/// perfectly height-balanced — every leaf sits at the same depth — and the
/// height changes at exactly one place: the root.
///
/// Every `unsafe` method on this type takes a `height` parameter, and its
/// safety contract is the same sentence: **`height` must be the true height
/// of the subtree rooted at this node.** A wrong height reinterprets one
/// node type as the other.
///
/// # Ownership
///
/// A `Node` owns its subtree, but has no drop glue: dropping the handle
/// silently leaks the subtree. Teardown is explicit, via
/// [`Node::drop_subtree`].
pub(crate) union Node<K: Key, V, const M: usize> {
    inner: NonNull<Inner<K, V, M>>,
    leaf: NonNull<Leaf<K, V, M>>,
}

/// Delegate methods on [`Node`] to the leaf that owns the key, threading the
/// subtree height down the descent: at `height == 0` the node is cast to
/// [`Leaf`] and the call terminates there, so each listed method must exist on
/// `Leaf` with the written signature; at `height > 0` the node is cast to
/// [`Inner`], routed to the child owning the key, and the call recurses on
/// `Node` at `height - 1`. Doc comments and other attributes carry over.
///
/// The first argument must be the routing key (`key: &K`) — it is what the
/// inner arm routes on.
///
/// The generated method is `unsafe`: `height` must be the true height of the
/// subtree rooted at this node, per the type's safety contract. A wrong
/// height reinterprets one pointee type as the other.
///
/// Methods whose signature mentions the pointee's own kind (e.g. `merge`,
/// which consumes a same-kind sibling) can't be delegated this way — the
/// same-kind argument or result needs erasing into a `Node`. Those are
/// written out by hand below.
macro_rules! delegate {
    () => {};
    (
        $(#[$attr:meta])*
        fn $name:ident(&self, $key:ident: &K $(, $arg:ident: $ty:ty)*) -> $ret:ty;
        $($rest:tt)*
    ) => {
        $(#[$attr])*
        ///
        /// # Safety
        ///
        /// `height` must be the height of the subtree rooted at this node.
        pub(crate) unsafe fn $name(&self, height: u8, $key: &K $(, $arg: $ty)*) -> $ret {
            if height == 0 {
                // SAFETY: height 0 ⇒ this node is a leaf (caller vouches
                // for `height`).
                unsafe { self.as_leaf().$name($key $(, $arg)*) }
            } else {
                // SAFETY: height > 0 ⇒ this node is inner (caller vouches
                // for `height`), and the child it routes to roots a subtree
                // of height - 1.
                unsafe { self.as_inner().$name(height, $key $(, $arg)*) }
            }
        }
        delegate! { $($rest)* }
    };
    (
        $(#[$attr:meta])*
        fn $name:ident(&mut self, $key:ident: &K $(, $arg:ident: $ty:ty)*) -> $ret:ty;
        $($rest:tt)*
    ) => {
        $(#[$attr])*
        ///
        /// # Safety
        ///
        /// `height` must be the height of the subtree rooted at this node.
        pub(crate) unsafe fn $name(&mut self, height: u8, $key: &K $(, $arg: $ty)*) -> $ret {
            if height == 0 {
                // SAFETY: height 0 ⇒ this node is a leaf (caller vouches
                // for `height`).
                unsafe { self.as_leaf_mut().$name($key $(, $arg)*) }
            } else {
                // SAFETY: height > 0 ⇒ this node is inner (caller vouches
                // for `height`), and the child it routes to roots a subtree
                // of height - 1.
                unsafe {
                    self.as_inner_mut().$name(height, $key $(, $arg)*)
                }
            }
        }
        delegate! { $($rest)* }
    };
}

/// Generate the typed views of the erased handle for one pointee kind —
/// the shared borrow, the exclusive borrow, and the by-value conversion
/// (which retires the node's slot to the allocator; every free in the
/// crate funnels through it). Every accessor checks the pointee's
/// debug-only kind tag before trusting the caller's cast; all three
/// share one safety contract, stated once here: **the caller must
/// ensure `self` points at the named kind** — it is not enforced by the
/// type system.
macro_rules! cast_accessors {
    ($Kind:ident, $field:ident: $as_ref:ident, $as_mut:ident, $into:ident) => {
        /// # Safety
        ///
        /// The caller must ensure that `self` is a
        #[doc = concat!("`", stringify!($Kind), "`.")]
        /// This is not enforced by the type system.
        #[inline(always)]
        #[track_caller]
        unsafe fn $as_ref(&self) -> &$Kind<K, V, M> {
            #[cfg(debug_assertions)]
            self.assert_kind(NodeKind::$Kind);
            // SAFETY: this accessor's contract — the caller vouches the
            // pointee is a live node of this kind (debug-asserted above);
            // the shared borrow of the handle keeps it valid.
            unsafe { self.$field.as_ref() }
        }

        /// # Safety
        ///
        /// The caller must ensure that `self` is a
        #[doc = concat!("`", stringify!($Kind), "`.")]
        /// This is not enforced by the type system.
        #[inline(always)]
        #[track_caller]
        unsafe fn $as_mut(&mut self) -> &mut $Kind<K, V, M> {
            #[cfg(debug_assertions)]
            self.assert_kind(NodeKind::$Kind);
            // SAFETY: this accessor's contract — the caller vouches the
            // pointee is a live node of this kind (debug-asserted above);
            // the handle owns its subtree, so the exclusive borrow of the
            // handle makes the pointee exclusively reachable.
            unsafe { self.$field.as_mut() }
        }

        /// # Safety
        ///
        /// The caller must ensure that `self` is a
        #[doc = concat!("`", stringify!($Kind), "`,")]
        /// which is not enforced by the type system, and that `alloc` is
        /// the allocator this node came from.
        #[track_caller]
        unsafe fn $into<A: SlotAllocator<$Kind<K, V, M>>>(self, alloc: &mut A) -> $Kind<K, V, M> {
            #[cfg(debug_assertions)]
            self.assert_kind(NodeKind::$Kind);
            // SAFETY: the pointer came from `alloc` (caller vouches) and
            // `self` is consumed — no other path to this slot remains.
            unsafe { alloc.deallocate(self.$field) }
        }
    };
}

/// Dispatch a niladic method on [`Node`] to whichever pointee `height`
/// implies: a single cast — no routing key, no recursion, no threading
/// of `height` down a descent (that is [`delegate!`]'s business). Each
/// listed method must exist on both [`Leaf`] and [`Inner`] with the written
/// signature. Doc comments and other attributes carry over.
///
/// The generated method is `unsafe`: `height` must be the true height
/// of the subtree rooted at this node, per the type's safety contract.
macro_rules! dispatch {
    () => {};
    (
        $(#[$attr:meta])*
        fn $name:ident(&self) -> $ret:ty;
        $($rest:tt)*
    ) => {
        $(#[$attr])*
        ///
        /// # Safety
        ///
        /// `height` must be the height of the subtree rooted at this node.
        pub(crate) unsafe fn $name(&self, height: u8) -> $ret {
            if height == 0 {
                // SAFETY: height 0 ⇒ this node is a leaf (caller vouches
                // for `height`).
                unsafe { self.as_leaf().$name() }
            } else {
                // SAFETY: height > 0 ⇒ this node is inner (caller vouches
                // for `height`).
                unsafe { self.as_inner().$name() }
            }
        }
        dispatch! { $($rest)* }
    };
}

impl<K: Key, V, const M: usize> Node<K, V, M> {
    pub(crate) fn from_inner<A: SlotAllocator<Inner<K, V, M>>>(
        inner: Inner<K, V, M>,
        alloc: &mut A,
    ) -> Self {
        Self { inner: alloc.allocate(inner) }
    }

    /// Wrap an already-boxed leaf: the right sibling handed back by
    /// [`Leaf::insert_at`] on split.
    pub(crate) fn from_leaf_ptr(leaf: NonNull<Leaf<K, V, M>>) -> Self {
        Self { leaf }
    }

    /// Wrap an already-boxed inner node: the right sibling handed back by
    /// [`Inner::splitting_insert_child`] on split.
    pub(crate) fn from_inner_ptr(inner: NonNull<Inner<K, V, M>>) -> Self {
        Self { inner }
    }

    /// Read the debug-only kind tag through the erased pointer.
    #[cfg(debug_assertions)]
    fn kind(&self) -> NodeKind {
        // SAFETY: in debug builds both pointee types are repr(C) with
        // NodeKind as their first field, so this read is valid whichever
        // type the handle actually points at. (Union fields share offset
        // 0; which field we read the pointer from is immaterial.)
        unsafe { *self.leaf.as_ptr().cast::<NodeKind>() }
    }

    /// Debug-only teeth for the depth-type invariant: every cast accessor
    /// checks the height-inferred kind against the pointee's tag.
    #[cfg(debug_assertions)]
    #[track_caller]
    fn assert_kind(&self, expected: NodeKind) {
        assert_eq!(
            self.kind(),
            expected,
            "depth-type invariant violated: cast to {expected:?}, but the pointee's tag \
             disagrees — a wrong height was passed down this call path"
        );
    }

    cast_accessors! { Leaf, leaf: as_leaf, as_leaf_mut, into_leaf }
    cast_accessors! { Inner, inner: as_inner, as_inner_mut, into_inner }

    delegate! {
        /// Get a reference to a value in the subtree rooted at this node, if
        /// it is present.
        ///
        /// Test-only — the production descent is iterative; fixtures in
        /// the node-layer tests read subtrees through it.
        #[cfg(test)]
        fn get(&self, key: &K) -> Option<&V>;
    }

    /// Remove a key from the subtree rooted at this node, if it exists.
    ///
    /// Test-only — the production descent is iterative; the
    /// node-layer tests drive `rebalance` through it.
    ///
    /// (Hand-written rather than a `delegate!` entry: removal can merge
    /// nodes, so it threads the allocator, and the macro has no slot for
    /// a generic parameter.)
    ///
    /// # Safety
    ///
    /// `height` must be the height of the subtree rooted at this node.
    #[cfg(test)]
    pub(crate) unsafe fn remove<A: NodeAllocator<K, V, M>>(
        &mut self,
        height: u8,
        key: &K,
        alloc: &mut A,
    ) -> Option<V> {
        if height == 0 {
            // SAFETY: height 0 ⇒ this node is a leaf (caller vouches
            // for `height`).
            unsafe { self.as_leaf_mut().remove(key) }
        } else {
            // SAFETY: height > 0 ⇒ this node is inner (caller vouches
            // for `height`), and the child it routes to roots a subtree
            // of height - 1.
            unsafe { self.as_inner_mut().remove(height, key, alloc) }
        }
    }

    dispatch! {
        /// The occupancy of THIS node — not the entry count of the subtree
        /// under it (that's `BPlusTree.len`, per the decided note above):
        /// key/value pairs for a leaf, children for an inner. This is the
        /// number the parent reads after a recursive `remove` returns, to
        /// decide whether the child underflowed and needs merging.
        fn len(&self) -> usize;

        /// True if this node is below the minimum occupancy the classical-
        /// rebalancing invariant demands of a NON-ROOT node of its kind
        /// (pairs for a leaf, children for an inner) — the parent's rebalance
        /// trigger after a recursive `remove` returns. The root is exempt and
        /// must not be judged by this.
        fn is_deficient(&self) -> bool;
    }

    /// Fold `other` — the immediate right sibling of `self` under their
    /// shared parent — into `self`, consuming and freeing `other`'s node.
    /// `sep` is the separator that sat between the two in the parent; its
    /// fate diverges by level, which is why it travels with the merge: at
    /// the leaf level it is discarded (the leaf keys carry the data, and
    /// `K: Copy` leaves nothing to drop), at inner levels it is demoted
    /// into the merged node's key gap.
    ///
    /// The caller (the parent, on `remove`'s underflow path) remains
    /// responsible for its own bookkeeping: removing `other`'s child slot
    /// and the separator from its arrays.
    ///
    /// (Hand-written rather than a `delegate!` entry: it consumes a second
    /// node of the same kind by value, and there is no routing key and no
    /// recursion — `height` is consumed by a single cast.)
    ///
    /// # Safety
    ///
    /// - `height` must be the true height of the subtrees rooted at BOTH
    ///   nodes (as same-parent siblings they necessarily share it).
    /// - `other` must be `self`'s immediate right sibling under the same
    ///   parent, with `sep` the separator between them: every key in
    ///   `self`'s subtree is `< sep`, every key in `other`'s is `>= sep`.
    /// - `other` must own its subtree; ownership of its contents transfers
    ///   to `self`, its node is freed here, and no other handle may be used
    ///   to reach it afterward.
    /// - The merged occupancy must fit; see [`Leaf::merge`] and
    ///   [`Inner::merge`] for each level's precise bound (each level's
    ///   remaining preconditions apply through this call).
    pub(crate) unsafe fn merge<A: NodeAllocator<K, V, M>>(
        &mut self,
        height: u8,
        sep: K,
        other: Self,
        alloc: &mut A,
    ) {
        if height == 0 {
            // SAFETY: height 0 ⇒ both nodes are leaves (caller vouches for
            // `height`, shared by both siblings). `other` is moved out of
            // and its slot reclaimed by `into_leaf`; `Leaf::merge`'s
            // remaining preconditions are the caller's, forwarded. The
            // separator is discarded — nothing to drop, `K: Copy`.
            unsafe { self.as_leaf_mut().merge(other.into_leaf(alloc)) }
        } else {
            // SAFETY: height > 0 ⇒ both nodes are inner (caller vouches for
            // `height`, shared by both siblings). `other` is moved out of
            // and its slot reclaimed by `into_inner`; `Inner::merge`'s
            // remaining preconditions are the caller's, forwarded.
            unsafe { self.as_inner_mut().merge(sep, other.into_inner(alloc)) }
        }
    }

    /// Borrow one unit from `right`, `self`'s immediate right sibling
    /// under their shared parent — a pair at the leaf level, a child at
    /// inner levels. Returns the replacement separator, which the caller
    /// writes over the old one in place. `sep` is the current separator
    /// between the pair: consumed (demoted) by the inner rotation, unused
    /// at the leaf level, where the new separator is minted from the
    /// donor's own contents.
    ///
    /// (Hand-written rather than a `delegate!` entry: it takes a second
    /// node of the same kind, and `height` is consumed by a single cast —
    /// no routing key, no recursion.)
    ///
    /// # Safety
    ///
    /// - `height` must be the true height of the subtrees rooted at BOTH
    ///   nodes (as same-parent siblings they necessarily share it).
    /// - `right` must be `self`'s immediate right sibling under the same
    ///   parent, with `sep` the separator between them; each level's
    ///   remaining preconditions ([`Leaf::steal_from_right`],
    ///   [`Inner::rotate_from_right`]) apply through this call.
    pub(crate) unsafe fn steal_from_right(&mut self, height: u8, sep: K, right: &mut Self) -> K {
        if height == 0 {
            // The separator is unused at this level — nothing to drop,
            // `K: Copy`.
            let _ = sep;
            // SAFETY: height 0 ⇒ both nodes are leaves (caller vouches
            // for `height`, shared by the pair).
            unsafe { self.as_leaf_mut().steal_from_right(right.as_leaf_mut()) }
        } else {
            // SAFETY: height > 0 ⇒ both nodes are inner (caller vouches
            // for `height`, shared by the pair).
            unsafe { self.as_inner_mut().rotate_from_right(sep, right.as_inner_mut()) }
        }
    }

    /// Mirror of [`Node::steal_from_right`]: borrow one unit from `left`,
    /// `self`'s immediate LEFT sibling, `sep` being the separator between
    /// `left` and `self`.
    ///
    /// # Safety
    ///
    /// As [`Node::steal_from_right`], mirrored; each level's remaining
    /// preconditions ([`Leaf::steal_from_left`],
    /// [`Inner::rotate_from_left`]) apply through this call.
    pub(crate) unsafe fn steal_from_left(&mut self, height: u8, sep: K, left: &mut Self) -> K {
        if height == 0 {
            // The separator is unused at this level.
            let _ = sep;
            // SAFETY: height 0 ⇒ both nodes are leaves (caller vouches
            // for `height`, shared by the pair).
            unsafe { self.as_leaf_mut().steal_from_left(left.as_leaf_mut()) }
        } else {
            // SAFETY: height > 0 ⇒ both nodes are inner (caller vouches
            // for `height`, shared by the pair).
            unsafe { self.as_inner_mut().rotate_from_left(sep, left.as_inner_mut()) }
        }
    }

    /// Tear down the subtree rooted at this node: free every descendant
    /// node, dropping the values they hold, exactly once. `Node` has no
    /// drop glue — a handle that goes out of scope without passing through
    /// this method leaks its whole subtree.
    ///
    /// # Safety
    ///
    /// - `height` must be the height of the subtree rooted at this node.
    /// - This handle must own the subtree: it must not have been torn down
    ///   already, and no other handle may be used to reach it afterward.
    /// - No leaf outside this subtree may hold a sibling link into it: the
    ///   caller must have spliced the leaf chain past this subtree first (or
    ///   be tearing down everything that links into it, as whole-tree drop
    ///   does). Teardown never touches `next`, so a stale inbound link
    ///   dangles silently until iteration walks into it.
    ///
    /// # Panic during teardown
    ///
    /// NOT panic-safe. A value [`Drop`] that unwinds partway through abandons
    /// every node and value not yet reached in this subtree — they leak. This
    /// is also the path taken by [`BPlusTree`]'s `Drop`, so a panic escaping it
    /// during an already-unwinding drop double-panics and aborts.
    pub(crate) unsafe fn drop_subtree<A: NodeAllocator<K, V, M>>(self, height: u8, alloc: &mut A) {
        if height == 0 {
            // SAFETY:
            // - `height` is 0, so `self` is a `Leaf`.
            // - `self` is moved out of, so no other code can access it.
            unsafe {
                drop(self.into_leaf(alloc));
            }
        } else {
            // SAFETY:
            // - `height` > 0, so `self` is an `Inner`.
            unsafe {
                self.into_inner(alloc).drop_subtree(height, alloc);
            }
        }
    }

    /// Test-only: recursively verify the structural invariants of the
    /// subtree rooted at this node, panicking with a description on the
    /// first violation. Per node it checks: occupancy bounds (a non-root
    /// leaf holds `Leaf::MIN_OCCUPANCY..=M` pairs, a non-root inner
    /// `Inner::MIN_OCCUPANCY..=M` children; the root is exempt down to 0
    /// pairs / 2 children), strict key ordering, separator correctness
    /// (child `i`'s keys `< keys[i] <= ` child `i + 1`'s keys), and that
    /// the leaf chain links consecutive children's leaves in order.
    ///
    /// Returns the subtree's key range (`None` only for an empty root
    /// leaf) and its first and last leaves, so a caller can check the
    /// chain across sibling subtrees and the terminal `next` itself.
    ///
    /// # Safety
    ///
    /// `height` must be the height of the subtree rooted at this node.
    #[cfg(test)]
    #[allow(clippy::type_complexity)]
    #[track_caller]
    pub(crate) unsafe fn check_invariants(
        &self,
        height: u8,
        is_root: bool,
    ) -> (Option<(K, K)>, NonNull<Leaf<K, V, M>>, NonNull<Leaf<K, V, M>>) {
        if height == 0 {
            // SAFETY: height 0 ⇒ leaf (caller vouches for `height`).
            let leaf = unsafe { self.as_leaf() };
            let keys = leaf.test_keys();
            assert!(leaf.len() <= M, "a leaf holds at most M pairs, found {}", leaf.len());
            assert!(
                is_root || leaf.len() >= Leaf::<K, V, M>::MIN_OCCUPANCY,
                "a non-root leaf must hold at least MIN_OCCUPANCY ({}) pairs, found {}",
                Leaf::<K, V, M>::MIN_OCCUPANCY,
                leaf.len()
            );
            assert!(keys.windows(2).all(|w| w[0] < w[1]), "leaf keys must be strictly sorted");
            let range = (!keys.is_empty()).then(|| (keys[0], *keys.last().unwrap()));
            let ptr = NonNull::from(leaf);
            (range, ptr, ptr)
        } else {
            // SAFETY: height > 0 ⇒ inner (caller vouches for `height`).
            let inner = unsafe { self.as_inner() };
            let n = inner.len();
            assert!(n <= M, "an inner holds at most M children, found {n}");
            let min = if is_root { 2 } else { Inner::<K, V, M>::MIN_OCCUPANCY };
            assert!(
                n >= min,
                "an inner must hold at least {min} children (root exempt down to 2), found {n}"
            );
            let keys = inner.test_keys();
            assert!(keys.windows(2).all(|w| w[0] < w[1]), "inner keys must be strictly sorted");

            let mut first_leaf = None;
            let mut prev_last: Option<NonNull<Leaf<K, V, M>>> = None;
            let mut range: Option<(K, K)> = None;
            for (i, child) in inner.test_children().iter().enumerate() {
                // SAFETY: the children of a height-`height` inner root
                // subtrees of `height - 1`, and are never the root.
                let (child_range, first, last) =
                    unsafe { child.check_invariants(height - 1, false) };

                if let Some(prev) = prev_last {
                    // SAFETY: the previous child's last leaf is live.
                    assert_eq!(
                        unsafe { prev.as_ref() }.next(),
                        Some(first),
                        "the leaf chain must link child {} to child {i}",
                        i - 1
                    );
                }
                first_leaf.get_or_insert(first);
                prev_last = Some(last);

                let (lo, hi) =
                    child_range.expect("non-root nodes have occupancy > 0, hence a key range");
                if i > 0 {
                    assert!(
                        keys[i - 1] <= lo,
                        "child {i}'s keys must be >= the separator on its left"
                    );
                }
                if i < n - 1 {
                    assert!(hi < keys[i], "child {i}'s keys must be < the separator on its right");
                }
                range = match range {
                    None => Some((lo, hi)),
                    Some((rlo, _)) => Some((rlo, hi)),
                };
            }
            (range, first_leaf.unwrap(), prev_last.unwrap())
        }
    }
}

/// A B+Tree.
///
/// The fanout `M` must be exactly [`Key::FANOUT`] for `K` — trees are
/// instantiated as `BPlusTree<K, V, { K::FANOUT }>`, and a mismatched
/// `M` must be rejected at compile time, where the tree's nodes are
/// born:
///
/// ```compile_fail
/// // <u64 as beets::Key>::FANOUT is not 7 — this must not compile.
/// let tree: beets::BPlusTree<u64, u64, 7> = beets::BPlusTree::new();
/// ```
///
/// The tree is [`Send`] exactly when its parts are; a non-`Send`
/// constituent must deny it:
///
/// ```compile_fail
/// // Rc is not Send — a tree holding Rc values must not be either.
/// fn require_send<T: Send>() {}
/// require_send::<beets::BPlusTree<u64, core::rc::Rc<u8>, { <u64 as beets::Key>::FANOUT }>>();
/// ```
///
/// [`Sync`] likewise — and `Send` parts alone must not make a
/// shareable tree:
///
/// ```compile_fail
/// // Cell is Send but not Sync — a tree of Cell values must not be Sync.
/// fn require_sync<T: Sync>() {}
/// require_sync::<beets::BPlusTree<u64, core::cell::Cell<u8>, { <u64 as beets::Key>::FANOUT }>>();
/// ```
pub struct BPlusTree<K: Key, V, const M: usize, A: NodeAllocator<K, V, M> = Slabs<K, V, M, Global>>
{
    // The handle IS the pointer; boxing it would be double indirection.
    root: Node<K, V, M>,
    height: u8,
    len: usize,

    // Declared last so any by-value teardown order is values-first; the
    // real guarantee is `Drop`, which walks the tree through `&mut self
    // .allocator` before the field itself drops.
    allocator: A,
}

// SAFETY: sending the tree sends exclusive ownership of everything it
// reaches. The `NonNull`s that suppress the auto-impl (the root
// union's node pointers and the leaf chain) all target nodes this tree
// allocated from its own `allocator` field and never shares: no node
// is reachable from two trees, and every alias the crate creates
// (descents, iterators, entries) is borrow-bound, so none outlives a
// move. Nothing in the tree is tied to its birth thread; moving it
// moves the nodes' `K`/`V` payloads and the allocator along with it,
// which is exactly what the three `Send` bounds sign for.
unsafe impl<K, V, const M: usize, A> Send for BPlusTree<K, V, M, A>
where
    K: Key + Send,
    V: Send,
    A: NodeAllocator<K, V, M> + Send,
{
}

// SAFETY: sharing `&BPlusTree` shares a read-only tree. Every `&self`
// method is a pure read of the node graph — descents, gets, iteration;
// none mutates node memory through the `NonNull`s — and no `&self`
// path can reach the allocator: [`SlotAllocator`]'s receivers are
// `&mut self`, unreachable through a shared borrow, and nothing hands
// out `&A`. The tree itself has no interior mutability, so while
// shared borrows exist, no thread can write anything a reader
// dereferences. What readers DO reach — `&K`s and `&V`s — is what the
// `Sync` bounds sign for (`A: Sync` is defensive; no `&self` path
// reads it today).
//
// Every future `&self` feature re-signs this contract; a
// `&self`-written cache (the leaf-cache TODO atop this file) is the
// standing example of what would break it.
unsafe impl<K, V, const M: usize, A> Sync for BPlusTree<K, V, M, A>
where
    K: Key + Sync,
    V: Sync,
    A: NodeAllocator<K, V, M> + Sync,
{
}

/// Walk from the root down to a leaf, choosing one child per inner
/// level: the loop body runs exactly `$tree.height` times, so by the
/// depth-type invariant the node in hand is an [`Inner`] on every
/// iteration and a [`Leaf`] after the last — each cast inside is justified
/// by the tree layer's height invariant (see the impl block below).
/// `ref`/`mut` picks the borrow flavor; `$inner` names the current inner
/// node inside `$pick`, which must evaluate to the child to descend
/// into.
/// The recorded path of one root-to-leaf descent: for each inner level
/// `h`, `path[h]` holds the visited node and the child index routed
/// through. Slots outside the descended range stay uninitialized.
type TreePath<K, V, const M: usize> = [MaybeUninit<(NonNull<Node<K, V, M>>, usize)>; MAX_LEVELS];

macro_rules! descend {
    ($tree:expr, ref |$inner:ident| $pick:expr) => {{
        let mut node = &$tree.root;
        for _ in 0..$tree.height {
            // SAFETY: if we're in this loop, the node sits above the leaf
            // level (height invariant).
            let $inner = unsafe { node.as_inner() };
            node = $pick;
        }
        // SAFETY: `height` levels below the root is the leaf level.
        unsafe { node.as_leaf() }
    }};
    ($tree:expr, mut |$inner:ident| $pick:expr) => {{
        let mut node = &mut $tree.root;
        for _ in 0..$tree.height {
            // SAFETY: if we're in this loop, the node sits above the leaf
            // level (height invariant).
            let $inner = unsafe { node.as_inner_mut() };
            node = $pick;
        }
        // SAFETY: `height` levels below the root is the leaf level.
        unsafe { node.as_leaf_mut() }
    }};
}

/// One recorded descent for a key: the tree it walked, the [`path`](Descent::path) it
/// recorded, the leaf it landed at, and the key's slot there. The state
/// shared by the two commit halves ([`Descent::commit_insert`],
/// [`Descent::commit_remove`]) and, above them, the entry API
/// (`entry.rs`): search once, then mutate through what the search
/// recorded — no re-traversal.
///
/// # Validity
///
/// A descent is a bundle of raw pointers into the tree. It is valid only
/// under the `&mut` borrow of the tree that [`BPlusTree::descend_into`]
/// derived it from, and only until the tree is next mutated — a commit
/// half ends it (they take `&mut self` purely to avoid moving the
/// recorded path; a committed descent must not be used again). Until then, the tree may be reached through the
/// descent's pointers ONLY. In Stacked Borrows terms: every pointer here
/// descends from `tree`, and any fresh `&mut` retag of the tree — a
/// re-derived path from the root, or the tree passing a method boundary
/// by `&mut` — pops the whole family. That is why the commit halves live
/// on `Descent` (not on [`BPlusTree`], whose `&mut self` at the call
/// boundary would be exactly such a retag), and why holders carry the
/// originating borrow as `PhantomData<&'a mut BPlusTree>` rather than a
/// live reference field, which would be retagged at each of THEIR
/// method boundaries.
pub(crate) struct Descent<K: Key, V, const M: usize, A: NodeAllocator<K, V, M> = Global> {
    /// The descended tree, as the raw handle every other pointer here
    /// descends from. The commit halves reach the tree's own fields
    /// through this: the scalars (`height`, `len`) via raw projections
    /// disjoint from every recorded pointer's pointee, the `allocator`
    /// (shared, never written) likewise, and the `root` slot only after
    /// the path's last use.
    tree: NonNull<BPlusTree<K, V, M, A>>,
    /// The recorded path (see [`TreePath`]); slots `1..=tree.height` are
    /// initialized. Private to this module: only the commit halves
    /// replay it.
    path: TreePath<K, V, M>,
    /// The leaf the descent landed at.
    pub(crate) leaf: NonNull<Leaf<K, V, M>>,
    /// `key`'s slot in that leaf, per [`Leaf::find_key`]: the match if
    /// `exact`, else the insertion point.
    pub(crate) partition: usize,
    /// True iff `partition` holds exactly the sought key.
    pub(crate) exact: bool,
}

// The commit halves: the back ends of `insert` and `remove`, run
// against a recorded descent. They live on `Descent` rather than on
// `BPlusTree` deliberately — taking `&mut self` on the tree at this
// boundary would freshly retag it and invalidate every pointer the
// descent recorded (see `Descent`'s validity rules). The tree's own
// fields are reached through the descent's raw handle instead: the
// scalars (`height`, `len`) via projections disjoint from every
// recorded pointer's pointee, and the `root` slot only after the
// path's last use.
impl<K: Key + Ord, V, const M: usize, A: NodeAllocator<K, V, M>> Descent<K, V, M, A> {
    /// Commit a replacement, consuming the descent.
    ///
    /// Safety:
    ///
    /// The caller must ensure that `self.exact` is set.
    #[inline]
    pub(crate) unsafe fn commit_replace(&mut self, val: V) -> V {
        debug_assert!(self.exact);
        // Present: replace the value in place — no structural
        // change, no len change.
        let mut leaf = self.leaf;
        // SAFETY: `exact` vouches that `partition` holds `key`.
        let slot = unsafe { leaf.as_mut() }.val_mut_unchecked(self.partition);
        core::mem::replace(slot, val)
    }

    /// The back half of insertion, for a key known ABSENT: insert
    /// `key`/`val` at the recorded slot, replay the path upward to
    /// house any splits, grow the root if one escapes it, and count the
    /// new pair in `len`.
    ///
    /// Returns the inserted value's slot — final by the time this
    /// returns: nodes never move on the heap, and the upward replay
    /// moves child HANDLES between inner nodes, never leaf contents. The
    /// pointer may be dereferenced for as long as the `&mut` borrow this
    /// descent lives under lasts and the tree is not otherwise touched.
    ///
    /// # Safety
    ///
    /// - This descent came from `BPlusTree::descend(&key)`, the `&mut`
    ///   borrow it was derived under is still live, and the tree has not
    ///   been mutated since.
    /// - `self.exact` is false: `key` is absent, and `self.partition`
    ///   is its insertion point in `self.leaf`.
    /// - The commit ends the descent: `self` must not be used again
    ///   after this call.
    // `&mut self`, not `self`, and the path read through it in place —
    // although a commit logically consumes the descent, taking it by
    // value compiles to a 1 KiB move at every call boundary (and
    // destructuring `path` out of it to another). `always` because LLVM
    // declines a plain `#[inline]` hint at this size.
    #[inline(always)]
    pub(crate) unsafe fn commit_insert(&mut self, key: K, val: V) -> NonNull<V> {
        let tree = self.tree.as_ptr();
        let partition = self.partition;
        debug_assert!(!self.exact, "commit_insert requires a vacant descent");

        // SAFETY: as the scalar-access note below — the allocator field is
        // disjoint from every recorded pointer's pointee, so this exclusive
        // reference aliases none of the node mutations below. It is also
        // the commit's only allocator borrow — every helper that allocates
        // or frees receives a reborrow of it — so its exclusivity holds
        // for its whole life.
        let alloc = unsafe { &mut (*tree).allocator };

        // SAFETY: the descent's leaf is live and exclusively reachable
        // under the borrow the caller vouches for.
        let (val_ptr, split) = unsafe { self.leaf.as_mut() }.insert_at(partition, key, val, alloc);

        // If we have a split, we need to insert it into the parent node.
        // We do this by replaying the recorded path from the bottom up.

        let mut split = split.map(|new_child| {
            // SAFETY: `insert_at` hands back a valid, initialized right
            // leaf, and this is its only handle.
            (*unsafe { new_child.as_ref() }.first_key(), Node::from_leaf_ptr(new_child))
        });

        let mut height = 0;
        // SAFETY: for `(*tree).height`, and `len` below — the borrow the
        // caller vouches for keeps the tree alive and exclusive, and the
        // scalar fields are disjoint from every recorded pointer's
        // pointee, so these raw accesses never invalidate the path.
        while height < unsafe { (*tree).height } as usize
            && let Some((sep, new_child)) = split
        {
            // SAFETY: the descent initialized path slots
            // `1..=tree.height` with valid inner nodes.
            let (mut node, child_idx) = unsafe { self.path[height + 1].assume_init() };

            // SAFETY: node is owned by this tree. We have exclusive access to
            // this tree.
            let node = unsafe { node.as_mut().as_inner_mut() };
            split = node
                .insert_child(child_idx, sep, new_child, alloc)
                .map(|(sep, new_inner)| (sep, Node::from_inner_ptr(new_inner)));

            height += 1;
        }

        // If we still have a split, we need to insert it into the root.
        // If the root is full, we need to increase tree height. This is
        // the commit's only touch of the root slot, after the path's
        // last use.
        if let Some((sep, node)) = split {
            // SAFETY: the caller's `&mut` borrow keeps the tree alive and
            // exclusive. The old root moves into the new one exactly once
            // (read, then the slot is overwritten), and the scalar
            // `height` bump is disjoint from every recorded pointee.
            unsafe {
                let old_root = core::ptr::read(&raw const (*tree).root);
                let inner = Inner::from_pair(sep, old_root, node);
                (&raw mut (*tree).root).write(Node::from_inner(inner, alloc));
                (*tree).height += 1;
            }
        }

        // The key was absent (the caller vouches), so this is a new pair.
        // SAFETY: see the scalar-access note above.
        unsafe { (*tree).len += 1 };

        val_ptr
    }

    /// The back half of removal, for a key known PRESENT: remove the
    /// pair at the recorded slot, replay the path upward repairing
    /// whatever the removal left deficient, hoist a lone-child root, and
    /// discount the pair from `len`. Returns the removed pair (the key
    /// comes back out, not just the value).
    ///
    /// # Safety
    ///
    /// - This descent came from `BPlusTree::descend(key)`, the `&mut`
    ///   borrow it was derived under is still live, and the tree has not
    ///   been mutated since.
    /// - `self.exact` is true: `self.partition` holds exactly the
    ///   sought key in `self.leaf`.
    /// - The commit ends the descent: `self` must not be used again
    ///   after this call.
    // `&mut self`, not `self`, and the path read through it in place —
    // although a commit logically consumes the descent, taking it by
    // value compiles to a 1 KiB move at every call boundary (and
    // destructuring `path` out of it to another). `always` because LLVM
    // declines a plain `#[inline]` hint at this size.
    #[inline(always)]
    pub(crate) unsafe fn commit_remove(&mut self) -> (K, V) {
        let tree = self.tree.as_ptr();
        debug_assert!(self.exact, "commit_remove requires an occupied descent");

        // SAFETY: as the scalar-access note below — the allocator field is
        // disjoint from every recorded pointer's pointee, so this exclusive
        // reference aliases none of the node mutations below. It is also
        // the commit's only allocator borrow — every helper that allocates
        // or frees receives a reborrow of it — so its exclusivity holds
        // for its whole life.
        let alloc = unsafe { &mut (*tree).allocator };

        // SAFETY: the descent's leaf is live and exclusively reachable
        // under the borrow the caller vouches for.
        let leaf = unsafe { self.leaf.as_mut() };
        let pair = leaf.remove_at(self.partition);

        // Replay the recorded path upward: repair while the removal (or
        // the repair one level down — a merge shrinks the parent) left
        // the current level's node deficient. The loop stops at the
        // root, which is exempt. NB: a merge at the level below may have
        // freed `leaf`'s node; it is not touched past this read.
        let mut deficient = leaf.is_deficient();
        let mut height = 0;
        // SAFETY: for `(*tree).height` here and `root`/`len` below — the
        // borrow the caller vouches for keeps the tree alive and
        // exclusive; the scalar fields are disjoint from every recorded
        // pointer's pointee, and the root slot is touched only after the
        // path's last use.
        while height < unsafe { (*tree).height } as usize && deficient {
            // SAFETY: the descent initialized path slots
            // `1..=tree.height` with valid inner nodes; height is
            // propagated correctly.
            unsafe {
                let (mut parent, child_idx) = self.path[height + 1].assume_init();
                let parent = parent.as_mut();
                parent.as_inner_mut().rebalance(height as u8 + 1, child_idx, alloc);
                deficient = parent.is_deficient(height as u8 + 1);
            }
            height += 1;
        }

        // If the root is deficient, that's okay. If it's a single entry, we
        // collapse.
        // SAFETY:
        // height invariant: `(*tree).height` is the height of the root's
        // subtree. The ptr reads are safe because the root slot is owned
        // data, read exactly once and overwritten with its replacement.
        unsafe {
            // if there's exactly 1 child of the root, decompose
            if (*tree).height > 0 && (*tree).root.len((*tree).height) == 1 {
                let new_root =
                    core::ptr::read(&raw const (*tree).root).into_inner(alloc).into_only_child();
                (&raw mut (*tree).root).write(new_root);
                (*tree).height -= 1;
            }
        }

        // The key was present (the caller vouches), so a pair left.
        // SAFETY: see the scalar-access note above.
        unsafe { (*tree).len -= 1 };

        pair
    }
}

impl<K: Key, V, const M: usize, A: NodeAllocator<K, V, M>> Drop for BPlusTree<K, V, M, A> {
    // Panic during teardown: NOT panic-safe. Teardown walks the tree dropping
    // values as it goes; a value `Drop` that unwinds leaks every node and
    // value not yet reached. Because this is itself `Drop` glue, such a panic
    // during an already-unwinding drop double-panics and aborts.
    fn drop(&mut self) {
        if const { Self::CAN_SKIP_DROP } {
            return;
        }

        // SAFETY:
        // - `root` is a node.
        // - `height` is maintained as the exact height of `root`'s subtree
        //   (it changes only where the root grows or shrinks).
        // - `root` is read exactly once and never touched again: the tree is
        //   mid-drop and `Node` has no drop glue of its own.
        // - The walk finishes before `self.allocator` itself drops — the
        //   values-first teardown order the allocator contract demands.
        unsafe { core::ptr::read(&self.root).drop_subtree(self.height, &mut self.allocator) }
    }
}

// The type-level invariant every method below signs: `height` is exactly
// the height of `root`'s subtree, and `len` is exactly the number of pairs
// in it. `height` changes in exactly two places — `insert`'s root grow and
// `remove`'s root shrink — and every unsafe `Node` call justifies its
// height argument by this invariant.
impl<K: Key + Ord, V, const M: usize, A: NodeAllocator<K, V, M>> BPlusTree<K, V, M, A> {
    /// A heuristic max height.
    pub const MAX_HEIGHT: usize = ((usize::BITS - 2) / M.div_ceil(2).ilog(2)) as usize;

    pub const CAN_SKIP_DROP: bool = !core::mem::needs_drop::<V>()
        && <A as SlotAllocator<Leaf<K, V, M>>>::OWNS_ALL
        && <A as SlotAllocator<Inner<K, V, M>>>::OWNS_ALL;

    /// Creates a tree whose root is a single empty leaf.
    pub fn new() -> Self
    where
        A: Default,
    {
        Self::new_in(A::default())
    }

    /// As [`Self::new`], but allocating nodes from `allocator` for the
    /// tree's whole life.
    pub fn new_in(mut allocator: A) -> Self {
        let root = Node::from_leaf_ptr(allocator.allocate(Leaf::new(None)));
        Self { root, height: 0, len: 0, allocator }
    }

    /// The number of key/value pairs in the tree.
    pub fn len(&self) -> usize {
        self.len
    }

    /// True if the tree holds no pairs.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get a reference to the first leaf of the tree.
    pub(crate) fn first_leaf(&self) -> &Leaf<K, V, M> {
        descend!(self, ref |inner| &inner.children_ref()[0])
    }

    /// Get a mutable reference to the first leaf of the tree.
    pub(crate) fn first_leaf_mut(&mut self) -> &mut Leaf<K, V, M> {
        descend!(self, mut |inner| &mut inner.children_mut()[0])
    }

    /// Get a reference to the last leaf of the tree.
    pub(crate) fn last_leaf(&self) -> &Leaf<K, V, M> {
        descend!(self, ref |inner| inner.children_ref().last().expect("no empty inner nodes"))
    }

    /// Find the leaf whose range contains the key. That leaf may or may not
    /// contain a value at that key
    pub(crate) fn find_leaf(&self, key: &K) -> &Leaf<K, V, M> {
        descend!(self, ref |inner| inner.child_for_key(key))
    }

    /// Find the leaf whose range contains the key. That leaf may or may not
    /// contain a value at that key
    pub(crate) fn find_leaf_mut(&mut self, key: &K) -> &mut Leaf<K, V, M> {
        descend!(self, mut |inner| inner.child_for_key_mut(key))
    }

    /// Get a reference to the value for `key`, if it is present.
    pub fn get(&self, key: &K) -> Option<&V> {
        self.find_leaf(key).get(key)
    }

    /// Get a mutable reference to the value for `key`, if it is present.
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        self.find_leaf_mut(key).get_mut(key)
    }

    /// True if `key` is present.
    pub fn contains_key(&self, key: &K) -> bool {
        self.get(key).is_some()
    }

    /// Descend from the root to the leaf whose range contains `key`,
    /// recording the path — each visited inner node and the child index
    /// routed through — in `path[h]` for every inner level `h`, and
    /// returning the leaf. The shared front half of [`insert`](Self::insert) (which
    /// replays the path upward to adopt splits) and [`remove`](Self::remove) (which
    /// replays it to rebalance whatever the removal left deficient);
    /// levels at and below the leaf are left untouched.
    #[inline]
    #[track_caller]
    fn descend_recording(
        &mut self,
        key: &K,
        path: &mut TreePath<K, V, M>,
    ) -> NonNull<Node<K, V, M>> {
        let mut height = self.height as usize;
        let mut node = NonNull::from_mut(&mut self.root);

        while height > 0 {
            // SAFETY: height is non-0, the pointer is valid. Root is always ok.
            let n = unsafe { node.as_mut().as_inner_mut() };
            let child_idx = n.child_idx_for_key(key);
            path[height].write((node, child_idx));

            node = NonNull::from_mut(&mut n.children_mut()[child_idx]);

            height -= 1;
        }

        node
    }

    /// As [`Self::descend_recording`], but building the descent in caller-provided
    /// storage — the way in for the plain ops. Returning a [`Descent`] by
    /// value compiles to a 1 KiB copy per call (Rust guarantees no NRVO),
    /// which is fine for the entry API, whose entry stores the descent by
    /// value anyway, and pure loss for [`insert`](Self::insert)/[`remove`](Self::remove), which commit it
    /// on the spot.
    // `always` because LLVM declines a plain `#[inline]` hint at this
    // size, and out of line every caller pays call-boundary stack
    // traffic for the slot.
    #[inline(always)]
    #[track_caller]
    pub(crate) fn descend_into<'s>(
        &mut self,
        key: &K,
        slot: &'s mut MaybeUninit<Descent<K, V, M, A>>,
    ) -> &'s mut Descent<K, V, M, A> {
        let mut tree = NonNull::from_mut(self);

        // Initialize field by field, in place: writing a whole `Descent`
        // value into the slot would be exactly the copy this out-param
        // exists to avoid. `path` is `MaybeUninit` slots and needs no
        // writes.
        let ptr = slot.as_mut_ptr();
        // SAFETY: `ptr` is the caller's `MaybeUninit` slot — the field
        // projections are in bounds, and raw writes into uninitialized
        // storage need no prior validity.
        unsafe {
            (&raw mut (*ptr).tree).write(tree);
            (&raw mut (*ptr).leaf).write(NonNull::dangling());
            (&raw mut (*ptr).partition).write(0);
            (&raw mut (*ptr).exact).write(false);
        }
        // SAFETY: every always-initialized field was written above;
        // `path`'s slots are `MaybeUninit` and carry no validity
        // requirement.
        let descent = unsafe { slot.assume_init_mut() };

        // Re-derive the tree reference from the raw handle the descent
        // will carry, so every pointer recorded below descends from
        // `tree` — NOT from the `&mut self` at this function's boundary,
        // whose tag family dies at the next `&mut self` method call (see
        // the field docs on `Descent::tree`).
        // SAFETY: `tree` is `self`: live and exclusively borrowed.
        let mut node = unsafe { tree.as_mut() }.descend_recording(key, &mut descent.path);

        // SAFETY: `descend_recording` ran the descent to the bottom, so
        // the node in hand is the leaf (height invariant).
        let leaf = unsafe { node.as_mut().as_leaf_mut() };
        (descent.partition, descent.exact) = leaf.probe(key);
        descent.leaf = NonNull::from_mut(leaf);

        descent
    }

    /// Insert a key-value pair, returning the previous value if the key was
    /// already present.
    pub fn insert(&mut self, key: K, val: V) -> Option<V> {
        let mut slot = MaybeUninit::uninit();
        let descent = self.descend_into(&key, &mut slot);

        if descent.exact {
            // SAFETY:
            //
            // `descent.exact`` is set
            return Some(unsafe { descent.commit_replace(val) });
        }

        // SAFETY: the descent is fresh from `descend` under this borrow,
        // the tree untouched since, and `exact` is false.
        unsafe { descent.commit_insert(key, val) };
        None
    }

    /// Remove `key`, returning its value if it was present.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        let mut slot = MaybeUninit::uninit();
        let descent = self.descend_into(key, &mut slot);
        if !descent.exact {
            return None;
        }

        // SAFETY: the descent is fresh from `descend` under this borrow,
        // the tree untouched since, and `exact` is true.
        Some(unsafe { descent.commit_remove() }.1)
    }

    /// Drop every pair, resetting to the empty tree.
    pub fn clear(&mut self) {
        if const { Self::CAN_SKIP_DROP } {
            // SAFETY:
            // A::OWNS_ALL is checked by CAN_SKIP_DROP
            // We immediately invalidate the tree by overwriting the root
            // We kinow that V has no drop glue (checked by CAN_SKIP_DROP
            unsafe {
                SlotAllocator::<Leaf<K, V, M>>::clear_all(&mut self.allocator);
                SlotAllocator::<Inner<K, V, M>>::clear_all(&mut self.allocator);
            }
        } else {
            // SAFETY:
            // - `height` is the impl-block invariant — exactly the
            // height of `root`'s subtree. `root` is overwritten with a fresh
            // node immediately below, before anything can read it.
            unsafe {
                let tree = core::ptr::read(&self.root);
                tree.drop_subtree(self.height, &mut self.allocator);
            }
        }

        self.root = Node::from_leaf_ptr(self.allocator.allocate(Leaf::new(None)));
        self.height = 0;
        self.len = 0;
    }

    /// Assemble a tree directly from its parts — the bulk loader's
    /// (`bulk.rs`) way in, since the fields are private to this module.
    ///
    /// # Safety
    ///
    /// The caller signs this impl block's invariant: `height` must be
    /// exactly the height of `root`'s subtree, and `len` exactly the
    /// number of pairs in it. A wrong height reinterprets node types
    /// throughout the tree (see [`Node`]); a wrong `len` misreports but
    /// is not unsound. Additionally, every node of `root`'s subtree must
    /// have been allocated from `allocator`.
    pub(crate) unsafe fn from_parts(
        root: Node<K, V, M>,
        height: u8,
        len: usize,
        allocator: A,
    ) -> Self {
        Self { root, height, len, allocator }
    }

    /// Iterate over all KV pairs.
    pub fn iter<'a>(&'a self) -> iter::FullIterator<'a, K, V, M> {
        iter::FullIterator::new(self)
    }

    /// Iterate over the pairs whose keys fall in `range`, in ascending
    /// key order.
    pub fn range<'a, R: core::ops::RangeBounds<K>>(&'a self, range: R) -> iter::Range<'a, K, V, M> {
        iter::Range::new(self, range)
    }

    /// Iterate over key/value pairs with mutable values.
    pub fn iter_mut<'a>(&'a mut self) -> iter::FullIteratorMut<'a, K, V, M> {
        iter::FullIteratorMut::new(self)
    }

    /// Iterate over the pairs whose keys fall in `range`, in ascending
    /// key order, with mutable values.
    pub fn range_mut<'a, R: core::ops::RangeBounds<K>>(
        &'a mut self,
        range: R,
    ) -> iter::RangeMut<'a, K, V, M> {
        iter::RangeMut::new(self, range)
    }

    /// The minimum-key pair, or `None` if the tree is empty.
    pub fn first_key_value(&self) -> Option<(&K, &V)> {
        // Only the empty tree's root leaf is empty; any other first
        // leaf holds its subtree's minimum at index 0.
        let leaf = self.first_leaf();
        (leaf.len() > 0).then(|| leaf.kv_ref_unchecked(0))
    }

    /// The maximum-key pair, or `None` if the tree is empty.
    pub fn last_key_value(&self) -> Option<(&K, &V)> {
        let leaf = self.last_leaf();
        leaf.len().checked_sub(1).map(|last| leaf.kv_ref_unchecked(last))
    }

    /// Iterate over the keys, in ascending order.
    pub fn keys(&self) -> impl core::iter::Iterator<Item = &K> {
        self.iter().map(|(k, _)| k)
    }

    /// Iterate over the values, in ascending key order.
    pub fn values(&self) -> impl core::iter::Iterator<Item = &V> {
        self.iter().map(|(_, v)| v)
    }

    /// Iterate over the values mutably, in ascending key order.
    pub fn values_mut(&mut self) -> impl core::iter::Iterator<Item = &mut V> {
        self.iter_mut().map(|(_, v)| v)
    }
}

impl<K: Key + Ord, V, const M: usize, A: NodeAllocator<K, V, M> + Default> Default
    for BPlusTree<K, V, M, A>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V, const M: usize, A: NodeAllocator<K, V, M>> core::fmt::Debug for BPlusTree<K, V, M, A>
where
    K: Key + core::fmt::Debug,
    V: core::fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl<K: Key + Ord, V, const M: usize, A: NodeAllocator<K, V, M> + Default> FromIterator<(K, V)>
    for BPlusTree<K, V, M, A>
{
    /// Builds through the bulk loader, not an insert loop: collect, sort,
    /// dedup, then [`BPlusTree::from_sorted_iter`]. The loaded tree is
    /// fully packed (every leaf at `M` pairs, up to the tail) where
    /// repeated [`insert`](Self::insert)s settle around ~2/3 occupancy — denser and
    /// shallower for the same pairs.
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let mut pairs: alloc::vec::Vec<(K, V)> = iter.into_iter().collect();
        // Stable sort, so duplicate keys stay in arrival order for the
        // dedup below.
        pairs.sort_by_key(|pair| pair.0);
        // `from_sorted_iter` demands strictly ascending keys. Collapse
        // each duplicate run to one pair with `insert`'s overwrite
        // semantics — the first-arrived key, the last-arrived value.
        // (`dedup_by` keeps the first element of a run and passes the
        // later one on the LEFT; the swap walks the newest value into
        // the kept slot.)
        pairs.dedup_by(|later, kept| {
            let dup = later.0 == kept.0;
            if dup {
                core::mem::swap(&mut later.1, &mut kept.1);
            }
            dup
        });
        Self::from_sorted_iter(pairs)
    }
}

impl<K: Key + Ord, V, const M: usize, A: NodeAllocator<K, V, M>> Extend<(K, V)>
    for BPlusTree<K, V, M, A>
{
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        for (key, val) in iter {
            self.insert(key, val);
        }
    }
}

#[cfg(test)]
impl<K: Key + Ord, V, const M: usize, A: NodeAllocator<K, V, M>> BPlusTree<K, V, M, A> {
    /// Test-only, for tests outside this module (which cannot reach the
    /// private fields): the full-strength invariant net — the structural
    /// walk ([`Node::check_invariants`]) plus the two facts only this
    /// layer can vouch for: `len` equals the pairs actually on the
    /// chain, and the chain terminates at the last leaf.
    pub(crate) fn check(&self) {
        // SAFETY: `height` is the impl-block invariant — exactly the
        // height of `root`'s subtree.

        if self.is_empty() {
            return;
        }

        let (_, first, last) = unsafe { self.root.check_invariants(self.height, true) };

        let mut total = 0;
        let mut hops = 0;
        let mut cur = Some(first);
        while let Some(ptr) = cur {
            hops += 1;
            assert!(hops <= self.len + 1, "the leaf chain must terminate within the tree's size");
            // SAFETY: every leaf on a valid tree's chain is live.
            let leaf = unsafe { ptr.as_ref() };
            total += leaf.len();
            cur = leaf.next();
        }
        assert_eq!(total, self.len, "tree.len must equal the pairs actually on the chain");
        // SAFETY: `last` is the walk's final live leaf.
        assert_eq!(
            unsafe { last.as_ref() }.next(),
            None,
            "the tree's last leaf must terminate the chain"
        );
    }
}

#[cfg(test)]
mod tests {
    //! Contract tests for the tree layer, plus the height-0 teardown
    //! contract for `Node` (heights 1 and 2 are pinned alongside the
    //! `Inner` tests).
    //!
    //! The `BPlusTree` tests come in two flavors. Hand-built-fixture tests
    //! assemble trees directly through the private fields (this module can)
    //! and pin single behaviors: routing, `len` bookkeeping, root shrink.
    //! Public-API tests drive everything through `new`/`insert`/`remove`/
    //! `get` and are RED until the whole stack under them lands — they are
    //! the specification for `insert`'s grow path and `remove`'s shrink
    //! path, capped by the model-mirroring churn test. `check_tree` is the
    //! full-strength net: the structural walk (`Node::check_invariants`)
    //! plus the two facts only this layer can vouch for — `len` equals the
    //! pairs on the chain, and the chain terminates at the last leaf.

    use alloc::sync::Arc;
    use core::sync::atomic::{AtomicIsize, Ordering::Relaxed};

    use super::*;
    use crate::test_util::{Counted, IMIN, LMIN, M, counted_leaf, minimal_inner, v, xorshift};

    /// Tearing down a height-0 node must drop the leaf it owns — observed
    /// through the leaf's values dropping exactly once.
    #[test]
    fn drop_subtree_at_height_zero_drops_the_leafs_values_exactly_once() {
        let live = Arc::new(AtomicIsize::new(0));

        let mut leaf: Leaf<u64, Counted, M> = Leaf::new(None);
        for k in 0..3 {
            leaf.raw_append(k, Counted::new(k, &live));
        }
        let node: Node<u64, Counted, M> = Node::from_leaf_ptr(Global.allocate(leaf));
        assert_eq!(live.load(Relaxed), 3, "one live value per stored key before teardown");

        // SAFETY: `node` roots a single leaf (height 0) and owns it.
        unsafe { node.drop_subtree(0, &mut Global) };
        assert_eq!(
            live.load(Relaxed),
            0,
            "drop_subtree(0) must drop the leaf's every value exactly once \
             (positive = leak, negative = double-drop)"
        );
    }

    // ── tree-layer fixtures and the full-strength invariant net ────────

    /// Full-tree invariant check: delegates to [`BPlusTree::check`], the
    /// cross-module net (the structural walk plus the tree-layer
    /// bookkeeping only `BPlusTree` can vouch for). Kept as a free
    /// function so this module's many call sites read unchanged.
    fn check_tree<K: Key + Ord, V, const N: usize, A: NodeAllocator<K, V, N>>(
        tree: &BPlusTree<K, V, N, A>,
    ) {
        tree.check()
    }

    // ── BPlusTree: hand-built fixtures ─────────────────────────────────

    /// A fresh tree is empty: zero pairs, misses on every read, and a
    /// clean teardown.
    #[test]
    fn new_tree_is_empty_and_reads_miss() {
        let tree: BPlusTree<u64, u64, M> = BPlusTree::new();
        assert_eq!(tree.len(), 0, "a new tree holds no pairs");
        assert!(tree.is_empty());
        assert!(tree.get(&0).is_none(), "a new tree must miss on any key");
        assert!(!tree.contains_key(&0));
        check_tree(&tree);
    }

    /// Reads route through a hand-built height-2 tree: hits return the
    /// right values, misses return None, and nothing is disturbed.
    #[test]
    fn get_routes_through_a_hand_built_height_two_tree() {
        let live = Arc::new(AtomicIsize::new(0));
        let (b, b_first) = minimal_inner(IMIN, 1_000_000, &live, None);
        let (a, _) = minimal_inner(IMIN, 0, &live, Some(b_first));
        let len = 2 * IMIN * LMIN;
        let root = Inner::test_from_parts(
            vec![1_000_000],
            vec![Node::from_inner(a, &mut Global), Node::from_inner(b, &mut Global)],
        );
        let tree = BPlusTree {
            root: Node::from_inner(root, &mut Global),
            height: 2,
            len,
            allocator: Global,
        };

        check_tree(&tree);
        for k in [0u64, 10, 1_000, 1_000_000, 1_000_000 + 1_000 * (IMIN as u64 - 1)] {
            let got = tree.get(&k);
            assert!(got.is_some_and(|c| c.0 == k), "key {k} must be reachable through the tree");
            assert!(tree.contains_key(&k));
        }
        for k in [5u64, 999, 2_000_000] {
            assert!(tree.get(&k).is_none(), "absent key {k} must miss");
        }

        drop(tree);
        assert_eq!(
            live.load(Relaxed),
            0,
            "dropping the tree must drop every value exactly once \
             (positive = leak, negative = double-drop)"
        );
    }

    /// `get_mut` exposes a value for in-place mutation, observable through
    /// a subsequent `get`.
    #[test]
    fn get_mut_mutates_in_place() {
        let live = Arc::new(AtomicIsize::new(0));
        let (inner, _) = minimal_inner(2, 0, &live, None);
        let len = 2 * LMIN;
        let mut tree = BPlusTree {
            root: Node::from_inner(inner, &mut Global),
            height: 1,
            len,
            allocator: Global,
        };

        tree.get_mut(&10).expect("present key must be gettable").0 = 424_242;
        assert_eq!(
            tree.get(&10).map(|c| c.0),
            Some(424_242),
            "a get_mut write must be visible to the next get"
        );
        assert_eq!(tree.len(), len, "get_mut must not change the pair count");
        check_tree(&tree);
    }

    /// Tree-level remove: the value comes back, `len` ticks down, and the
    /// invariants hold afterwards.
    #[test]
    fn remove_returns_the_value_and_updates_len() {
        let live = Arc::new(AtomicIsize::new(0));
        let (inner, _) = minimal_inner(2, 0, &live, None);
        let len = 2 * LMIN;
        let mut tree = BPlusTree {
            root: Node::from_inner(inner, &mut Global),
            height: 1,
            len,
            allocator: Global,
        };

        let got = tree.remove(&0);
        assert!(got.is_some_and(|c| c.0 == 0), "removing a present key must return its value");
        assert_eq!(tree.len(), len - 1, "a hit must decrement len");
        assert!(tree.remove(&0).is_none(), "a second remove of the same key must miss");
        assert_eq!(tree.len(), len - 1, "a miss must not change len");
        check_tree(&tree);

        drop(tree);
        assert_eq!(live.load(Relaxed), 0, "teardown must drop every survivor exactly once");
    }

    /// A cascade merge that leaves the root with one child must hoist it:
    /// the tree gets one level shorter and stays fully valid.
    #[test]
    fn remove_hoists_a_single_child_root() {
        let live = Arc::new(AtomicIsize::new(0));
        let (b, b_first) = minimal_inner(IMIN, 1_000_000, &live, None);
        let (a, _) = minimal_inner(IMIN, 0, &live, Some(b_first));
        let len = 2 * IMIN * LMIN;
        let root = Inner::test_from_parts(
            vec![1_000_000],
            vec![Node::from_inner(a, &mut Global), Node::from_inner(b, &mut Global)],
        );
        let mut tree = BPlusTree {
            root: Node::from_inner(root, &mut Global),
            height: 2,
            len,
            allocator: Global,
        };

        // One remove: leaf merge inside `a` → `a` deficient → both inners
        // minimal → they merge → the root has one child → hoist.
        let got = tree.remove(&0);
        assert!(got.is_some_and(|c| c.0 == 0), "removing a present key must return its value");
        assert_eq!(tree.len(), len - 1);
        assert_eq!(tree.height, 1, "hoisting the merged child must shorten the tree by one level");
        check_tree(&tree);
        assert!(
            tree.get(&1_000_000).is_some(),
            "keys from the absorbed side must remain reachable"
        );

        drop(tree);
        assert_eq!(
            live.load(Relaxed),
            0,
            "teardown after a hoist must drop every survivor exactly once \
             (positive = leak, negative = double-drop)"
        );
    }

    /// `clear` empties the tree in one call and drops every value exactly
    /// once, leaving a usable empty tree.
    #[test]
    fn clear_resets_to_an_empty_tree() {
        let live = Arc::new(AtomicIsize::new(0));
        let (inner, _) = minimal_inner(2, 0, &live, None);
        let len = 2 * LMIN;
        let mut tree = BPlusTree {
            root: Node::from_inner(inner, &mut Global),
            height: 1,
            len,
            allocator: Global,
        };

        tree.clear();
        assert_eq!(live.load(Relaxed), 0, "clear must drop every value exactly once");
        assert_eq!(tree.len(), 0);
        assert!(tree.is_empty());
        assert!(tree.get(&0).is_none());
        check_tree(&tree);
    }

    // ── BPlusTree: the public-API specification (insert-driven) ────────

    /// Inserting distinct keys grows the tree through leaf and root
    /// splits; every pair stays reachable, `len` is exact, and the
    /// invariants hold at every size.
    #[test]
    fn insert_get_roundtrip_grows_through_splits() {
        const N: u64 = 2_000;
        let mut tree: BPlusTree<u64, u64, M> = BPlusTree::new();

        for i in 0..N {
            let k = (i * 37) % N; // coprime stride: a shuffled bijection
            assert_eq!(tree.insert(k, v(k)), None, "first insert of key {k} must return None");
        }
        assert_eq!(tree.len(), N as usize, "len must count every distinct insert");
        assert!(
            tree.height >= 2,
            "{N} pairs at fanout {M} must have grown the root at least twice"
        );
        check_tree(&tree);

        for k in 0..N {
            assert_eq!(tree.get(&k), Some(&v(k)), "key {k} must round-trip");
        }
        assert!(tree.get(&N).is_none(), "an absent key must miss");
    }

    /// Re-inserting a key replaces its value in place: the old value comes
    /// back, and the pair count does not change.
    #[test]
    fn insert_replaces_and_returns_the_previous_value() {
        let mut tree: BPlusTree<u64, u64, M> = BPlusTree::new();
        assert_eq!(tree.insert(7, 1), None);
        assert_eq!(tree.insert(7, 2), Some(1), "a replace must return the previous value");
        assert_eq!(tree.len(), 1, "a replace must not change the pair count");
        assert_eq!(tree.get(&7), Some(&2));
        check_tree(&tree);
    }

    #[test]
    fn from_iterator_collects_every_pair() {
        let tree: BPlusTree<u64, u64, M> = (0..300u64).map(|k| (k, v(k))).collect();
        assert_eq!(tree.len(), 300);
        check_tree(&tree);
        for k in 0..300 {
            assert_eq!(tree.get(&k), Some(&v(k)), "collected key {k} must round-trip");
        }
    }

    /// `from_sorted_iter` must build a structurally valid tree at every
    /// awkward size: empty, single pair, exactly one leaf, one-past (the
    /// worst ragged tail), exact multiples (whose final level is one
    /// exactly-full node), and multi-level trees whose tails cascade
    /// repairs at both the leaf and inner levels (`M² + 1`, `M³ + 1`).
    /// `check_tree` is the net: occupancy invariants, separator routing,
    /// leaf-chain integrity, and `len` bookkeeping.
    ///
    /// Under Miri the `m³` sizes are skipped: interpreting them costs
    /// tens of minutes, and the deep build path is Miri-covered by
    /// `bulk::tests::deep_cascade_loads_own_every_value_exactly_once`.
    /// Regular CI runs the full list.
    #[test]
    fn from_sorted_iter_builds_valid_trees_at_awkward_sizes() {
        let m = M as u64;
        #[rustfmt::skip]
        let sizes = [
            0, 1, 2, m - 1, m, m + 1, 2 * m, 2 * m + 1,
            m * m, m * m + 1, m * m + m + 3,
            m * m * m, m * m * m + 1,
        ];
        for n in sizes {
            if cfg!(miri) && n >= m * m * m {
                continue;
            }
            let tree: BPlusTree<u64, u64, M> =
                BPlusTree::from_sorted_iter((0..n).map(|k| (k, v(k))));

            assert_eq!(tree.len(), n as usize, "len must count the drained pairs (n={n})");
            let want_height = if n <= m {
                0
            } else if n <= m * m {
                1
            } else if n <= m * m * m {
                2
            } else {
                3
            };
            assert_eq!(tree.height, want_height, "height must match the level count (n={n})");
            check_tree(&tree);

            for k in 0..n {
                assert_eq!(tree.get(&k), Some(&v(k)), "key {k} lost in a bulk load of {n}");
            }
            assert_eq!(tree.get(&n), None, "unknown keys must miss (n={n})");
        }
    }

    /// A bulk-loaded tree owns its values end to end: the load itself
    /// drops nothing, lookups see every value, and dropping the tree
    /// drops each exactly once.
    #[test]
    fn from_sorted_iter_drops_values_exactly_once() {
        // One-past-two-full-levels: leaf and inner tail repairs both fire.
        let n = M * M + 1;
        let live = Arc::new(AtomicIsize::new(0));
        {
            let tree: BPlusTree<u64, Counted, M> =
                BPlusTree::from_sorted_iter((0..n as u64).map(|k| (k, Counted::new(k, &live))));
            assert_eq!(live.load(Relaxed), n as isize, "the load itself must not drop anything");
            check_tree(&tree);
        }
        assert_eq!(
            live.load(Relaxed),
            0,
            "dropping the tree must drop every bulk-loaded value exactly once \
             (positive = leak, negative = double-drop)"
        );
    }

    /// Draining a grown tree through the public API shrinks it back to an
    /// empty root leaf: every removal returns its value, and at the end
    /// the tree is empty at height 0.
    #[test]
    fn remove_drains_the_tree_back_to_height_zero() {
        const N: u64 = 600;
        let mut tree: BPlusTree<u64, u64, M> = BPlusTree::new();
        for k in 0..N {
            tree.insert(k, v(k));
        }
        assert!(tree.height >= 1, "{N} pairs must not fit a single leaf");
        check_tree(&tree);

        for i in 0..N {
            let k = (i * 7) % N; // coprime stride
            assert_eq!(
                tree.remove(&k),
                Some(v(k)),
                "removing present key {k} must return its value"
            );
            assert_eq!(tree.len(), (N - 1 - i) as usize, "len must tick down on every hit");
            if i % 100 == 0 {
                check_tree(&tree);
            }
        }
        assert!(tree.is_empty(), "a drained tree is empty");
        assert_eq!(tree.height, 0, "a drained tree must have shrunk back to a root leaf");
        check_tree(&tree);
    }

    /// The payoff test: a deterministic insert/remove churn mirrored
    /// against `alloc::collections::BTreeMap`. Every operation must agree
    /// with the model — return values, lengths, and final contents — with
    /// the invariants checked throughout.
    #[test]
    fn churn_mirrors_btreemap() {
        use alloc::collections::BTreeMap;

        let mut tree: BPlusTree<u64, u64, M> = BPlusTree::new();
        let mut model: BTreeMap<u64, u64> = BTreeMap::new();
        let mut state: u64 = 0x5EED_CAFE_F00D_D00D;

        for step in 0..1_500u64 {
            let r = xorshift(&mut state);
            let key = r % 200;
            if (r >> 32) % 5 < 3 {
                assert_eq!(
                    tree.insert(key, step),
                    model.insert(key, step),
                    "insert({key}) must agree with the model at step {step}"
                );
            } else {
                assert_eq!(
                    tree.remove(&key),
                    model.remove(&key),
                    "remove({key}) must agree with the model at step {step}"
                );
            }
            assert_eq!(tree.len(), model.len(), "len must agree with the model at step {step}");
            if step % 128 == 0 {
                check_tree(&tree);
            }
        }

        check_tree(&tree);
        for (k, val) in &model {
            assert_eq!(tree.get(k), Some(val), "key {k} must match the model at the end");
        }
        assert!(tree.get(&10_000).is_none());
    }

    /// The whole stack drops values exactly once when the tree does —
    /// grown through the public API, torn down through `Drop`.
    #[test]
    fn tree_drop_drops_every_value_exactly_once() {
        let live = Arc::new(AtomicIsize::new(0));
        {
            let mut tree: BPlusTree<u64, Counted, M> = BPlusTree::new();
            for k in 0..(3 * M as u64) {
                tree.insert(k, Counted::new(k, &live));
            }
            assert_eq!(live.load(Relaxed), 3 * M as isize, "one live value per inserted key");
            check_tree(&tree);
        }
        assert_eq!(
            live.load(Relaxed),
            0,
            "dropping the tree must drop every value exactly once \
             (positive = leak, negative = double-drop)"
        );
    }

    // ── remove: rebalance-path pins ─────────────────────────────────────
    //
    // One contract per test, so a red test names the violated behavior on
    // its own: a miss leaves the tree untouched, a deficient node borrows
    // before it merges, and the upward repair stops at the first healthy
    // level. (The hit/hoist/drain/churn contracts are pinned with the
    // other public-API tests above.)

    /// A probe for an absent key returns `None` and leaves the tree
    /// untouched: same `len`, same reachable pairs, invariants intact.
    #[test]
    fn remove_miss_leaves_the_tree_untouched() {
        let live = Arc::new(AtomicIsize::new(0));
        let (inner, _) = minimal_inner(2, 0, &live, None);
        let len = 2 * LMIN;
        let mut tree = BPlusTree {
            root: Node::from_inner(inner, &mut Global),
            height: 1,
            len,
            allocator: Global,
        };

        for k in [5u64, 999, u64::MAX] {
            assert!(tree.remove(&k).is_none(), "absent key {k} must miss");
        }
        assert_eq!(tree.len(), len, "misses must not change len");
        check_tree(&tree);
        let seeds = (0..2).flat_map(|c| (0..LMIN as u64).map(move |j| 1_000 * c + 10 * j));
        for k in seeds {
            assert!(tree.get(&k).is_some_and(|c| c.0 == k), "key {k} must remain reachable");
        }
    }

    /// A deficient leaf with a sibling strictly above its minimum is
    /// repaired by borrowing, not merging: the root keeps both children
    /// and every survivor stays reachable.
    #[test]
    fn remove_repairs_a_deficient_leaf_by_borrowing() {
        let live = Arc::new(AtomicIsize::new(0));
        // Child 0 minimal, child 1 one above minimum: removing from
        // child 0 must borrow from child 1.
        let right = counted_leaf(1_000, LMIN + 1, &live, None);
        let left = counted_leaf(0, LMIN, &live, Some(right));
        let root = Inner::test_from_parts(
            vec![1_000],
            vec![Node::from_leaf_ptr(left), Node::from_leaf_ptr(right)],
        );
        let len = 2 * LMIN + 1;
        let mut tree = BPlusTree {
            root: Node::from_inner(root, &mut Global),
            height: 1,
            len,
            allocator: Global,
        };

        let got = tree.remove(&0);
        assert!(got.is_some_and(|c| c.0 == 0), "removing a present key must return its value");
        assert_eq!(tree.height, 1, "borrowing must not change the tree's shape");
        // SAFETY: height-1 subtree, judged as a root.
        let _ = unsafe { tree.root.check_invariants(1, true) };
        assert_eq!(
            unsafe { tree.root.len(1) },
            2,
            "a repair by borrowing must keep both children alive"
        );
        check_tree(&tree);

        drop(tree);
        assert_eq!(live.load(Relaxed), 0, "teardown must drop every survivor exactly once");
    }

    /// Two minimal leaf siblings merge when one dips below minimum — and
    /// under a root with children to spare, the merge is absorbed: no
    /// hoist, chain intact, all survivors reachable.
    #[test]
    fn remove_merges_minimal_leaf_siblings() {
        let live = Arc::new(AtomicIsize::new(0));
        let (inner, _) = minimal_inner(3, 0, &live, None);
        let len = 3 * LMIN;
        let mut tree = BPlusTree {
            root: Node::from_inner(inner, &mut Global),
            height: 1,
            len,
            allocator: Global,
        };

        let got = tree.remove(&0);
        assert!(got.is_some_and(|c| c.0 == 0), "removing a present key must return its value");
        assert_eq!(tree.height, 1, "a merge absorbed by the root must not shrink the tree");
        assert_eq!(
            unsafe { tree.root.len(1) },
            2,
            "merging one minimal pair under a 3-child root must leave 2 children"
        );
        check_tree(&tree);

        drop(tree);
        assert_eq!(live.load(Relaxed), 0, "teardown must drop every survivor exactly once");
    }

    /// A leaf repair below a HEALTHY parent must not climb further: with
    /// the parent above its minimum, one removal repairs the leaf level
    /// and stops, leaving every level valid. (Height 2, so there is a
    /// level above the repair to get this wrong in.)
    #[test]
    fn remove_stops_rebalancing_at_the_first_healthy_level() {
        let live = Arc::new(AtomicIsize::new(0));
        // Right subtree: IMIN minimal leaves. Left subtree: IMIN + 1, so
        // the leaf merge inside it leaves it at IMIN — still legal, and
        // the cascade must stop there.
        let (b, b_first) = minimal_inner(IMIN, 1_000_000, &live, None);
        let (a, _) = minimal_inner(IMIN + 1, 0, &live, Some(b_first));
        let len = (2 * IMIN + 1) * LMIN;
        let root = Inner::test_from_parts(
            vec![1_000_000],
            vec![Node::from_inner(a, &mut Global), Node::from_inner(b, &mut Global)],
        );
        let mut tree = BPlusTree {
            root: Node::from_inner(root, &mut Global),
            height: 2,
            len,
            allocator: Global,
        };

        let got = tree.remove(&0);
        assert!(got.is_some_and(|c| c.0 == 0), "removing a present key must return its value");
        assert_eq!(tree.len(), len - 1);
        assert_eq!(tree.height, 2, "a repair absorbed at the leaf level must not shrink the tree");
        check_tree(&tree);
        for k in [10u64, 1_000, 1_000_000] {
            assert!(tree.get(&k).is_some_and(|c| c.0 == k), "key {k} must remain reachable");
        }

        drop(tree);
        assert_eq!(
            live.load(Relaxed),
            0,
            "teardown must drop every survivor exactly once \
             (positive = leak, negative = double-drop)"
        );
    }

    // ── beyond the default fanout ──────────────────────────────────────

    /// The fanout each byte-array key size is chosen to produce
    /// (`M = 512 / (S + 8)`): the key size selects the fanout.
    const _: () = assert!(<[u8; 121] as Key>::FANOUT == 3);
    const _: () = assert!(<[u8; 120] as Key>::FANOUT == 4);
    const _: () = assert!(<[u8; 94] as Key>::FANOUT == 5);

    fn bkey<const S: usize>(k: u8) -> [u8; S] {
        [k; S]
    }

    /// The full public-API lifecycle at fanout `N`: shuffled inserts
    /// growing the tree to at least `min_height` (small fanouts make
    /// depth cheap — this covers splits and remove cascades through
    /// multiple inner levels, which the u64 tests never reach), every
    /// key served, then a differently-shuffled drain back to the empty
    /// root leaf, with the invariant net thrown repeatedly and values
    /// dropping exactly once.
    fn lifecycle_at_fanout<K: Key + Ord, const N: usize>(mk: impl Fn(u8) -> K, min_height: u8) {
        const KEYS: usize = 120;
        let live = Arc::new(AtomicIsize::new(0));
        {
            let mut tree: BPlusTree<K, Counted, N> = BPlusTree::new();

            let mut keys: Vec<u8> = (0..KEYS as u8).collect();
            keys.sort_by_key(|k| (*k as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            for (i, &k) in keys.iter().enumerate() {
                assert!(
                    tree.insert(mk(k), Counted::new(k as u64, &live)).is_none(),
                    "key {k} is fresh — there is no value to replace (M={N}, insert #{i})"
                );
                assert_eq!(
                    tree.len(),
                    i + 1,
                    "len must count every inserted pair (M={N}, insert #{i})"
                );
                if i % 16 == 0 {
                    check_tree(&tree);
                }
            }
            check_tree(&tree);
            assert!(
                tree.height >= min_height,
                "at fanout {N}, {KEYS} keys must build a tree of height >= {min_height} (got {})",
                tree.height
            );
            assert_eq!(
                live.load(Relaxed),
                KEYS as isize,
                "one live value per inserted key (M={N})"
            );
            for k in 0..KEYS as u8 {
                assert!(
                    tree.get(&mk(k)).is_some_and(|v| v.0 == k as u64),
                    "key {k} must be served from the deep tree (M={N})"
                );
            }

            keys.sort_by_key(|k| (*k as u64).wrapping_mul(0xD1B5_4A32_D192_ED03));
            for (i, &k) in keys.iter().enumerate() {
                let got = tree.remove(&mk(k));
                assert!(
                    got.is_some_and(|v| v.0 == k as u64),
                    "removing stored key {k} must return its value (M={N}, removal #{i})"
                );
                assert_eq!(
                    tree.len(),
                    KEYS - i - 1,
                    "len must shrink by one (M={N}, removal #{i})"
                );
                if i % 16 == 0 {
                    check_tree(&tree);
                }
            }
            check_tree(&tree);
            assert!(tree.is_empty(), "a full drain must empty the tree (M={N})");
            assert_eq!(tree.height, 0, "a drained tree must shrink back to a root leaf (M={N})");
            assert_eq!(
                live.load(Relaxed),
                0,
                "every removed value must have dropped exactly once via the returned \
                 handle (M={N})"
            );
        }
        assert_eq!(
            live.load(Relaxed),
            0,
            "dropping the drained tree must drop nothing further \
             (positive = leak, negative = double-drop; M={N})"
        );
    }

    /// M == 3, the minimum: `⌈M/2⌉` arithmetic at its tightest.
    #[test]
    fn full_lifecycle_at_minimum_fanout() {
        lifecycle_at_fanout::<[u8; 121], 3>(bkey::<121>, 3);
    }

    /// M == 4, the smallest EVEN fanout: the split midpoint and
    /// min-occupancy land on the other parity from 3 and 5.
    #[test]
    fn full_lifecycle_at_even_fanout() {
        lifecycle_at_fanout::<[u8; 120], 4>(bkey::<120>, 3);
    }

    /// M == 5: with 3, 4, and the u64 default 32, both parities are
    /// covered on each side of the `div_ceil` midpoint.
    #[test]
    fn full_lifecycle_at_odd_fanout() {
        lifecycle_at_fanout::<[u8; 94], 5>(bkey::<94>, 2);
    }

    /// The churn contract must hold regardless of seed: several seeds,
    /// each mirrored against `BTreeMap` from a fresh tree. Skipped under
    /// Miri — the single-seed churn above already runs there; this one
    /// buys breadth, not memory-safety coverage.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn churn_mirrors_btreemap_across_seeds() {
        use alloc::collections::BTreeMap;

        for seed in [1u64, 42, 0xDEAD_BEEF, 0xABCD_EF01_2345_6789, u64::MAX / 7] {
            let mut tree: BPlusTree<u64, u64, M> = BPlusTree::new();
            let mut model: BTreeMap<u64, u64> = BTreeMap::new();
            let mut state = seed;

            for step in 0..600u64 {
                let r = xorshift(&mut state);
                let key = r % 300;
                if (r >> 32) % 5 < 3 {
                    assert_eq!(
                        tree.insert(key, step),
                        model.insert(key, step),
                        "insert({key}) must agree with the model (seed {seed:#x}, step {step})"
                    );
                } else {
                    assert_eq!(
                        tree.remove(&key),
                        model.remove(&key),
                        "remove({key}) must agree with the model (seed {seed:#x}, step {step})"
                    );
                }
                assert_eq!(
                    tree.len(),
                    model.len(),
                    "len must agree with the model (seed {seed:#x}, step {step})"
                );
                if step % 128 == 0 {
                    check_tree(&tree);
                }
            }

            check_tree(&tree);
            for (k, val) in &model {
                assert_eq!(
                    tree.get(k),
                    Some(val),
                    "key {k} must match the model at the end (seed {seed:#x})"
                );
            }
        }
    }

    // ── property-based differential testing ────────────────────────────
    //
    // proptest generates random op sequences and, on failure, SHRINKS
    // the sequence to a minimal reproduction, persisted under
    // proptest-regressions/ — COMMIT those files; each is a permanent
    // regression test. The property is the churn contract with the
    // generator and shrinker upgraded: every operation agrees with
    // `BTreeMap`, and the invariant net holds afterwards.

    use proptest::prelude::*;

    /// One step against both the tree and the model.
    #[derive(Debug, Clone, Copy)]
    enum Op {
        Insert(u64, u64),
        Remove(u64),
        Get(u64),
    }

    /// Keys mostly from a small domain, so collisions, replacements, and
    /// re-inserts of removed keys actually happen.
    fn key_strategy() -> impl Strategy<Value = u64> + Clone {
        prop_oneof![3 => 0u64..64, 1 => any::<u64>()]
    }

    /// Weighted toward inserts so trees grow deep enough to split.
    fn op_strategy() -> impl Strategy<Value = Op> + Clone {
        prop_oneof![
            3 => (key_strategy(), any::<u64>()).prop_map(|(k, v)| Op::Insert(k, v)),
            2 => key_strategy().prop_map(Op::Remove),
            1 => key_strategy().prop_map(Op::Get),
        ]
    }

    /// Order-preserving widen of a u64 into the fanout-3 key type:
    /// big-endian bytes sort like the integers they came from.
    fn wide(k: u64) -> [u8; 121] {
        let mut key = [0u8; 121];
        key[..8].copy_from_slice(&k.to_be_bytes());
        key
    }

    /// Apply `ops` to a fresh tree and a `BTreeMap`, asserting every
    /// return value and `len` agree step-for-step, then throw the
    /// invariant net and sweep the survivors.
    fn run_differential<K: Key + Ord, const N: usize>(mk: impl Fn(u64) -> K, ops: &[Op]) {
        use alloc::collections::BTreeMap;

        let mut tree: BPlusTree<K, u64, N> = BPlusTree::new();
        let mut model: BTreeMap<u64, u64> = BTreeMap::new();

        for (i, &op) in ops.iter().enumerate() {
            match op {
                Op::Insert(k, v) => assert_eq!(
                    tree.insert(mk(k), v),
                    model.insert(k, v),
                    "insert({k}) must agree with the model (op #{i})"
                ),
                Op::Remove(k) => assert_eq!(
                    tree.remove(&mk(k)),
                    model.remove(&k),
                    "remove({k}) must agree with the model (op #{i})"
                ),
                Op::Get(k) => assert_eq!(
                    tree.get(&mk(k)),
                    model.get(&k),
                    "get({k}) must agree with the model (op #{i})"
                ),
            }
            assert_eq!(tree.len(), model.len(), "len must agree with the model (op #{i})");
        }

        check_tree(&tree);
        for (k, v) in &model {
            assert_eq!(tree.get(&mk(*k)), Some(v), "key {k} must match the model at the end");
        }
    }

    /// `Debug` renders the tree exactly like the reference map renders
    /// the same pairs: `debug_map` shape, ascending key order.
    #[test]
    fn debug_formats_like_btreemap() {
        use alloc::{collections::BTreeMap, format};

        let n = 2 * M as u64 + 3;
        let tree: BPlusTree<u64, u64, M> = (0..n).map(|k| (k, v(k))).collect();
        let model: BTreeMap<u64, u64> = (0..n).map(|k| (k, v(k))).collect();
        assert_eq!(
            format!("{tree:?}"),
            format!("{model:?}"),
            "Debug must render the pairs in ascending key order, debug_map-shaped"
        );

        let empty: BPlusTree<u64, u64, M> = BPlusTree::new();
        assert_eq!(format!("{empty:?}"), "{}", "an empty tree must render as an empty map");
    }

    /// `first_key_value`/`last_key_value`: `None` on the empty tree, the
    /// extreme pairs otherwise — tracking growth and removal.
    #[test]
    fn first_and_last_key_value_return_the_extreme_pairs() {
        let mut tree: BPlusTree<u64, u64, M> = BPlusTree::new();
        assert_eq!(tree.first_key_value(), None, "an empty tree has no first pair");
        assert_eq!(tree.last_key_value(), None, "an empty tree has no last pair");

        tree.insert(5, v(5));
        assert_eq!(tree.first_key_value(), Some((&5, &v(5))), "a lone pair is both extremes");
        assert_eq!(tree.last_key_value(), Some((&5, &v(5))), "a lone pair is both extremes");

        let n = (M * M + 1) as u64;
        let mut tree: BPlusTree<u64, u64, M> =
            BPlusTree::from_sorted_iter((0..n).map(|k| (k, v(k))));
        assert_eq!(tree.first_key_value(), Some((&0, &v(0))), "first must be the minimum pair");
        assert_eq!(
            tree.last_key_value(),
            Some((&(n - 1), &v(n - 1))),
            "last must be the maximum pair"
        );

        tree.remove(&0);
        tree.remove(&(n - 1));
        assert_eq!(
            tree.first_key_value(),
            Some((&1, &v(1))),
            "first must track removal of the minimum"
        );
        assert_eq!(
            tree.last_key_value(),
            Some((&(n - 2), &v(n - 2))),
            "last must track removal of the maximum"
        );
    }

    // ── first_leaf / last_leaf ──────────────────────────────────────

    /// A height-0 tree is its own chain: `first_leaf` and `last_leaf`
    /// are the root leaf itself — empty, and at every fill short of a
    /// split.
    #[test]
    fn first_and_last_leaf_of_a_root_leaf_tree_are_the_root() {
        let mut tree: BPlusTree<u64, u64, M> = BPlusTree::new();
        assert!(
            core::ptr::eq(tree.first_leaf(), tree.last_leaf()),
            "an empty tree's one leaf must be both ends"
        );
        assert_eq!(tree.first_leaf().len(), 0, "the empty root leaf holds nothing");

        // Fill the root leaf to capacity without splitting it.
        for k in 0..M as u64 {
            tree.insert(k, v(k));
        }
        let first = tree.first_leaf();
        let last = tree.last_leaf();
        assert!(core::ptr::eq(first, last), "a lone full root leaf must still be both ends");
        assert_eq!(*first.first_key(), 0, "first_leaf must hold the minimum key");
        assert_eq!(
            first.test_keys().last(),
            Some(&(M as u64 - 1)),
            "last_leaf must hold the maximum key"
        );
    }

    /// `first_leaf` must hold the tree's minimum key and `last_leaf` its
    /// maximum — the chain's terminal leaf — on multi-level trees from
    /// both construction paths (insert-grown and bulk-loaded).
    #[test]
    fn first_and_last_leaf_hold_the_extreme_keys() {
        let n = (M * M + M) as u64;
        let mut grown: BPlusTree<u64, u64, M> = BPlusTree::new();
        for i in 0..n {
            // A stride-permuted insert order, so growth splits all over.
            let k = (i * 7919) % n;
            grown.insert(k, v(k));
        }
        let loaded: BPlusTree<u64, u64, M> = BPlusTree::from_sorted_iter((0..n).map(|k| (k, v(k))));

        for (tree, how) in [(&grown, "insert-grown"), (&loaded, "bulk-loaded")] {
            let first = tree.first_leaf();
            let last = tree.last_leaf();
            assert_eq!(*first.first_key(), 0, "first_leaf must hold the minimum key ({how})");
            assert_eq!(
                last.test_keys().last(),
                Some(&(n - 1)),
                "last_leaf must hold the maximum key ({how})"
            );
            assert!(last.next().is_none(), "last_leaf must be the end of the leaf chain ({how})");
        }
    }

    /// Walking the sibling links from `first_leaf` must visit every key
    /// in ascending order and arrive exactly at `last_leaf`.
    #[test]
    fn the_leaf_chain_runs_from_first_leaf_to_last_leaf() {
        let n = (M * M + 1) as u64;
        let tree: BPlusTree<u64, u64, M> = BPlusTree::from_sorted_iter((0..n).map(|k| (k, v(k))));

        let mut leaf = tree.first_leaf();
        let mut expect = 0u64;
        loop {
            for k in leaf.test_keys() {
                assert_eq!(*k, expect, "the chain must visit every key in ascending order");
                expect += 1;
            }
            let Some(next) = leaf.next() else { break };
            // SAFETY: chain links point at live leaves the tree owns,
            // borrowed here at the tree's lifetime.
            leaf = unsafe { next.as_ref() };
        }
        assert_eq!(expect, n, "the chain must visit every pair");
        assert!(
            core::ptr::eq(leaf, tree.last_leaf()),
            "the chain from first_leaf must terminate at last_leaf"
        );
    }

    /// The ends must track mutation: draining inward from both ends —
    /// through leaf borrows, merges, and root shrinks — `first_leaf` and
    /// `last_leaf` must always hold the current minimum and maximum,
    /// down to a lone root leaf again.
    #[test]
    fn first_and_last_leaf_track_removes() {
        let n = (M * M + 1) as u64;
        let mut tree: BPlusTree<u64, u64, M> =
            BPlusTree::from_sorted_iter((0..n).map(|k| (k, v(k))));

        let (mut lo, mut hi) = (0u64, n - 1);
        while lo < hi {
            assert_eq!(
                *tree.first_leaf().first_key(),
                lo,
                "first_leaf must hold the minimum after draining below {lo}"
            );
            assert_eq!(
                tree.last_leaf().test_keys().last(),
                Some(&hi),
                "last_leaf must hold the maximum after draining above {hi}"
            );
            tree.remove(&lo);
            tree.remove(&hi);
            lo += 1;
            hi -= 1;
        }

        // n is odd, so exactly one pair remains.
        assert_eq!(tree.len(), 1, "one pair must remain (fixture arithmetic otherwise)");
        let first = tree.first_leaf();
        let last = tree.last_leaf();
        assert!(core::ptr::eq(first, last), "the last pair must live in a lone root leaf");
        assert_eq!(*first.first_key(), lo, "the survivor must be the middle key");
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: if cfg!(miri) { 2 } else { 256 },
            // Persistence resolves paths through `getcwd`, which Miri's
            // isolation forbids; regression files are a native-run luxury.
            failure_persistence: if cfg!(miri) {
                None
            } else {
                ProptestConfig::default().failure_persistence
            },
            ..ProptestConfig::default()
        })]

        /// Any op sequence must agree with `BTreeMap` step-for-step at
        /// the default fanout (M == 32)...
        #[test]
        fn differential_vs_btreemap_at_default_fanout(
            ops in proptest::collection::vec(op_strategy(), 0..512)
        ) {
            run_differential::<u64, M>(|k| k, &ops);
        }

        /// ...and at the minimum fanout (M == 3), where the same
        /// sequences build deep trees and cascade through multiple inner
        /// levels.
        #[test]
        fn differential_vs_btreemap_at_minimum_fanout(
            ops in proptest::collection::vec(op_strategy(), 0..256)
        ) {
            run_differential::<[u8; 121], 3>(wide, &ops);
        }
    }
}
