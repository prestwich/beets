use core::ptr::NonNull;

use crate::{
    Inner, Key, Leaf,
    allocator::{NodeAllocator, SlotAllocator},
};

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
        pub(crate) unsafe fn $as_ref(&self) -> &$Kind<K, V, M> {
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
        pub(crate) unsafe fn $as_mut(&mut self) -> &mut $Kind<K, V, M> {
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
        pub(crate) unsafe fn $into<A: SlotAllocator<$Kind<K, V, M>>>(
            self,
            alloc: &mut A,
        ) -> $Kind<K, V, M> {
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
/// of `height` down a descent (that is the test-only `delegate!`'s business). Each
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

    dispatch! {
        /// The occupancy of THIS node — NOT the entry count of the subtree
        /// under it.
        fn len(&self) -> usize;

        /// True if this node is below minimum occupancy. When a remove makes
        /// a non-root node deficient, it will trigger rebalancing at the
        /// parent level via [`Inner::rebalance`].
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
    /// is also the path taken by [`BPlusTree`](crate::BPlusTree)'s `Drop`, so a panic escaping it
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
}
