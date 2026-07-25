use crate::fanout;

mod sealed {
    /// Seals [`Key`](super::Key): only the blanket impl can satisfy it,
    /// so the associated consts always keep their defaults.
    pub trait Sealed {}
    impl<T: Ord + Copy> Sealed for T {}
}

/// A type usable as a B+tree key.
///
/// Sealed, and blanket-implemented for every `Ord + Copy` type. [`Copy`] is
/// load-bearing: separators are duplicated freely between nodes (a leaf
/// split copies the new right sibling's minimum key into the parent),
/// and none of the node machinery runs drop glue for keys.
pub trait Key: sealed::Sealed + Ord + Copy + Sized {
    /// The key's size in memory, in bytes.
    const SIZE: usize = size_of::<Self>();

    /// The node fanout for this key type: how many children an inner
    /// node — and pairs a leaf — can hold, sized so a node's arrays fit
    /// `NODE_BUDGET` bytes: `NODE_BUDGET / (SIZE + 8)`, charging 8 bytes
    /// per entry for the value or erased child handle. Trees are
    /// instantiated as `BPlusTree<K, V, { K::FANOUT }>`, and the node
    /// constructors const-assert `M == K::FANOUT`.
    const FANOUT: usize = fanout(Self::SIZE);

    /// Compile-time bound on the key size: `1..128`. With the 512-byte
    /// `NODE_BUDGET` this keeps [`Self::FANOUT`] within `3..=56` — and 3
    /// is the hard floor below which min-occupancy rebalancing
    /// degenerates (asserted again, per-M, where nodes are born).
    const __SIZE_IN_BOUND: () = {
        assert!(Self::SIZE > 0);
        assert!(Self::SIZE < 128);
    };
}

impl<T> Key for T where T: sealed::Sealed + Ord + Copy + Sized {}
