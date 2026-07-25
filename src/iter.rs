use core::{ops::Bound, ptr::NonNull};

use crate::allocator::NodeAllocator;
use crate::{BPlusTree, Key, Leaf};

fn check_bound<K: PartialOrd>(bound: &Bound<K>, key: &K) -> bool {
    match bound {
        Bound::Included(b) => key <= b,
        Bound::Excluded(b) => key < b,
        Bound::Unbounded => true,
    }
}

/// Resolve a range's start bound within the leaf that owns its key: the
/// index of the first pair at or above `key`, and whether the cursor
/// must additionally step once past an exact hit (an [`Excluded`](Bound::Excluded) start).
fn start_in_leaf<K: Key, V, const M: usize>(
    leaf: &Leaf<K, V, M>,
    key: &K,
    excluded: bool,
) -> (usize, bool) {
    let idx = leaf.find_key(key);
    let advance = excluded && idx < leaf.len() && leaf.keys_ref()[idx] == *key;
    (idx, advance)
}

/// A full-tree cursor made exact-sized: wraps either flavor of cursor
/// ([`Iterator`], [`IteratorMut`]) with the remaining-pair count.
pub struct Full<I> {
    len: usize,
    inner: I,
}

/// An iterator over the Full tree
pub type FullIterator<'a, K, V, const M: usize> = Full<Iterator<'a, K, V, M>>;

/// An iterator over the Full tree, yielding mut values
pub type FullIteratorMut<'a, K, V, const M: usize> = Full<IteratorMut<'a, K, V, M>>;

impl<'a, K: Key, V, const M: usize> FullIterator<'a, K, V, M> {
    pub(crate) fn new<A: NodeAllocator<K, V, M>>(tree: &'a BPlusTree<K, V, M, A>) -> Self {
        Self { len: tree.len(), inner: Iterator::from_start(tree) }
    }
}

impl<'a, K: Key, V, const M: usize> FullIteratorMut<'a, K, V, M> {
    pub(crate) fn new<A: NodeAllocator<K, V, M>>(tree: &'a mut BPlusTree<K, V, M, A>) -> Self {
        Self { len: tree.len(), inner: IteratorMut::from_start(tree) }
    }
}

impl<I: core::iter::Iterator> core::iter::Iterator for Full<I> {
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.inner.next();
        self.len -= item.is_some() as usize;
        item
    }
}

impl<I: core::iter::Iterator> core::iter::ExactSizeIterator for Full<I> {
    fn len(&self) -> usize {
        self.len
    }
}

// Exhaustion is terminal: both wrapped cursors are themselves fused
// (each parks on its own terminal state).
impl<I: core::iter::FusedIterator> core::iter::FusedIterator for Full<I> {}

/// Iterator over `(&K, &mut V)`.
///
/// Exclusivity is spent once per leaf, not once per item: on entering a
/// leaf, the cursor immediately splits it into disjoint per-pair borrows
/// ([`Leaf::pairs_mut_from`]) and rides [`slice::IterMut`](core::slice::IterMut) through them, so
/// no later whole-leaf `&mut` retags over pairs already yielded.
pub struct IteratorMut<'a, K: Key, V, const M: usize> {
    /// The current leaf's remaining pairs.
    current: core::iter::Zip<core::slice::Iter<'a, K>, core::slice::IterMut<'a, V>>,
    /// The link out of the current leaf, read at entry — before the leaf
    /// is decomposed — and not dereferenced until the cursor gets there.
    next_leaf: Option<NonNull<Leaf<K, V, M>>>,
}

impl<'a, K: Key, V, const M: usize> IteratorMut<'a, K, V, M> {
    /// Start mid-tree: enter `leaf` at pair index `from`.
    pub(crate) fn new(leaf: &'a mut Leaf<K, V, M>, from: usize) -> Self {
        let next_leaf = leaf.next();
        Self { next_leaf, current: leaf.pairs_mut_from(from) }
    }

    /// Start at the tree's first pair.
    pub(crate) fn from_start<A: NodeAllocator<K, V, M>>(
        tree: &'a mut BPlusTree<K, V, M, A>,
    ) -> Self {
        Self::new(tree.first_leaf_mut(), 0)
    }
}

impl<'a, K: Key, V, const M: usize> core::iter::Iterator for IteratorMut<'a, K, V, M> {
    type Item = (&'a K, &'a mut V);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(pair) = self.current.next() {
                return Some(pair);
            }

            // Advance to the next leaf.
            let ptr = self.next_leaf.take()?;
            // SAFETY: the iterator holds the tree's unique borrow for
            // 'a, and the chain visits each leaf exactly once, so this
            // exclusive borrow is disjoint from every borrow already
            // handed out.
            let leaf: &'a mut Leaf<K, V, M> = unsafe { &mut *ptr.as_ptr() };
            self.next_leaf = leaf.next();
            self.current = leaf.pairs_mut_from(0);
        }
    }
}

// Exhaustion is terminal: the zip empties and `next_leaf` is `None`.
impl<K: Key, V, const M: usize> core::iter::FusedIterator for IteratorMut<'_, K, V, M> {}

/// A positioned cursor stopped at a range's end bound: wraps either
/// flavor of cursor, already positioned at the range's start, and
/// checks each candidate key against the end bound before yield. Keys
/// are [`Copy`], so the stored [`Bound<K>`](Bound) owns its key rather than
/// borrowing the caller's range.
///
/// An inverted range (start above end) yields nothing: the end-bound
/// check runs before the first yield.
pub struct Bounded<I, K> {
    /// Cursor over the remaining pairs, positioned at the start bound.
    inner: I,
    /// The upper stop, checked against each candidate key before yield.
    end: Bound<K>,
}

/// Iterator over the pairs whose keys fall in a caller-given range:
/// `(&K, &V)` in ascending key order, from the range's start bound
/// through its end bound.
pub type Range<'a, K, V, const M: usize> = Bounded<Iterator<'a, K, V, M>, K>;

/// Mutable mirror of [`Range`]: `(&K, &mut V)` over the in-range pairs,
/// in ascending key order.
pub type RangeMut<'a, K, V, const M: usize> = Bounded<IteratorMut<'a, K, V, M>, K>;

impl<'a, K: Key, V, const M: usize> Range<'a, K, V, M> {
    /// Resolve `range`'s bounds against `tree`: position the cursor at
    /// the first in-range pair and store the end bound.
    pub(crate) fn new<R: core::ops::RangeBounds<K>, A: NodeAllocator<K, V, M>>(
        tree: &'a BPlusTree<K, V, M, A>,
        range: R,
    ) -> Self {
        let start = range.start_bound();
        let inner = match start {
            Bound::Unbounded => Iterator::from_start(tree),
            Bound::Included(key) | Bound::Excluded(key) => {
                let current = tree.find_leaf(key);

                let (idx, advance) =
                    start_in_leaf(current, key, matches!(start, Bound::Excluded(_)));

                let mut i = Iterator::new(Some(current), idx);
                if advance {
                    let _ = i.next();
                }
                i
            }
        };

        Self { inner, end: range.end_bound().cloned() }
    }
}

impl<'a, K: Key, V, const M: usize> RangeMut<'a, K, V, M> {
    /// Resolve `range`'s bounds against `tree`: position the cursor at
    /// the first in-range pair and store the end bound.
    pub(crate) fn new<R: core::ops::RangeBounds<K>, A: NodeAllocator<K, V, M>>(
        tree: &'a mut BPlusTree<K, V, M, A>,
        range: R,
    ) -> Self {
        let start = range.start_bound();
        let inner = match start {
            Bound::Unbounded => IteratorMut::from_start(tree),
            Bound::Included(key) | Bound::Excluded(key) => {
                let current = tree.find_leaf_mut(key);

                let (idx, advance) =
                    start_in_leaf(current, key, matches!(start, Bound::Excluded(_)));

                let mut i = IteratorMut::new(current, idx);
                if advance {
                    let _ = i.next();
                }
                i
            }
        };

        Self { inner, end: range.end_bound().cloned() }
    }
}

impl<'a, K: Key + 'a, V, I> core::iter::Iterator for Bounded<I, K>
where
    I: core::iter::Iterator<Item = (&'a K, V)>,
{
    type Item = (&'a K, V);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().filter(|(k, _)| check_bound(&self.end, k))
    }
}

// Fused by key order: the walk ascends and the end bound is fixed, so
// once one key falls outside the bound, every later key does too —
// `next()` is `None` from then on.
impl<'a, K: Key + 'a, V, I> core::iter::FusedIterator for Bounded<I, K> where
    I: core::iter::Iterator<Item = (&'a K, V)>
{
}

/// Iterator over keys and value refs.
pub struct Iterator<'a, K: Key, V, const M: usize> {
    current: Option<&'a Leaf<K, V, M>>,
    idx_in_leaf: usize,
}

impl<'a, K: Key, V, const M: usize> Iterator<'a, K, V, M> {
    pub(crate) fn new(current: Option<&'a Leaf<K, V, M>>, idx_in_leaf: usize) -> Self {
        Self { current, idx_in_leaf }
    }

    /// Instantiate a new iterator.
    pub(crate) fn from_start<A: NodeAllocator<K, V, M>>(tree: &'a BPlusTree<K, V, M, A>) -> Self {
        Self { current: Some(tree.first_leaf()), idx_in_leaf: 0 }
    }
}

impl<'a, K: Key, V, const M: usize> core::iter::Iterator for Iterator<'a, K, V, M> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let Some(current) = self.current else {
                // Reached the end of the leaf set.
                return None;
            };

            // Still in the leaf?
            let idx = self.idx_in_leaf;
            if idx < current.len() {
                self.idx_in_leaf += 1;
                // The condition is checked by the if block.
                return Some(current.kv_ref_unchecked(idx));
            }

            // Advance to next leaf
            self.idx_in_leaf = 0;
            // SAFETY: `next` links only live leaves of the same tree, and
            // the iterator's shared borrow of the tree keeps them alive
            // and unmutated for `'a`.
            self.current = current.next().map(|non_null| unsafe { non_null.as_ref() })
        }
    }
}

// Exhaustion is terminal: the cursor parks on `current: None`.
impl<K: Key, V, const M: usize> core::iter::FusedIterator for Iterator<'_, K, V, M> {}

impl<'a, K: Key, V, const M: usize, A: NodeAllocator<K, V, M>> IntoIterator
    for &'a BPlusTree<K, V, M, A>
{
    type Item = (&'a K, &'a V);

    type IntoIter = FullIterator<'a, K, V, M>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, K: Key, V, const M: usize, A: NodeAllocator<K, V, M>> IntoIterator
    for &'a mut BPlusTree<K, V, M, A>
{
    type Item = (&'a K, &'a mut V);

    type IntoIter = FullIteratorMut<'a, K, V, M>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

#[cfg(test)]
mod tests {
    //! Contract tests for the iterator family, driven entirely through
    //! the public API (`iter`, `iter_mut`, `iter_mut_from_key`,
    //! `iter_range`, `keys`, `values`, `values_mut`).
    //!
    //! The shared contract: iteration yields pairs in ascending key
    //! order, exactly the pairs the call promises — all `len()` of them
    //! for the full iterators, the in-range window for `iter_range` —
    //! whatever shape the tree is in and however it was built. The
    //! mutable iterators additionally promise every yielded `&mut V` is
    //! the pair's real value: writes through it must be visible to
    //! every later read.

    use alloc::vec::Vec;
    use core::ops::Bound;

    use crate::BPlusTree;
    use crate::test_util::{M, v};

    /// A tree of `n` pairs grown by scattered inserts, so leaf
    /// occupancies vary (a sorted load would pack every leaf full).
    fn grown(n: u64) -> BPlusTree<u64, u64, M> {
        let mut tree = BPlusTree::new();
        for i in 0..n {
            let k = (i * 7919) % n; // coprime stride: a permutation
            tree.insert(k, v(k));
        }
        tree
    }

    /// An empty tree iterates as the empty sequence.
    #[test]
    fn iter_of_an_empty_tree_yields_nothing() {
        let tree: BPlusTree<u64, u64, M> = BPlusTree::new();
        assert_eq!(tree.iter().count(), 0, "an empty tree must yield no pairs");
    }

    /// A height-0 tree yields its pairs in ascending key order with the
    /// right values, however they were inserted.
    #[test]
    fn iter_yields_a_root_leafs_pairs_in_order() {
        let mut tree: BPlusTree<u64, u64, M> = BPlusTree::new();
        for k in (0..M as u64).rev() {
            tree.insert(k, v(k));
        }
        let got: Vec<(u64, u64)> = tree.iter().map(|(k, val)| (*k, *val)).collect();
        let want: Vec<(u64, u64)> = (0..M as u64).map(|k| (k, v(k))).collect();
        assert_eq!(got, want, "iteration must yield every pair in ascending key order");
    }

    /// Iteration must cross every leaf boundary: a multi-level tree
    /// yields exactly `len()` pairs, in ascending key order, from both
    /// construction paths (bulk-loaded and insert-grown).
    #[test]
    fn iter_walks_the_whole_tree_in_order() {
        let n = (M * M + 1) as u64;
        let loaded: BPlusTree<u64, u64, M> = BPlusTree::from_sorted_iter((0..n).map(|k| (k, v(k))));

        for (tree, how) in [(&loaded, "bulk-loaded"), (&grown(n), "insert-grown")] {
            let mut expect = 0u64;
            for (k, val) in tree.iter() {
                assert_eq!(*k, expect, "iteration must visit every key in ascending order ({how})");
                assert_eq!(*val, v(expect), "each key must carry its own value ({how})");
                expect += 1;
            }
            assert_eq!(expect, n, "iteration must yield exactly len() pairs ({how})");
        }
    }

    /// `iter().len()` must report the pairs REMAINING, per
    /// `ExactSizeIterator`'s contract: full at the start, shrinking as
    /// pairs are consumed, zero at exhaustion.
    #[test]
    fn iter_len_reports_the_remaining_pairs() {
        let n = 2 * M + 1;
        let tree: BPlusTree<u64, u64, M> =
            BPlusTree::from_sorted_iter((0..n as u64).map(|k| (k, v(k))));

        let mut it = tree.iter();
        assert_eq!(it.len(), n, "a fresh iterator's len must be the tree's len");
        it.next();
        assert_eq!(it.len(), n - 1, "len must shrink as pairs are consumed");
        for _ in it.by_ref().take(M) {}
        assert_eq!(it.len(), n - 1 - M, "len must track consumption across a leaf hop");
        for _ in it.by_ref() {}
        assert_eq!(it.len(), 0, "an exhausted iterator's len must be zero");
    }

    // ── iter_mut / values_mut ───────────────────────────────────────────

    /// `iter_mut` visits every pair in ascending key order, and each
    /// yielded `&mut V` is the pair's real value: the writes must be
    /// visible to every later read.
    #[test]
    fn iter_mut_mutates_every_pair_in_order() {
        let n = (M * M + 1) as u64;
        let loaded: BPlusTree<u64, u64, M> = BPlusTree::from_sorted_iter((0..n).map(|k| (k, v(k))));

        for (mut tree, how) in [(loaded, "bulk-loaded"), (grown(n), "insert-grown")] {
            let mut expect = 0u64;
            for (k, val) in tree.iter_mut() {
                assert_eq!(*k, expect, "iter_mut must visit every key in ascending order ({how})");
                assert_eq!(*val, v(expect), "each key must carry its own value ({how})");
                *val += 1;
                expect += 1;
            }
            assert_eq!(expect, n, "iter_mut must yield exactly len() pairs ({how})");

            for (k, val) in tree.iter() {
                assert_eq!(*val, v(*k) + 1, "the write through key {k}'s &mut must stick ({how})");
            }
        }
    }

    /// `iter_mut` on an empty tree yields nothing.
    #[test]
    fn iter_mut_of_an_empty_tree_yields_nothing() {
        let mut tree: BPlusTree<u64, u64, M> = BPlusTree::new();
        assert_eq!(tree.iter_mut().count(), 0, "an empty tree must yield no pairs");
    }

    /// `values_mut` walks values in ascending key order and its writes
    /// stick.
    #[test]
    fn values_mut_mutates_every_value() {
        let n = 2 * M as u64 + 1;
        let mut tree: BPlusTree<u64, u64, M> =
            BPlusTree::from_sorted_iter((0..n).map(|k| (k, v(k))));

        for val in tree.values_mut() {
            *val ^= 0xF00D;
        }
        for (k, val) in tree.iter() {
            assert_eq!(*val, v(*k) ^ 0xF00D, "the write through key {k}'s value must stick");
        }
    }

    // ── keys / values ───────────────────────────────────────────────────

    /// `keys` and `values` are the pair iteration, projected: same
    /// order, same count, matching halves.
    #[test]
    fn keys_and_values_project_the_pairs_in_order() {
        let n = 2 * M as u64 + 1;
        let tree: BPlusTree<u64, u64, M> = BPlusTree::from_sorted_iter((0..n).map(|k| (k, v(k))));

        let keys: Vec<u64> = tree.keys().copied().collect();
        let want_keys: Vec<u64> = (0..n).collect();
        assert_eq!(keys, want_keys, "keys() must yield every key in ascending order");

        let values: Vec<u64> = tree.values().copied().collect();
        let want_values: Vec<u64> = (0..n).map(v).collect();
        assert_eq!(values, want_values, "values() must yield every value in key order");
    }

    /// The iterator family is fused: each type declares
    /// `FusedIterator` (the helper's bound makes that a compile-time
    /// check), and an exhausted iterator keeps returning `None`.
    #[test]
    fn the_iterator_family_is_fused() {
        fn fused<I: core::iter::FusedIterator>(it: I) -> I {
            it
        }

        let n = 2 * M as u64 + 1;
        let mut tree: BPlusTree<u64, u64, M> =
            BPlusTree::from_sorted_iter((0..n).map(|k| (k, v(k))));

        let mut it = fused(tree.iter());
        for _ in it.by_ref() {}
        assert!(it.next().is_none(), "an exhausted iter must stay exhausted");
        assert!(it.next().is_none(), "and stay that way");

        let mut it = fused(tree.range(1..2 * M as u64));
        for _ in it.by_ref() {}
        assert!(it.next().is_none(), "an exhausted iter_range must stay exhausted");
        assert!(it.next().is_none(), "and stay that way");

        let mut it = fused(tree.iter_mut());
        for _ in it.by_ref() {}
        assert!(it.next().is_none(), "an exhausted iter_mut must stay exhausted");
        assert!(it.next().is_none(), "and stay that way");
    }

    // ── iter_range ──────────────────────────────────────────────────────

    /// A `start..` range on a present key starts at that key, inclusive,
    /// and yields the complete sorted tail — wherever in its leaf the
    /// key sits.
    #[test]
    fn range_from_a_present_key_yields_the_tail_from_that_key() {
        let n = (M * M + 1) as u64;
        let tree: BPlusTree<u64, u64, M> = BPlusTree::from_sorted_iter((0..n).map(|k| (k, v(k))));

        // Starts spanning leaf interiors and boundaries.
        for start in [0, 1, M as u64 - 1, M as u64, n / 2, n - 1] {
            let got: Vec<u64> = tree.range(start..).map(|(k, _)| *k).collect();
            let want: Vec<u64> = (start..n).collect();
            assert_eq!(got, want, "the tail from present key {start} must be complete");
        }
    }

    /// A `start..` range on an absent key starts at the next greater
    /// key; an explicitly excluded start skips an exact hit.
    #[test]
    fn range_start_bounds_are_honored() {
        // Store the even keys, probe odd ones between them.
        let n = 3 * M as u64;
        let tree: BPlusTree<u64, u64, M> =
            BPlusTree::from_sorted_iter((0..n).map(|k| (2 * k, v(2 * k))));

        for probe in [1, M as u64 * 2 - 1, n - 1] {
            assert_eq!(
                tree.range(probe..).next().map(|(k, _)| *k),
                Some(probe + 1),
                "the tail from absent key {probe} must start at its successor"
            );
        }
        assert_eq!(
            tree.range((Bound::Excluded(4u64), Bound::Unbounded)).next().map(|(k, _)| *k),
            Some(6),
            "an excluded start must skip its exact hit"
        );
    }

    /// A bounded window yields exactly the keys inside it: the end bound
    /// excludes (`..end`) or includes (`..=end`) its endpoint.
    #[test]
    fn range_end_bounds_are_honored() {
        let n = (M * M + 1) as u64;
        let tree: BPlusTree<u64, u64, M> = BPlusTree::from_sorted_iter((0..n).map(|k| (k, v(k))));

        // Windows spanning leaf interiors and boundaries.
        for (start, end) in [(1, 5), (0, M as u64), (M as u64 - 1, M as u64 + 1), (n / 2, n - 1)] {
            let got: Vec<u64> = tree.range(start..end).map(|(k, _)| *k).collect();
            let want: Vec<u64> = (start..end).collect();
            assert_eq!(got, want, "{start}..{end} must yield exactly the window");

            let got: Vec<u64> = tree.range(start..=end).map(|(k, _)| *k).collect();
            let want: Vec<u64> = (start..=end).collect();
            assert_eq!(got, want, "{start}..={end} must include its endpoint");
        }
    }

    /// The unbounded range is the full iteration, and ranges that cover
    /// no keys — past the maximum, or inverted — yield nothing.
    #[test]
    fn range_degenerate_cases() {
        let n = 2 * M as u64;
        let tree: BPlusTree<u64, u64, M> = BPlusTree::from_sorted_iter((0..n).map(|k| (k, v(k))));

        assert_eq!(tree.range(..).count(), n as usize, ".. must yield every pair");
        assert_eq!(tree.range(n..).count(), 0, "a range above every key must yield nothing");
        assert_eq!(
            tree.range(u64::MAX..).count(),
            0,
            "a range at the key-space maximum must yield nothing"
        );
        assert_eq!(tree.range(5..5).count(), 0, "an empty window must yield nothing");
        #[allow(clippy::reversed_empty_ranges)]
        let inverted = 7..3;
        assert_eq!(tree.range(inverted).count(), 0, "an inverted range must yield nothing");

        let empty: BPlusTree<u64, u64, M> = BPlusTree::new();
        assert_eq!(empty.range(0..).count(), 0, "an empty tree must yield nothing");
    }

    /// An excluded start above every stored key is an empty range: it
    /// must yield nothing.
    #[test]
    fn range_excluded_start_above_the_maximum_yields_nothing() {
        let n = 2 * M as u64;
        let tree: BPlusTree<u64, u64, M> = BPlusTree::from_sorted_iter((0..n).map(|k| (k, v(k))));
        assert_eq!(
            tree.range((Bound::Excluded(n + 5), Bound::Unbounded)).count(),
            0,
            "an excluded start above every key must yield nothing"
        );
    }

    // ── range_mut ───────────────────────────────────────────────────────

    /// `range_mut` yields exactly the in-window pairs, in ascending key
    /// order, and its writes stick — pairs outside the window must be
    /// untouched.
    #[test]
    fn range_mut_mutates_exactly_the_window() {
        let n = (M * M + 1) as u64;
        // Windows spanning leaf interiors and boundaries.
        for (start, end) in [(1, 5), (0, M as u64), (M as u64 - 1, M as u64 + 1), (n / 2, n - 1)] {
            let mut tree: BPlusTree<u64, u64, M> =
                BPlusTree::from_sorted_iter((0..n).map(|k| (k, v(k))));

            let mut expect = start;
            for (k, val) in tree.range_mut(start..end) {
                assert_eq!(*k, expect, "{start}..{end} must visit the window in order");
                *val += 1;
                expect += 1;
            }
            assert_eq!(expect, end, "{start}..{end} must visit the whole window");

            for (k, val) in tree.iter() {
                let want = if (start..end).contains(k) { v(*k) + 1 } else { v(*k) };
                assert_eq!(*val, want, "only the window {start}..{end} may change (key {k})");
            }
        }
    }

    /// `range_mut` honors the same bound conventions as `range`: an
    /// inclusive end includes its endpoint, and an excluded start skips
    /// an exact hit.
    #[test]
    fn range_mut_bounds_are_honored() {
        let n = 2 * M as u64;
        let mut tree: BPlusTree<u64, u64, M> =
            BPlusTree::from_sorted_iter((0..n).map(|k| (k, v(k))));

        for (_, val) in tree.range_mut(2..=5) {
            *val += 1;
        }
        for (_, val) in tree.range_mut((Bound::Excluded(M as u64), Bound::Unbounded)) {
            *val += 10;
        }

        for (k, val) in tree.iter() {
            let mut want = v(*k);
            if (2..=5).contains(k) {
                want += 1;
            }
            if *k > M as u64 {
                want += 10;
            }
            assert_eq!(*val, want, "bounds must be honored exactly (key {k})");
        }
    }

    /// Ranges that cover no keys mutate nothing: empty and inverted
    /// windows, starts above every key (included or excluded), and the
    /// empty tree.
    #[test]
    fn range_mut_degenerate_cases_mutate_nothing() {
        let n = 2 * M as u64;
        let mut tree: BPlusTree<u64, u64, M> =
            BPlusTree::from_sorted_iter((0..n).map(|k| (k, v(k))));

        assert_eq!(tree.range_mut(5..5).count(), 0, "an empty window must yield nothing");
        #[allow(clippy::reversed_empty_ranges)]
        let inverted = 7..3;
        assert_eq!(tree.range_mut(inverted).count(), 0, "an inverted range must yield nothing");
        assert_eq!(tree.range_mut(n..).count(), 0, "a range above every key must yield nothing");
        assert_eq!(
            tree.range_mut((Bound::Excluded(n + 5), Bound::Unbounded)).count(),
            0,
            "an excluded start above every key must yield nothing"
        );

        for (k, val) in tree.iter() {
            assert_eq!(*val, v(*k), "degenerate ranges must not mutate anything (key {k})");
        }

        let mut empty: BPlusTree<u64, u64, M> = BPlusTree::new();
        assert_eq!(empty.range_mut(0..).count(), 0, "an empty tree must yield nothing");
    }
}
