use core::{ops::Bound, ptr::NonNull};

use crate::{BPlusTree, Key, Leaf, allocator::NodeAllocator};

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
    pub(crate) fn new<A: NodeAllocator<K, V, M>, const H: usize>(
        tree: &'a BPlusTree<K, V, M, A, H>,
    ) -> Self {
        Self { len: tree.len(), inner: Iterator::from_start(tree) }
    }
}

impl<'a, K: Key, V, const M: usize> FullIteratorMut<'a, K, V, M> {
    pub(crate) fn new<A: NodeAllocator<K, V, M>, const H: usize>(
        tree: &'a mut BPlusTree<K, V, M, A, H>,
    ) -> Self {
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
    pub(crate) fn from_start<A: NodeAllocator<K, V, M>, const H: usize>(
        tree: &'a mut BPlusTree<K, V, M, A, H>,
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
    pub(crate) fn new<R: core::ops::RangeBounds<K>, A: NodeAllocator<K, V, M>, const H: usize>(
        tree: &'a BPlusTree<K, V, M, A, H>,
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
    pub(crate) fn new<R: core::ops::RangeBounds<K>, A: NodeAllocator<K, V, M>, const H: usize>(
        tree: &'a mut BPlusTree<K, V, M, A, H>,
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
    pub(crate) fn from_start<A: NodeAllocator<K, V, M>, const H: usize>(
        tree: &'a BPlusTree<K, V, M, A, H>,
    ) -> Self {
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

impl<'a, K: Key, V, const M: usize, A: NodeAllocator<K, V, M>, const H: usize> IntoIterator
    for &'a BPlusTree<K, V, M, A, H>
{
    type Item = (&'a K, &'a V);

    type IntoIter = FullIterator<'a, K, V, M>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, K: Key, V, const M: usize, A: NodeAllocator<K, V, M>, const H: usize> IntoIterator
    for &'a mut BPlusTree<K, V, M, A, H>
{
    type Item = (&'a K, &'a mut V);

    type IntoIter = FullIteratorMut<'a, K, V, M>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

#[cfg(test)]
#[path = "../tests/iter.rs"]
mod tests;
