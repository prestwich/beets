use core::{iter::FusedIterator, ops::RangeBounds};

use crate::{
    BPlusTree, FullIterator, IntoIter as TreeIntoIter, Key, Range, Slabs,
    allocator::{Global, NodeAllocator},
};

fn drop_1<T, U>(pair: (T, U)) -> T {
    pair.0
}

fn set_1<T>(item: T) -> (T, ()) {
    (item, ())
}

fn transform_iter<T, I: IntoIterator<Item = T>>(iter: I) -> impl IntoIterator<Item = (T, ())> {
    iter.into_iter().map(set_1)
}

/// A sorted set of unique keys.
///
/// Backed by a [`BPlusTree`] whose values are all `()` — every method
/// here is a thin wrapper over the tree's own API, dropping the value
/// half of each pair. See [`BPlusTree`] for the meaning of `M`, `A`,
/// and `H`.
///
/// ```rust,ignore
/// use beets::{BPlusSet, Key};
///
/// let mut set = BPlusSet::<u64, { <u64 as Key>::FANOUT }>::new();
/// set.insert(3);
/// set.insert(1);
///
/// assert!(set.contains(&1));
/// assert_eq!(set.len(), 2);
/// assert!(set.iter().eq([&1, &3]));
/// ```
#[repr(transparent)]
pub struct BPlusSet<
    K: Key,
    const M: usize,
    A: NodeAllocator<K, (), M> = Slabs<K, (), M, Global>,
    const H: usize = { crate::DEFAULT_MAX_LEVELS },
>(BPlusTree<K, (), M, A, H>);

impl<K: Key + Ord, const M: usize, A: NodeAllocator<K, (), M>, const H: usize>
    BPlusSet<K, M, A, H>
{
    /// A heuristic max height. See [`BPlusTree::MAX_HEIGHT`].
    pub const MAX_HEIGHT: usize = crate::max_height(M);

    /// Creates an empty set.
    pub fn new() -> Self
    where
        A: Default,
    {
        Self(BPlusTree::<K, (), M, A, H>::new())
    }

    /// As [`Self::new`], but allocating nodes from `allocator` for the
    /// set's whole life.
    pub fn new_in(allocator: A) -> Self {
        Self(BPlusTree::<K, (), M, A, H>::new_in(allocator))
    }

    /// The number of keys in the set.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// True if the set holds no keys.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// True if `key` is present.
    pub fn contains(&self, key: &K) -> bool {
        self.0.contains_key(key)
    }

    /// Get a reference to the stored copy of `key`, if present.
    ///
    /// Useful when `K`'s `Eq`/`Ord` impl considers values equal that are
    /// not identical (e.g. an interned or wrapped type) and the caller
    /// wants the copy the set actually holds.
    pub fn get(&self, key: &K) -> Option<&K> {
        self.0.get_key_value(key).map(drop_1)
    }

    /// The minimum key, or `None` if the set is empty.
    pub fn first(&self) -> Option<&K> {
        self.0.first_key_value().map(drop_1)
    }

    /// The maximum key, or `None` if the set is empty.
    pub fn last(&self) -> Option<&K> {
        self.0.last_key_value().map(drop_1)
    }

    /// Insert `key`. Returns `true` if it was newly inserted, `false` if
    /// it was already present (and left unchanged — sets have no
    /// second value to overwrite).
    pub fn insert(&mut self, key: K) -> bool {
        self.0.insert(key, ()).is_none()
    }

    /// Remove `key`. Returns `true` if it was present.
    pub fn remove(&mut self, key: &K) -> bool {
        self.0.remove(key).is_some()
    }

    /// Remove and return the minimum key, or `None` if the set is empty.
    pub fn pop_first(&mut self) -> Option<K> {
        self.0.pop_first().map(drop_1)
    }

    /// Remove and return the maximum key, or `None` if the set is empty.
    pub fn pop_last(&mut self) -> Option<K> {
        self.0.pop_last().map(drop_1)
    }

    /// Remove every key, resetting to the empty set.
    pub fn clear(&mut self) {
        self.0.clear()
    }

    /// Remove the key equal to `key`, returning the ACTUAL stored copy
    /// — not necessarily identical to the query, when `K`'s `Ord`/`Eq`
    /// considers values equal that aren't identical (see the note on
    /// [`Self::get`]). Matches
    /// [`BTreeSet::take`](https://doc.rust-lang.org/std/collections/struct.BTreeSet.html#method.take).
    ///
    /// No new [`BPlusTree`] primitive needed: `self.0.remove_key_value(key)`
    /// already hands back the stored `(K, ())` pair.
    pub fn take(&mut self, key: &K) -> Option<K> {
        self.0.remove_key_value(key).map(drop_1)
    }

    /// Insert `key`, replacing the stored copy if an equal one was
    /// already present, and returning that OLD copy (`None` if newly
    /// inserted). Distinct from [`Self::insert`]: [`BPlusTree::insert`]
    /// (like `BTreeMap::insert`) never touches an already-present key,
    /// only the paired value, so getting `replace`'s identity swap
    /// takes an explicit remove-then-insert. Matches
    /// [`BTreeSet::replace`](https://doc.rust-lang.org/std/collections/struct.BTreeSet.html#method.replace).
    pub fn replace(&mut self, key: K) -> Option<K> {
        let old = self.take(&key);
        self.insert(key);
        old
    }

    /// See [`BPlusTree::append`] — same pair-by-pair semantics (and the
    /// same reason a structural splice across two allocator instances
    /// isn't available).
    pub fn append(&mut self, other: &mut Self) {
        while let Some(k) = other.pop_first() {
            self.insert(k);
        }
    }

    /// See [`BPlusTree::split_off`] — same fresh-allocator requirement
    /// and the same reason it isn't a structural `O(log n)` split.
    pub fn split_off(&mut self, _key: &K) -> Self
    where
        A: Default,
    {
        todo!("build `Self::new()`, then move every key >= `key` from `self` into it")
    }

    /// Keep only the keys for which `f` returns `true`, dropping the
    /// rest. Sets have no values to hand `f` mutable access to, so `f`
    /// takes just `&K` — matches
    /// [`BTreeSet::retain`](https://doc.rust-lang.org/std/collections/struct.BTreeSet.html#method.retain).
    /// See [`BPlusTree::retain`] for why this is a collect-then-remove
    /// pass rather than a structural in-place walk.
    pub fn retain<F: FnMut(&K) -> bool>(&mut self, mut _f: F) {
        todo!("collect keys where `!f(k)` via `iter`, then `remove` each")
    }

    /// Iterate over all keys, in ascending order.
    pub fn iter(&self) -> Iter<FullIterator<'_, K, (), M>> {
        Iter(self.0.iter())
    }

    /// Iterate over the keys that fall in `range`, in ascending order.
    pub fn range<R: RangeBounds<K>>(&self, range: R) -> Iter<Range<'_, K, (), M>> {
        Iter(self.0.range(range))
    }

    /// True if every key in `self` is also in `other`.
    pub fn is_subset(&self, other: &Self) -> bool {
        self.intersection(other).count() == self.len()
    }

    /// True if every key in `other` is also in `self`.
    pub fn is_superset(&self, other: &Self) -> bool {
        other.is_subset(self)
    }

    /// True if `self` and `other` share no keys.
    pub fn is_disjoint(&self, other: &Self) -> bool {
        self.intersection(other).next().is_none()
    }

    /// Iterate over the keys in `self`, `other`, or both, in ascending
    /// order, without duplicates.
    pub fn union<'a>(&'a self, other: &'a Self) -> Union<FullIterator<'a, K, (), M>> {
        Union(self.0.iter().peekable(), other.0.iter().peekable())
    }

    /// Iterate over the keys in both `self` and `other`, in ascending
    /// order.
    pub fn intersection<'a>(&'a self, other: &'a Self) -> Intersection<FullIterator<'a, K, (), M>> {
        Intersection(self.0.iter().peekable(), other.0.iter().peekable())
    }

    /// Iterate over the keys in `self` but not `other`, in ascending
    /// order.
    pub fn difference<'a>(&'a self, other: &'a Self) -> Difference<FullIterator<'a, K, (), M>> {
        Difference(self.0.iter().peekable(), other.0.iter().peekable())
    }

    /// Iterate over the keys in `self` or `other` but not both, in
    /// ascending order.
    pub fn symmetric_difference<'a>(
        &'a self,
        other: &'a Self,
    ) -> SymmetricDifference<FullIterator<'a, K, (), M>> {
        SymmetricDifference(self.0.iter().peekable(), other.0.iter().peekable())
    }

    /// Bulk-load a set from a stream of keys sorted strictly ascending.
    /// See [`BPlusTree::from_sorted_iter`].
    pub fn from_sorted_iter<I: IntoIterator<Item = K>>(iter: I) -> Self
    where
        A: Default,
    {
        Self(BPlusTree::from_sorted_iter(transform_iter(iter)))
    }

    /// As [`Self::from_sorted_iter`], but allocating nodes from
    /// `allocator` for the set's whole life.
    pub fn from_sorted_iter_in<I: IntoIterator<Item = K>>(iter: I, allocator: A) -> Self {
        Self(BPlusTree::from_sorted_iter_in(transform_iter(iter), allocator))
    }
}

impl<K: Key + Ord, const M: usize, A: NodeAllocator<K, (), M> + Default, const H: usize> Default
    for BPlusSet<K, M, A, H>
{
    fn default() -> Self {
        Self::new()
    }
}

/// See [`BPlusTree`]'s `Clone` — same bulk-rebuild-into-a-fresh-`A::default()`
/// approach (this crate's allocators have no `Clone` of their own),
/// same denser-than-`insert`-built-sets caveat.
impl<K: Key + Ord, const M: usize, A: NodeAllocator<K, (), M> + Default, const H: usize> Clone
    for BPlusSet<K, M, A, H>
{
    fn clone(&self) -> Self {
        Self::from_sorted_iter(self.iter().copied())
    }
}

impl<K, const M: usize, A: NodeAllocator<K, (), M>, const H: usize> core::fmt::Debug
    for BPlusSet<K, M, A, H>
where
    K: Key + core::fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

// NOT `#[derive(...)]` for any of `PartialEq`/`Eq`/`PartialOrd`/`Ord`/`Hash`:
// see the parallel note on [`BPlusTree`]'s impls — a derived impl would
// compare/hash the `BPlusTree` field (in turn its `root` handle) rather
// than content. `K: Ord` already gives `K: Eq` (a supertrait), so
// unlike the map these need no extra per-field bound: sets carry no
// `V` for the derive to (mis)reach in the first place.
impl<K: Key + Ord, const M: usize, A: NodeAllocator<K, (), M>, const H: usize> PartialEq
    for BPlusSet<K, M, A, H>
{
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().eq(other.iter())
    }
}

impl<K: Key + Ord, const M: usize, A: NodeAllocator<K, (), M>, const H: usize> Eq
    for BPlusSet<K, M, A, H>
{
}

/// Lexicographic key order, matching
/// [`BTreeSet`](https://doc.rust-lang.org/std/collections/struct.BTreeSet.html)'s
/// own `PartialOrd`/`Ord`. Same same-`A`/`H`-only scope as `PartialEq`
/// above.
///
/// Unlike `BPlusTree`'s split (there, `PartialOrd`'s `V: PartialOrd`
/// bound is strictly weaker than `Ord`'s `V: Ord`, so it can't borrow
/// `Self::cmp`): a set's only per-element bound is `K: Key + Ord`
/// either way, so `Self: Ord` is always available here, and delegating
/// is both canonical and correct.
impl<K: Key + Ord, const M: usize, A: NodeAllocator<K, (), M>, const H: usize> PartialOrd
    for BPlusSet<K, M, A, H>
{
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<K: Key + Ord, const M: usize, A: NodeAllocator<K, (), M>, const H: usize> Ord
    for BPlusSet<K, M, A, H>
{
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.iter().cmp(other.iter())
    }
}

impl<K: Key + Ord + core::hash::Hash, const M: usize, A: NodeAllocator<K, (), M>, const H: usize>
    core::hash::Hash for BPlusSet<K, M, A, H>
{
    fn hash<Hr: core::hash::Hasher>(&self, state: &mut Hr) {
        self.len().hash(state);
        for key in self.iter() {
            key.hash(state);
        }
    }
}

impl<K: Key + Ord, const M: usize, A: NodeAllocator<K, (), M> + Default, const H: usize>
    FromIterator<K> for BPlusSet<K, M, A, H>
{
    fn from_iter<I: IntoIterator<Item = K>>(iter: I) -> Self {
        Self(BPlusTree::from_iter(transform_iter(iter)))
    }
}

impl<K: Key + Ord, const M: usize, A: NodeAllocator<K, (), M>, const H: usize> Extend<K>
    for BPlusSet<K, M, A, H>
{
    fn extend<I: IntoIterator<Item = K>>(&mut self, iter: I) {
        self.0.extend(transform_iter(iter))
    }
}

impl<'a, K: Key + Ord + 'a, const M: usize, A: NodeAllocator<K, (), M>, const H: usize>
    Extend<&'a K> for BPlusSet<K, M, A, H>
{
    fn extend<I: IntoIterator<Item = &'a K>>(&mut self, iter: I) {
        self.0.extend(transform_iter(iter.into_iter().copied()))
    }
}

// `&BPlusSet op &BPlusSet -> BPlusSet`, matching `BTreeSet`'s own
// `BitAnd`/`BitOr`/`BitXor`/`Sub` — thin wrappers over the existing
// `intersection`/`union`/`symmetric_difference`/`difference` cursors,
// materialized into a fresh set via the same bulk `FromIterator` path
// `collect` already uses (hence the extra `A: Default`, matching
// `FromIterator`'s own bound).
impl<K: Key + Ord, const M: usize, A: NodeAllocator<K, (), M> + Default, const H: usize>
    core::ops::BitAnd<&Self> for &BPlusSet<K, M, A, H>
{
    type Output = BPlusSet<K, M, A, H>;

    /// The intersection of the two sets.
    fn bitand(self, rhs: &Self) -> Self::Output {
        self.intersection(rhs).copied().collect()
    }
}

impl<K: Key + Ord, const M: usize, A: NodeAllocator<K, (), M> + Default, const H: usize>
    core::ops::BitOr<&Self> for &BPlusSet<K, M, A, H>
{
    type Output = BPlusSet<K, M, A, H>;

    /// The union of the two sets.
    fn bitor(self, rhs: &Self) -> Self::Output {
        self.union(rhs).copied().collect()
    }
}

impl<K: Key + Ord, const M: usize, A: NodeAllocator<K, (), M> + Default, const H: usize>
    core::ops::BitXor<&Self> for &BPlusSet<K, M, A, H>
{
    type Output = BPlusSet<K, M, A, H>;

    /// The keys in exactly one of the two sets.
    fn bitxor(self, rhs: &Self) -> Self::Output {
        self.symmetric_difference(rhs).copied().collect()
    }
}

impl<K: Key + Ord, const M: usize, A: NodeAllocator<K, (), M> + Default, const H: usize>
    core::ops::Sub<&Self> for &BPlusSet<K, M, A, H>
{
    type Output = BPlusSet<K, M, A, H>;

    /// The keys in `self` but not `rhs`.
    fn sub(self, rhs: &Self) -> Self::Output {
        self.difference(rhs).copied().collect()
    }
}

impl<'a, K: Key, const M: usize, A: NodeAllocator<K, (), M>, const H: usize> IntoIterator
    for &'a BPlusSet<K, M, A, H>
{
    type Item = &'a K;
    type IntoIter = Iter<FullIterator<'a, K, (), M>>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<K: Key, const M: usize, A: NodeAllocator<K, (), M>, const H: usize> IntoIterator
    for BPlusSet<K, M, A, H>
{
    type Item = K;
    type IntoIter = IntoIter<K, M, A, H>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter(self.0.into_iter())
    }
}

/// Owned iterator over a [`BPlusSet`]'s keys, in ascending order — the
/// by-value counterpart to [`Iter`] (which iterates by reference).
/// Just unwraps the tree's own owned iterator's `()` half; the real
/// unsafe teardown work lives entirely in [`BPlusTree`]'s `IntoIter`
/// (`tree/iter.rs`) — see that type's doc comment for the safety
/// contract a real implementation has to satisfy.
pub struct IntoIter<K: Key, const M: usize, A: NodeAllocator<K, (), M>, const H: usize>(
    TreeIntoIter<K, (), M, A, H>,
);

impl<K: Key, const M: usize, A: NodeAllocator<K, (), M>, const H: usize> Iterator
    for IntoIter<K, M, A, H>
{
    type Item = K;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(drop_1)
    }
}

impl<K: Key, const M: usize, A: NodeAllocator<K, (), M>, const H: usize> ExactSizeIterator
    for IntoIter<K, M, A, H>
{
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl<K: Key, const M: usize, A: NodeAllocator<K, (), M>, const H: usize> FusedIterator
    for IntoIter<K, M, A, H>
{
}

/// Iterates over the keys of a [`BPlusSet`], discarding the paired `()`
/// value that the underlying tree carries alongside every key.
///
/// Generic over the wrapped cursor so it serves both
/// [`BPlusSet::iter`] (a [`FullIterator`]) and [`BPlusSet::range`] (a
/// [`Range`]).
pub struct Iter<I>(I);

impl<'a, K: 'a, I> Iterator for Iter<I>
where
    I: Iterator<Item = (&'a K, &'a ())>,
{
    type Item = &'a K;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(drop_1)
    }
}

impl<'a, K: 'a, I> ExactSizeIterator for Iter<I>
where
    I: ExactSizeIterator<Item = (&'a K, &'a ())>,
{
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl<'a, K: 'a, I> FusedIterator for Iter<I> where I: FusedIterator<Item = (&'a K, &'a ())> {}

/// Iterates over the keys in either of two sets (or both), in ascending
/// order, without duplicates. Built by walking each set's key/value cursor in
/// lockstep, merge-style, dropping the paired `()` values.
pub struct Union<I: Iterator>(core::iter::Peekable<I>, core::iter::Peekable<I>);

impl<'a, K, I> Iterator for Union<I>
where
    K: Ord + 'a,
    I: Iterator<Item = (&'a K, &'a ())>,
{
    type Item = &'a K;

    fn next(&mut self) -> Option<Self::Item> {
        // If left is exhausted, return right.
        let Some((this, _)) = self.0.peek() else {
            return self.1.next().map(drop_1);
        };

        let Some((that, _)) = self.1.peek() else {
            return self.0.next().map(drop_1);
        };

        // if same, pop both. return left.
        // return lesser of the others.
        if this == that {
            let _ = self.1.next();
            self.0.next().map(drop_1)
        } else if this < that {
            self.0.next().map(drop_1)
        } else {
            self.1.next().map(drop_1)
        }
    }
}

impl<'a, K, I> FusedIterator for Union<I>
where
    K: Ord + 'a,
    I: FusedIterator<Item = (&'a K, &'a ())>,
{
}

/// Iterates over the keys present in both sets, in ascending order.
/// Built by walking each set's key/value cursor in lockstep,
/// merge-style, dropping the paired `()` values.
pub struct Intersection<I: Iterator>(core::iter::Peekable<I>, core::iter::Peekable<I>);

impl<'a, K, I> Iterator for Intersection<I>
where
    K: Ord + 'a,
    I: Iterator<Item = (&'a K, &'a ())>,
{
    type Item = &'a K;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // If either is exhausted, we're done.
            let (this, _) = self.0.peek()?;
            let (that, _) = self.1.peek()?;

            // if same, pop both, return left.
            // otherwise, pop lesser, loop.
            if this == that {
                let _ = self.1.next();
                return self.0.next().map(drop_1);
            } else if this < that {
                let _ = self.0.next();
            } else {
                let _ = self.1.next();
            }
        }
    }
}

impl<'a, K, I> FusedIterator for Intersection<I>
where
    K: Ord + 'a,
    I: FusedIterator<Item = (&'a K, &'a ())>,
{
}

/// Iterates over the keys in the first set but not the second, in
/// ascending order. Built by walking each set's key/value cursor in lockstep,
/// merge-style, dropping the paired `()` values.
pub struct Difference<I: Iterator>(core::iter::Peekable<I>, core::iter::Peekable<I>);

impl<'a, K, I> Iterator for Difference<I>
where
    K: Ord + 'a,
    I: Iterator<Item = (&'a K, &'a ())>,
{
    type Item = &'a K;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // If this is exhausted, we're done.
            let (this, _) = self.0.peek()?;
            // If that is exhausted, we're at the tail.
            let Some((that, _)) = self.1.peek() else {
                return Some(this);
            };

            // If they're the same, advance both and loop.
            // If this < that return this
            // If that < this advance that and reloop.
            if this == that {
                let _ = self.0.next();
                let _ = self.1.next();
            } else if this < that {
                return self.0.next().map(drop_1);
            } else {
                let _ = self.1.next();
            }
        }
    }
}

impl<'a, K, I> FusedIterator for Difference<I>
where
    K: Ord + 'a,
    I: FusedIterator<Item = (&'a K, &'a ())>,
{
}

/// Iterates over the keys in exactly one of the two sets, in ascending
/// order. Built by walking each set's key/value cursor in lockstep,
/// merge-style, dropping the paired `()` values.
pub struct SymmetricDifference<I: Iterator>(core::iter::Peekable<I>, core::iter::Peekable<I>);

impl<'a, K, I> Iterator for SymmetricDifference<I>
where
    K: Ord + 'a,
    I: Iterator<Item = (&'a K, &'a ())>,
{
    type Item = &'a K;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // If left is exhausted, then we just need the tail of right.
            let Some((this, _)) = self.0.peek() else {
                return self.1.next().map(drop_1);
            };
            // If right is exhausted, then we just need the tail of left.
            let Some((that, _)) = self.1.peek() else {
                return self.0.next().map(drop_1);
            };

            // If equal, advance and reloop
            if this == that {
                let _ = self.0.next();
                let _ = self.1.next();
            } else if this < that {
                return self.0.next().map(drop_1);
            } else {
                return self.1.next().map(drop_1);
            }
        }
    }
}

impl<'a, K, I> FusedIterator for SymmetricDifference<I>
where
    K: Ord + 'a,
    I: FusedIterator<Item = (&'a K, &'a ())>,
{
}

#[cfg(test)]
#[path = "tests/set.rs"]
mod tests;
