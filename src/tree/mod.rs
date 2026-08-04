use core::mem::MaybeUninit;

use crate::{
    Key, Slabs,
    allocator::{Global, NodeAllocator},
};

mod bulk;

mod descent;

mod inner;
pub use inner::Inner;

mod iter;
pub(crate) use iter::{FullIterator, IntoIter, Range};

mod leaf;
pub use leaf::Leaf;

mod node;
pub(crate) use node::Node;
#[cfg(debug_assertions)]
pub(crate) use node::NodeKind;

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
/// The max levels `H` determines the amount of scratch space used for
/// descending the tree (e.g. for each insert/remove) or when bulk-loading the
/// tree. It defaults to [`usize::BITS`] levels, which admits any tree. If
/// your application is memory sensitive, you can tune this parameter to use
/// less stack memory for mutation operations. A tree of height `h` occupies
/// `h + 1` levels, so `H` is also a cap on how far the tree can grow: an
/// operation that must reach more levels than `H` provides panics rather
/// than writing out of bounds. `H` must be in `1..=usize::BITS`, checked at
/// compile time, where the tree is born:
///
/// ```compile_fail
/// // H = 0 has no room for even the root's level — this must not compile.
/// const M: usize = <u64 as beets::Key>::FANOUT;
/// let tree: beets::BPlusTree<u64, u64, M, beets::Slabs<u64, u64, M>, 0> =
///     beets::BPlusTree::new();
/// ```
///
/// A reasonable setting is `{ beets::max_levels(M) }` — the worst-case level
/// count for the fanout, safe for a tree of any size:
///
/// ```
/// use beets::{BPlusTree, Key, Slabs, max_levels};
///
/// const M: usize = <u64 as Key>::FANOUT;
/// let mut tree: BPlusTree<u64, &str, M, Slabs<u64, &str, M>, { max_levels(M) }> =
///     BPlusTree::new();
///
/// tree.insert(7, "seven");
/// assert_eq!(tree.get(&7), Some(&"seven"));
/// ```
///
/// Applications with known, fixed-size trees may set even lower values —
/// particularly useful for embedded applications. The application vouches
/// that the tree stays under the cap; one grown past it panics:
///
/// ```
/// use beets::{BPlusTree, Key, Slabs};
///
/// const M: usize = <u64 as Key>::FANOUT;
/// // A table of at most a few hundred entries never grows past
/// // height 2, so three level slots cover it — the mutation scratch
/// // shrinks from `usize::BITS` path slots to 3.
/// let mut tree: BPlusTree<u64, u64, M, Slabs<u64, u64, M>, 3> = BPlusTree::new();
///
/// for k in 0..300 {
///     tree.insert(k, k * 2);
/// }
/// assert_eq!(tree.len(), 300);
/// assert_eq!(tree.get(&250), Some(&500));
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
pub struct BPlusTree<
    K: Key,
    V,
    const M: usize,
    A: NodeAllocator<K, V, M> = Slabs<K, V, M, Global>,
    const H: usize = { crate::DEFAULT_MAX_LEVELS },
> {
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
unsafe impl<K, V, const M: usize, A, const H: usize> Send for BPlusTree<K, V, M, A, H>
where
    K: Key + Send,
    V: Send,
    A: NodeAllocator<K, V, M> + Send,
{
}

// SAFETY: sharing `&BPlusTree` shares a read-only tree. Every `&self`
// method is a pure read of the node graph — descents, gets, iteration;
// none mutates node memory through the `NonNull`s — and no `&self`
// path can MUTATE the allocator: [`NodeAllocator`]'s slot-traffic
// receivers are `&mut self`, unreachable through a shared borrow (its
// `&self` capacity queries are pure reads of plain fields). The tree
// itself has no interior mutability, so while shared borrows exist, no
// thread can write anything a reader dereferences. What readers DO
// reach — `&K`s and `&V`s — is what the `Sync` bounds sign for
// (`A: Sync` is defensive; no `&self` path reads it today).
unsafe impl<K, V, const M: usize, A, const H: usize> Sync for BPlusTree<K, V, M, A, H>
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

impl<K: Key, V, const M: usize, A: NodeAllocator<K, V, M>, const H: usize> Drop
    for BPlusTree<K, V, M, A, H>
{
    // Panic during teardown: NOT panic-safe. Teardown walks the tree dropping
    // values as it goes; a value `Drop` that unwinds leaks every node and
    // value not yet reached. Because this is itself `Drop` glue, such a panic
    // during an already-unwinding drop double-panics and aborts.
    fn drop(&mut self) {
        // SAFETY: the tree is mid-drop — no node pointer is read after
        // this, and the forgotten values were just checked drop-free.
        if !core::mem::needs_drop::<V>() && unsafe { self.allocator.reclaim_all() } {
            // Keys never need `drop`. When values don't need drop AND the
            // allocator can reclaim all memory itself, we can skip subtree
            // traversal.
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
impl<K: Key + Ord, V, const M: usize, A: NodeAllocator<K, V, M>, const H: usize>
    BPlusTree<K, V, M, A, H>
{
    /// A heuristic max height.
    pub const MAX_HEIGHT: usize = crate::max_height(M);

    const __LEVEL_CAP: () =
        assert!(H >= 1 && H <= usize::BITS as usize, "the level cap H must be in 1..=usize::BITS");

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
        const { Self::__LEVEL_CAP };

        let root = Node::from_leaf_ptr(allocator.alloc_leaf(Leaf::new(None)));
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

    /// Get a reference to the last leaf of the tree.
    pub(crate) fn last_leaf_mut(&mut self) -> &mut Leaf<K, V, M> {
        descend!(self, mut |inner| {
            let last = inner.len() - 1;
            &mut inner.children_mut()[last]
        })
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

    /// Get a reference to the stored key and value for `key`, if it is present
    pub fn get_key_value(&self, key: &K) -> Option<(&K, &V)> {
        self.find_leaf(key).get_kv(key)
    }

    /// Get a reference to the value for `key`, if it is present.
    pub fn get(&self, key: &K) -> Option<&V> {
        self.find_leaf(key).get(key)
    }

    /// True if `key` is present.
    pub fn contains_key(&self, key: &K) -> bool {
        self.get(key).is_some()
    }

    /// Get a mutable reference to the value for `key`, if it is present.
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        self.find_leaf_mut(key).get_mut(key)
    }

    /// The minimum-key pair, or `None` if the tree is empty.
    pub fn first_key_value(&self) -> Option<(&K, &V)> {
        // If the tree is non-empty, the first leaf is non empty
        (!self.is_empty()).then(|| {
            let leaf = self.first_leaf();
            leaf.kv_ref_unchecked(0)
        })
    }

    /// Copy the first key, and get a mutable reference to its value.
    pub fn first_key_value_mut(&mut self) -> Option<(K, &mut V)> {
        (!self.is_empty()).then(|| {
            let leaf = self.first_leaf_mut();
            leaf.kv_mut_unchecked(0)
        })
    }

    /// The maximum-key pair, or `None` if the tree is empty.
    pub fn last_key_value(&self) -> Option<(&K, &V)> {
        let leaf = self.last_leaf();
        leaf.len().checked_sub(1).map(|last| leaf.kv_ref_unchecked(last))
    }

    /// Copy the last key, and get a mutable reference to its value.
    pub fn last_key_value_mut(&mut self) -> Option<(K, &mut V)> {
        let leaf = self.last_leaf_mut();
        leaf.len().checked_sub(1).map(|last| leaf.kv_mut_unchecked(last))
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

    /// Remove `key`, returning the stored copy of the key as well as the value.
    pub fn remove_key_value(&mut self, key: &K) -> Option<(K, V)> {
        let mut slot = MaybeUninit::uninit();
        let descent = self.descend_into(key, &mut slot);
        if !descent.exact {
            return None;
        }

        // SAFETY: the descent is fresh from `descend` under this borrow,
        // the tree untouched since, and `exact` is true.
        Some(unsafe { descent.commit_remove() })
    }

    /// Remove `key`, returning its value if it was present.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        self.remove_key_value(key).map(|(_, v)| v)
    }

    /// Pop the first element of the tree, returning the KV pair.
    pub fn pop_first(&mut self) -> Option<(K, V)> {
        if self.is_empty() {
            return None;
        };
        let mut slot = MaybeUninit::uninit();
        let descent = self.descend_into_first(&mut slot);

        // SAFETY: the descent is fresh from `descend` under this borrow,
        // the tree untouched since, and `exact` is always true when
        // descending into first.
        Some(unsafe { descent.commit_remove() })
    }

    /// Pop the last element of the tree, returning the KV pair.
    pub fn pop_last(&mut self) -> Option<(K, V)> {
        if self.is_empty() {
            return None;
        };
        let mut slot = MaybeUninit::uninit();
        let descent = self.descend_into_last(&mut slot);

        // SAFETY: the descent is fresh from `descend` under this borrow,
        // the tree untouched since, and `exact` is always true when
        // descending into last.
        Some(unsafe { descent.commit_remove() })
    }

    /// Drop every pair, resetting to the empty tree.
    pub fn clear(&mut self) {
        // Wholesale reset when values carry no drop glue (`K: Copy`
        // always; `V` checked here) and the allocator can forget every
        // slot at once; the per-node walk otherwise.
        // SAFETY (reclaim_all): every node pointer it invalidates is
        // dead — the root is overwritten immediately below, before
        // anything can read it — and the forgotten values were just
        // checked drop-free.
        let reclaimed = !core::mem::needs_drop::<V>() && unsafe { self.allocator.reclaim_all() };

        if !reclaimed {
            // SAFETY:
            // - `height` is the impl-block invariant — exactly the
            // height of `root`'s subtree. `root` is overwritten with a fresh
            // node immediately below, before anything can read it.
            unsafe {
                let tree = core::ptr::read(&self.root);
                tree.drop_subtree(self.height, &mut self.allocator);
            }
        }

        self.root = Node::from_leaf_ptr(self.allocator.alloc_leaf(Leaf::new(None)));
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

    /// Move every pair of `other` into `self`, leaving `other` empty.
    /// Where a key exists in both, `self`'s previous value is dropped
    /// and `other`'s is kept — [`insert`](Self::insert)'s overwrite
    /// semantics, applied pairwise.
    ///
    /// Unlike [`BTreeMap::append`](https://doc.rust-lang.org/std/collections/struct.BTreeMap.html#method.append),
    /// which splices `other`'s nodes into `self` directly
    /// (`O(other.len() * log(self.len() / other.len()))`, no individual
    /// pair touched), this crate can't reparent nodes between two
    /// `allocator` instances the same way — each node an allocator hands
    /// out must be retired through that SAME allocator (see
    /// [`NodeAllocator`]'s contract), and `other`'s nodes belong to
    /// `other.allocator`. So the only correct route is pair-by-pair:
    /// drain `other` (its own allocator frees each node as it goes) and
    /// insert each pair into `self` — `O(other.len() * log(self.len()))`.
    pub fn append(&mut self, other: &mut Self) {
        while let Some((k, v)) = other.pop_first() {
            let _ = self.insert(k, v);
        }
    }
}

impl<K: Key + Ord, V, const M: usize, A: NodeAllocator<K, V, M> + Default, const H: usize> Default
    for BPlusTree<K, V, M, A, H>
{
    fn default() -> Self {
        Self::new()
    }
}

// NOT `#[derive(Clone)]`: a derived impl would require `A: Clone` and
// clone the `root`/`height`/`len` fields verbatim — copying the
// `root` handle would alias two trees onto the same nodes, and
// dropping both would double-free. Neither of this crate's allocators
// (`Slabs`, `Global`) implements `Clone` anyway (an arena's slots
// aren't safely duplicable); `Default` is the one they both give you.
// So this clones by CONTENT, through the same bulk-load path
// `FromIterator` already uses, into a freshly defaulted allocator —
// not a structural copy of `self`'s node layout. One consequence: the
// clone is bulk-packed (dense, near-`M`-per-leaf) even if `self` sits
// at `insert`-built ~2/3 occupancy.
impl<K: Key + Ord, V: Clone, const M: usize, A: NodeAllocator<K, V, M> + Default, const H: usize>
    Clone for BPlusTree<K, V, M, A, H>
{
    fn clone(&self) -> Self {
        Self::from_sorted_iter(self.iter().map(|(k, v)| (*k, v.clone())))
    }
}

impl<K, V, const M: usize, A: NodeAllocator<K, V, M>, const H: usize> core::fmt::Debug
    for BPlusTree<K, V, M, A, H>
where
    K: Key + core::fmt::Debug,
    V: core::fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

// NOT `#[derive(PartialEq)]`: a derived impl would compare the
// `root`/`height`/`len`/`allocator` fields directly — `root` is a
// `NonNull`-bearing handle, so that would compare ADDRESSES, not
// content (two trees holding identical pairs but built differently —
// one bulk-loaded, one `insert`-built — would wrongly compare
// unequal; a moved-then-rebuilt copy of the same logical tree would
// too). Compares by content instead, in key order, the same shape as
// `BTreeMap`'s own `PartialEq`.
//
// Scoped to identical `A`/`H`: there's no `PartialEq<BPlusTree<K, V,
// M, A2, H2>>` for a differing allocator type or level cap. That's a
// default scope call (std has no equivalent extra parameters to
// choose over), not a hard limitation — additional cross-parameter
// impls could be added the same way if that's ever wanted.
impl<K: Key + Ord, V: PartialEq, const M: usize, A: NodeAllocator<K, V, M>, const H: usize>
    PartialEq for BPlusTree<K, V, M, A, H>
{
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().eq(other.iter())
    }
}

impl<K: Key + Ord, V: Eq, const M: usize, A: NodeAllocator<K, V, M>, const H: usize> Eq
    for BPlusTree<K, V, M, A, H>
{
}

/// Lexicographic order over the `(key, value)` sequence, matching
/// [`BTreeMap`](https://doc.rust-lang.org/std/collections/struct.BTreeMap.html)'s
/// own `PartialOrd`/`Ord`. Same same-`A`/`H`-only scope as `PartialEq`
/// above.
// `partial_cmp` can't canonically delegate to `Ord::cmp` (clippy's
// usual ask): this impl's bound is `V: PartialOrd`, strictly weaker
// than `Ord`'s `V: Ord` below — `f64` values, say, are `PartialOrd`
// but not `Ord`, and `Self: Ord` isn't available in that case. `Ord`'s
// own bound gives it a matching, independent implementation instead —
// mirroring `BTreeMap`'s own split for exactly this reason.
#[allow(clippy::non_canonical_partial_ord_impl)]
impl<K: Key + Ord, V: PartialOrd, const M: usize, A: NodeAllocator<K, V, M>, const H: usize>
    PartialOrd for BPlusTree<K, V, M, A, H>
{
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.iter().partial_cmp(other.iter())
    }
}

impl<K: Key + Ord, V: Ord, const M: usize, A: NodeAllocator<K, V, M>, const H: usize> Ord
    for BPlusTree<K, V, M, A, H>
{
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.iter().cmp(other.iter())
    }
}

// NOT `#[derive(Hash)]`: a derived impl would hash the `root` handle
// (an address) alongside `height`/`len`/`allocator` — inconsistent
// with the content-based `Eq` above (the `Hash`/`Eq` contract demands
// equal values hash equal, and two content-equal trees can disagree
// on address, height, and allocator state). Hashes the length, then
// every pair in key order instead — matching `BTreeMap`'s own `Hash`,
// and consistent with `Eq` regardless of `A`/`H` or internal shape.
impl<K, V, const M: usize, A: NodeAllocator<K, V, M>, const H: usize> core::hash::Hash
    for BPlusTree<K, V, M, A, H>
where
    K: Key + Ord + core::hash::Hash,
    V: core::hash::Hash,
{
    fn hash<Hr: core::hash::Hasher>(&self, state: &mut Hr) {
        self.len().hash(state);
        for pair in self.iter() {
            pair.hash(state);
        }
    }
}

impl<K: Key + Ord, V, const M: usize, A: NodeAllocator<K, V, M>, const H: usize>
    core::ops::Index<&K> for BPlusTree<K, V, M, A, H>
{
    type Output = V;

    /// Panics if `key` is absent — matches
    /// [`BTreeMap`](https://doc.rust-lang.org/std/collections/struct.BTreeMap.html)'s
    /// `Index`. Takes `&K` rather than `&Q where K: Borrow<Q>`: this
    /// crate has no heterogeneous-lookup path yet (see the
    /// `get`/`get_mut`/`contains_key`/`remove`/`get_key_value`/
    /// `remove_key_value`/`range`/`range_mut` callsite list) — widen
    /// this alongside that work if it lands, rather than assuming `&K`
    /// is final.
    fn index(&self, key: &K) -> &Self::Output {
        self.get(key).expect("no entry found for key")
    }
}

/// `BTreeMap` deliberately has NO `IndexMut` (`map[k] = v` would be
/// ambiguous on a missing key: insert, or panic?). This crate adds one
/// anyway, at the user's request — scoped to the unambiguous half of
/// that question: panic on a missing key, `&mut V` on a hit, no
/// implicit insert. Flagging the divergence rather than silently
/// diverging from std.
impl<K: Key + Ord, V, const M: usize, A: NodeAllocator<K, V, M>, const H: usize>
    core::ops::IndexMut<&K> for BPlusTree<K, V, M, A, H>
{
    fn index_mut(&mut self, key: &K) -> &mut Self::Output {
        self.get_mut(key).expect("no entry found for key")
    }
}

impl<K: Key + Ord, V, const M: usize, A: NodeAllocator<K, V, M> + Default, const H: usize>
    FromIterator<(K, V)> for BPlusTree<K, V, M, A, H>
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

impl<K: Key + Ord, V, const M: usize, A: NodeAllocator<K, V, M>, const H: usize> Extend<(K, V)>
    for BPlusTree<K, V, M, A, H>
{
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        for (key, val) in iter {
            self.insert(key, val);
        }
    }
}

// Under plain `testutils` (the fuzz targets' build, no `cfg(test)`)
// this module is just the harness's test-only views into the private
// fields; the contract tests inside are `#[cfg(test)]`-gated.
#[cfg(any(test, feature = "testutils"))]
#[path = "../tests/tree.rs"]
mod tests;
