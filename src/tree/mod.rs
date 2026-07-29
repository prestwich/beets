use core::mem::MaybeUninit;

use crate::{
    Key, Slabs,
    allocator::{Global, NodeAllocator, SlotAllocator},
};

mod bulk;

mod descent;

mod inner;
pub use inner::Inner;

mod iter;

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

// Under plain `testutils` (the fuzz targets' build, no `cfg(test)`)
// this module is just the harness's test-only views into the private
// fields; the contract tests inside are `#[cfg(test)]`-gated.
#[cfg(any(test, feature = "testutils"))]
#[path = "../tests/tree.rs"]
mod tests;
