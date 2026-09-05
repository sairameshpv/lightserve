"""Prefix caching: if a new request's prompt starts with the same tokens as
a prompt we've already computed, reuse that earlier computation instead of
redoing it.

How it works, in plain terms:

  - Tokens are grouped into fixed-size "blocks" -- the same blocks
    BlockManager already uses for memory. We only ever match/cache whole
    blocks, never part of one.
  - Every block gets a hash. The hash depends on that block's own tokens
    AND every block before it. This way, two blocks only count as "the
    same" if their entire history matches -- not just that one block's
    tokens happen to look the same somewhere else.
  - These hashes form a tree (a "trie"). Each path down from the root is
    one possible sequence of blocks. A new request walks this tree to see
    how much of its own prompt already exists.
  - When a match is found, the request reuses those same physical memory
    blocks instead of recomputing them -- that's the "skip prefill" part.
  - Because a block can now be used by more than one request at a time, we
    count how many requests are currently relying on it (`ref_count`). We
    only allow a block to be reused for something else once nobody needs
    it anymore.
  - A block that nobody needs right now isn't deleted immediately -- it's
    kept around in case a future request matches it too. It only actually
    gets reclaimed ("evicted") once we run out of free blocks, and we
    always reclaim the least-recently-used one first.
  - We only ever cache prompt tokens, never generated output tokens. Two
    requests can share the same prompt but still generate different
    replies, so caching output tokens wouldn't be safe to reuse.
  - We keep simple counters so a hit rate can be reported (see
    PrefixCacheStats below).
"""
from dataclasses import dataclass, field
from typing import Callable, Optional

ROOT_HASH = None  # stand-in "parent hash" for a sequence's very first block


def hash_block(parent_hash: Optional[int], block_token_ids: tuple) -> int:
    """Computes a hash for one block, chained to its parent block's hash
    (see the module docstring for why chaining matters).

    We use Python's plain `hash()` here instead of something like sha256 --
    it's fast, and good enough, because we never *trust* the hash alone:
    `RadixTrie.match`/`insert` both double-check the actual tokens too. If
    two different blocks ever hash to the same value (a "collision"), we
    still catch it there and treat it as no match -- a collision can never
    cause a wrong answer, only a missed opportunity to share.

    `hash_fn` on `RadixTrie` can be swapped out in tests to force a
    collision on purpose, so that safety check can be tested directly.
    """
    return hash((parent_hash, block_token_ids))


@dataclass
class TrieNode:
    """One block of tokens whose KV cache has been fully computed and is
    now cached. `physical_block_id` points at the actual memory block that
    holds its data.
    """
    block_hash: int
    parent_hash: Optional[int]
    token_ids: tuple
    physical_block_id: int
    ref_count: int = 0  # how many requests are currently relying on this block
    children: dict = field(default_factory=dict)  # child block_hash -> TrieNode
    last_used: int = 0  # a simple counter, bumped every time this block is touched


@dataclass
class PrefixMatch:
    """The result of checking a prompt against the cache. `physical_block_ids`
    are the blocks to reuse and `num_matched_tokens` is how many tokens they
    cover. `_nodes` is only for RadixTrie's own internal use.
    """
    physical_block_ids: list
    num_matched_tokens: int
    _nodes: list = field(default_factory=list, repr=False)


@dataclass
class PrefixCacheStats:
    """Simple hit-rate counters, added up over the whole life of the cache."""
    num_lookup_tokens: int
    num_hit_tokens: int
    num_lookups: int
    num_hits: int

    @property
    def hit_rate(self) -> float:
        """Fraction of all prompt tokens we looked up that were actually
        served from the cache instead of recomputed. This is the main
        number to look at.
        """
        if self.num_lookup_tokens == 0:
            return 0.0
        return self.num_hit_tokens / self.num_lookup_tokens

    @property
    def request_hit_rate(self) -> float:
        """Fraction of requests that got *any* cache hit at all, even a
        small one. A secondary, easier-to-eyeball number.
        """
        if self.num_lookups == 0:
            return 0.0
        return self.num_hits / self.num_lookups


class RadixTrie:
    """A tree of cached token blocks, used to find out how much of a new
    prompt has already been computed by some earlier request.

    Matching only ever happens in whole blocks -- never part of one --
    because the underlying KV-cache memory is itself organized in whole
    blocks (see block_manager.py).
    """

    def __init__(self, block_size: int, hash_fn: Callable = hash_block):
        self.block_size = block_size
        self.hash_fn = hash_fn
        self._roots: dict = {}    # hash of a sequence's first block -> TrieNode
        self._by_hash: dict = {}  # every node in the tree, keyed by its own hash
        self._clock = 0           # simple counter used to track "least recently used"
        self._num_lookup_tokens = 0
        self._num_hit_tokens = 0
        self._num_lookups = 0
        self._num_hits = 0

    # -- Lookup ---------------------------------------------------------

    def match(self, token_ids: list) -> PrefixMatch:
        """Checks how much of `token_ids` is already cached -- a read-only
        "just looking" operation. It never changes ref counts or stats, so
        it's safe to call as many times as we like (for example: checking
        once to see if there's room, then checking again right before
        actually committing to using the match -- see `acquire`).

        Walks block by block from the top of the tree. Stops as soon as a
        block doesn't match (either the hash is different, or the hash
        happens to collide but the actual tokens differ -- see hash_block's
        docstring) or fewer than a full block's worth of tokens is left.
        """
        matched_nodes = []
        parent_hash = ROOT_HASH
        level = self._roots
        n_full_blocks = len(token_ids) // self.block_size
        for i in range(n_full_blocks):
            block = tuple(token_ids[i * self.block_size:(i + 1) * self.block_size])
            h = self.hash_fn(parent_hash, block)
            node = level.get(h)
            if node is None or node.token_ids != block:
                break
            matched_nodes.append(node)
            parent_hash = h
            level = node.children
        return PrefixMatch(
            physical_block_ids=[n.physical_block_id for n in matched_nodes],
            num_matched_tokens=len(matched_nodes) * self.block_size,
            _nodes=matched_nodes,
        )

    def acquire(self, match: PrefixMatch, num_lookup_tokens: int) -> None:
        """Confirms we're actually going to use a match found by `match()`.

        Call this exactly once, at the moment a request is genuinely
        admitted -- never from a "just checking" call. It does two things:

          1. Marks every matched block as "in use" (increases its
             ref_count), so it can't be evicted out from under this
             request while it still needs it.
          2. Records the attempt in our hit-rate stats -- whether it was a
             hit or a total miss, so the hit rate reflects every real
             attempt, not just the successful ones.

        `num_lookup_tokens` is just the prompt length we tried to match --
        it's the denominator for the hit-rate calculation.
        """
        self._clock += 1
        for node in match._nodes:
            node.ref_count += 1
            node.last_used = self._clock
        self._num_lookups += 1
        self._num_lookup_tokens += num_lookup_tokens
        if match._nodes:
            self._num_hits += 1
            self._num_hit_tokens += match.num_matched_tokens

    def release(self, nodes: list) -> None:
        """Called when a request is done with some cached blocks (it
        finished, was freed, or was preempted). Lowers each block's
        ref_count by one -- it does NOT remove the block from the cache.
        A block reaching ref_count 0 just becomes *eligible* to be evicted
        later, if the space is ever needed (see `evict_one_lru`).
        """
        for node in nodes:
            if node.ref_count <= 0:
                raise ValueError(
                    f"release() on block {node.block_hash} with ref_count already {node.ref_count} "
                    "-- caller released more times than it acquired"
                )
            node.ref_count -= 1

    # -- Insertion --------------------------------------------------------

    def insert(self, token_ids: list, physical_block_ids: list, num_computed_tokens: int,
               known_prefix: tuple = ()) -> list:
        """Registers newly-computed blocks into the cache, so a future
        request's `match()` can find and reuse them.

        Only call this for tokens that have genuinely already been
        computed by a real forward pass -- never for tokens that are only
        planned. `num_computed_tokens` says how many tokens (from the
        start) are done; only whole blocks within that count get cached.

        Every block walked here (i.e. everything beyond `known_prefix`,
        see below) gets its ref_count bumped -- this is a hold, the same
        as acquire()'s, just reached via "I just computed this" instead of
        "I matched something someone else computed". Two different
        requests independently computing the same content (e.g.
        concurrent admission before either has registered anything, so
        neither matched the other at admission time) both get their own
        hold on the resulting shared node, exactly like two requests that
        matched it via `acquire()` would.

        `known_prefix`: this exact caller's own already-established prefix
        nodes, in order -- pass `request.cached_prefix_nodes` (whatever it
        currently holds: `()` for a request that's never registered
        anything, the matched nodes from admission-time `allocate()`, or
        the return value of an earlier `insert()` call for this same
        request). The walk resumes right after these, skipping re-hashing
        and re-looking-up every block the caller already knows about --
        needed both for correctness (this can safely be called again and
        again as more of a request's prompt gets computed across chunked
        prefill steps, so without skipping already-known blocks, every
        repeat call would re-bump their ref_count, inflating it far past
        the number of requests actually depending on the block) and for
        performance (a request whose entire prompt prefix was matched at
        admission has already paid for that walk once, in `match()` --
        re-walking all of it again here, every step until its tiny
        unmatched remainder finishes, is pure waste that scales with
        matched-prefix length instead of with what's actually new; this
        showed up as real, measured multi-second stalls -- see
        benchmarks/prefix_caching/README.md).

        Returns the list of blocks now registered: `known_prefix` plus
        whatever's newly found/created beyond it. May end up shorter than
        `num_computed_tokens // block_size` if a hash collision is hit
        partway through -- see below.

        Hash collision safety: if a block's hash matches an existing entry
        but the actual tokens are different, we stop right there instead of
        overwriting that entry. Overwriting would corrupt an
        already-cached block that some other request might still be using.
        """
        n_blocks = num_computed_tokens // self.block_size
        nodes = list(known_prefix)
        start = len(nodes)
        if start >= n_blocks:
            return nodes[:n_blocks]  # nothing new -- num_computed_tokens didn't grow
        self._clock += 1
        parent_hash = nodes[-1].block_hash if nodes else ROOT_HASH
        level = nodes[-1].children if nodes else self._roots
        for i in range(start, n_blocks):
            block = tuple(token_ids[i * self.block_size:(i + 1) * self.block_size])
            h = self.hash_fn(parent_hash, block)
            node = level.get(h)
            if node is not None and node.token_ids != block:
                break  # hash collision with different content -- see docstring above
            if node is None:
                node = TrieNode(
                    block_hash=h,
                    parent_hash=parent_hash,
                    token_ids=block,
                    physical_block_id=physical_block_ids[i],
                    ref_count=0,
                    last_used=self._clock,
                )
                level[h] = node
                self._by_hash[h] = node
            node.ref_count += 1
            node.last_used = self._clock
            nodes.append(node)
            parent_hash = h
            level = node.children
        return nodes

    # -- Eviction ---------------------------------------------------------

    @property
    def num_evictable_blocks(self) -> int:
        """How many cached blocks could be reclaimed right now: nobody is
        using them (`ref_count == 0`) and nothing comes after them in the
        tree (no children). This can undercount slightly -- a block with
        children that are *themselves* all evictable will only show up
        here once those children are actually evicted first (see
        `evict_one_lru`). That's fine: it just means we're never overly
        optimistic about how much space is really free.
        """
        return sum(1 for n in self._by_hash.values() if n.ref_count == 0 and not n.children)

    def evict_one_lru(self) -> Optional[int]:
        """Reclaims one cached block to free up space, and returns its
        physical block id -- or `None` if nothing can be reclaimed.

        We only ever reclaim a block with no children (a "leaf"). Reclaiming
        a block that still has cached blocks after it in the tree would
        leave those blocks pointing at data that no longer means anything.
        Calling this repeatedly naturally works from the outside in: once a
        leaf is gone, its parent may become a leaf itself and become
        reclaimable on the next call.

        Among everything that's currently reclaimable, we always pick the
        one that hasn't been used in the longest time.
        """
        candidates = [n for n in self._by_hash.values() if n.ref_count == 0 and not n.children]
        if not candidates:
            return None
        victim = min(candidates, key=lambda n: n.last_used)
        self._remove(victim)
        return victim.physical_block_id

    def _remove(self, node: TrieNode) -> None:
        """Deletes one node from the tree entirely."""
        level = self._roots if node.parent_hash is ROOT_HASH else self._by_hash[node.parent_hash].children
        del level[node.block_hash]
        del self._by_hash[node.block_hash]

    # -- Metrics ------------------------------------------------------------

    def stats(self) -> PrefixCacheStats:
        """Returns a snapshot of the hit-rate counters collected so far."""
        return PrefixCacheStats(
            num_lookup_tokens=self._num_lookup_tokens,
            num_hit_tokens=self._num_hit_tokens,
            num_lookups=self._num_lookups,
            num_hits=self._num_hits,
        )