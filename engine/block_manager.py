"""Block-based (paged) KV-cache allocator -- vLLM's BlockManager (v0) /
BlockPool + KVCacheManager (v1), studied from engine/README.md's source
pointers. Physical KV-cache memory is carved into `num_gpu_blocks`
fixed-size slots by whatever owns the real GPU buffers (not this module --
see README.md's "What's not wired up"); this module only tracks which
integer block ids are free vs. owned by which request's logical sequence.

Free blocks live in a plain list-as-stack, not vLLM's doubly-linked
FreeKVCacheBlockQueue -- LRU eviction order isn't needed here since it
lives entirely in engine/prefix_cache.py's RadixTrie instead.

Prefix caching (vLLM's `cached_block_hash_to_block`): opt-in via
CacheConfig.enable_prefix_caching, wiring in RadixTrie as
`self.prefix_cache` (None when disabled). `allocate` reuses another
request's already-computed blocks via `match_prefix`; `free` releases a
request's hold on shared blocks instead of always returning them to
`_free_block_ids`. See each method's docstring for the exact contract.

Not implemented, flagged rather than silently missing:
  - Copy-on-write / fork (vLLM's `BlockManager.fork`): sharing a
    block-table prefix across sibling sequences until they diverge.
  - Swap-to-CPU preemption (vLLM's `swap_out`/`swap_in`): scheduler.py
    only does recompute-based preemption.
"""
from dataclasses import dataclass, field
from typing import Optional

from engine.prefix_cache import PrefixMatch, RadixTrie
from engine.request import Request


class OutOfMemoryError(RuntimeError):
    """Raised by allocate()/append_slot() when called without first
    checking can_allocate()/can_append_slot(). The Scheduler always checks
    first, so hitting this means a scheduler bug -- treat it as a bug
    report, not something to catch and retry.
    """


@dataclass
class BlockManager:
    block_size: int
    num_gpu_blocks: int
    watermark_blocks: int = 0
    enable_prefix_caching: bool = False

    # Stack of genuinely free block ids. Order doesn't matter -- O(1)
    # allocate/free. LRU-ordered reuse is handled separately by
    # self.prefix_cache, not here.
    _free_block_ids: list = field(init=False, repr=False)
    # request_id -> ordered physical block ids. Index i holds tokens
    # [i*block_size, (i+1)*block_size) of the request's logical sequence.
    block_tables: dict = field(init=False, default_factory=dict)

    def __post_init__(self):
        self._free_block_ids = list(range(self.num_gpu_blocks))
        self.prefix_cache = RadixTrie(block_size=self.block_size) if self.enable_prefix_caching else None

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_block_ids)

    def get_block_table(self, request: Request) -> list:
        return self.block_tables.get(request.request_id, [])

    # -- Admission (prefill / resumed-after-preemption prefill) -----------

    def match_prefix(self, token_ids: list) -> Optional[PrefixMatch]:
        """Side-effect-free peek at how much of `token_ids` is already
        cached (safe to call any number of times). Returns None, not an
        empty PrefixMatch, when prefix caching is disabled, so callers can
        branch on `is None` and skip cache logic entirely.
        """
        if self.prefix_cache is None:
            return None
        return self.prefix_cache.match(token_ids)

    def can_allocate(self, request: Request, match: Optional[PrefixMatch] = None) -> bool:
        """Watermark reserve applies only to new admissions, not an
        already-running request's next decode slot (see can_append_slot) --
        protects running requests' progress over admitting one more that
        might need to be preempted again.

        `match` reduces how many fresh blocks are needed; evictable cached
        blocks count toward availability too, since allocate() reclaims
        them on demand.
        """
        matched = len(match.physical_block_ids) if match else 0
        needed_fresh = max(request.num_blocks_needed(self.block_size) - matched, 0)
        evictable = self.prefix_cache.num_evictable_blocks if self.prefix_cache is not None else 0
        return (self.num_free_blocks + evictable) - needed_fresh >= self.watermark_blocks

    def allocate(self, request: Request, match: Optional[PrefixMatch] = None) -> None:
        """`match`: an already-computed PrefixMatch, or None to look one up
        here -- passing it in avoids a redundant trie walk when the caller
        already has one from can_allocate(). Either way this calls
        RadixTrie.acquire() exactly once per admission, so hit-rate stats
        stay correct even if a caller skips match_prefix().
        """
        if request.request_id in self.block_tables:
            raise ValueError(
                f"{request.request_id} already has a block table -- call append_slot, not allocate"
            )
        if self.prefix_cache is not None and match is None:
            match = self.prefix_cache.match(request.prompt_token_ids)
        matched_ids = list(match.physical_block_ids) if match else []
        needed_fresh = request.num_blocks_needed(self.block_size) - len(matched_ids)
        if self.prefix_cache is not None:
            # Acquire before draining evictions below -- otherwise
            # evict_one_lru() could reclaim one of this request's own
            # matched blocks if its ref_count is currently 0 (e.g. the
            # donor that computed it already finished and freed).
            self.prefix_cache.acquire(match, num_lookup_tokens=len(request.prompt_token_ids))
            request.cached_prefix_nodes = list(match._nodes)
        self._drain_evictions(needed_fresh)
        if self.num_free_blocks < needed_fresh:
            raise OutOfMemoryError(
                f"need {needed_fresh} fresh blocks for {request.request_id}, only "
                f"{self.num_free_blocks} free -- caller must check can_allocate() first"
            )
        fresh = [self._free_block_ids.pop() for _ in range(needed_fresh)]
        table = matched_ids + fresh
        self.block_tables[request.request_id] = table
        request.block_table = table

    def insert_computed_prefix(self, request: Request) -> None:
        """Registers whatever of this request's prompt has actually been
        computed, so a future match_prefix() can reuse it. Call only AFTER
        a real forward pass has run this step (see model/llm_engine.py's
        step()) -- calling it early would let a future match reuse a block
        that doesn't hold what it claims to. No-op if prefix caching is
        disabled or less than one full block is computed.

        Safe to call repeatedly across chunked-prefill steps. Always
        replaces request.cached_prefix_nodes wholesale -- insert() resumes
        from wherever request.cached_prefix_nodes already leaves off (its
        `known_prefix` param), so the result is always the complete set of
        everything this request depends on, without re-walking blocks
        already established -- whether from an earlier call of this same
        method (chunked prefill) or an admission-time match (see
        RadixTrie.insert's docstring: re-walking already-known blocks on
        every step is both unnecessary and, for a long matched prefix,
        measurably slow).
        """
        if self.prefix_cache is None:
            return
        num_computed = min(request.num_computed_tokens, len(request.prompt_token_ids))
        if num_computed < self.block_size:
            return
        request.cached_prefix_nodes = self.prefix_cache.insert(
            request.prompt_token_ids, request.block_table, num_computed,
            known_prefix=request.cached_prefix_nodes,
        )

    def _drain_evictions(self, num_needed: int) -> None:
        """Reclaims cached-but-unreferenced blocks from the RadixTrie back
        into `_free_block_ids`, LRU-first, until `num_needed` are free or
        nothing more can be evicted. No-op when prefix caching is disabled.
        """
        if self.prefix_cache is None:
            return
        while self.num_free_blocks < num_needed:
            freed_id = self.prefix_cache.evict_one_lru()
            if freed_id is None:
                break
            self._free_block_ids.append(freed_id)

    # -- Steady-state decode ------------------------------------------------

    def can_append_slot(self, request: Request) -> bool:
        """True if the next token fits in the last block's reserved room,
        or one more block is free. No watermark reserve here (see
        can_allocate).

        Not eviction-aware, unlike can_allocate/allocate: this can raise
        OutOfMemoryError even when evictable blocks exist. Decode growth
        is one block at a time and rarely blocks on this in practice, and
        Scheduler already has a preemption path for it
        (_schedule_running's can_append_slot loop).
        """
        if self._needs_new_block(request):
            return self.num_free_blocks >= 1
        return True

    def append_slot(self, request: Request) -> None:
        """Reserves a block for the request's newest token if its last
        block is full; no-op otherwise.

        Must be called with request.get_len() already reflecting the new
        token (Scheduler calls this before advancing num_computed_tokens --
        see _needs_new_block).
        """
        table = self.block_tables.get(request.request_id)
        if table is None:
            raise ValueError(f"{request.request_id} has no block table -- call allocate() first")
        if self._needs_new_block(request):
            if not self._free_block_ids:
                raise OutOfMemoryError(
                    f"{request.request_id} needs a new block, none free -- "
                    "caller must check can_append_slot() first"
                )
            table.append(self._free_block_ids.pop())

    def _needs_new_block(self, request: Request) -> bool:
        """True when the request's token count exceeds its block table's
        capacity -- the new token doesn't fit in an already-reserved block.
        Depends only on request.get_len(), not num_computed_tokens, so it's
        safe to call before or after that advances.
        """
        table = self.block_tables.get(request.request_id, [])
        capacity = len(table) * self.block_size
        return request.get_len() > capacity

    # -- Teardown ------------------------------------------------------------

    def free(self, request: Request) -> None:
        """Returns every block this request privately owns to the free
        pool. No-op if the request was never allocated or already freed --
        Scheduler calls this on every preemption and finish/abort path.

        Blocks held via a prefix-cache ref (request.cached_prefix_nodes)
        are owned by the RadixTrie, not this free list, since another
        request may still depend on them -- only their ref_count is
        decremented (via RadixTrie.release). They re-enter the free pool
        later via _drain_evictions, if needed. Pushing them here directly
        would let a live TrieNode and the free stack both claim the same
        physical id -- silent KV corruption.
        """
        table = self.block_tables.pop(request.request_id, None)
        if table is None:
            return
        if self.prefix_cache is not None and request.cached_prefix_nodes:
            cache_owned_ids = {n.physical_block_id for n in request.cached_prefix_nodes}
            self.prefix_cache.release(request.cached_prefix_nodes)
            request.cached_prefix_nodes = []
            table = [bid for bid in table if bid not in cache_owned_ids]
        self._free_block_ids.extend(table)
        request.block_table = []