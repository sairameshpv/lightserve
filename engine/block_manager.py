"""Block-based (paged) KV-cache allocator -- the piece vLLM calls
BlockManager (v0) / BlockPool + KVCacheManager (the current v1 engine, see
engine/README.md for the source pointers this design is studied from).
Physical KV-cache memory is carved into `num_gpu_blocks` fixed-size slots up
front by whatever owns the real GPU buffers (not this module -- kernels/
flash_attention.py's kernel doesn't take a block table yet, see README.md's
"What's not wired up"); this module only tracks *which* integer block ids
are free vs. owned by which request's logical sequence.

Same free-list-of-ints abstraction vLLM's `KVCacheBlock` objects /
`FreeKVCacheBlockQueue` implement with an actual doubly-linked list (needed
there for O(1) LRU-ordered eviction so prefix-cache hits get reused before
truly-cold blocks). This version uses a plain `list` as a stack instead --
no eviction/reuse-ordering policy to preserve, since prefix caching is a
documented non-goal here (see below), so any free block is as good as any
other.

Not implemented, on purpose -- each is a real vLLM feature, flagged rather
than silently missing:

  - Prefix caching (vLLM's `cached_block_hash_to_block`): reusing another
    request's already-computed blocks for a shared prompt prefix. Needs a
    content hash per block and a lookup table; real value, no correctness
    dependency for the rest of this design, cut for scope.
  - Copy-on-write / fork (vLLM's `BlockManager.fork`): sharing a block-table
    prefix across sibling sequences (beam search, parallel sampling `n>1`)
    until they diverge. Every Request here owns private blocks.
  - Swap-to-CPU preemption (vLLM's `swap_out`/`swap_in`): scheduler.py only
    does recompute-based preemption (free the blocks, redo the prefix as a
    fresh prefill later).
"""
from dataclasses import dataclass, field

from engine.request import Request


class OutOfMemoryError(RuntimeError):
    """Raised by allocate()/append_slot() if called without first checking
    can_allocate()/can_append_slot(). The Scheduler is expected to always
    check first (see scheduler.py's _schedule_waiting/_schedule_running), so
    hitting this means a scheduler bug, not an expected/handled runtime
    condition -- callers outside this package should treat it as a bug
    report, not something to catch and retry.
    """


@dataclass
class BlockManager:
    block_size: int
    num_gpu_blocks: int
    watermark_blocks: int = 0

    # Stack of free physical block ids. Order doesn't matter for correctness
    # (no eviction/reuse-locality policy in this no-prefix-caching design,
    # see module docstring) -- a list-as-stack is the simplest thing that
    # gives O(1) allocate/free.
    _free_block_ids: list = field(init=False, repr=False)
    # request_id -> ordered list of physical block ids. Index i holds tokens
    # [i*block_size, (i+1)*block_size) of that request's logical sequence
    # (prompt_token_ids + output_token_ids, in that order).
    block_tables: dict = field(init=False, default_factory=dict)

    def __post_init__(self):
        self._free_block_ids = list(range(self.num_gpu_blocks))

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_block_ids)

    def get_block_table(self, request: Request) -> list:
        return self.block_tables.get(request.request_id, [])

    # -- Admission (prefill / resumed-after-preemption prefill) -----------

    def can_allocate(self, request: Request) -> bool:
        """Watermark reserve is subtracted from what's "available" here, for
        a *new* admission -- not for an already-running request's next
        decode slot (see can_append_slot, no watermark there). Same
        asymmetry as vLLM's watermark: protect running requests' forward
        progress over admitting one more request that might immediately need
        to be preempted again.
        """
        needed = request.num_blocks_needed(self.block_size)
        return self.num_free_blocks - needed >= self.watermark_blocks

    def allocate(self, request: Request) -> None:
        if request.request_id in self.block_tables:
            raise ValueError(
                f"{request.request_id} already has a block table -- call append_slot, not allocate"
            )
        needed = request.num_blocks_needed(self.block_size)
        if self.num_free_blocks < needed:
            raise OutOfMemoryError(
                f"need {needed} blocks for {request.request_id}, only {self.num_free_blocks} free -- "
                "caller must check can_allocate() first"
            )
        table = [self._free_block_ids.pop() for _ in range(needed)]
        self.block_tables[request.request_id] = table
        request.block_table = table

    # -- Steady-state decode ------------------------------------------------

    def can_append_slot(self, request: Request) -> bool:
        """True if this request's next token fits in its last block's
        already-reserved room, or -- if it doesn't -- one more block is
        free. No watermark reserve here (see can_allocate's docstring on the
        asymmetry).
        """
        if self._needs_new_block(request):
            return self.num_free_blocks >= 1
        return True

    def append_slot(self, request: Request) -> None:
        """Reserve a physical block for the request's newest token if its
        current last block is already full; no-op otherwise, since the
        token then lands in a slot the block table already owns.

        Must be called with request.get_len() already reflecting the new
        token (Scheduler calls this before advancing num_computed_tokens,
        see scheduler.py's _schedule_running) -- see _needs_new_block.
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
        """True exactly when the request's current token count exceeds its
        block table's current capacity -- i.e. the token about to be
        computed is the first one that doesn't fit in an already-reserved
        block. Depends only on request.get_len() vs. the existing table, not
        on num_computed_tokens, so it's safe to call both before and after
        that gets advanced.
        """
        table = self.block_tables.get(request.request_id, [])
        capacity = len(table) * self.block_size
        return request.get_len() > capacity

    # -- Teardown ------------------------------------------------------------

    def free(self, request: Request) -> None:
        """Returns every block this request owns to the free pool. A no-op,
        not an error, if the request was never allocated (e.g. aborted while
        still WAITING) or already freed -- Scheduler calls this on every
        preemption and every finish/abort path, and shouldn't have to track
        which ones already had blocks.
        """
        table = self.block_tables.pop(request.request_id, None)
        if table is None:
            return
        self._free_block_ids.extend(table)
        request.block_table = []