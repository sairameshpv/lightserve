"""Engine-wide config objects: how big a KV-cache block is, how many
physical blocks exist, and how many requests/tokens a scheduler step may
admit.

Deliberately not here: the physical GPU-memory math (block_size * 2 (K and
V) * n_layers * n_kv_heads * head_dim * dtype_bytes bytes per block, how
many of those fit in whatever fraction of GPU memory is budgeted for the KV
cache) -- that's deployment-time arithmetic that belongs next to whatever
loads the real model shape (model/minimal_llama.py's LlamaConfig,
kernels/flash_attention.py's D), not hardcoded in this CPU-only package.
CacheConfig.num_gpu_blocks is a plain int this module treats as given.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class CacheConfig:
    """block_size: tokens per physical KV-cache block. vLLM's own default is
    16 -- see engine/README.md's block-size tradeoff section for why a
    number in that range and not, say, 1 or 512.

    num_gpu_blocks: total physical blocks the BlockManager is allowed to
    hand out. Computed elsewhere from a GPU memory budget; taken as given
    here.

    watermark_blocks: blocks reserved and never handed to a *new* request's
    admission, even when technically free -- guards against admitting a
    request that fits this instant but has no room left for its own next
    decode-step block on the very next iteration, which would force
    preempting a request that had barely started. vLLM's own `watermark`
    config is the same idea, expressed as a fraction of total blocks instead
    of an absolute count.

    enable_prefix_caching: opt-in flag wiring engine/prefix_cache.py's
    RadixTrie into BlockManager. Off by default so existing behavior/tests
    are unaffected; when on, BlockManager.allocate reuses another request's
    already-computed blocks for a shared prompt prefix instead of always
    allocating fresh ones (see block_manager.py's module docstring).
    """
    block_size: int = 16
    num_gpu_blocks: int = 0
    watermark_blocks: int = 0
    enable_prefix_caching: bool = False

    def __post_init__(self):
        assert self.block_size > 0, "block_size must be positive"
        assert self.num_gpu_blocks >= 0, "num_gpu_blocks must be non-negative"
        assert 0 <= self.watermark_blocks <= self.num_gpu_blocks, (
            "watermark_blocks must fit within num_gpu_blocks"
        )


@dataclass(frozen=True)
class SchedulerConfig:
    """max_num_seqs: cap on requests RUNNING at once, independent of block
    availability -- a batch-size ceiling for whatever eventually calls into
    the model-runner side (batching more sequences than this into one
    forward pass isn't attempted even if blocks are plentiful).

    max_num_batched_tokens: token budget per schedule() call. Caps how many
    prefill tokens (or, in this design's one-decode-token-per-running-
    request model, prefill tokens plus one per already-running request) one
    step processes -- the mechanism that keeps one huge prefill from
    starving every other request's decode latency for a whole step.

    This is also the chunk size: a prompt longer than what's left of this
    budget on the step it's admitted (or resumed) doesn't wait for a step
    that can fit it whole -- Scheduler schedules as many of its tokens as
    fit this step and picks up the rest on a later one (see
    engine/README.md's "Chunked prefill" section). Set this equal to (or
    above) the longest prompt you expect to admit if you want every prefill
    to still land in a single step, matching this design's pre-chunking
    behavior.

    max_cache_hit_context_tokens: a second, separate per-step budget --
    None (default) reuses max_num_batched_tokens's value -- covering a gap
    max_num_batched_tokens can't see on its own: model/model_runner.py's
    _attention still pays O(context_length^2) attention cost for any
    request continuing past position 0, REGARDLESS of how few new tokens
    it has this step (see its docstring on the padded-Q trade-off for
    reusing the existing flash-attention kernel unmodified). A prefix-cache
    hit (CacheConfig.enable_prefix_caching) can make a request's *new*
    token count tiny (just its unmatched suffix) while its real context
    length stays huge (the matched prefix) -- max_num_batched_tokens, which
    only charges by new tokens, would happily admit many such requests
    into the very same step, each still paying full O(context_length^2)
    attention. max_cache_hit_context_tokens caps total matched-prefix
    length admitted per step instead, closing that gap -- see
    Scheduler._schedule_waiting. Only ever matters when caching is on and
    a match actually has matched tokens; otherwise inert.
    """
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 2048
    max_cache_hit_context_tokens: int = None

    def __post_init__(self):
        assert self.max_num_seqs > 0, "max_num_seqs must be positive"
        assert self.max_num_batched_tokens > 0, "max_num_batched_tokens must be positive"
        assert self.max_cache_hit_context_tokens is None or self.max_cache_hit_context_tokens > 0, (
            "max_cache_hit_context_tokens must be positive if set"
        )
