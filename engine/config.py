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
    """
    block_size: int = 16
    num_gpu_blocks: int = 0
    watermark_blocks: int = 0

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
    step processes -- the mechanism a real engine uses to keep one huge
    prefill from starving every other request's decode latency for a whole
    step. Chunked prefill (splitting one large prompt's prefill across
    multiple steps so it doesn't have to fit in one token budget) is a
    documented non-goal here, see engine/README.md.
    """
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 2048

    def __post_init__(self):
        assert self.max_num_seqs > 0, "max_num_seqs must be positive"
        assert self.max_num_batched_tokens > 0, "max_num_batched_tokens must be positive"
