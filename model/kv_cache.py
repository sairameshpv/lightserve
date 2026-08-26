"""Physical, GPU-resident backing store for the block ids engine/block_manager.py
hands out. BlockManager only tracks which integer block ids are free vs.
owned by which request (see its module docstring); this is the "actual GPU
memory a block id refers to" it explicitly says isn't its job.

One tensor per layer per K/V, shaped
`[num_gpu_blocks, block_size, n_heads, head_dim]` -- `write`/`read` translate
a request's logical token positions into `(physical_block_id, offset_in_block)`
pairs via its `block_table` (BlockManager.allocate/append_slot already sized
and populated that list; this module only ever reads it) and gather/scatter
with one vectorized advanced-indexing op per call, not a per-token Python
loop.

No new Triton kernel: `model/model_runner.py` reads a request's full
K/V-so-far out of here as one dense `[seq_len, n_heads, head_dim]` tensor and
hands that straight to `kernels/flash_attention.py`'s existing
`flash_attention_forward` -- the "gather, then reuse the existing kernel"
design engine/README.md's "What's not wired up" section flagged as the
alternative to writing a new paged-attention kernel from scratch. The cost:
one dense gather per request per layer per step, and attention runs in a
per-request Python loop (see model_runner.py) rather than one kernel call
batching every request's ragged K/V lengths at once -- real vLLM's
PagedAttention kernel does that gather *inside* the kernel across the whole
batch; this doesn't, on purpose, to avoid new Triton kernel-authoring/tuning
work here.

Same plain-MHA assumption as minimal_llama.py: `n_heads` here is also the
KV head count (no GQA broadcast) -- see that file's module docstring.
"""
import torch

from engine.config import CacheConfig
from engine.request import Request
from model.minimal_llama import LlamaConfig


class PagedKVCache:
    def __init__(self, cache_config: CacheConfig, model_config: LlamaConfig, device: str = "cuda"):
        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        self.device = device
        shape = (
            model_config.n_layers, cache_config.num_gpu_blocks, cache_config.block_size,
            model_config.n_heads, model_config.head_dim,
        )
        # zeros, not empty: a never-written slot (e.g. a block's tail past a
        # request's real length) must read back as inert, not NaN/garbage --
        # matters if anything ever reads a whole block rather than exactly
        # `seq_len` positions (nothing here does today, but cheap insurance).
        self.k_cache = torch.zeros(shape, dtype=model_config.dtype, device=device)
        self.v_cache = torch.zeros(shape, dtype=model_config.dtype, device=device)

    def _physical_locations(self, request: Request, start: int, end: int):
        """positions [start, end) -> (physical_block_ids, offsets), both
        [end-start] long tensors, ready to index k_cache/v_cache's
        (block, offset) dims at once.
        """
        positions = torch.arange(start, end, device=self.device)
        block_idx = positions // self.block_size
        offset = positions % self.block_size
        table = torch.as_tensor(request.block_table, dtype=torch.long, device=self.device)
        physical_block_ids = table[block_idx]
        return physical_block_ids, offset

    def write(self, layer_idx: int, request: Request, start: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Scatter this step's freshly computed k/v for `request` into the
        physical blocks its block_table already reserves, at logical
        positions [start, start + k.shape[0]). k, v: [num_new_tokens,
        n_heads, head_dim]. `start` is the *sequence* position of the first
        new token -- 0 for a fresh/resumed prefill's first chunk (the whole
        prompt in one step, or just its first slice under chunked prefill,
        see engine/README.md), num_computed_tokens (pre-this-step) for a
        steady-state decode step or a later chunked-prefill continuation --
        the caller
        (model_runner.py) derives it from ScheduledRequest.num_scheduled_tokens
        since Scheduler.schedule() has already advanced
        request.num_computed_tokens to the post-step value by the time this
        runs (its synchronous-scheduling design, see scheduler.py's
        docstring).
        """
        num_new = k.shape[0]
        physical_block_ids, offset = self._physical_locations(request, start, start + num_new)
        self.k_cache[layer_idx, physical_block_ids, offset] = k
        self.v_cache[layer_idx, physical_block_ids, offset] = v

    def read(self, layer_idx: int, request: Request, seq_len: int):
        """Gather `request`'s first `seq_len` logical positions' K/V back
        into one dense [seq_len, n_heads, head_dim] tensor each -- includes
        whatever `write` just stored this same step, since write-then-read
        against the same block ids is exactly how a decode step's new token
        ends up included in its own attention call's K/Nkv.
        """
        physical_block_ids, offset = self._physical_locations(request, 0, seq_len)
        k = self.k_cache[layer_idx, physical_block_ids, offset]
        v = self.v_cache[layer_idx, physical_block_ids, offset]
        return k, v
