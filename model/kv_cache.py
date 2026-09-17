"""Physical, GPU-resident backing store for the block ids engine/block_manager.py
hands out. BlockManager only tracks which integer block ids are free vs.
owned by which request (see its module docstring); this is the "actual GPU
memory a block id refers to" it explicitly says isn't its job.

One tensor per layer per K/V, shaped
`[num_gpu_blocks, block_size, num_kv_heads, head_dim]` -- `write`/`read` translate
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

Stores K/V at `num_kv_heads` (== `n_heads` under plain MHA, fewer under GQA
-- see minimal_llama.py's module docstring). The repeat_kv broadcast up to
`n_heads` happens in model_runner.py, after read(), not here -- this file
only ever stores/gathers what the model actually projected.
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
            model_config.num_kv_heads, model_config.head_dim,
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
        num_kv_heads, head_dim]. `start` is the *sequence* position of the first
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
        into one dense [seq_len, num_kv_heads, head_dim] tensor each -- includes
        whatever `write` just stored this same step, since write-then-read
        against the same block ids is exactly how a decode step's new token
        ends up included in its own attention call's K/Nkv.
        """
        physical_block_ids, offset = self._physical_locations(request, 0, seq_len)
        k = self.k_cache[layer_idx, physical_block_ids, offset]
        v = self.v_cache[layer_idx, physical_block_ids, offset]
        return k, v

    # -- Cross-instance transfer (P/D disaggregation) ------------------------
    #
    # export/import move one request's whole K/V between two *separate*
    # PagedKVCache instances -- the prefiller's and the decoder's, in
    # prefill/decode disaggregation (see model/pd_disaggregation.py). Both
    # are thin loops over read/write above rather than anything new,
    # because read already hands back a **dense, logically-ordered**
    # [seq_len, ...] tensor and write already takes that same shape at a
    # logical start. Everything block-layout-specific stays behind that
    # interface, so the two instances need share nothing: not the physical
    # block ids (arbitrary either side -- block_manager.py's free list is
    # an explicitly order-agnostic stack), not the block_table, and not
    # even block_size. What they *must* agree on is the model shape, which
    # import_request_kv asserts rather than trusts.

    def export_request_kv(self, request: Request, seq_len: int):
        """`request`'s first `seq_len` positions' K/V across every layer,
        as two [n_layers, seq_len, num_kv_heads, head_dim] CPU tensors.

        CPU, not GPU: this is the handoff point a real cross-node
        transport would serialize at, so paying the device->host copy
        here keeps the seam honest (see model/pd_disaggregation.py's
        module docstring on what this does and doesn't simulate). Callers
        wanting a same-device copy can .to() them back.
        """
        n_layers = self.k_cache.shape[0]
        ks, vs = [], []
        for layer_idx in range(n_layers):
            k, v = self.read(layer_idx, request, seq_len)
            ks.append(k)
            vs.append(v)
        return torch.stack(ks).cpu(), torch.stack(vs).cpu()

    def import_request_kv(self, request: Request, k: torch.Tensor, v: torch.Tensor) -> None:
        """Inverse of export_request_kv: scatter an exported bundle into
        `request`'s *own* block_table in this cache. `request` must
        already have blocks allocated (BlockManager.allocate) covering at
        least k.shape[1] positions.

        Asserts shape/dtype compatibility instead of trusting it: a
        silent mismatch here would land as wrong-but-finite KV values,
        which surface only as subtly wrong generated tokens much later --
        far harder to trace back than an assert at the transfer itself.
        """
        assert k.shape == v.shape, f"k/v shape mismatch: {k.shape} vs {v.shape}"
        n_layers, seq_len, num_kv_heads, head_dim = k.shape
        expected = (self.k_cache.shape[0], self.k_cache.shape[3], self.k_cache.shape[4])
        assert (n_layers, num_kv_heads, head_dim) == expected, (
            f"exported KV is shaped for (n_layers, num_kv_heads, head_dim)="
            f"{(n_layers, num_kv_heads, head_dim)}, but this cache is {expected} -- "
            "the two instances aren't running the same model shape"
        )
        assert k.dtype == self.k_cache.dtype, (
            f"exported KV dtype {k.dtype} != this cache's {self.k_cache.dtype}"
        )
        k = k.to(self.device)
        v = v.to(self.device)
        for layer_idx in range(n_layers):
            self.write(layer_idx, request, 0, k[layer_idx], v[layer_idx])
