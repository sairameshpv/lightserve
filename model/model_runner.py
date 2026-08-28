"""Wires engine/scheduler.py's SchedulerOutput to an actual GPU forward pass:
the piece engine/README.md's "What's not wired up" section names as
"nothing here calls into model/minimal_llama.py's forward pass".

Why this isn't just "call llama_forward per request": minimal_llama.py's
`llama_forward` has no KV cache at all -- every call recomputes dense
self-attention over its whole input every time (see that file's and
cuda_graph_decode.py's docstrings on why: no Nq != Nkv kernel support).
Continuous batching needs the opposite -- a decode step must compute exactly
one new token's K/V and attend it against everything already cached, not
redo the whole sequence. So this module is a second forward-pass
implementation, not a reuse of `llama_forward`: same per-layer math (RMSNorm
-> QKV -> RoPE -> attention -> O-proj -> RMSNorm -> SwiGLU MLP), same
building-block kernels (`matmul`, `fused_add_rmsnorm`,
`flash_attention_forward`), but restructured around model/kv_cache.py's
PagedKVCache and a flattened, ragged-length batch instead of a dense [B, N]
tensor.

Batching strategy, and its real limit: every request scheduled this step
(prefill and decode alike) is concatenated along one flat token dimension
`T = sum(num_scheduled_tokens)` -- embedding, RMSNorm, every Linear, and the
MLP all run as a *single* batched call over all T rows, since none of those
ops mix information across rows. Attention is the one op that can't be
batched this way: each request needs its own K/V (gathered from its own
block_table, a different length per request) and its own causal-vs-not
treatment, so it runs in a Python loop, one flash_attention_forward call per
request per layer (B=1 each) -- see kv_cache.py's module docstring for why
this trades a real vectorization opportunity for reusing the existing FA
kernel unmodified rather than writing a new ragged-batch/paged one.

Q/K shape mismatch on a decode step, and how this works around it without
touching the kernel: flash_attention_forward *asserts*
`q.shape == k.shape == v.shape` -- exactly the Nq==Nkv-only limitation
engine/README.md's "What's not wired up" section names. A decode step's
real query is 1 token attending against `se` (>1) cached keys, which fails
that assert outright. Rather than relaxing the kernel (ruled out -- the
whole point of the chosen approach is reusing it unmodified, see
kv_cache.py's docstring), `_attention` pads Q up to K's length `se` with
dummy zero rows *in front of* the real query row(s), so the shapes match
and `causal=True` can be used uniformly for every request, prefill or
decode: the real query row ends up at the same relative position its real
tokens occupy in K (the last `L` rows), so the kernel's existing causal
mask -- comparing each row's position against each key's position, both
*relative to this call* -- restricts it to exactly the keys at or before
its own true position, same as an ordinary dense prefill. The dummy rows'
own outputs are nonsense and never read (`_attention` slices off only the
last `L` real rows before returning) -- they're numerically harmless: each
query row's softmax only depends on that row, so garbage in a discarded
row can't leak into a real one, and causal masking always leaves at least
the diagonal key unmasked (kernel's own no-NaN invariant, see its
docstring), so no row divides by zero either.

This has a real cost, and it isn't hidden: computing full `O(se)`
(padded to `O(se^2)` FLOPs, since the kernel doesn't know most of those
query rows are dummies) attention every decode step is the same "recompute,
don't incrementally extend" trade-off model/cuda_graph_decode.py's module
docstring already makes for the identical reason (no Nq != Nkv kernel
support) -- this version is strictly better than that one's fixed
`max_seq_len` bound (cost here scales with the request's *actual* length
`se`, not a fixed static buffer), but it is not the `O(1)`-per-step
incremental decode a dedicated Nq=1-against-paged-Nkv kernel would give.
That kernel is the real follow-up (see engine/README.md); this is the
"ship correct end-to-end generation now, with the existing kernel,
correctness proven against a step-by-step dense-recompute reference" path.

Sampling is deliberately just argmax (greedy) -- engine/request.py's
SamplingParams module docstring already scopes "temperature/top-p/top-k...
not the scheduler's [concern]" to whatever the model-runner side turns out
to be; this is that side, and greedy is the minimum needed to prove the
engine loop runs end to end. A real sampler (temperature/top-p/top-k) is a
drop-in replacement for the one argmax call in `_sample`.
"""
from dataclasses import replace

import torch
import torch.nn.functional as F

from engine.scheduler import SchedulerOutput
from kernels.flash_attention import flash_attention_forward
from kernels.fused_rmsnorm_residual import fused_add_rmsnorm
from kernels.tiled_matmul import matmul
from model.kv_cache import PagedKVCache
from model.minimal_llama import LlamaConfig, LlamaWeights, precompute_rope


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope_at_positions(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                              positions: torch.Tensor):
    """q, k: [T, n_heads, head_dim] -- a flat, cross-request batch, unlike
    minimal_llama.py's apply_rope (which assumes one contiguous [0:n) range
    shared by every row). `positions`: [T] long, each row's own *absolute*
    sequence position -- the reason this can't reuse that function: two
    different requests' rows sitting side by side in the same flat batch
    need two different RoPE angles for the same batch index.
    """
    cos_p = cos[positions].unsqueeze(1)  # [T, 1, head_dim]
    sin_p = sin[positions].unsqueeze(1)
    q_rot = q * cos_p + _rotate_half(q) * sin_p
    k_rot = k * cos_p + _rotate_half(k) * sin_p
    return q_rot, k_rot


class ModelRunner:
    """Owns the weights and the physical KV cache; `execute_model` is the
    once-per-step call the engine loop (model/llm_engine.py) makes after
    Scheduler.schedule(). Mutates each scheduled request's
    `output_token_ids`/`status` directly (same "caller does the bookkeeping
    inline" style engine/scheduler.py itself uses) rather than returning a
    separate output struct -- there's no async-scheduling split to preserve
    here yet (see scheduler.py's docstring on why that split doesn't exist).
    """

    def __init__(self, config: LlamaConfig, weights: LlamaWeights, kv_cache: PagedKVCache,
                 max_model_len: int = None, device: str = "cuda"):
        self.config = config
        self.weights = weights
        self.kv_cache = kv_cache
        self.device = device
        # RoPE's cos/sin table must cover every position any in-flight
        # sequence can reach, not just config.max_seq_len (that field is
        # minimal_llama.py's own "one dense buffer" sizing, orthogonal to
        # how long a continuous-batched generation can run) -- default to
        # it only when the caller doesn't say otherwise.
        rope_config = replace(config, max_seq_len=max_model_len or config.max_seq_len)
        self.cos, self.sin = precompute_rope(rope_config, device)

    def _build_flat_batch(self, scheduled: list):
        """Concatenates every scheduled request's new tokens into one flat
        [T] sequence. Returns the flat input_ids/positions tensors plus, per
        request, its (flat_start, flat_end) row range in that batch and its
        (seq_start, seq_end) *sequence* position range (the latter is what
        kv_cache.py's write/read and the causal-or-not test need).
        """
        input_ids, positions = [], []
        flat_starts, flat_ends, seq_starts, seq_ends = [], [], [], []
        cursor = 0
        for sr in scheduled:
            request = sr.request
            n = sr.num_scheduled_tokens
            seq_end = request.num_computed_tokens  # already advanced -- see scheduler.py's docstring
            seq_start = seq_end - n
            input_ids.extend(request.all_token_ids()[seq_start:seq_end])
            positions.extend(range(seq_start, seq_end))
            flat_starts.append(cursor)
            cursor += n
            flat_ends.append(cursor)
            seq_starts.append(seq_start)
            seq_ends.append(seq_end)
        if positions and positions[-1] >= self.cos.shape[0]:
            raise ValueError(
                f"sequence position {positions[-1]} exceeds this ModelRunner's max_model_len "
                f"({self.cos.shape[0]}) -- construct it with a larger max_model_len"
            )
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        positions = torch.tensor(positions, dtype=torch.long, device=self.device)
        return input_ids, positions, flat_starts, flat_ends, seq_starts, seq_ends

    def _attention(self, layer_idx, q, k, v, scheduled, flat_starts, flat_ends, seq_starts, seq_ends):
        """q, k, v: [T, n_heads, head_dim], already RoPE'd. Writes this
        step's new k/v into the KV cache and returns attn_out shaped like q
        -- see module docstring on why this loops per request instead of
        one batched call, and on the zero-pad-Q-to-K's-length trick used
        here to keep calling flash_attention_forward with its own
        q.shape == k.shape == v.shape contract intact.
        """
        attn_out = torch.empty_like(q)
        for i in range(len(scheduled)):
            fs, fe = flat_starts[i], flat_ends[i]
            ss, se = seq_starts[i], seq_ends[i]
            L = fe - fs
            request = scheduled[i].request

            self.kv_cache.write(layer_idx, request, ss, k[fs:fe], v[fs:fe])
            k_full, v_full = self.kv_cache.read(layer_idx, request, se)  # [se, n_heads, head_dim]

            q_new = q[fs:fe]  # [L, n_heads, head_dim] -- this step's real query rows
            if L < se:
                # Decode, or a chunked-prefill continuation (any step where
                # this request already has computed tokens behind it, ss >
                # 0): pad the front with dummy rows so Q's length matches
                # K/V's -- see module docstring. L == se only when ss == 0,
                # i.e. this step covers the request's sequence from position
                # 0 -- a fresh/resumed prefill's first chunk, whether that
                # chunk is the whole prompt or (chunked prefill, see
                # engine/README.md) just the first slice of it -- where no
                # padding is needed at all.
                pad = q_new.new_zeros(se - L, *q_new.shape[1:])
                q_full = torch.cat([pad, q_new], dim=0)
            else:
                q_full = q_new

            q_i = q_full.permute(1, 0, 2).unsqueeze(0).contiguous()  # [1, n_heads, se, head_dim]
            k_i = k_full.permute(1, 0, 2).unsqueeze(0).contiguous()  # [1, n_heads, se, head_dim]
            v_i = v_full.permute(1, 0, 2).unsqueeze(0).contiguous()

            out_i = flash_attention_forward(q_i, k_i, v_i, causal=True)  # [1, n_heads, se, head_dim]
            attn_out[fs:fe] = out_i.squeeze(0).permute(1, 0, 2)[-L:]  # keep only the real rows
        return attn_out

    def execute_model(self, scheduler_output: SchedulerOutput) -> list:
        """Runs one forward pass covering every request in
        scheduler_output.scheduled_new + scheduled_running, samples (greedy)
        each one's next token, appends it to output_token_ids, and calls
        Request.maybe_finish() on it. Returns the stepped Request objects
        (scheduled_new's, then scheduled_running's, same order
        SchedulerOutput carries them in) -- preempted requests aren't in
        `scheduled` at all (nothing to run for them this step, see
        scheduler.py's SchedulerOutput.preempted docstring).
        """
        scheduled = list(scheduler_output.scheduled_new) + list(scheduler_output.scheduled_running)
        if not scheduled:
            return []

        input_ids, positions, flat_starts, flat_ends, seq_starts, seq_ends = self._build_flat_batch(scheduled)

        x = F.embedding(input_ids, self.weights.embed_tokens)  # [T, H]
        residual = x
        # Same "feed zero, not a double-counted x" trick as llama_forward --
        # see minimal_llama.py's llama_forward docstring for why.
        x = torch.zeros_like(x)

        for layer_idx, layer in enumerate(self.weights.layers):
            normed, residual = fused_add_rmsnorm(x, residual, layer.input_layernorm_weight, eps=self.config.rms_eps)

            q = matmul(normed, layer.q_proj).reshape(-1, self.config.n_heads, self.config.head_dim)
            k = matmul(normed, layer.k_proj).reshape(-1, self.config.n_heads, self.config.head_dim)
            v = matmul(normed, layer.v_proj).reshape(-1, self.config.n_heads, self.config.head_dim)
            q, k = _apply_rope_at_positions(q, k, self.cos, self.sin, positions)

            attn_out = self._attention(layer_idx, q, k, v, scheduled, flat_starts, flat_ends, seq_starts, seq_ends)
            attn_out_2d = attn_out.reshape(-1, self.config.n_heads * self.config.head_dim)
            x = matmul(attn_out_2d, layer.o_proj)  # [T, H]

            normed2, residual = fused_add_rmsnorm(
                x, residual, layer.post_attention_layernorm_weight, eps=self.config.rms_eps,
            )
            gate = matmul(normed2, layer.gate_proj)
            up = matmul(normed2, layer.up_proj)
            mlp_hidden = F.silu(gate) * up
            x = matmul(mlp_hidden, layer.down_proj)

        final_normed, _ = fused_add_rmsnorm(x, residual, self.weights.norm_weight, eps=self.config.rms_eps)

        # Only the last row of each request's flat range predicts its next
        # token -- no need to run lm_head (a [*, H] @ [H, vocab_size] matmul)
        # over every prefilled prompt position, just the one that matters.
        last_idx = torch.tensor([e - 1 for e in flat_ends], dtype=torch.long, device=self.device)
        last_hidden = final_normed.index_select(0, last_idx)  # [num_requests, H]
        logits = matmul(last_hidden, self.weights.lm_head)  # [num_requests, vocab_size]
        next_token_ids = self._sample(logits)

        for i, sr in enumerate(scheduled):
            sr.request.output_token_ids.append(int(next_token_ids[i].item()))
            sr.request.maybe_finish()
        return [sr.request for sr in scheduled]

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Greedy argmax -- see module docstring on why nothing fancier is
        implemented here.
        """
        return logits.argmax(dim=-1)
