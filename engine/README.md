# engine

CPU-only bookkeeping for a single-GPU continuous-batching inference engine:
request lifecycle, a paged KV-cache block allocator, and the scheduler that
ties them together. No torch import anywhere in this package on purpose --
everything here is testable without CUDA (see `engine/tests/`), unlike
`kernels/` and `model/`.

This is a scoped-down study of [vLLM](https://github.com/vllm-project/vllm)'s
v1 engine core, specifically:

- `vllm/v1/core/sched/interface.py` (`SchedulerInterface` ABC) and
  `vllm/v1/core/sched/scheduler.py` (`Scheduler`) → `scheduler.py`
- vLLM's `BlockManager` (v0) / `BlockPool` + `KVCacheManager` (current v1) →
  `block_manager.py`

Read the module docstrings first -- each file explains its own scope and its
specific vLLM counterpart. This doc is the connective tissue: the request
lifecycle end to end, how scheduling decisions get made, the block-size
tradeoff, and -- important -- exactly where this stops being a working
engine.

## Request lifecycle

```
WAITING --admitted (_schedule_waiting)--> RUNNING
RUNNING --preempted (_preempt, under memory pressure)--> PREEMPTED
PREEMPTED --requeued at front of `waiting`--> WAITING
RUNNING --Request.maybe_finish() sees a stop condition--> FINISHED_STOPPED
                                                        or FINISHED_LENGTH_CAPPED
(any state) --Scheduler.abort_requests()--> FINISHED_ABORTED
```

- **WAITING → RUNNING** is admission: `Scheduler._schedule_waiting` pops
  from the FIFO `waiting` deque, checks `BlockManager.can_allocate`, and if
  it fits, allocates blocks for the whole prompt and marks the request
  RUNNING. FIFO means if the head of the queue can't be admitted, nothing
  behind it jumps ahead either -- `_schedule_waiting` stops scanning rather
  than admitting a smaller request out of order.
- **RUNNING → PREEMPTED → WAITING** is recompute-based preemption
  (`Scheduler._preempt`, see "Preemption" below). The request keeps its
  token ids (`output_token_ids` isn't touched) but loses its KV cache: every
  block it owned is freed and `num_computed_tokens` resets to 0. It's
  requeued at the *front* of `waiting`, not the back, so it's first in line
  once space frees up -- ahead of requests that arrived later but were
  never running. Its next admission redoes the whole sequence-so-far as a
  fresh full prefill.
- **RUNNING → FINISHED_STOPPED / FINISHED_LENGTH_CAPPED**: whatever appends
  a newly-sampled token to `output_token_ids` (the model-runner-facing side
  of the engine loop -- see "What's not wired up") calls
  `Request.maybe_finish()` right after. It checks two stop conditions --
  `output_token_ids[-1] == sampling_params.eos_token_id`, or
  `len(output_token_ids) >= sampling_params.max_tokens` -- and updates
  `status` if either fires.
- **FINISHED_ABORTED**: `Scheduler.abort_requests()` cancels a request
  regardless of lifecycle stage (still WAITING, RUNNING, or already
  finished).

Terminal states are swept out of `running` by
`Scheduler.free_finished_requests()`, which frees their blocks and returns
them to the caller -- meant to be called once per engine step, after
processing model output.

## Scheduler

`Scheduler.schedule()` is the once-per-engine-step decision. It:

1. **Runs `_schedule_running` first.** Every already-RUNNING request gets
   considered, in `self.running`'s existing order (arrival order). Each
   needs `request.get_num_new_tokens()` new tokens computed -- almost
   always 1 (steady-state decode); more only if it's still mid-prefill and
   re-entering this path after a partial admission. If a request's next
   token doesn't fit in its last block, `BlockManager.append_slot` needs one
   more free block; if none is free, something gets preempted (see below).
2. **Runs `_schedule_waiting` only if nothing was preempted this step.**
   Same rule vLLM's `Scheduler` follows: a step that just freed blocks
   under memory pressure shouldn't immediately hand them to a brand-new
   admission and risk preempting it right back next step (thrashing).
3. Returns a `SchedulerOutput`: `scheduled_new` (fresh admissions, full
   prefill), `scheduled_running` (steady-state decode, or resumed prefill),
   and `preempted` (bumped back to `waiting`; the caller doesn't need to do
   anything about these beyond not expecting output from them this step --
   `Scheduler` has already freed their blocks).

`ScheduledRequest.num_scheduled_tokens > 1` means a prefill-shaped step
(`request.is_prefill()` was `True` going in); `== 1` means steady-state
decode -- the distinction a model-runner needs to pick which attention-
kernel call shape applies (see "What's not wired up").

### Preemption

Preemption only happens inside `_schedule_running`, when a request can't get
the one additional block its next token needs. Victims are popped from the
*tail* of `pending` (the running requests not yet decided this step, in
priority order) -- lowest priority among those not yet processed, never from
a request whose scheduling decision was already made earlier in the same
step. So a block a already-scheduled request just claimed is never clawed
back later in the same `schedule()` call, even if handing it to a different,
still-pending request instead would have avoided a preemption entirely. If
the request needing a block is itself the last one left in `pending` (no one
lower-priority to evict), it preempts itself.

This is a real, load-bearing consequence, not just an implementation detail
-- `engine/tests/test_scheduler.py`'s
`test_preempted_request_resumes_as_a_fresh_prefill` exercises it directly:
two requests each holding the pool's last free block, the lower-priority one
evicted to make room for the other, and only able to resume once the winner
separately finishes and frees its blocks.

Recompute (not swap-to-CPU) is the only preemption strategy here -- see
`block_manager.py`'s module docstring for why.

## Block-size tradeoff

`CacheConfig.block_size` (tokens per physical KV-cache block) defaults to
16, matching vLLM's own default. It's a knob between two costs:

- **Too small** (e.g. 1): minimal internal fragmentation (a request's last
  block is never wasting much room), but the block table grows one entry
  per token, and every block boundary is a potential allocation event --
  more bookkeeping, more chances to need `append_slot` on a decode step
  that would otherwise be a no-op.
- **Too large** (e.g. 512): far fewer allocation events and a much smaller
  block table, but a short request wastes most of its last block (internal
  fragmentation), and coarser granularity means preemption/admission
  decisions move in bigger, lumpier increments -- exactly the effect the
  block-size choice in `test_preempted_request_resumes_as_a_fresh_prefill`
  depends on (crossing a block boundary is what triggers preemption at all).

16 sits in the middle: small enough to keep fragmentation reasonable, large
enough that most requests don't touch `append_slot` on every single decode
step.

## What's not wired up

This package is scheduling and block-accounting logic only -- pairing it
with an actual GPU forward pass is `model/`'s job, not this package's (see
model/README.md's "Model runner" section for what's wired up there and how).
Specifically:

- **Model runner: now wired, in `model/`, not here.**
  `model/model_runner.py`'s `ModelRunner.execute_model` turns a
  `SchedulerOutput` into a real forward pass, and `model/llm_engine.py`'s
  `LLMEngine` drives `Scheduler.schedule() -> execute_model() ->
  free_finished_requests()` in a loop for actual end-to-end generation. It
  lives under `model/`, not `engine/`, because it needs torch -- this
  package's own modules are explicit about staying torch-free (see e.g.
  `request.py`'s docstring) so their tests run without CUDA.
- **No *dedicated* decode-shaped attention kernel -- worked around, not
  solved.** `kernels/flash_attention.py`'s `flash_attention_forward` is
  still prefill-shaped only (`q.shape == k.shape == v.shape` asserted, no
  block table). `ModelRunner` doesn't relax that: it pads a decode step's
  single real query row up to the cached K/V's length with dummy rows,
  reusing `causal=True` unmodified (see `model_runner.py`'s module
  docstring for the padding trick and its correctness argument). That makes
  decode correct and real, but not `O(1)`-per-step -- it costs `O(seq_len)`
  attention (same "recompute instead of incrementally extend" trade-off
  `model/cuda_graph_decode.py` already makes, for the identical reason). A
  real `Nq == 1`-against-*paged*-`Nkv` kernel (gathering block-table K/V
  *inside* the kernel, across the whole batch at once) is the actual
  follow-up this gap still points at -- `Request.is_prefill()`'s docstring
  flags the same gap.
- **Physical KV-cache buffers: now real, in `model/kv_cache.py`.**
  `BlockManager` still only tracks which integer block ids are free vs.
  owned by which request (unchanged, and still not this package's job); the
  actual GPU memory each block id refers to is `model/kv_cache.py`'s
  `PagedKVCache` -- one `[num_gpu_blocks, block_size, n_heads, head_dim]`
  tensor per layer per K/V, sized from `CacheConfig` and
  `model/minimal_llama.py`'s `LlamaConfig` (`kernels/flash_attention.py`'s
  `D` is that `head_dim`). Same plain-MHA assumption as `minimal_llama.py`
  throughout: no separate KV-head count, no GQA.
- **Synchronous scheduling.** `Scheduler.schedule()` both decides the batch
  *and* immediately advances `Request.num_computed_tokens` / the block
  tables for it, as if the model runner is guaranteed to honor exactly that
  decision this step. Real vLLM splits this into `SchedulerOutput` (what to
  run) and a separate `update_from_output(model_runner_output)` call after
  the forward pass actually completes -- what lets it pipeline scheduling
  for step N+1 while step N's forward pass is still running on the GPU
  (async scheduling). `LLMEngine.step()` now exists (a real model runner to
  split against) but still calls `execute_model()` synchronously right
  after `schedule()`, same as this bullet always described -- splitting
  scheduling from completion the way async scheduling needs remains a real
  follow-up, just no longer blocked on "no model runner to split against."

## Non-goals (cut for scope, not oversights)

Each of these is a real vLLM feature, deliberately left out:

- **Prefix caching** (vLLM's `cached_block_hash_to_block`): reusing another
  request's already-computed blocks for a shared prompt prefix. Needs a
  content hash per block and a lookup table. This is also why
  `BlockManager` uses a plain `list` as a free-block stack instead of
  vLLM's `KVCacheBlock` / `FreeKVCacheBlockQueue` doubly-linked list -- that
  structure exists for O(1) LRU-ordered eviction so prefix-cache hits get
  reused before truly-cold blocks; with no reuse-ordering policy to
  preserve here, any free block is as good as any other.
- **Copy-on-write / fork** (vLLM's `BlockManager.fork`): sharing a
  block-table prefix across sibling sequences (beam search, parallel
  sampling `n>1`) until they diverge. Every `Request` here owns private
  blocks.
- **Swap-to-CPU preemption** (vLLM's `swap_out`/`swap_in`): `scheduler.py`
  only does recompute-based preemption (free the blocks, redo the prefix as
  a fresh prefill later).
- **Chunked prefill**: splitting one large prompt's prefill across multiple
  steps so it doesn't have to fit in one `max_num_batched_tokens` budget.
- **Priority scheduling**: vLLM supports a `priority` policy alongside FIFO;
  `waiting` here is strictly FIFO (preempted requests requeued at the
  front).
- **Speculative decoding, multimodal encoder cache, distributed KV
  connector**: no analog of any of these exists in this design.
