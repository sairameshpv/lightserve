"""Prefill/decode (P/D) disaggregation: runs a request's prefill to
completion on one LLMEngine and resumes it for decode on a *second*,
independent LLMEngine -- handing its KV cache across via model/kv_cache.py's
export_request_kv/import_request_kv instead of recomputing it there.

Mechanism only, on this project's single L40S -- flagged up front, not
discovered later. P/D disaggregation's real payoff is hardware isolation
(prefill is compute-bound, decode is memory-bandwidth-bound; together on
one GPU they contend, see benchmarks/chunked_prefill/README.md for the
concrete numbers on that contention). Two LLMEngines here still share the
same SMs and memory bandwidth -- this proves the *transfer* is correct
(byte-identical to a single-engine run, see model/tests/
test_pd_disaggregation.py and benchmarks/pd_disaggregation/), not that
disaggregation is faster here. No speedup should be expected or claimed
from this module; a real deployment needs separate GPUs (or nodes) behind
a real transport, which is exactly what export_request_kv/
import_request_kv's CPU round-trip stands in for -- see model/kv_cache.py's
comment on why that boundary is CPU-shaped rather than a same-device copy.

Handoff contents -- the one piece of arithmetic worth stating precisely,
since getting it wrong is the likeliest way this silently produces wrong
tokens instead of an error: by the time a scheduled step's forward pass
actually runs, Scheduler.schedule() has already advanced
request.num_computed_tokens synchronously (its documented invariant, see
engine/scheduler.py's module docstring) -- so on the step that finishes a
prompt's prefill, request.is_prefill() is already False *before*
execute_model runs, and that step therefore samples and appends this
request's first output token (model/model_runner.py's execute_model does
not special-case "just finished prefill" -- see its `is_prefill()` check).
So what crosses the split is: the prompt, that first sampled token, and
KV for positions [0, len(prompt)). The decoder seeds num_computed_tokens =
len(prompt), giving get_num_new_tokens() == 1 (the first output token's
own position, still needing its KV computed as input to the next forward
pass) -- the exact same invariant steady-state decode already runs under
elsewhere in this repo, not a special case. This also keeps the decoder
off a real crash this project already hit and fixed once: a request
scheduled for *zero* rows has no hidden state anywhere to sample from
(engine/scheduler.py's prefix-cache seed is explicitly capped at
`len(prompt) - 1` for exactly this reason, see its comment and
model/model_runner.py's zero-row guard). Carrying the first output token
across is what avoids that path here without needing the same cap.

Bypassing Scheduler.schedule()'s normal admission path is deliberate, not
a workaround: resume_decode() hand-places the resumed Request directly
into engine.scheduler.running/requests, the same way model/
draft_proposer.py already bypasses admission for the draft side (see its
module docstring) and the same direct running/requests manipulation
engine/tests/test_scheduler.py uses throughout to set up scenarios. Once
placed, the request is indistinguishable from an ordinary steady-state
decode continuation to every later engine.step() call -- nothing under
engine/ or model/model_runner.py needs to know P/D disaggregation exists.

Out of scope here: server/. EngineWorker's own docstring states its
engine "is only ever touched from this one thread, so none of it needs
its own locking" (server/engine_worker.py) -- running two engines behind
one HTTP server means confronting that invariant, which this module
doesn't attempt.
"""
from dataclasses import dataclass

from engine.request import Request, RequestStatus, SamplingParams
from model.llm_engine import GenerationOutput, LLMEngine


@dataclass
class PrefillHandoff:
    """Everything resume_decode() needs to continue a request on a second
    engine without recomputing its prefill. Built by prefill(), consumed
    exactly once by resume_decode() -- a plain transport-shaped bundle
    (see module docstring), not a live handle into either engine.
    """
    request_id: str
    prompt_token_ids: list
    first_token_id: int
    sampling_params: SamplingParams
    k: object  # CPU tensor, [n_layers, len(prompt_token_ids), num_kv_heads, head_dim]
    v: object


def prefill(engine: LLMEngine, prompt_token_ids: list, sampling_params: SamplingParams = None,
            request_id: str = None):
    """Runs `prompt_token_ids` through `engine` until its prefill is done,
    exports its KV, and removes the request from `engine` entirely
    (frees its blocks there) -- the request no longer exists on `engine`
    once this returns.

    Returns a PrefillHandoff for resume_decode() to pick up on a
    *different* engine -- or, if the request already finished during the
    very step that completed its prefill (e.g. sampling_params.max_tokens
    == 1: one token is sampled and immediately hits the cap, see module
    docstring), a GenerationOutput directly, since there is then nothing
    left to hand off.
    """
    request = engine.add_request(prompt_token_ids, sampling_params, request_id=request_id)
    while request.is_prefill():
        engine.step()

    if request.is_finished():
        # Already swept out of engine.scheduler by step()'s own
        # free_finished_requests() call -- this Request object is now
        # only reachable through our own reference. See module docstring.
        return GenerationOutput(
            request_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            output_token_ids=request.output_token_ids,
            finish_reason=request.status.name,
        )

    # request.num_computed_tokens == len(request.prompt_token_ids) here --
    # the step that finished prefill advanced it exactly that far (see
    # module docstring) -- so this exports precisely the prompt's KV.
    seq_len = len(request.prompt_token_ids)
    k, v = engine.kv_cache.export_request_kv(request, seq_len)

    handoff = PrefillHandoff(
        request_id=request.request_id,
        prompt_token_ids=list(request.prompt_token_ids),
        first_token_id=request.output_token_ids[-1],
        sampling_params=request.sampling_params,
        k=k, v=v,
    )
    engine.scheduler.abort_requests([request.request_id])  # frees this request's blocks on `engine`
    return handoff


def resume_decode(engine: LLMEngine, handoff: PrefillHandoff) -> Request:
    """Continues `handoff` on `engine` -- a different LLMEngine instance
    than the one prefill() ran on, possibly with a different block_size/
    num_gpu_blocks (see model/kv_cache.py's import_request_kv, which
    asserts model-shape compatibility but not block-layout compatibility,
    since none is required). Allocates a fresh block table sized for
    prompt+1 tokens, imports the KV directly into it -- no recompute --
    and hand-places the resulting Request into engine.scheduler's running
    state (see module docstring on why that bypass is deliberate and
    precedented).

    Returns the live Request -- same shape of handle add_request() would
    have returned, so the caller can poll it or drive engine.step() in a
    loop exactly like any other request (see generate_disaggregated()
    below for that loop).
    """
    request = Request(
        request_id=handoff.request_id,
        prompt_token_ids=list(handoff.prompt_token_ids),
        sampling_params=handoff.sampling_params,
        output_token_ids=[handoff.first_token_id],
        status=RequestStatus.RUNNING,
    )
    # Sized off request.get_len() == len(prompt)+1 (the first output
    # token above already counts) -- exactly enough room for positions
    # [0, len(prompt)], the prompt's imported KV plus the one new
    # position the next decode step computes. Same sizing BlockManager.
    # allocate() would give a fresh admission with one output token
    # already committed (a resumed-after-preemption request); nothing
    # P/D-specific about this call itself.
    engine.scheduler.block_manager.allocate(request)
    engine.kv_cache.import_request_kv(request, handoff.k, handoff.v)
    # Matches steady-state decode's own invariant exactly: get_len() -
    # num_computed_tokens == 1, the newly-sampled token still needing its
    # own position computed as input to the next forward pass. See
    # module docstring.
    request.num_computed_tokens = len(handoff.prompt_token_ids)

    engine.scheduler.requests[request.request_id] = request
    engine.scheduler.running.append(request)
    return request


def generate_disaggregated(prefill_engine: LLMEngine, decode_engine: LLMEngine, prompt_token_ids: list,
                            sampling_params: SamplingParams = None, request_id: str = None,
                            max_steps: int = 100_000) -> GenerationOutput:
    """One request, start to finish, across the split: prefill() on
    prefill_engine, resume_decode() onto decode_engine, then steps
    decode_engine until this request finishes. Thin composition of the
    two primitives above -- holds no state of its own, and (like
    LLMEngine.generate()'s own max_steps) isn't a solo-request-only
    assumption: decode_engine.step() advances every request running on
    it, not just this one, so this composes fine alongside other
    concurrent decode-side traffic.

    max_steps is the same safety-cap-not-normal-exit-path convention
    LLMEngine.generate() uses -- hitting it means sampling_params.
    max_tokens was left unreasonably high, not a normal outcome.
    """
    result = prefill(prefill_engine, prompt_token_ids, sampling_params, request_id=request_id)
    if isinstance(result, GenerationOutput):
        return result  # finished during prefill -- see prefill()'s docstring

    request = resume_decode(decode_engine, result)
    while not request.is_finished() and max_steps > 0:
        decode_engine.step()
        max_steps -= 1
    return GenerationOutput(
        request_id=request.request_id,
        prompt_token_ids=request.prompt_token_ids,
        output_token_ids=request.output_token_ids,
        finish_reason=request.status.name,
    )