"""The top-level entry point: Allocator (engine/block_manager.py, owned by
Scheduler) + Scheduler (engine/scheduler.py) + ModelRunner
(model/model_runner.py) driven in a loop until every submitted prompt has a
finished generation. This is the "wiring" engine/README.md's "What's not
wired up" section describes as the natural follow-up once a model runner
exists -- `LLMEngine.step()` is the one method that makes that whole stack
run for real, once per engine iteration:

    schedule() -> execute_model(output) -> free_finished_requests()

`generate()` is a batch convenience on top: submit every prompt up front
(so the scheduler gets to interleave their prefills and decodes exactly
like a real serving workload would, not one at a time), then call `step()`
until nothing's left running or waiting.

When CacheConfig.enable_prefix_caching is on, `step()` has one more stage
after execute_model: registering whatever prefill progress just happened
for real into the RadixTrie (engine/prefix_cache.py), so a later request's
admission can match against it. This can't happen any earlier than here --
Scheduler.schedule() only *decides* the batch (advancing
Request.num_computed_tokens synchronously, before the forward pass has
actually run, per scheduler.py's own documented invariant); this is the
first point where model_runner has actually computed real KV data for
those tokens.

Lives under model/, not engine/, on purpose: engine/'s own module
docstrings are explicit about staying torch-free so that package's tests run
without CUDA (see e.g. engine/request.py's docstring). This file imports
torch transitively (via ModelRunner/PagedKVCache) and needs a real GPU to do
anything, so it belongs on the model/ side of that split, alongside
model_runner.py and kv_cache.py.
"""
from dataclasses import dataclass

from engine.config import CacheConfig, SchedulerConfig, SpeculativeConfig
from engine.request import Request, RequestStatus, SamplingParams
from engine.scheduler import Scheduler
from model.draft_proposer import DraftProposer
from model.kv_cache import PagedKVCache
from model.minimal_llama import LlamaConfig, LlamaWeights, init_weights
from model.model_runner import ModelRunner

# load_hf_checkpoint is imported lazily inside __init__, not here -- it
# pulls in the safetensors package, which only speculative_config's real-
# checkpoint path actually needs. Importing it unconditionally would make
# every LLMEngine user need safetensors installed even when never using
# speculative decoding at all.


@dataclass
class GenerationOutput:
    """One finished (or, if generate() hit max_steps, still-running)
    request's result -- prompt/output token ids kept separate (like
    Request.all_token_ids() distinguishes them) so a caller can decode just
    the newly generated continuation without re-slicing the prompt back off.
    """
    request_id: str
    prompt_token_ids: list
    output_token_ids: list
    finish_reason: str  # RequestStatus member name, e.g. "FINISHED_STOPPED"


class LLMEngine:
    def __init__(self, cache_config: CacheConfig, scheduler_config: SchedulerConfig,
                 model_config: LlamaConfig, weights: LlamaWeights = None,
                 max_model_len: int = None, device: str = "cuda", seed: int = 0,
                 speculative_config: SpeculativeConfig = None):
        self.scheduler = Scheduler(cache_config, scheduler_config)
        self.kv_cache = PagedKVCache(cache_config, model_config, device=device)
        self.weights = weights if weights is not None else init_weights(model_config, device=device, seed=seed)
        # max_model_len defaults to model_config.max_seq_len inside
        # ModelRunner if left None -- for llama3_8b_shape() that's a
        # deliberately truncated 128 (see minimal_llama.py's module
        # docstring), not a real generation-length bound, so callers running
        # anything longer than that should pass this explicitly.
        self.model_runner = ModelRunner(
            model_config, self.weights, self.kv_cache, max_model_len=max_model_len, device=device,
        )
        self._next_id = 0

        # speculative_config absent (default) is the off switch: every new
        # step() branch below is gated on self.draft_proposer, so behavior
        # is completely unchanged when this stays None.
        self.draft_proposer = None
        if speculative_config is not None:
            from model.hf_loader import load_hf_checkpoint  # see top-of-file note on why this is lazy
            draft_config, draft_weights = load_hf_checkpoint(speculative_config.draft_model_path, device=device)
            # Own CacheConfig, not cache_config -- the draft's smaller
            # shape needs far fewer blocks per token (see
            # SpeculativeConfig.draft_num_gpu_blocks's docstring).
            draft_cache_config = CacheConfig(
                block_size=cache_config.block_size,
                num_gpu_blocks=speculative_config.draft_num_gpu_blocks,
            )
            draft_kv_cache = PagedKVCache(draft_cache_config, draft_config, device=device)
            draft_model_runner = ModelRunner(
                draft_config, draft_weights, draft_kv_cache, max_model_len=max_model_len, device=device,
            )
            self.draft_proposer = DraftProposer(draft_model_runner, speculative_config.num_speculative_tokens)

    def add_request(self, prompt_token_ids: list, sampling_params: SamplingParams = None,
                     request_id: str = None) -> Request:
        if request_id is None:
            request_id = f"req-{self._next_id}"
            self._next_id += 1
        request = Request(
            request_id=request_id,
            prompt_token_ids=list(prompt_token_ids),
            sampling_params=sampling_params or SamplingParams(),
        )
        self.scheduler.add_request(request)
        return request

    def step(self):
        """One iteration: schedule this step's batch, run it, register any
        genuinely-computed prefill progress into the prefix cache, correct
        speculative over-advance (see below), sweep out whatever finished,
        propose next round's draft tokens. Returns (SchedulerOutput,
        finished_requests) -- mirrors Scheduler.schedule()/
        free_finished_requests()'s own return shapes, since this is just
        sequencing them with the forward pass (and, when enabled, the
        prefix-cache registration and speculative decoding steps) run in
        between.
        """
        output = self.scheduler.schedule()

        # A request with draft_token_ids just got scheduled for exactly
        # num_scheduled_tokens == len(draft_token_ids) + 1 (Scheduler.
        # _schedule_running's spec branch), pre-advancing num_computed_tokens
        # by that full amount regardless of how many rows execute_model
        # below actually accepts (model/model_runner.py's _accept_reject).
        # Snapshot each one's output length now -- the only way to recover
        # the real accepted count afterward, since execute_model's return
        # value doesn't carry it.
        spec_pre_len = {
            sr.request.request_id: len(sr.request.output_token_ids)
            for sr in output.scheduled_running if sr.request.draft_token_ids
        }

        self.model_runner.execute_model(output)
        if self.scheduler.block_manager.prefix_cache is not None:
            for sr in list(output.scheduled_new) + list(output.scheduled_running):
                request = sr.request
                # num_computed_tokens is already post-step (schedule()
                # advances it synchronously before this forward pass ran --
                # see scheduler.py's docstring); subtract this step's own
                # contribution back out to get the pre-step value, so this
                # only fires for a request that had genuine prefill compute
                # in the step whose forward pass just ran above -- never
                # before real KV exists, and never wastefully on a pure
                # steady-state decode step.
                pre_step_computed = request.num_computed_tokens - sr.num_scheduled_tokens
                if pre_step_computed < len(request.prompt_token_ids):
                    self.scheduler.block_manager.insert_computed_prefix(request)

        # Correct this step's speculative over-advance: however many of
        # the K+1 scheduled rows actually got committed (1..K+1) is what
        # output_token_ids really grew by, not num_scheduled_tokens --
        # wind num_computed_tokens back to match, clear draft_token_ids
        # (cleared every round regardless of acceptance count, per its
        # docstring in engine/request.py), and correct the draft's own
        # shadow the same way. Must run after the prefix-cache block
        # above, not before -- that block's arithmetic relies on
        # num_computed_tokens still holding schedule()'s pre-advanced
        # value, matching sr.num_scheduled_tokens exactly.
        for sr in output.scheduled_running:
            request = sr.request
            pre_len = spec_pre_len.get(request.request_id)
            if pre_len is None:
                continue
            accepted = len(request.output_token_ids) - pre_len

            # _accept_reject (model_runner.py) only knows about draft-vs-
            # target matching, nothing about eos_token_id/max_tokens -- a
            # multi-token commit can straddle a stop condition that would
            # have ended generation partway through it, appending tokens
            # dense decoding never would have produced. Trim back to the
            # true first-stop point if the committed batch overshot one
            # (a single-token, non-speculative commit can never trigger
            # this: stop_at, when found, always equals accepted - 1 then).
            new_tokens = request.output_token_ids[pre_len:]
            eos = request.sampling_params.eos_token_id
            stop_at = next(
                (i for i, tok in enumerate(new_tokens)
                 if tok == eos or pre_len + i + 1 >= request.sampling_params.max_tokens),
                None,
            )
            if stop_at is not None and stop_at < accepted - 1:
                del request.output_token_ids[pre_len + stop_at + 1:]
                accepted = stop_at + 1
                request.status = (
                    RequestStatus.FINISHED_STOPPED if new_tokens[stop_at] == eos
                    else RequestStatus.FINISHED_LENGTH_CAPPED
                )

            request.num_computed_tokens -= sr.num_scheduled_tokens - accepted
            request.draft_token_ids = []
            if self.draft_proposer is not None:
                self.draft_proposer.rollback(request.request_id, accepted)

        finished = self.scheduler.free_finished_requests()

        # Propose next round's draft tokens for every still-running request
        # that has a real committed token to extend from (is_prefill()
        # guards a request still mid-prompt -- nothing to speculate past
        # yet). self.scheduler.running is already finished-swept above.
        if self.draft_proposer is not None:
            for request in self.scheduler.running:
                if not request.is_prefill():
                    request.draft_token_ids = self.draft_proposer.propose(request)

        return output, finished

    def generate(self, prompts: list, sampling_params: SamplingParams = None,
                 max_steps: int = 100_000) -> list:
        """prompts: list of token-id lists (this repo has no tokenizer --
        see model/README.md's scope notes on random-not-real-checkpoint
        weights; callers own tokenization). Returns one GenerationOutput per
        prompt, same order as `prompts`. `max_steps` is a safety cap, not a
        normal exit path -- every request's own sampling_params.max_tokens
        (or eos) is what should actually end it; hitting max_steps first
        means either max_tokens was left unreasonably high or the cache
        can't fit anything into `max_num_seqs`'s worth of concurrent
        requests at all (a config bug, not expected steady state).
        """
        requests = [self.add_request(p, sampling_params) for p in prompts]
        while self.scheduler.has_unfinished_requests() and max_steps > 0:
            self.step()
            max_steps -= 1
        return [
            GenerationOutput(
                request_id=r.request_id,
                prompt_token_ids=r.prompt_token_ids,
                output_token_ids=r.output_token_ids,
                finish_reason=r.status.name,
            )
            for r in requests
        ]
