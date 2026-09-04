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

from engine.config import CacheConfig, SchedulerConfig
from engine.request import Request, SamplingParams
from engine.scheduler import Scheduler
from model.kv_cache import PagedKVCache
from model.minimal_llama import LlamaConfig, LlamaWeights, init_weights
from model.model_runner import ModelRunner


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
                 max_model_len: int = None, device: str = "cuda", seed: int = 0):
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
        genuinely-computed prefill progress into the prefix cache, sweep
        out whatever finished. Returns (SchedulerOutput, finished_requests)
        -- mirrors Scheduler.schedule()/free_finished_requests()'s own
        return shapes, since this is just sequencing them with the forward
        pass (and, when enabled, the prefix-cache registration step) run in
        between.
        """
        output = self.scheduler.schedule()
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
        finished = self.scheduler.free_finished_requests()
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
