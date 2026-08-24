"""FakeEngine: stands in for model/llm_engine.py's LLMEngine in tests that
must run without CUDA/Triton -- same add_request/step/scheduler-shaped
surface EngineWorker actually calls (see server/engine_worker.py's module
docstring), built on the real, already-torch-free engine/scheduler.py, so
admission/preemption/teardown behave exactly like the real engine. Only
`step()`'s per-token math is fake: instead of a real forward pass, each
scheduled request's next token is just a running count of its own output
tokens so far -- deterministic and trivially checkable, mirroring
model/model_runner.py's ModelRunner.execute_model's own contract (mutate
output_token_ids, call maybe_finish()) without needing torch at all.
"""
from engine.config import CacheConfig, SchedulerConfig
from engine.request import Request, SamplingParams
from engine.scheduler import Scheduler


class FakeEngine:
    def __init__(self, cache_config: CacheConfig = None, scheduler_config: SchedulerConfig = None):
        self.scheduler = Scheduler(
            cache_config or CacheConfig(block_size=4, num_gpu_blocks=64),
            scheduler_config or SchedulerConfig(max_num_seqs=8, max_num_batched_tokens=64),
        )
        self._next_id = 0

    def add_request(self, prompt_token_ids, sampling_params=None, request_id=None) -> Request:
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
        output = self.scheduler.schedule()
        for sr in list(output.scheduled_new) + list(output.scheduled_running):
            request = sr.request
            request.output_token_ids.append(len(request.output_token_ids))
            request.maybe_finish()
        finished = self.scheduler.free_finished_requests()
        return output, finished
