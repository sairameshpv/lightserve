"""Correctness tests for model/model_runner.py: the incremental,
KV-cache-backed forward pass wired to engine/scheduler.py's SchedulerOutput.
Requires CUDA -- see test_minimal_llama.py's module docstring.

Ground truth throughout is model/minimal_llama.py's `reference_llama_forward`
(plain PyTorch, dense, no KV cache, no padding tricks) re-run from scratch
over the full tokens-so-far after every step. If the incremental/gathered/
padded-causal path here (see model_runner.py's module docstring) agrees with
a dense from-scratch recompute at every single step, RoPE's per-token
absolute positions, the KV-cache write/read addressing, and the
pad-Q-to-K's-length causal trick are all correct together, not just each in
isolation.
"""
from dataclasses import replace

import pytest
import torch

from engine.config import CacheConfig, SchedulerConfig
from engine.request import Request, SamplingParams
from engine.scheduler import Scheduler, SchedulerOutput
from model.kv_cache import PagedKVCache
from model.minimal_llama import TOY_CONFIG, init_weights, reference_llama_forward
from model.model_runner import ModelRunner

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="ModelRunner needs Triton kernels on a real CUDA GPU"
)


def _reference_next_token(weights, config, token_ids):
    input_ids = torch.tensor([token_ids], device="cuda")
    logits = reference_llama_forward(weights, config, input_ids, causal=True)
    return int(logits[0, -1].argmax().item())


def _make_scheduler_and_runner(config, weights, block_size=4, num_gpu_blocks=64,
                                max_num_seqs=8, max_num_batched_tokens=64):
    cache_config = CacheConfig(block_size=block_size, num_gpu_blocks=num_gpu_blocks)
    kv_cache = PagedKVCache(cache_config, config, device="cuda")
    runner = ModelRunner(config, weights, kv_cache, max_model_len=config.max_seq_len, device="cuda")
    scheduler = Scheduler(
        cache_config, SchedulerConfig(max_num_seqs=max_num_seqs, max_num_batched_tokens=max_num_batched_tokens),
    )
    return scheduler, runner


@requires_cuda
class TestIncrementalGenerationMatchesReference:
    def test_single_request_step_by_step(self):
        torch.manual_seed(0)
        config = replace(TOY_CONFIG, dtype=torch.float32)
        weights = init_weights(config, device="cuda", seed=0)
        scheduler, runner = _make_scheduler_and_runner(config, weights)

        prompt = [1, 2, 3, 4, 5]
        req = Request(request_id="r0", prompt_token_ids=prompt, sampling_params=SamplingParams(max_tokens=6))
        scheduler.add_request(req)

        reference_tokens = list(prompt)
        steps = 0
        while scheduler.has_unfinished_requests() and steps < 20:
            output = scheduler.schedule()
            runner.execute_model(output)
            scheduler.free_finished_requests()
            steps += 1
            reference_tokens.append(_reference_next_token(weights, config, reference_tokens))

        assert req.output_token_ids == reference_tokens[len(prompt):]

    def test_mixed_batch_prefill_and_decode_in_one_execute_model_call(self):
        # One request already RUNNING (about to take a decode step) and one
        # freshly WAITING (about to be admitted as a prefill) scheduled
        # together -- exercises _build_flat_batch/_attention's heterogeneous
        # num_scheduled_tokens handling within a single call.
        torch.manual_seed(0)
        config = replace(TOY_CONFIG, dtype=torch.float32)
        weights = init_weights(config, device="cuda", seed=0)
        scheduler, runner = _make_scheduler_and_runner(config, weights)

        running_req = Request(
            request_id="running", prompt_token_ids=[1, 2, 3], sampling_params=SamplingParams(max_tokens=10),
        )
        scheduler.add_request(running_req)
        output = scheduler.schedule()  # admits + prefills running_req
        runner.execute_model(output)
        assert len(running_req.output_token_ids) == 1

        waiting_req = Request(
            request_id="waiting", prompt_token_ids=[9, 8, 7, 6], sampling_params=SamplingParams(max_tokens=10),
        )
        scheduler.add_request(waiting_req)

        output = scheduler.schedule()  # running_req decodes, waiting_req prefills -- same step
        assert len(output.scheduled_running) == 1
        assert len(output.scheduled_new) == 1
        runner.execute_model(output)

        assert len(running_req.output_token_ids) == 2
        assert len(waiting_req.output_token_ids) == 1

        expected_running = _reference_next_token(
            weights, config, running_req.prompt_token_ids + running_req.output_token_ids[:1],
        )
        expected_waiting = _reference_next_token(weights, config, waiting_req.prompt_token_ids)
        assert running_req.output_token_ids[1] == expected_running
        assert waiting_req.output_token_ids[0] == expected_waiting

    def test_two_simultaneous_prefills_of_different_lengths_dont_contaminate(self):
        torch.manual_seed(0)
        config = replace(TOY_CONFIG, dtype=torch.float32)
        weights = init_weights(config, device="cuda", seed=0)
        scheduler, runner = _make_scheduler_and_runner(config, weights)

        req_a = Request(request_id="a", prompt_token_ids=[1, 2, 3], sampling_params=SamplingParams(max_tokens=1))
        req_b = Request(
            request_id="b", prompt_token_ids=[9, 8, 7, 6, 5], sampling_params=SamplingParams(max_tokens=1),
        )
        scheduler.add_request(req_a)
        scheduler.add_request(req_b)

        output = scheduler.schedule()  # both admitted as prefill, same step, different lengths
        assert len(output.scheduled_new) == 2
        runner.execute_model(output)

        assert req_a.output_token_ids[0] == _reference_next_token(weights, config, req_a.prompt_token_ids)
        assert req_b.output_token_ids[0] == _reference_next_token(weights, config, req_b.prompt_token_ids)


@requires_cuda
def test_execute_model_returns_empty_list_for_an_empty_scheduler_output():
    config = replace(TOY_CONFIG, dtype=torch.float32)
    weights = init_weights(config, device="cuda", seed=0)
    cache_config = CacheConfig(block_size=4, num_gpu_blocks=10)
    kv_cache = PagedKVCache(cache_config, config, device="cuda")
    runner = ModelRunner(config, weights, kv_cache, device="cuda")

    assert runner.execute_model(SchedulerOutput()) == []
