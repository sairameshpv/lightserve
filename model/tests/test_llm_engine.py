"""End-to-end correctness for model/llm_engine.py: Allocator
(engine/block_manager.py, via Scheduler) + Scheduler + ModelRunner, all
driven by LLMEngine.generate() from raw prompt token ids to a finished
GenerationOutput. Requires CUDA -- see test_minimal_llama.py's module
docstring.

Same ground truth as test_model_runner.py: model/minimal_llama.py's
reference_llama_forward, re-run dense over tokens-so-far after every step.
"""
from dataclasses import replace

import pytest
import torch

from engine.config import CacheConfig, SchedulerConfig
from engine.request import RequestStatus, SamplingParams
from model.llm_engine import LLMEngine
from model.minimal_llama import TOY_CONFIG, init_weights, reference_llama_forward

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="LLMEngine needs Triton kernels on a real CUDA GPU"
)


def _reference_generate(weights, config, prompt, max_tokens, eos_token_id=None):
    tokens = list(prompt)
    generated = []
    for _ in range(max_tokens):
        input_ids = torch.tensor([tokens], device="cuda")
        logits = reference_llama_forward(weights, config, input_ids, causal=True)
        next_tok = int(logits[0, -1].argmax().item())
        tokens.append(next_tok)
        generated.append(next_tok)
        if eos_token_id is not None and next_tok == eos_token_id:
            break
    return generated


def _make_engine(config, weights, block_size=4, num_gpu_blocks=64, max_num_seqs=8, max_num_batched_tokens=64):
    cache_config = CacheConfig(block_size=block_size, num_gpu_blocks=num_gpu_blocks)
    scheduler_config = SchedulerConfig(max_num_seqs=max_num_seqs, max_num_batched_tokens=max_num_batched_tokens)
    return LLMEngine(cache_config, scheduler_config, config, weights=weights, device="cuda")


@requires_cuda
class TestGenerate:
    def test_single_prompt_matches_reference(self):
        torch.manual_seed(0)
        config = replace(TOY_CONFIG, dtype=torch.float32)
        weights = init_weights(config, device="cuda", seed=0)
        engine = _make_engine(config, weights)

        prompt = [1, 2, 3, 4]
        [output] = engine.generate([prompt], sampling_params=SamplingParams(max_tokens=5))

        assert output.output_token_ids == _reference_generate(weights, config, prompt, max_tokens=5)
        assert output.prompt_token_ids == prompt
        assert output.finish_reason == RequestStatus.FINISHED_LENGTH_CAPPED.name

    def test_concurrent_prompts_of_different_lengths_match_their_own_solo_reference(self):
        # Both submitted in the same generate() call -- the scheduler
        # interleaves their prefills/decodes, exactly the "batched prefill +
        # decode" behavior under test -- yet each must come out identical to
        # what it would've produced running alone, since the model has no
        # cross-sequence state to leak through.
        torch.manual_seed(0)
        config = replace(TOY_CONFIG, dtype=torch.float32)
        weights = init_weights(config, device="cuda", seed=0)
        engine = _make_engine(config, weights)

        prompt_a, prompt_b = [1, 2, 3], [9, 8, 7, 6, 5]
        out_a, out_b = engine.generate([prompt_a, prompt_b], sampling_params=SamplingParams(max_tokens=4))

        assert out_a.output_token_ids == _reference_generate(weights, config, prompt_a, max_tokens=4)
        assert out_b.output_token_ids == _reference_generate(weights, config, prompt_b, max_tokens=4)

    def test_eos_stops_generation_before_max_tokens(self):
        torch.manual_seed(0)
        config = replace(TOY_CONFIG, dtype=torch.float32)
        weights = init_weights(config, device="cuda", seed=0)
        engine = _make_engine(config, weights)

        prompt = [1, 2, 3]
        # Whatever token the reference path generates first, forcing that
        # exact id to be eos_token_id must stop generation after 1 token
        # even though max_tokens allows many more.
        first = _reference_generate(weights, config, prompt, max_tokens=1)[0]
        [output] = engine.generate([prompt], sampling_params=SamplingParams(max_tokens=10, eos_token_id=first))

        assert output.output_token_ids == [first]
        assert output.finish_reason == RequestStatus.FINISHED_STOPPED.name

    def test_preemption_under_tight_blocks_still_matches_reference(self):
        # block_size=2 / num_gpu_blocks=3 shared by 2 concurrent requests
        # that each grow to 6 tokens (3 blocks' worth) is tight enough to
        # force at least one real preemption+recompute along the way (same
        # "two requests, one pool" pressure engine/tests/test_scheduler.py's
        # TestPreemption sets up, just with a real model and KV cache behind
        # it this time). The point: recompute-based preemption must be
        # transparent to the model's actual output, not just to the
        # scheduler's own bookkeeping.
        torch.manual_seed(0)
        config = replace(TOY_CONFIG, dtype=torch.float32)
        weights = init_weights(config, device="cuda", seed=0)
        engine = _make_engine(config, weights, block_size=2, num_gpu_blocks=3)

        prompt_a, prompt_b = [1, 2], [3, 4]
        out_a, out_b = engine.generate([prompt_a, prompt_b], sampling_params=SamplingParams(max_tokens=4))

        assert out_a.output_token_ids == _reference_generate(weights, config, prompt_a, max_tokens=4)
        assert out_b.output_token_ids == _reference_generate(weights, config, prompt_b, max_tokens=4)
