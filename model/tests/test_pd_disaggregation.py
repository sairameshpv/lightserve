"""Correctness tests for model/pd_disaggregation.py: does prefill() on one
LLMEngine + resume_decode() on a *different* LLMEngine -- possibly with a
different block_size/num_gpu_blocks -- produce output byte-identical to a
single-engine run of the same prompt? Requires CUDA -- see
test_minimal_llama.py's module docstring.

Same ground truth as test_llm_engine.py: dense reference_llama_forward,
re-run from scratch every step. Each test file in this repo inlines its own
copy of that reference helper rather than importing across test modules
(see test_draft_proposer.py, test_llm_engine.py) -- this one follows suit.
"""
from dataclasses import replace

import pytest
import torch

from engine.config import CacheConfig, SchedulerConfig
from engine.request import RequestStatus, SamplingParams
from model.llm_engine import GenerationOutput, LLMEngine
from model.minimal_llama import TOY_CONFIG, init_weights, reference_llama_forward
from model.pd_disaggregation import generate_disaggregated, prefill

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
class TestPDDisaggregation:
    def test_matches_single_engine_dense_short_prompt(self):
        torch.manual_seed(0)
        config = replace(TOY_CONFIG, dtype=torch.float32)
        weights = init_weights(config, device="cuda", seed=0)

        prefill_engine = _make_engine(config, weights)
        decode_engine = _make_engine(config, weights)

        prompt = [1, 2, 3, 4]
        output = generate_disaggregated(prefill_engine, decode_engine, prompt,
                                         sampling_params=SamplingParams(max_tokens=10))

        assert output.output_token_ids == _reference_generate(weights, config, prompt, max_tokens=10)
        assert output.prompt_token_ids == prompt

    def test_matches_single_engine_dense_multi_step_chunked_prefill(self):
        # prompt_len=10, chunk=3 -- doesn't divide evenly, forcing several
        # schedule() calls (and several loop iterations inside prefill())
        # before is_prefill() finally goes False. Same shape benchmarks/
        # chunked_prefill/verify_multi_chunk_correctness.py uses to force
        # this path.
        torch.manual_seed(0)
        config = replace(TOY_CONFIG, dtype=torch.float32)
        weights = init_weights(config, device="cuda", seed=0)

        prefill_engine = _make_engine(config, weights, max_num_batched_tokens=3)
        decode_engine = _make_engine(config, weights)

        prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        output = generate_disaggregated(prefill_engine, decode_engine, prompt,
                                         sampling_params=SamplingParams(max_tokens=5))

        assert output.output_token_ids == _reference_generate(weights, config, prompt, max_tokens=5)

    def test_max_tokens_one_finishes_during_prefill(self):
        # The completing prefill step samples and appends a token in the
        # same step (see pd_disaggregation.py's module docstring) -- with
        # max_tokens=1 that immediately hits the cap, so there is nothing
        # left to hand off to a decoder at all.
        torch.manual_seed(0)
        config = replace(TOY_CONFIG, dtype=torch.float32)
        weights = init_weights(config, device="cuda", seed=0)
        prefill_engine = _make_engine(config, weights)

        prompt = [1, 2, 3]
        result = prefill(prefill_engine, prompt, sampling_params=SamplingParams(max_tokens=1))

        assert isinstance(result, GenerationOutput)
        assert result.output_token_ids == _reference_generate(weights, config, prompt, max_tokens=1)
        assert result.finish_reason == RequestStatus.FINISHED_LENGTH_CAPPED.name
        # Confirms free_finished_requests() already swept it -- nothing
        # leaked on the prefiller from a request that never got exported.
        assert not prefill_engine.scheduler.has_unfinished_requests()

    def test_decoder_with_different_block_layout(self):
        # The actual proof of block-table/block_size independence -- not
        # just asserted in kv_cache.py's docstring, exercised here through
        # a real forward pass on two engines that share nothing but the
        # model shape: different block_size AND different num_gpu_blocks.
        torch.manual_seed(0)
        config = replace(TOY_CONFIG, dtype=torch.float32)
        weights = init_weights(config, device="cuda", seed=0)

        prefill_engine = _make_engine(config, weights, block_size=8, num_gpu_blocks=32)
        decode_engine = _make_engine(config, weights, block_size=3, num_gpu_blocks=97)

        prompt = [5, 4, 3, 2, 1, 9, 9, 9, 1, 2, 3]  # 11 tokens -- not a multiple of either block_size
        output = generate_disaggregated(prefill_engine, decode_engine, prompt,
                                         sampling_params=SamplingParams(max_tokens=8))

        assert output.output_token_ids == _reference_generate(weights, config, prompt, max_tokens=8)

    def test_handoff_carries_exactly_the_prompt_and_first_token(self):
        # Direct structural check of prefill()'s contract (see its and the
        # module's docstrings), not just the end-to-end output -- pins the
        # handoff shape down explicitly.
        torch.manual_seed(0)
        config = replace(TOY_CONFIG, dtype=torch.float32)
        weights = init_weights(config, device="cuda", seed=0)
        prefill_engine = _make_engine(config, weights)

        prompt = [1, 2, 3, 4, 5]
        handoff = prefill(prefill_engine, prompt, sampling_params=SamplingParams(max_tokens=10))

        expected_first_token = _reference_generate(weights, config, prompt, max_tokens=1)[0]
        assert handoff.prompt_token_ids == prompt
        assert handoff.first_token_id == expected_first_token
        assert handoff.k.shape[1] == len(prompt)  # exactly the prompt's KV, not prompt+1
        assert handoff.k.device.type == "cpu"
        # prefill() aborts/frees the request from the prefiller once
        # exported -- nothing should be left running there.
        assert not prefill_engine.scheduler.has_unfinished_requests()