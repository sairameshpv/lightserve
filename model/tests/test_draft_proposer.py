"""Correctness tests for model/draft_proposer.py: does DraftProposer.
propose() produce exactly what greedy step-by-step decoding of the same
draft model would produce next -- not merely "some tokens"? Requires CUDA,
same reason as model/tests/test_model_runner.py (the draft ModelRunner
needs Triton kernels on a real GPU).
"""
from dataclasses import replace

import pytest
import torch

from engine.config import CacheConfig
from engine.request import Request, SamplingParams
from model.draft_proposer import DraftProposer
from model.kv_cache import PagedKVCache
from model.minimal_llama import TOY_CONFIG, init_weights, reference_llama_forward
from model.model_runner import ModelRunner

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="DraftProposer needs Triton kernels on a real CUDA GPU"
)


def _reference_next_token(weights, config, token_ids):
    input_ids = torch.tensor([token_ids], device="cuda")
    logits = reference_llama_forward(weights, config, input_ids, causal=True)
    return int(logits[0, -1].argmax().item())


def _reference_continuation(weights, config, seed_ids, num_new_tokens):
    """Independent step-by-step greedy decode of the same model/weights --
    what propose() is checked against, same "dense from-scratch recompute
    at every step" reference pattern test_model_runner.py's own tests use.
    """
    tokens = list(seed_ids)
    out = []
    for _ in range(num_new_tokens):
        next_id = _reference_next_token(weights, config, tokens)
        tokens.append(next_id)
        out.append(next_id)
    return out


def _make_draft_proposer(config, weights, num_speculative_tokens, block_size=4, num_gpu_blocks=64):
    cache_config = CacheConfig(block_size=block_size, num_gpu_blocks=num_gpu_blocks)
    kv_cache = PagedKVCache(cache_config, config, device="cuda")
    runner = ModelRunner(config, weights, kv_cache, max_model_len=config.max_seq_len, device="cuda")
    return DraftProposer(runner, num_speculative_tokens)


@requires_cuda
def test_propose_matches_step_by_step_reference():
    torch.manual_seed(0)
    config = replace(TOY_CONFIG, dtype=torch.float32)
    weights = init_weights(config, device="cuda", seed=0)
    proposer = _make_draft_proposer(config, weights, num_speculative_tokens=3)

    prompt = [1, 2, 3, 4, 5]
    request = Request(request_id="r0", prompt_token_ids=prompt, sampling_params=SamplingParams(max_tokens=50))

    proposed = proposer.propose(request)

    assert proposed == _reference_continuation(weights, config, prompt, num_new_tokens=3)


@requires_cuda
def test_propose_again_continues_from_newly_committed_tokens():
    torch.manual_seed(0)
    config = replace(TOY_CONFIG, dtype=torch.float32)
    weights = init_weights(config, device="cuda", seed=0)
    proposer = _make_draft_proposer(config, weights, num_speculative_tokens=2)

    prompt = [1, 2, 3, 4, 5]
    request = Request(request_id="r0", prompt_token_ids=prompt, sampling_params=SamplingParams(max_tokens=50))

    first_round = proposer.propose(request)
    # Simulate an all-accepted round: the target committed exactly what
    # the draft proposed (Stage D/E's job in the real engine, not built
    # yet -- the only scenario expressible without them, see module
    # docstring on why this file assumes committed tokens only grow).
    request.output_token_ids.extend(first_round)

    second_round = proposer.propose(request)

    assert second_round == _reference_continuation(weights, config, prompt + first_round, num_new_tokens=2)


@requires_cuda
def test_rollback_then_propose_continues_from_only_the_accepted_prefix():
    torch.manual_seed(0)
    config = replace(TOY_CONFIG, dtype=torch.float32)
    weights = init_weights(config, device="cuda", seed=0)
    proposer = _make_draft_proposer(config, weights, num_speculative_tokens=3)

    prompt = [1, 2, 3, 4, 5]
    request = Request(request_id="r0", prompt_token_ids=prompt, sampling_params=SamplingParams(max_tokens=50))

    first_round = proposer.propose(request)  # K=3 proposed
    # Simulate partial acceptance: only the first of the 3 proposed tokens
    # survived verification (Stage E's real job -- this is what it does).
    accepted = first_round[:1]
    request.output_token_ids.extend(accepted)
    proposer.rollback(request.request_id, num_accepted=len(accepted))

    second_round = proposer.propose(request)

    # If rollback hadn't corrected the shadow's num_computed_tokens, this
    # would either crash (negative num_new inside propose()'s loop) or
    # silently skip tokens the shadow wrongly believed were already
    # computed -- diverging from the reference below.
    assert second_round == _reference_continuation(weights, config, prompt + accepted, num_new_tokens=3)


@requires_cuda
def test_propose_isolates_different_requests():
    torch.manual_seed(0)
    config = replace(TOY_CONFIG, dtype=torch.float32)
    weights = init_weights(config, device="cuda", seed=0)
    proposer = _make_draft_proposer(config, weights, num_speculative_tokens=2)

    prompt_a, prompt_b = [1, 2, 3], [9, 8, 7, 6]
    request_a = Request(request_id="a", prompt_token_ids=prompt_a, sampling_params=SamplingParams(max_tokens=50))
    request_b = Request(request_id="b", prompt_token_ids=prompt_b, sampling_params=SamplingParams(max_tokens=50))

    proposed_a = proposer.propose(request_a)
    proposed_b = proposer.propose(request_b)

    assert proposed_a == _reference_continuation(weights, config, prompt_a, num_new_tokens=2)
    assert proposed_b == _reference_continuation(weights, config, prompt_b, num_new_tokens=2)