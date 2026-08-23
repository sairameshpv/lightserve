"""Correctness tests for engine/request.py's Request/RequestStatus state
machine. Pure Python, no torch/CUDA -- see this module's own docstring on
why it's fully unit-testable without a GPU. Referenced by name in
request.py's module docstring; scheduler.py's tests exercise these methods
indirectly through a live Scheduler, this file tests them directly and in
isolation.
"""
from engine.request import Request, RequestStatus, SamplingParams


def make_request(prompt_len=8, num_output=0, max_tokens=16, eos_token_id=None):
    req = Request(
        request_id="r0",
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=SamplingParams(max_tokens=max_tokens, eos_token_id=eos_token_id),
    )
    req.output_token_ids = list(range(num_output))
    return req


class TestLengthAccounting:
    def test_get_len_is_prompt_plus_output(self):
        req = make_request(prompt_len=8, num_output=3)
        assert req.get_len() == 11
        assert req.all_token_ids() == list(range(8)) + list(range(3))

    def test_get_num_new_tokens_before_any_computation(self):
        req = make_request(prompt_len=8)
        assert req.num_computed_tokens == 0
        assert req.get_num_new_tokens() == 8  # whole prompt is new

    def test_get_num_new_tokens_after_partial_computation(self):
        req = make_request(prompt_len=8, num_output=2)
        req.num_computed_tokens = 8  # prompt done, output tokens not yet
        assert req.get_num_new_tokens() == 2

    def test_get_num_new_tokens_zero_once_fully_computed(self):
        req = make_request(prompt_len=8)
        req.num_computed_tokens = 8
        assert req.get_num_new_tokens() == 0


class TestIsPrefill:
    def test_true_before_any_prompt_token_is_computed(self):
        req = make_request(prompt_len=8)
        assert req.is_prefill()

    def test_true_mid_prompt(self):
        req = make_request(prompt_len=8)
        req.num_computed_tokens = 4
        assert req.is_prefill()

    def test_false_once_whole_prompt_is_computed(self):
        req = make_request(prompt_len=8)
        req.num_computed_tokens = 8
        assert not req.is_prefill()

    def test_false_during_steady_state_decode(self):
        req = make_request(prompt_len=8, num_output=3)
        req.num_computed_tokens = 10  # prompt + 2 of 3 output tokens
        assert not req.is_prefill()


class TestNumBlocksNeeded:
    def test_zero_length_needs_zero_blocks(self):
        req = make_request(prompt_len=0)
        assert req.num_blocks_needed(block_size=4) == 0

    def test_exact_multiple_of_block_size(self):
        req = make_request(prompt_len=8)
        assert req.num_blocks_needed(block_size=4) == 2

    def test_rounds_up_for_a_partial_final_block(self):
        req = make_request(prompt_len=9)
        assert req.num_blocks_needed(block_size=4) == 3

    def test_output_tokens_count_toward_blocks_needed(self):
        req = make_request(prompt_len=8, num_output=1)  # 9 tokens total
        assert req.num_blocks_needed(block_size=4) == 3


class TestMaybeFinish:
    def test_no_stop_condition_stays_running(self):
        req = make_request(max_tokens=16)
        req.status = RequestStatus.RUNNING  # maybe_finish is only ever called on a RUNNING request
        req.output_token_ids.append(1)
        req.maybe_finish()
        assert req.status == RequestStatus.RUNNING
        assert not req.is_finished()

    def test_eos_token_finishes_as_stopped(self):
        req = make_request(eos_token_id=99)
        req.status = RequestStatus.RUNNING
        req.output_token_ids.append(99)
        req.maybe_finish()
        assert req.status == RequestStatus.FINISHED_STOPPED

    def test_non_eos_last_token_does_not_finish(self):
        req = make_request(eos_token_id=99)
        req.status = RequestStatus.RUNNING
        req.output_token_ids.append(1)
        req.maybe_finish()
        assert req.status == RequestStatus.RUNNING

    def test_max_tokens_finishes_as_length_capped(self):
        req = make_request(max_tokens=2)
        req.status = RequestStatus.RUNNING
        req.output_token_ids.extend([1, 2])
        req.maybe_finish()
        assert req.status == RequestStatus.FINISHED_LENGTH_CAPPED

    def test_eos_checked_before_length_cap_when_both_would_fire(self):
        # Last token is both the eos id and puts the request at max_tokens --
        # status should reflect the eos branch (checked first in
        # maybe_finish), not silently become inconsistent between the two.
        req = make_request(max_tokens=1, eos_token_id=99)
        req.status = RequestStatus.RUNNING
        req.output_token_ids.append(99)
        req.maybe_finish()
        assert req.status == RequestStatus.FINISHED_STOPPED

    def test_already_finished_status_is_left_alone(self):
        # An aborted request must not get silently flipped to a different
        # terminal state just because its token counts happen to satisfy a
        # stop condition too.
        req = make_request(max_tokens=1)
        req.status = RequestStatus.FINISHED_ABORTED
        req.output_token_ids.append(1)  # would otherwise hit the length cap
        req.maybe_finish()
        assert req.status == RequestStatus.FINISHED_ABORTED

    def test_is_finished_true_for_every_terminal_status(self):
        for status in (
            RequestStatus.FINISHED_STOPPED,
            RequestStatus.FINISHED_LENGTH_CAPPED,
            RequestStatus.FINISHED_ABORTED,
        ):
            req = make_request()
            req.status = status
            assert req.is_finished()

    def test_is_finished_false_for_non_terminal_statuses(self):
        for status in (RequestStatus.WAITING, RequestStatus.RUNNING, RequestStatus.PREEMPTED):
            req = make_request()
            req.status = status
            assert not req.is_finished()
