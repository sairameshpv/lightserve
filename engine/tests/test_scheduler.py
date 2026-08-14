"""Correctness tests for engine/scheduler.py's continuous-batching
schedule() loop: admission, steady-state decode, preemption-under-pressure,
and teardown. Pure Python, no torch/CUDA -- see test_block_manager.py's
module docstring on why these run for real on CI, not just import-checked.
"""
from engine.config import CacheConfig, SchedulerConfig
from engine.request import Request, RequestStatus, SamplingParams
from engine.scheduler import Scheduler


def make_scheduler(block_size=4, num_gpu_blocks=100, watermark_blocks=0,
                    max_num_seqs=256, max_num_batched_tokens=2048):
    return Scheduler(
        CacheConfig(block_size=block_size, num_gpu_blocks=num_gpu_blocks, watermark_blocks=watermark_blocks),
        SchedulerConfig(max_num_seqs=max_num_seqs, max_num_batched_tokens=max_num_batched_tokens),
    )


def make_request(request_id, prompt_len=8, max_tokens=16, eos_token_id=None):
    return Request(
        request_id=request_id,
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=SamplingParams(max_tokens=max_tokens, eos_token_id=eos_token_id),
    )


class TestAdmission:
    def test_add_request_lands_in_waiting(self):
        sched = make_scheduler()
        req = make_request("r0")
        sched.add_request(req)
        assert req.status == RequestStatus.WAITING
        assert req in sched.waiting
        assert sched.get_num_unfinished_requests() == 1

    def test_duplicate_request_id_rejected(self):
        sched = make_scheduler()
        sched.add_request(make_request("r0"))
        try:
            sched.add_request(make_request("r0"))
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_schedule_admits_a_waiting_request_as_prefill(self):
        sched = make_scheduler(block_size=4)
        req = make_request("r0", prompt_len=8)
        sched.add_request(req)

        output = sched.schedule()

        assert len(output.scheduled_new) == 1
        assert output.scheduled_new[0].num_scheduled_tokens == 8
        assert output.scheduled_running == []
        assert req.status == RequestStatus.RUNNING
        assert req.num_computed_tokens == 8
        assert req in sched.running
        assert req not in sched.waiting
        assert len(sched.block_manager.get_block_table(req)) == 2  # ceil(8/4)

    def test_schedule_admits_fifo_order(self):
        sched = make_scheduler()
        a, b = make_request("a"), make_request("b")
        sched.add_request(a)
        sched.add_request(b)
        output = sched.schedule()
        ids_in_order = [sr.request.request_id for sr in output.scheduled_new]
        assert ids_in_order == ["a", "b"]

    def test_admission_blocked_by_max_num_seqs(self):
        sched = make_scheduler(max_num_seqs=1)
        a, b = make_request("a"), make_request("b")
        sched.add_request(a)
        sched.add_request(b)
        output = sched.schedule()
        assert [sr.request.request_id for sr in output.scheduled_new] == ["a"]
        assert list(sched.waiting) == [b]

    def test_admission_blocked_by_token_budget(self):
        sched = make_scheduler(max_num_batched_tokens=10)
        a = make_request("a", prompt_len=8)
        b = make_request("b", prompt_len=8)  # 8+8=16 > budget of 10
        sched.add_request(a)
        sched.add_request(b)
        output = sched.schedule()
        assert [sr.request.request_id for sr in output.scheduled_new] == ["a"]
        assert list(sched.waiting) == [b]

    def test_fifo_head_of_line_blocking(self):
        """A small request behind a too-big one does NOT jump ahead of it --
        documented FIFO behavior, see scheduler.py's _schedule_waiting.
        """
        sched = make_scheduler(block_size=4, num_gpu_blocks=2)  # only 8 tokens' worth of blocks
        big = make_request("big", prompt_len=12)   # needs 3 blocks, won't fit
        small = make_request("small", prompt_len=2)  # needs 1 block, would fit alone
        sched.add_request(big)
        sched.add_request(small)
        output = sched.schedule()
        assert output.scheduled_new == []
        assert list(sched.waiting) == [big, small]


class TestSteadyStateDecode:
    def test_decode_step_schedules_exactly_one_new_token(self):
        sched = make_scheduler(block_size=4)
        req = make_request("r0", prompt_len=8)
        sched.add_request(req)
        sched.schedule()  # prefill
        assert req.num_computed_tokens == 8

        # Model runner would append the sampled token here.
        req.output_token_ids.append(1)
        output = sched.schedule()

        assert len(output.scheduled_running) == 1
        assert output.scheduled_running[0].num_scheduled_tokens == 1
        assert req.num_computed_tokens == 9
        assert req in sched.running

    def test_decode_grows_block_table_only_at_boundary(self):
        sched = make_scheduler(block_size=4)
        req = make_request("r0", prompt_len=4, max_tokens=100)  # 1 block, exactly full
        sched.add_request(req)
        sched.schedule()
        assert len(sched.block_manager.get_block_table(req)) == 1

        req.output_token_ids.append(1)  # len=5, crosses into a 2nd block
        sched.schedule()
        assert len(sched.block_manager.get_block_table(req)) == 2

        for _ in range(3):  # len 6, 7, 8 -- still fit in 2 blocks (capacity 8)
            req.output_token_ids.append(1)
            sched.schedule()
        assert len(sched.block_manager.get_block_table(req)) == 2


class TestPreemption:
    def test_preemption_frees_blocks_and_requeues_at_front(self):
        # 2 blocks total, block_size=4. Two 1-block requests fill the pool.
        sched = make_scheduler(block_size=4, num_gpu_blocks=2)
        victim = make_request("victim", prompt_len=4, max_tokens=100)
        survivor = make_request("survivor", prompt_len=4, max_tokens=100)
        newcomer = make_request("newcomer", prompt_len=4)

        sched.add_request(victim)
        sched.add_request(survivor)
        sched.schedule()  # both admitted, pool now full (2/2 blocks used)
        assert victim in sched.running and survivor in sched.running

        sched.add_request(newcomer)  # stays WAITING, no free blocks

        # Both running requests now need a 2nd block on the same step --
        # no free blocks exist, so the lower-priority one (survivor, added
        # after victim) must be preempted to make room for victim.
        victim.output_token_ids.append(1)
        survivor.output_token_ids.append(1)
        output = sched.schedule()

        assert len(output.preempted) == 1
        preempted_req = output.preempted[0]
        assert preempted_req.request_id == "survivor"
        assert preempted_req.status == RequestStatus.PREEMPTED
        assert preempted_req.num_computed_tokens == 0
        assert sched.block_manager.get_block_table(preempted_req) == []

        # Requeued at the FRONT of waiting -- ahead of newcomer, which
        # arrived later but was never running.
        assert list(sched.waiting)[0] is preempted_req

        assert victim in sched.running
        assert preempted_req not in sched.running
        # No new admissions happened this step -- a preemption occurred.
        assert output.scheduled_new == []

    def test_preempted_request_resumes_as_a_fresh_prefill(self):
        sched = make_scheduler(block_size=4, num_gpu_blocks=1)
        a = make_request("a", prompt_len=4, max_tokens=100)
        sched.add_request(a)
        sched.schedule()
        assert a.num_computed_tokens == 4

        # Manually preempt via a second competing request needing the only block.
        a.output_token_ids.append(1)  # a now needs a 2nd block
        b = make_request("b", prompt_len=4, max_tokens=100)
        sched.add_request(b)
        # a is RUNNING and gets processed before b is even considered
        # (b is still WAITING) -- force pressure by shrinking free blocks
        # to 0 directly is unnecessary; there's only 1 block total, held by
        # a, so a's own append_slot needs a 2nd block with nothing to
        # preempt from `pending` (a is the only running request) -> a
        # preempts itself.
        output = sched.schedule()
        assert len(output.preempted) == 1
        assert output.preempted[0].request_id == "a"
        assert a.status == RequestStatus.PREEMPTED
        assert a.num_computed_tokens == 0
        # 5 tokens (4 prompt + 1 output) survive the preemption -- recompute
        # strategy keeps the token ids, just loses the KV cache.
        assert a.get_len() == 5

        # Next step: a is re-admitted, whole 5-token sequence-so-far
        # scheduled as one fresh prefill.
        output2 = sched.schedule()
        assert len(output2.scheduled_new) == 1
        assert output2.scheduled_new[0].request is a
        assert output2.scheduled_new[0].num_scheduled_tokens == 5
        assert a.num_computed_tokens == 5


class TestTeardown:
    def test_maybe_finish_on_length_cap(self):
        sched = make_scheduler()
        req = make_request("r0", prompt_len=4, max_tokens=2)
        sched.add_request(req)
        sched.schedule()

        req.output_token_ids.append(1)
        req.maybe_finish()
        assert req.status == RequestStatus.RUNNING  # 1 output token, cap is 2

        req.output_token_ids.append(2)
        req.maybe_finish()
        assert req.status == RequestStatus.FINISHED_LENGTH_CAPPED
        assert req.is_finished()

    def test_maybe_finish_on_eos(self):
        req = make_request("r0", eos_token_id=99)
        req.output_token_ids.append(99)
        req.maybe_finish()
        assert req.status == RequestStatus.FINISHED_STOPPED

    def test_free_finished_requests_releases_blocks_and_removes_from_running(self):
        sched = make_scheduler(block_size=4, num_gpu_blocks=10)
        req = make_request("r0", prompt_len=4, max_tokens=1)
        sched.add_request(req)
        sched.schedule()
        assert sched.block_manager.num_free_blocks == 9

        req.output_token_ids.append(1)
        req.maybe_finish()
        assert req.status == RequestStatus.FINISHED_LENGTH_CAPPED

        freed = sched.free_finished_requests()
        assert freed == [req]
        assert req not in sched.running
        assert sched.block_manager.num_free_blocks == 10
        assert req.request_id not in sched.requests

    def test_abort_waiting_request(self):
        sched = make_scheduler()
        req = make_request("r0")
        sched.add_request(req)
        aborted = sched.abort_requests(["r0"])
        assert aborted == [req]
        assert req.status == RequestStatus.FINISHED_ABORTED
        assert req not in sched.waiting
        assert req.request_id not in sched.requests

    def test_abort_running_request_frees_its_blocks(self):
        sched = make_scheduler(block_size=4, num_gpu_blocks=10)
        req = make_request("r0", prompt_len=4)
        sched.add_request(req)
        sched.schedule()
        assert sched.block_manager.num_free_blocks == 9

        aborted = sched.abort_requests(["r0"])
        assert aborted == [req]
        assert req.status == RequestStatus.FINISHED_ABORTED
        assert req not in sched.running
        assert sched.block_manager.num_free_blocks == 10

    def test_abort_unknown_id_is_a_noop(self):
        sched = make_scheduler()
        sched.add_request(make_request("r0"))
        aborted = sched.abort_requests(["does-not-exist"])
        assert aborted == []
        assert sched.get_num_unfinished_requests() == 1