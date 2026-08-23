"""Correctness tests for engine/scheduler.py's continuous-batching
schedule() loop: admission, steady-state decode, preemption-under-pressure,
and teardown. Pure Python, no torch/CUDA -- see test_block_manager.py's
module docstring on why these run for real on CI, not just import-checked.
"""
from engine.config import CacheConfig, SchedulerConfig
from engine.request import Request, RequestStatus, SamplingParams
from engine.scheduler import ScheduledRequest, Scheduler, SchedulerOutput


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


class TestMixedBatching:
    """One schedule() call handling a running decode and a new admission
    together -- the actual "continuous batching, mixed request lengths"
    behavior: iteration-level scheduling means every step re-decides the
    whole batch, not just whichever queue happened to be non-empty.
    """

    def test_same_step_admits_new_prefill_alongside_running_decode(self):
        sched = make_scheduler(block_size=4, max_num_batched_tokens=2048)
        running_req = make_request("running", prompt_len=4)
        sched.add_request(running_req)
        sched.schedule()  # admits running_req: prefill, num_computed_tokens=4
        assert running_req in sched.running

        running_req.output_token_ids.append(1)  # now needs 1 decode token
        waiting_req = make_request("waiting", prompt_len=6)
        sched.add_request(waiting_req)  # added after, so still WAITING

        output = sched.schedule()

        # Steady-state decode (length 1) and a fresh prefill (length 6)
        # scheduled in the very same step -- exactly the mixed-length,
        # admit-every-step behavior under test.
        assert len(output.scheduled_running) == 1
        assert output.scheduled_running[0].request is running_req
        assert output.scheduled_running[0].num_scheduled_tokens == 1
        assert len(output.scheduled_new) == 1
        assert output.scheduled_new[0].request is waiting_req
        assert output.scheduled_new[0].num_scheduled_tokens == 6
        assert output.total_num_scheduled_tokens == 7
        assert waiting_req in sched.running

    def test_token_budget_is_shared_between_running_and_waiting_in_one_step(self):
        # Running decode's 1 token is deducted from the budget *before*
        # _schedule_waiting sees what's left -- a tight budget must block
        # the new admission even though it would fit the raw config max.
        sched = make_scheduler(block_size=4, max_num_batched_tokens=6)
        running_req = make_request("running", prompt_len=4)
        sched.add_request(running_req)
        sched.schedule()
        running_req.output_token_ids.append(1)

        waiting_req = make_request("waiting", prompt_len=6)
        sched.add_request(waiting_req)

        output = sched.schedule()  # budget 6: 1 for decode, only 5 left for a 6-token prefill

        assert len(output.scheduled_running) == 1  # decode still goes through
        assert output.scheduled_new == []  # prefill didn't fit in what remained
        assert waiting_req in sched.waiting

        # One more token of budget (7 total: 1 decode + 6 prefill) admits it.
        sched2 = make_scheduler(block_size=4, max_num_batched_tokens=7)
        running_req2 = make_request("running2", prompt_len=4)
        sched2.add_request(running_req2)
        sched2.schedule()
        running_req2.output_token_ids.append(1)
        waiting_req2 = make_request("waiting2", prompt_len=6)
        sched2.add_request(waiting_req2)

        output2 = sched2.schedule()
        assert len(output2.scheduled_running) == 1
        assert len(output2.scheduled_new) == 1
        assert output2.total_num_scheduled_tokens == 7


class TestSchedulerOutputProperties:
    def test_is_empty_true_with_no_scheduled_requests(self):
        output = SchedulerOutput()
        assert output.is_empty
        assert output.total_num_scheduled_tokens == 0

    def test_is_empty_true_even_with_a_preemption(self):
        # is_empty only looks at scheduled_new/scheduled_running -- a step
        # that preempted someone but admitted/decoded no one still counts as
        # empty (see test_preemption_frees_blocks_and_requeues_at_front,
        # which hits exactly this: scheduled_new == [] the same step a
        # preemption happens).
        req = make_request("r0")
        output = SchedulerOutput(preempted=[req])
        assert output.is_empty

    def test_total_num_scheduled_tokens_sums_both_lists(self):
        new_req, running_req = make_request("new"), make_request("running")
        output = SchedulerOutput(
            scheduled_new=[ScheduledRequest(new_req, 6)],
            scheduled_running=[ScheduledRequest(running_req, 1)],
        )
        assert output.total_num_scheduled_tokens == 7
        assert not output.is_empty


class TestRunningRequestOverBudget:
    def test_running_request_needing_more_tokens_than_budget_stays_running_untouched(self):
        # This design has no chunked prefill (see engine/README.md), so a
        # RUNNING request normally only ever needs exactly 1 new token
        # (steady-state decode) by the time _schedule_running sees it again.
        # scheduler.py's comment on this branch flags it as mattering only
        # for "a request still mid-prefill re-entering this path" -- construct
        # that state directly rather than waiting on a future chunked-prefill
        # feature to exercise scheduler.py:141-147's own-budget check.
        sched = make_scheduler(block_size=4, num_gpu_blocks=10, max_num_batched_tokens=5)
        req = make_request("mid_prefill", prompt_len=10, max_tokens=100)
        sched.block_manager.allocate(req)
        req.status = RequestStatus.RUNNING
        req.num_computed_tokens = 2  # 8 new tokens needed, budget is only 5
        sched.requests[req.request_id] = req
        sched.running.append(req)

        output = sched.schedule()

        assert output.scheduled_running == []
        assert output.scheduled_new == []
        assert output.preempted == []
        assert req in sched.running  # left untouched, retried next schedule() call
        assert req.num_computed_tokens == 2  # not advanced


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

    def test_self_preemption_when_no_lower_priority_candidate_exists(self):
        # Only one running request, one block total. It grows past its
        # block's capacity, needs a 2nd block, none free, and -- being the
        # only entry left in `pending` -- has no lower-priority victim to
        # evict, so it preempts itself (scheduler.py's _schedule_running:
        # "if pending: ... else: self._preempt(request, output)").
        sched = make_scheduler(block_size=4, num_gpu_blocks=1)
        solo = make_request("solo", prompt_len=4, max_tokens=100)
        sched.add_request(solo)
        sched.schedule()
        assert sched.block_manager.num_free_blocks == 0

        solo.output_token_ids.append(1)  # len=5, crosses into a 2nd block
        output = sched.schedule()

        assert output.preempted == [solo]
        assert solo.status == RequestStatus.PREEMPTED
        assert solo not in sched.running
        assert list(sched.waiting)[0] is solo
        assert sched.block_manager.num_free_blocks == 1  # its 1 block was freed

    def test_preempted_request_resumes_as_a_fresh_prefill(self):
        # 2 blocks total, block_size=4. b and a (admitted in that order)
        # each hold the sequence's only spare block, so free==0 once both
        # are running -- num_gpu_blocks=1 can't set this up at all: with
        # only 1 block to ever hand out, a's fresh 2-block re-admission
        # below would be unsatisfiable no matter what freed it.
        sched = make_scheduler(block_size=4, num_gpu_blocks=2)
        b = make_request("b", prompt_len=4, max_tokens=1)
        sched.add_request(b)
        sched.schedule()

        a = make_request("a", prompt_len=4, max_tokens=100)
        sched.add_request(a)
        sched.schedule()
        assert a.num_computed_tokens == 4

        # Both blocks are now spoken for (b: 1, a: 1). Grow both by one
        # output token: b crosses its block boundary and needs a 2nd
        # block, and -- being processed before a in `pending` -- evicts a
        # (the lower-priority request still pending this step) rather
        # than itself. b also now sits at its own max_tokens.
        b.output_token_ids.append(1)
        a.output_token_ids.append(1)
        output = sched.schedule()
        assert len(output.preempted) == 1
        assert output.preempted[0].request_id == "a"
        assert a.status == RequestStatus.PREEMPTED
        assert a.num_computed_tokens == 0
        # 5 tokens (4 prompt + 1 output) survive the preemption -- recompute
        # strategy keeps the token ids, just loses the KV cache.
        assert a.get_len() == 5
        assert b in sched.running
        assert a not in sched.running

        # b immediately finishes (it just hit max_tokens=1) and frees both
        # of its blocks -- the room a's fresh 2-block prefill needs below.
        b.maybe_finish()
        assert b.is_finished()
        assert sched.free_finished_requests() == [b]

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