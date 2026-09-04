"""Continuous-batching scheduler -- decides, once per engine step, which
requests get a forward pass this step and how many new tokens each gets KV
computed for. Mirrors vLLM's own split (a `SchedulerInterface` ABC in
vllm/v1/core/sched/interface.py, concretely implemented by `Scheduler` in
vllm/v1/core/sched/scheduler.py, delegating block accounting to a
KVCacheManager/BlockPool) at a scope that fits this repo: one GPU, no
speculative decoding, no multimodal encoder cache, no distributed KV
connector, no priority scheduling -- see block_manager.py's module
docstring and engine/README.md's "Non-goals" for the full list of what's
cut and why. Prefix caching *is* implemented, opt-in via
CacheConfig.enable_prefix_caching (see block_manager.py's module docstring
and `_schedule_waiting` below). Chunked prefill *is* implemented (see
"Chunked prefill" below and engine/README.md) -- SchedulerConfig's own
max_num_batched_tokens doubles as the chunk size, no separate knob.

Simplification vLLM itself doesn't make, flagged here rather than left
implicit: this Scheduler is synchronous. schedule() both decides the batch
*and* immediately advances Request.num_computed_tokens / BlockManager's
block tables for it, as if the model runner is guaranteed to honor exactly
that decision this step. Real vLLM splits this into SchedulerOutput (what to
run) and a separate update_from_output(model_runner_output) call after the
forward pass actually completes -- what lets it pipeline scheduling for step
N+1 while step N's forward pass is still running on the GPU (async
scheduling). Wiring a real model runner (this repo's kernels don't yet
support the Nq=1-against-paged-Nkv decode shape that would call into -- see
README.md's "What's not wired up") is the natural point to split those
apart; premature here.

Chunked prefill: `_schedule_waiting` and `_schedule_running` both schedule
`min(request.get_num_new_tokens(), token_budget)` tokens for a request with
uncomputed prompt tokens, not `get_num_new_tokens()` outright -- a prompt
longer than what's left of a step's budget gets however much fits *this*
step, and picks up the rest on a later call once it re-enters
`_schedule_running` still mid-prefill (`request.is_prefill()` stays `True`
until every prompt token is computed). Block allocation is unaffected: a
request's full block table -- sized off `request.get_len()`, prompt+output
-- is still reserved by `BlockManager.allocate` in one shot at first
admission (see `_schedule_waiting`), same as before this feature existed;
with prefix caching on, some of that table's *entries* may be reused
physical blocks rather than freshly popped ones, but the table is still
allocated whole, upfront. Chunking only throttles *compute* (how many
tokens one step's forward pass covers), not *memory* -- so a request that
can't get all its blocks reserved up front still doesn't get admitted at
all, chunked or not (see `test_fifo_head_of_line_blocking`).

Prefix caching plugs into this same admission path with no separate
scheduling logic of its own: `_schedule_waiting` looks up a match before
allocating, seeds `num_computed_tokens` from it, and the chunked-prefill
math above (`get_num_new_tokens()`, already chunk-aware) naturally
schedules only the unmatched remainder -- a full cache hit just looks like
a request that arrives with most of its prefill already done. See
block_manager.py's module docstring for where the RadixTrie itself lives
and model/llm_engine.py's step() for where matched blocks get registered
back into it once real compute has happened.
"""
from collections import deque
from dataclasses import dataclass, field

from engine.block_manager import BlockManager
from engine.config import CacheConfig, SchedulerConfig
from engine.request import Request, RequestStatus


@dataclass
class ScheduledRequest:
    """One request's slice of a SchedulerOutput: which request, and how many
    of its tokens get KV computed this step. num_scheduled_tokens > 1
    usually means this is a prefill-shaped step (request.is_prefill() was
    True going in) -- either a fresh admission or a chunked-prefill
    continuation (see this module's docstring); == 1 usually means a
    steady-state decode step. "Usually", not "always", now that chunked
    prefill exists: a prompt admitted right at the tail end of a step's
    token budget can get a 1-token first chunk and still be mid-prefill --
    request.is_prefill() (checked before schedule() ran) is the precise
    signal a model-runner needs to pick which of this repo's attention-
    kernel call shapes applies (see engine/README.md), not
    num_scheduled_tokens alone.
    """
    request: Request
    num_scheduled_tokens: int


@dataclass
class SchedulerOutput:
    """What one schedule() call decided.

    scheduled_new: requests admitted from `waiting` this step -- their first
    (or, after a preemption, first-since-being-requeued) scheduled step. Not
    necessarily a full-sequence prefill in one shot any more: chunked
    prefill (see this module's docstring) may only cover part of the
    prompt if the step's token budget is tight, leaving the request
    `is_prefill() == True` and due to continue via scheduled_running on a
    later step.
    scheduled_running: requests that were already RUNNING and got scheduled
    again this step -- steady-state decode (num_scheduled_tokens == 1) most
    of the time, or a chunked-prefill continuation (num_scheduled_tokens
    > 1, request.is_prefill() still True) for a request whose prompt didn't
    fully fit an earlier step's budget.
    preempted: requests bumped from `running` back to `waiting` this step to
    free blocks for someone else. The caller doesn't need to *do* anything
    about these beyond not expecting output from them this step -- Scheduler
    has already updated their state and freed their blocks.
    """
    scheduled_new: list = field(default_factory=list)
    scheduled_running: list = field(default_factory=list)
    preempted: list = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (self.scheduled_new or self.scheduled_running)

    @property
    def total_num_scheduled_tokens(self) -> int:
        return sum(sr.num_scheduled_tokens for sr in self.scheduled_new + self.scheduled_running)


class Scheduler:
    """See engine/README.md's "Request lifecycle" and "Scheduler" sections
    for the full picture this class implements. `waiting` is FIFO (arrival
    order, preempted requests requeued at the front -- see _preempt);
    priority scheduling (vLLM supports a `priority` policy alongside FIFO)
    is a documented non-goal here.
    """

    def __init__(self, cache_config: CacheConfig, scheduler_config: SchedulerConfig):
        self.cache_config = cache_config
        self.scheduler_config = scheduler_config
        self.block_manager = BlockManager(
            block_size=cache_config.block_size,
            num_gpu_blocks=cache_config.num_gpu_blocks,
            watermark_blocks=cache_config.watermark_blocks,
            enable_prefix_caching=cache_config.enable_prefix_caching,
        )
        self.waiting: deque = deque()
        self.running: list = []
        self.requests: dict = {}

    def add_request(self, request: Request) -> None:
        if request.request_id in self.requests:
            raise ValueError(f"duplicate request_id {request.request_id!r}")
        request.status = RequestStatus.WAITING
        self.requests[request.request_id] = request
        self.waiting.append(request)

    def get_num_unfinished_requests(self) -> int:
        return len(self.waiting) + len(self.running)

    def has_unfinished_requests(self) -> bool:
        return self.get_num_unfinished_requests() > 0

    # -- The main decision ---------------------------------------------------

    def schedule(self) -> SchedulerOutput:
        output = SchedulerOutput()
        token_budget = self.scheduler_config.max_num_batched_tokens

        token_budget = self._schedule_running(output, token_budget)
        # Only admit new requests if nothing was preempted this step -- same
        # rule vLLM's Scheduler follows: a step that just freed blocks under
        # memory pressure shouldn't immediately hand them to a brand-new
        # request and risk preempting it right back next step (thrashing).
        if not output.preempted:
            self._schedule_waiting(output, token_budget)
        return output

    def _schedule_running(self, output: SchedulerOutput, token_budget: int) -> int:
        # `pending`: running requests not yet decided this step, in
        # priority order (arrival order -- self.running's existing order).
        # Preemption victims are popped from pending's *tail* (lowest
        # priority among those not yet processed this step), never from
        # requests already scheduled earlier in this same loop -- so a
        # scheduling decision, once made, is never undone later in the same
        # step.
        pending = deque(self.running)
        scheduled = []
        while pending:
            request = pending.popleft()
            num_new = request.get_num_new_tokens()
            if num_new <= 0:
                # Nothing new to compute -- shouldn't normally happen for a
                # RUNNING request, but harmless if it does. Keep it running,
                # skip it this step.
                scheduled.append(request)
                continue
            if token_budget <= 0:
                # Genuinely nothing left to give it this step -- leave it
                # RUNNING, retry next schedule() call. Steady-state decode's
                # num_new is always 1, so this only bites a request still
                # mid-prefill (chunked or not) re-entering this path.
                scheduled.append(request)
                continue
            # Chunked prefill: take whatever's left of this step's budget,
            # not necessarily all of num_new -- see this module's docstring.
            # For steady-state decode num_new == 1 == chunk always.
            chunk = min(num_new, token_budget)
            while not self.block_manager.can_append_slot(request):
                if pending:
                    self._preempt(pending.pop(), output)
                else:
                    self._preempt(request, output)
                    break
            else:
                self.block_manager.append_slot(request)
                # += chunk, not = request.get_len(): during a chunked-prefill
                # continuation get_len() is the *whole* prompt length (no
                # output tokens yet), not this step's partial progress.
                request.num_computed_tokens += chunk
                output.scheduled_running.append(ScheduledRequest(request, chunk))
                token_budget -= chunk
                scheduled.append(request)
        self.running = scheduled
        return token_budget

    def _schedule_waiting(self, output: SchedulerOutput, token_budget: int) -> int:
        still_waiting: deque = deque()
        while self.waiting:
            request = self.waiting.popleft()
            # Side-effect-free peek (see BlockManager.match_prefix's
            # docstring) -- safe to compute even if this request ends up not
            # admitted this step (the `break` below), since nothing is
            # committed (ref counts bumped, stats recorded) until allocate().
            match = self.block_manager.match_prefix(request.prompt_token_ids)
            fits_seq_budget = len(self.running) < self.scheduler_config.max_num_seqs
            if (
                not fits_seq_budget
                or token_budget <= 0
                or not self.block_manager.can_allocate(request, match)
            ):
                still_waiting.append(request)
                # FIFO: if the head of the line can't be admitted at all --
                # no seq slot, no budget left, or (unaffected by chunking,
                # see this module's docstring) not enough blocks for the
                # whole prompt -- nothing behind it should jump ahead of it
                # either. Stop scanning rather than admitting a smaller
                # request out of order. A request that merely doesn't fit
                # the *remaining* budget in full isn't blocked any more --
                # see the chunked-prefill admission below.
                break
            self.block_manager.allocate(request, match)
            request.status = RequestStatus.RUNNING
            # Prefix caching: a match means some of this prompt's KV is
            # already computed -- seed num_computed_tokens with it *before*
            # computing num_new below, so the chunked-prefill math already
            # in get_num_new_tokens() naturally schedules only the unmatched
            # remainder. Must happen before num_new is read, not after --
            # reading it first (this design's pre-prefix-caching order)
            # would schedule the matched tokens for compute all over again.
            if match is not None:
                request.num_computed_tokens = match.num_matched_tokens
            num_new = request.get_num_new_tokens()
            # Chunked prefill: admit with whatever's left of this step's
            # budget, not necessarily the whole prompt -- blocks are still
            # reserved for the full prompt above, so a later step's
            # _schedule_running just continues where this leaves off (see
            # this module's docstring).
            chunk = min(num_new, token_budget)
            request.num_computed_tokens += chunk
            output.scheduled_new.append(ScheduledRequest(request, chunk))
            token_budget -= chunk
            self.running.append(request)
        still_waiting.extend(self.waiting)  # whatever wasn't popped before the break
        self.waiting = still_waiting
        return token_budget

    def _preempt(self, request: Request, output: SchedulerOutput) -> None:
        """Recompute-based preemption (see block_manager.py's module
        docstring on why not swap-to-CPU): free every physical block this
        request owns and reset num_computed_tokens to 0, so its next
        admission redoes the whole sequence-so-far as a fresh prefill rather
        than trying to resume mid-sequence against a KV cache that's gone.
        Requeued at the *front* of `waiting`, not the back, so it's first in
        line once space frees up -- ahead of requests that arrived later but
        were never running.

        No special prefix-cache handling needed here: block_manager.free
        already releases any cache ref this request holds (see its
        docstring), and any prefill progress this request made before this
        step was already registered into the RadixTrie at the end of the
        LLMEngine.step() that produced it (see model/llm_engine.py) -- there
        is never a window, given this Scheduler's synchronous one-step-at-a-
        time cadence, where genuinely-computed-but-unregistered progress
        exists at the moment a preemption decision runs. Revisit this if
        scheduling ever becomes async (see this module's docstring).
        """
        self.block_manager.free(request)
        request.num_computed_tokens = 0
        request.status = RequestStatus.PREEMPTED
        self.waiting.appendleft(request)
        output.preempted.append(request)

    # -- Teardown --------------------------------------------------------------

    def free_finished_requests(self) -> list:
        """Call once per engine step, after processing model output and
        marking finished requests' status via Request.maybe_finish() (both
        out of scope here -- see this module's docstring). Returns the freed
        requests so the caller can do whatever "return this response to the
        client" step needs.
        """
        finished = [r for r in self.running if r.is_finished()]
        if finished:
            self.running = [r for r in self.running if not r.is_finished()]
            for r in finished:
                self.block_manager.free(r)
                del self.requests[r.request_id]
        return finished

    def abort_requests(self, request_ids) -> list:
        """Cancel requests regardless of lifecycle stage: still WAITING
        (never allocated -- block_manager.free is a no-op for these, see its
        docstring), RUNNING (blocks freed here), or already finished
        (ignored). Returns the newly-aborted requests.
        """
        ids = set(request_ids)
        aborted = []

        still_waiting: deque = deque()
        for request in self.waiting:
            if request.request_id in ids:
                request.status = RequestStatus.FINISHED_ABORTED
                aborted.append(request)
                del self.requests[request.request_id]
            else:
                still_waiting.append(request)
        self.waiting = still_waiting

        for request in self.running:
            if request.request_id in ids:
                request.status = RequestStatus.FINISHED_ABORTED
        aborted.extend(self.free_finished_requests())
        return aborted