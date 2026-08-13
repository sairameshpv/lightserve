"""A request's lifecycle state machine and the data it carries end to end --
independent of any GPU/kernel concern. No torch import here on purpose: this
whole module is CPU-side bookkeeping, so it's fully unit-testable without
CUDA (see engine/tests/test_request.py), unlike everything under kernels/
and model/.
"""
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class RequestStatus(Enum):
    """See engine/README.md's "Request lifecycle" section for the full state
    diagram. WAITING -> RUNNING is admission (Scheduler._schedule_waiting);
    RUNNING -> PREEMPTED -> WAITING is the recompute-based preemption path
    (Scheduler._preempt); RUNNING -> one of the FINISHED_* terminal states
    happens once Request.maybe_finish() sees a stop condition, swept out of
    `running` by Scheduler.free_finished_requests.
    """
    WAITING = auto()
    RUNNING = auto()
    PREEMPTED = auto()
    FINISHED_STOPPED = auto()        # hit eos_token_id
    FINISHED_LENGTH_CAPPED = auto()  # hit sampling_params.max_tokens
    FINISHED_ABORTED = auto()        # caller cancelled it (Scheduler.abort_requests)

    @property
    def finished(self) -> bool:
        return self in (
            RequestStatus.FINISHED_STOPPED,
            RequestStatus.FINISHED_LENGTH_CAPPED,
            RequestStatus.FINISHED_ABORTED,
        )


@dataclass
class SamplingParams:
    """Deliberately tiny. Temperature/top-p/top-k sampling itself is a
    model-runner concern (whatever calls kernels/tiled_matmul.py's lm_head
    matmul and turns logits into a token id), not the scheduler's -- the
    scheduler only needs to know when a request is *done*, hence max_tokens
    and eos_token_id here and nothing about sampling strategy.
    """
    max_tokens: int = 16
    eos_token_id: Optional[int] = None


@dataclass
class Request:
    """One in-flight generation request.

    `block_table` is populated by BlockManager.allocate/append_slot, not set
    directly by this class -- request.py doesn't import block_manager.py (no
    circular dependency, and it keeps "which physical block ids" out of the
    object a request-lifecycle unit test shouldn't need to know about).
    """
    request_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    arrival_time: float = 0.0

    output_token_ids: list[int] = field(default_factory=list)
    status: RequestStatus = RequestStatus.WAITING
    block_table: list[int] = field(default_factory=list)

    # How many of this request's tokens (prompt + output, in that order)
    # already have a KV-cache entry computed. Scheduler.schedule() advances
    # this by num_scheduled_tokens each time the request is scheduled for a
    # prefill/decode step; Scheduler._preempt resets it to 0 (recompute
    # strategy -- see block_manager.py's module docstring on why not swap).
    num_computed_tokens: int = 0

    def all_token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.output_token_ids

    def get_len(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    def get_num_new_tokens(self) -> int:
        """Tokens that still need a KV-cache entry computed: the whole
        prompt (minus whatever a prior partial admission already covered) on
        a first/resumed admission, exactly 1 once steady-state decoding.
        """
        return self.get_len() - self.num_computed_tokens

    def is_prefill(self) -> bool:
        """True while this request still has uncomputed prompt tokens --
        for a future model-runner to decide whether a scheduled step needs a
        prefill-shaped (Nq == Nkv, this repo's flash_attention_forward as-is)
        or an incremental decode-shaped (Nq=1 against a paged Nkv, not yet
        supported by any kernel in this repo -- see engine/README.md) call.
        """
        return self.num_computed_tokens < len(self.prompt_token_ids)

    def num_blocks_needed(self, block_size: int) -> int:
        length = self.get_len()
        if length == 0:
            return 0
        return (length + block_size - 1) // block_size

    def is_finished(self) -> bool:
        return self.status.finished

    def maybe_finish(self) -> None:
        """Checks this design's two stop conditions and updates `status` if
        either fires. Called by whatever appends a newly-sampled token to
        `output_token_ids` (the model-runner-facing side of the engine loop,
        not written yet -- see engine/README.md's scope note) right after
        appending it.
        """
        if self.status.finished:
            return
        eos = self.sampling_params.eos_token_id
        if eos is not None and self.output_token_ids and self.output_token_ids[-1] == eos:
            self.status = RequestStatus.FINISHED_STOPPED
        elif len(self.output_token_ids) >= self.sampling_params.max_tokens:
            self.status = RequestStatus.FINISHED_LENGTH_CAPPED
