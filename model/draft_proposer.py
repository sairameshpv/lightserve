"""Speculative decoding's draft side: DraftProposer.propose(request) runs
K sequential single-token decode steps on a cheap draft ModelRunner,
seeded from `request`'s currently-committed tokens, and returns the K
proposed token ids. The expensive target model verifies all K in one pass
(Stage D, not built yet) and accepts the longest greedy-matching prefix.

Why a *shadow* Request, not `request` itself: `request.block_table` is
already owned by the target's own PagedKVCache/BlockManager (see
model/kv_cache.py, engine/block_manager.py). The draft needs an
independent block table into the *draft's own* PagedKVCache (already
owned by the `draft_model_runner` passed in here) for the same logical
sequence. PagedKVCache.write/read only ever touch `request.block_table`
(never anything else on the Request), so a second, private `Request`
instance -- reusing the dataclass as-is, just a stand-in that this file
never lets the engine/scheduler see -- is a complete, correct handle for
driving a second KV cache. This lets DraftProposer reuse BlockManager.
allocate/append_slot, Request.get_num_new_tokens, and ModelRunner.
execute_model entirely unmodified.

Draft never sees rejected tokens: each round, the caller (Stage E) is
responsible for keeping `request.output_token_ids` -- what propose()
resyncs the shadow to on every call -- reflecting only what the target
actually accepted, and clearing `request.draft_token_ids` (engine/
request.py) regardless of how many of the last round's proposals were
accepted. This file only ever reads `request`; it never mutates it.
"""
from engine.block_manager import BlockManager
from engine.request import Request, RequestStatus
from engine.scheduler import ScheduledRequest, SchedulerOutput
from model.model_runner import ModelRunner


class DraftProposer:
    def __init__(self, draft_model_runner: ModelRunner, num_speculative_tokens: int):
        self.model_runner = draft_model_runner
        self.num_speculative_tokens = num_speculative_tokens
        # Sized off the draft's own PagedKVCache, not a separate
        # CacheConfig -- keeps this constructor to exactly the two things
        # the caller already has to build anyway.
        kv_cache = draft_model_runner.kv_cache
        self.block_manager = BlockManager(block_size=kv_cache.block_size, num_gpu_blocks=kv_cache.num_gpu_blocks)
        self._shadow_requests: dict = {}  # real request_id -> this proposer's own private Request

    def _ensure_capacity(self, shadow: Request) -> None:
        """Grows `shadow`'s block table (via the ordinary public
        BlockManager.append_slot, one block at a time) until it covers
        shadow.get_len() -- append_slot alone only ever adds a single
        block per call (sized for steady-state decode's one-token-at-a-
        time growth, see its docstring), which isn't enough right after a
        multi-token resync in propose() below.
        """
        target_blocks = shadow.num_blocks_needed(self.block_manager.block_size)
        while len(self.block_manager.get_block_table(shadow)) < target_blocks:
            self.block_manager.append_slot(shadow)

    def _get_or_create_shadow(self, request: Request) -> Request:
        shadow = self._shadow_requests.get(request.request_id)
        if shadow is not None:
            # Resync to whatever's newly committed since the last call --
            # see module docstring on why this file assumes that only ever
            # grows (Stage E's job to guarantee).
            shadow.output_token_ids = list(request.output_token_ids)
            return shadow

        shadow = Request(
            request_id=request.request_id,
            prompt_token_ids=list(request.prompt_token_ids),
            # Seeded with whatever's *already* committed, not just the
            # prompt -- a request can have real output tokens before
            # speculative decoding starts proposing for it at all.
            output_token_ids=list(request.output_token_ids),
            sampling_params=request.sampling_params,  # no point drafting past what the real generation could ever use
            status=RequestStatus.RUNNING,
        )
        self.block_manager.allocate(shadow)  # sized off get_len(), which already includes output_token_ids above
        self._shadow_requests[request.request_id] = shadow
        return shadow

    def propose(self, request: Request) -> list:
        """Runs num_speculative_tokens sequential decode steps on the
        draft model, seeded from `request`'s currently-committed tokens
        (prompt + output_token_ids), and returns that many proposed token
        ids. See module docstring for the shadow-Request design and its
        assumptions.
        """
        shadow = self._get_or_create_shadow(request)

        proposed = []
        for _ in range(self.num_speculative_tokens):
            self._ensure_capacity(shadow)
            # The first iteration's num_new is the whole resync delta (a
            # prefill-shaped call covering it in one shot, not chunked --
            # see module docstring on why execute_model still records its
            # sample); every iteration after is a normal 1-new-token
            # decode step, identical in shape to a real engine step.
            num_new = shadow.get_num_new_tokens()
            shadow.num_computed_tokens += num_new  # advance-before-calling, same convention Scheduler itself uses
            self.model_runner.execute_model(
                SchedulerOutput(scheduled_running=[ScheduledRequest(shadow, num_new)])
            )
            proposed.append(shadow.output_token_ids[-1])
        return proposed