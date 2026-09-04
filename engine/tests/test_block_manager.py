"""Correctness tests for engine/block_manager.py. Pure Python, no torch/CUDA
-- unlike kernels/tests and model/tests, these run for real (not just
import-checked) on GitHub-hosted CI runners, since block accounting has no
GPU dependency at all.
"""
import pytest

from engine.block_manager import BlockManager, OutOfMemoryError
from engine.request import Request


def make_request(request_id="r0", prompt_len=10, num_output=0):
    req = Request(request_id=request_id, prompt_token_ids=list(range(prompt_len)))
    req.output_token_ids = list(range(num_output))
    return req


class TestAllocate:
    def test_allocates_ceil_div_blocks(self):
        bm = BlockManager(block_size=4, num_gpu_blocks=10)
        req = make_request(prompt_len=9)  # ceil(9/4) = 3
        assert bm.can_allocate(req)
        bm.allocate(req)
        assert len(bm.get_block_table(req)) == 3
        assert bm.num_free_blocks == 7

    def test_exact_multiple_of_block_size(self):
        bm = BlockManager(block_size=4, num_gpu_blocks=10)
        req = make_request(prompt_len=8)  # exactly 2 blocks, no remainder
        bm.allocate(req)
        assert len(bm.get_block_table(req)) == 2

    def test_block_table_written_back_onto_request(self):
        bm = BlockManager(block_size=4, num_gpu_blocks=10)
        req = make_request(prompt_len=5)
        bm.allocate(req)
        assert req.block_table == bm.get_block_table(req)
        assert len(req.block_table) == 2

    def test_can_allocate_respects_watermark(self):
        bm = BlockManager(block_size=4, num_gpu_blocks=10, watermark_blocks=2)
        # 8 blocks usable before the watermark; this needs exactly 8.
        req = make_request(prompt_len=32)
        assert bm.can_allocate(req)
        # One more token pushes it to 9 needed blocks -- watermark blocks it.
        req2 = make_request(request_id="r1", prompt_len=33)
        assert not bm.can_allocate(req2)

    def test_allocate_raises_when_insufficient_blocks(self):
        bm = BlockManager(block_size=4, num_gpu_blocks=2)
        req = make_request(prompt_len=9)  # needs 3, only 2 exist
        assert not bm.can_allocate(req)
        with pytest.raises(OutOfMemoryError):
            bm.allocate(req)

    def test_double_allocate_raises(self):
        bm = BlockManager(block_size=4, num_gpu_blocks=10)
        req = make_request(prompt_len=4)
        bm.allocate(req)
        with pytest.raises(ValueError):
            bm.allocate(req)


class TestAppendSlot:
    def test_no_new_block_within_current_capacity(self):
        bm = BlockManager(block_size=4, num_gpu_blocks=10)
        req = make_request(prompt_len=4)  # 1 block, capacity 4, exactly full
        bm.allocate(req)
        free_before = bm.num_free_blocks
        # Next token (5th) exceeds capacity 4 -> needs a new block.
        req.output_token_ids = [99]
        assert bm.can_append_slot(req)
        bm.append_slot(req)
        assert len(bm.get_block_table(req)) == 2
        assert bm.num_free_blocks == free_before - 1

    def test_grows_only_at_block_boundary(self):
        bm = BlockManager(block_size=4, num_gpu_blocks=10)
        req = make_request(prompt_len=4)
        bm.allocate(req)  # 1 block (capacity 4)

        # Token 5 (get_len()=5 > capacity 4) needs a new block.
        req.output_token_ids = [1]
        bm.append_slot(req)
        assert len(bm.get_block_table(req)) == 2  # capacity now 8

        # Tokens 6, 7, 8 (get_len() 6,7,8, all <= capacity 8) need none.
        for n in (2, 3, 4):
            req.output_token_ids = list(range(n))
            bm.append_slot(req)
            assert len(bm.get_block_table(req)) == 2

        # Token 9 (get_len()=9 > capacity 8) needs a 3rd block.
        req.output_token_ids = list(range(5))
        bm.append_slot(req)
        assert len(bm.get_block_table(req)) == 3

    def test_append_slot_without_allocate_raises(self):
        bm = BlockManager(block_size=4, num_gpu_blocks=10)
        req = make_request(prompt_len=4)
        with pytest.raises(ValueError):
            bm.append_slot(req)

    def test_can_append_slot_false_when_pool_exhausted(self):
        bm = BlockManager(block_size=4, num_gpu_blocks=1)
        req = make_request(prompt_len=4)  # takes the only block
        bm.allocate(req)
        req.output_token_ids = [1]  # needs a 2nd block, none free
        assert not bm.can_append_slot(req)
        with pytest.raises(OutOfMemoryError):
            bm.append_slot(req)


class TestFree:
    def test_free_returns_blocks_to_pool(self):
        bm = BlockManager(block_size=4, num_gpu_blocks=10)
        req = make_request(prompt_len=9)  # 3 blocks
        bm.allocate(req)
        assert bm.num_free_blocks == 7
        bm.free(req)
        assert bm.num_free_blocks == 10
        assert bm.get_block_table(req) == []
        assert req.block_table == []

    def test_freed_blocks_are_reusable(self):
        bm = BlockManager(block_size=4, num_gpu_blocks=4)
        req_a = make_request("a", prompt_len=16)  # all 4 blocks
        bm.allocate(req_a)
        assert bm.num_free_blocks == 0
        bm.free(req_a)

        req_b = make_request("b", prompt_len=16)
        assert bm.can_allocate(req_b)
        bm.allocate(req_b)
        assert bm.num_free_blocks == 0

    def test_free_never_allocated_request_is_a_noop(self):
        bm = BlockManager(block_size=4, num_gpu_blocks=4)
        req = make_request(prompt_len=4)
        bm.free(req)  # must not raise
        assert bm.num_free_blocks == 4

    def test_free_twice_is_a_noop(self):
        bm = BlockManager(block_size=4, num_gpu_blocks=4)
        req = make_request(prompt_len=4)
        bm.allocate(req)
        bm.free(req)
        bm.free(req)  # must not raise, must not double-credit the pool
        assert bm.num_free_blocks == 4


class TestPrefixCaching:
    """enable_prefix_caching=True wiring: match_prefix/allocate/free/eviction.
    make_request's prompt_token_ids is always list(range(prompt_len)), so
    two requests with the same prompt_len share identical content and will
    match; a distinct range (e.g. range(100, ...)) is used wherever a test
    needs content that deliberately does *not* match.
    """

    def test_disabled_by_default(self):
        bm = BlockManager(block_size=4, num_gpu_blocks=10)
        assert bm.prefix_cache is None
        assert bm.match_prefix(list(range(8))) is None

    def test_match_finds_a_donors_inserted_blocks(self):
        bm = BlockManager(block_size=4, num_gpu_blocks=10, enable_prefix_caching=True)
        donor = make_request("donor", prompt_len=8)  # 2 whole blocks
        bm.allocate(donor)
        donor.num_computed_tokens = 8
        bm.insert_computed_prefix(donor)

        follower = make_request("follower", prompt_len=8)  # identical content
        match = bm.match_prefix(follower.prompt_token_ids)
        assert match.num_matched_tokens == 8
        assert match.physical_block_ids == bm.get_block_table(donor)

    def test_allocate_with_match_only_pops_fresh_blocks_for_the_remainder(self):
        bm = BlockManager(block_size=4, num_gpu_blocks=10, enable_prefix_caching=True)
        donor = make_request("donor", prompt_len=8)
        bm.allocate(donor)
        donor.num_computed_tokens = 8
        bm.insert_computed_prefix(donor)
        free_before = bm.num_free_blocks

        # Shares donor's 8-token prefix, plus 4 more unique tokens (1 more block).
        follower = Request(request_id="follower", prompt_token_ids=list(range(8)) + [999] * 4)
        match = bm.match_prefix(follower.prompt_token_ids)
        bm.allocate(follower, match)

        table = bm.get_block_table(follower)
        assert len(table) == 3
        assert table[:2] == bm.get_block_table(donor)  # matched slots reused, not fresh
        assert bm.num_free_blocks == free_before - 1  # only the unmatched block popped fresh

    def test_free_does_not_return_still_shared_blocks_to_the_pool(self):
        bm = BlockManager(block_size=4, num_gpu_blocks=10, enable_prefix_caching=True)
        donor = make_request("donor", prompt_len=8)
        bm.allocate(donor)
        donor.num_computed_tokens = 8
        bm.insert_computed_prefix(donor)

        follower = make_request("follower", prompt_len=8)
        match = bm.match_prefix(follower.prompt_token_ids)
        bm.allocate(follower, match)

        free_before = bm.num_free_blocks
        bm.free(donor)
        assert bm.num_free_blocks == free_before  # still referenced by follower -- not returned
        assert bm.get_block_table(donor) == []  # table entry itself is gone though
        assert donor.cached_prefix_nodes == []

    def test_eviction_reclaims_a_block_only_once_all_holders_free_it(self):
        bm = BlockManager(block_size=4, num_gpu_blocks=10, enable_prefix_caching=True)
        donor = make_request("donor", prompt_len=8)
        bm.allocate(donor)
        donor.num_computed_tokens = 8
        bm.insert_computed_prefix(donor)

        follower = make_request("follower", prompt_len=8)
        match = bm.match_prefix(follower.prompt_token_ids)
        bm.allocate(follower, match)

        bm.free(donor)
        assert bm.prefix_cache.num_evictable_blocks == 0  # follower still holds a ref
        bm.free(follower)
        # Both blocks are now unreferenced, but they're chained (block 0 is
        # block 1's parent) -- only the tail leaf counts as evictable until
        # it's actually evicted, per RadixTrie.num_evictable_blocks's own
        # documented undercount (a parent with children isn't a leaf yet).
        assert bm.prefix_cache.num_evictable_blocks == 1
        assert bm.prefix_cache.evict_one_lru() is not None  # the tail block
        assert bm.prefix_cache.evict_one_lru() is not None  # now-leaf parent block
        assert bm.prefix_cache.evict_one_lru() is None

    def test_allocate_drains_lru_evictions_when_free_stack_is_empty(self):
        # A single block (prompt_len == block_size) avoids RadixTrie's
        # parent/child leaf-counting subtlety entirely -- see the
        # "chained blocks" note in test_eviction_reclaims_a_block_only_once_
        # all_holders_free_it for why a multi-block chain's
        # num_evictable_blocks would undercount here instead.
        bm = BlockManager(block_size=4, num_gpu_blocks=1, enable_prefix_caching=True)
        donor = make_request("donor", prompt_len=4)  # the pool's only block
        bm.allocate(donor)
        donor.num_computed_tokens = 4
        bm.insert_computed_prefix(donor)
        bm.free(donor)  # the block is now cache-owned, evictable, and NOT in the free stack
        assert bm.num_free_blocks == 0
        assert bm.prefix_cache.num_evictable_blocks == 1

        other = Request(request_id="other", prompt_token_ids=list(range(100, 104)))  # no shared prefix
        match = bm.match_prefix(other.prompt_token_ids)
        assert match.num_matched_tokens == 0
        assert bm.can_allocate(other, match)  # 0 free + 1 evictable - 1 needed >= 0
        bm.allocate(other, match)
        assert len(bm.get_block_table(other)) == 1
        assert bm.num_free_blocks == 0
        assert bm.prefix_cache.num_evictable_blocks == 0

    def test_allocate_still_raises_oom_when_eviction_cannot_cover_the_need(self):
        bm = BlockManager(block_size=4, num_gpu_blocks=2, enable_prefix_caching=True)
        donor = make_request("donor", prompt_len=8)
        bm.allocate(donor)
        donor.num_computed_tokens = 8
        bm.insert_computed_prefix(donor)
        bm.free(donor)  # 2 evictable, 0 free, pool has only 2 blocks total

        # Needs 3 blocks -- more than exist in the whole pool, even fully evicted.
        big = Request(request_id="big", prompt_token_ids=list(range(100, 112)))
        match = bm.match_prefix(big.prompt_token_ids)
        assert not bm.can_allocate(big, match)
        with pytest.raises(OutOfMemoryError):
            bm.allocate(big, match)

    def test_acquire_happens_before_eviction_protects_the_match_from_being_evicted(self):
        """Regression test for a specific ordering bug: if allocate() drained
        evictions *before* acquiring the match, evict_one_lru()'s LRU policy
        would pick the donor's blocks (inserted earlier, so a smaller
        last_used) over the throwaway request's blocks (inserted later),
        evicting exactly the blocks this allocation is about to reuse.
        Acquiring first protects them (bumps ref_count off 0) before any
        eviction runs.
        """
        bm = BlockManager(block_size=4, num_gpu_blocks=3, enable_prefix_caching=True)

        donor = make_request("donor", prompt_len=8)  # 2 blocks, to be matched later
        bm.allocate(donor)
        donor.num_computed_tokens = 8
        bm.insert_computed_prefix(donor)
        donor_blocks = bm.get_block_table(donor)

        # Uses the pool's 3rd (last free) block; freed after, becoming the
        # only legitimately-evictable block once donor's match is protected.
        throwaway = Request(request_id="throwaway", prompt_token_ids=list(range(200, 204)))
        bm.allocate(throwaway)
        throwaway.num_computed_tokens = 4
        bm.insert_computed_prefix(throwaway)
        bm.free(throwaway)

        bm.free(donor)  # donor's 2 blocks become evictable too -- pool is now fully saturated
        assert bm.num_free_blocks == 0
        # donor's 2 blocks are chained (a parent-child pair), so only the
        # tail leaf counts as evictable right now -- see
        # RadixTrie.num_evictable_blocks's documented undercount. Plus
        # throwaway's single standalone block: 2 evictable total.
        assert bm.prefix_cache.num_evictable_blocks == 2

        # Shares donor's 8-token prefix + 4 unique tokens -> needs exactly 1 fresh block.
        follower = Request(request_id="follower", prompt_token_ids=list(range(8)) + [999] * 4)
        match = bm.match_prefix(follower.prompt_token_ids)
        assert match.num_matched_tokens == 8
        bm.allocate(follower, match)

        table = bm.get_block_table(follower)
        assert table[:2] == donor_blocks  # donor's matched blocks survived, weren't evicted
        assert bm.prefix_cache.num_evictable_blocks == 0  # throwaway's block was the one reclaimed