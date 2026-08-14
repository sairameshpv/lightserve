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