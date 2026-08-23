"""Correctness tests for model/kv_cache.py's PagedKVCache: the physical
per-layer K/V tensors the block ids engine/block_manager.py hands out
actually point into. Requires a real CUDA GPU (this module allocates real
device tensors at construction time) -- see model/tests/test_minimal_llama.py's
module docstring for why these are skipped, not run, on a machine without
one.
"""
import pytest
import torch

from engine.config import CacheConfig
from engine.request import Request
from model.kv_cache import PagedKVCache
from model.minimal_llama import TOY_CONFIG

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="PagedKVCache allocates real CUDA tensors"
)


def make_request(request_id="r0", block_table=None):
    req = Request(request_id=request_id, prompt_token_ids=[])
    req.block_table = block_table or []
    return req


def _kv(n, fill=None, device="cuda"):
    shape = (n, TOY_CONFIG.n_heads, TOY_CONFIG.head_dim)
    if fill is None:
        return torch.randn(shape, device=device, dtype=TOY_CONFIG.dtype)
    return torch.full(shape, float(fill), device=device, dtype=TOY_CONFIG.dtype)


@requires_cuda
class TestWriteRead:
    def test_round_trip_within_one_block(self):
        cache = PagedKVCache(CacheConfig(block_size=4, num_gpu_blocks=10), TOY_CONFIG, device="cuda")
        req = make_request(block_table=[3])  # arbitrary physical block id
        k, v = _kv(4), _kv(4)

        cache.write(layer_idx=0, request=req, start=0, k=k, v=v)
        k_read, v_read = cache.read(layer_idx=0, request=req, seq_len=4)

        torch.testing.assert_close(k_read, k)
        torch.testing.assert_close(v_read, v)

    def test_write_spans_a_block_boundary(self):
        cache = PagedKVCache(CacheConfig(block_size=4, num_gpu_blocks=10), TOY_CONFIG, device="cuda")
        req = make_request(block_table=[1, 7])  # 2 blocks -> 8 logical positions
        k, v = _kv(6), _kv(6)  # positions 0..5 -- crosses from block 1 into block 7 at position 4

        cache.write(layer_idx=0, request=req, start=0, k=k, v=v)
        k_read, v_read = cache.read(layer_idx=0, request=req, seq_len=6)

        torch.testing.assert_close(k_read, k)
        torch.testing.assert_close(v_read, v)

    def test_incremental_decode_style_writes_accumulate(self):
        # Prefill writes positions [0, 4); 3 further decode-style
        # single-token writes at positions 4, 5, 6. read(seq_len=7) must see
        # everything written across all 4 calls, not just the last one --
        # this is exactly the write-then-read-every-step pattern
        # model_runner.py's _attention uses.
        cache = PagedKVCache(CacheConfig(block_size=4, num_gpu_blocks=10), TOY_CONFIG, device="cuda")
        req = make_request(block_table=[2, 5])
        full_k, full_v = _kv(7), _kv(7)

        cache.write(0, req, 0, full_k[0:4], full_v[0:4])
        for pos in range(4, 7):
            cache.write(0, req, pos, full_k[pos:pos + 1], full_v[pos:pos + 1])

        k_read, v_read = cache.read(0, req, seq_len=7)
        torch.testing.assert_close(k_read, full_k)
        torch.testing.assert_close(v_read, full_v)

    def test_different_requests_use_disjoint_physical_blocks(self):
        cache = PagedKVCache(CacheConfig(block_size=4, num_gpu_blocks=10), TOY_CONFIG, device="cuda")
        req_a = make_request("a", block_table=[0])
        req_b = make_request("b", block_table=[1])
        k_a, k_b = _kv(4, fill=1.0), _kv(4, fill=2.0)

        cache.write(0, req_a, 0, k_a, k_a)
        cache.write(0, req_b, 0, k_b, k_b)

        k_read_a, _ = cache.read(0, req_a, seq_len=4)
        k_read_b, _ = cache.read(0, req_b, seq_len=4)
        torch.testing.assert_close(k_read_a, k_a)
        torch.testing.assert_close(k_read_b, k_b)

    def test_layers_are_isolated(self):
        cache = PagedKVCache(CacheConfig(block_size=4, num_gpu_blocks=10), TOY_CONFIG, device="cuda")
        req = make_request(block_table=[0])
        k0, k1 = _kv(4, fill=1.0), _kv(4, fill=2.0)

        cache.write(0, req, 0, k0, k0)
        cache.write(1, req, 0, k1, k1)  # TOY_CONFIG.n_layers == 2, so layer 1 is valid

        k_read0, _ = cache.read(0, req, seq_len=4)
        k_read1, _ = cache.read(1, req, seq_len=4)
        torch.testing.assert_close(k_read0, k0)
        torch.testing.assert_close(k_read1, k1)
