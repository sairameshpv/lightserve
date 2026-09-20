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


# int8_kv is lossy by construction -- these mirror TestWriteRead's own
# cases but with a loosened tolerance, not exact equality. The tolerance
# is derived, not arbitrary: per-token scale is amax/127, so the worst-case
# per-element rounding error is scale/2. For torch.randn data over
# TOY_CONFIG.head_dim=64, amax is typically ~3-3.5, giving scale ~0.03 and
# a rounding-error bound ~0.015 -- atol/rtol=0.05 below has real margin
# above that, loose enough to never spuriously fail on quantization noise,
# tight enough to still catch a real bug (wrong shape, forgotten scale,
# scale/value mismatch) that would show up as a much larger error.
_INT8_ATOL, _INT8_RTOL = 0.05, 0.05


@requires_cuda
class TestInt8Kv:
    def test_round_trip_is_close_not_exact(self):
        cache = PagedKVCache(CacheConfig(block_size=4, num_gpu_blocks=10, int8_kv=True),
                              TOY_CONFIG, device="cuda")
        req = make_request(block_table=[3])
        k, v = _kv(4), _kv(4)

        cache.write(layer_idx=0, request=req, start=0, k=k, v=v)
        k_read, v_read = cache.read(layer_idx=0, request=req, seq_len=4)

        # Close, but this is the one place in this file where exact
        # equality would be the wrong assertion -- int8_kv is lossy by
        # design (see model/kv_cache.py's _quantize docstring).
        torch.testing.assert_close(k_read, k, atol=_INT8_ATOL, rtol=_INT8_RTOL)
        torch.testing.assert_close(v_read, v, atol=_INT8_ATOL, rtol=_INT8_RTOL)
        assert not torch.equal(k_read, k), (
            "got exact equality from a lossy path -- suspicious, check _quantize/_dequantize "
            "are actually being called"
        )

    def test_storage_dtype_is_int8_but_read_returns_compute_dtype(self):
        cache = PagedKVCache(CacheConfig(block_size=4, num_gpu_blocks=10, int8_kv=True),
                              TOY_CONFIG, device="cuda")
        req = make_request(block_table=[0])
        k, v = _kv(4), _kv(4)

        assert cache.k_cache.dtype == torch.int8
        assert cache.v_cache.dtype == torch.int8

        cache.write(0, req, 0, k, v)
        k_read, v_read = cache.read(0, req, seq_len=4)
        assert k_read.dtype == TOY_CONFIG.dtype
        assert v_read.dtype == TOY_CONFIG.dtype

    def test_incremental_decode_style_writes_still_accumulate(self):
        # Same shape as TestWriteRead's own version of this test -- each
        # write() call quantizes independently (no cross-call state), so
        # this also confirms per-token scales don't bleed across writes.
        cache = PagedKVCache(CacheConfig(block_size=4, num_gpu_blocks=10, int8_kv=True),
                              TOY_CONFIG, device="cuda")
        req = make_request(block_table=[2, 5])
        full_k, full_v = _kv(7), _kv(7)

        cache.write(0, req, 0, full_k[0:4], full_v[0:4])
        for pos in range(4, 7):
            cache.write(0, req, pos, full_k[pos:pos + 1], full_v[pos:pos + 1])

        k_read, v_read = cache.read(0, req, seq_len=7)
        torch.testing.assert_close(k_read, full_k, atol=_INT8_ATOL, rtol=_INT8_RTOL)
        torch.testing.assert_close(v_read, full_v, atol=_INT8_ATOL, rtol=_INT8_RTOL)

    def test_different_requests_dont_share_scale_storage(self):
        cache = PagedKVCache(CacheConfig(block_size=4, num_gpu_blocks=10, int8_kv=True),
                              TOY_CONFIG, device="cuda")
        req_a = make_request("a", block_table=[0])
        req_b = make_request("b", block_table=[1])
        # Deliberately very different magnitudes -- if scales were shared
        # or mixed up, one request's dequantized values would be wildly
        # (not just quantization-noise) off.
        k_a, k_b = _kv(4, fill=1.0), _kv(4, fill=50.0)

        cache.write(0, req_a, 0, k_a, k_a)
        cache.write(0, req_b, 0, k_b, k_b)

        k_read_a, _ = cache.read(0, req_a, seq_len=4)
        k_read_b, _ = cache.read(0, req_b, seq_len=4)
        torch.testing.assert_close(k_read_a, k_a, atol=_INT8_ATOL, rtol=_INT8_RTOL)
        torch.testing.assert_close(k_read_b, k_b, atol=_INT8_ATOL, rtol=_INT8_RTOL)

    def test_default_is_still_bf16_unaffected(self):
        # int8_kv defaults to False -- every existing TestWriteRead case
        # above already covers this, but a direct construction-time check
        # here makes the "opt-in, zero risk to the default path" claim
        # explicit rather than merely implied.
        cache = PagedKVCache(CacheConfig(block_size=4, num_gpu_blocks=10), TOY_CONFIG, device="cuda")
        assert cache.k_cache.dtype == TOY_CONFIG.dtype
        assert not hasattr(cache, "k_scale")
