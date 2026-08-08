"""Empirical roofline measurement: times the actual matmul/attention shapes
used by compute_roofline.py's analytical model on whatever GPU this runs on,
via CUDA-event timing. Meant to run inside the vllm-openai container (already
has torch+CUDA+flash-attn matching what the server itself uses):

    docker run --rm --gpus all --ipc=host \
      -v /path/to/measure_roofline.py:/measure_roofline.py \
      --entrypoint python3 vllm/vllm-openai:latest /measure_roofline.py \
      > results.json

Shapes/FLOP-byte formulas are kept identical to compute_roofline.py so the
measured numbers are directly comparable to the theoretical ceiling.
"""
import json
import statistics
import sys

import torch
import torch.nn.functional as F

HIDDEN = 4096
INTERMEDIATE = 14336
N_HEADS = 32
N_KV_HEADS = 8
HEAD_DIM = HIDDEN // N_HEADS
Q_DIM = N_HEADS * HEAD_DIM
KV_DIM = N_KV_HEADS * HEAD_DIM

SEQ_LENS = [1, 8, 32, 128, 512, 2048, 8192]
DEVICE = "cuda"
DTYPE = torch.bfloat16
WARMUP = 15
ITERS = 50
REPEATS = 3  # take the median of this many timed blocks


def time_block(fn, iters=ITERS, warmup=WARMUP, repeats=REPEATS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    per_call_ms = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        per_call_ms.append(start.elapsed_time(end) / iters)
    return statistics.median(per_call_ms)


def bench_matmul(S, K, N):
    x = torch.randn(S, K, device=DEVICE, dtype=DTYPE)
    w = torch.randn(K, N, device=DEVICE, dtype=DTYPE)
    ms = time_block(lambda: x @ w)
    flops = 2 * S * K * N
    bytes_ = 2 * (S * K + K * N + S * N)
    return ms, flops, bytes_


def bench_ffn(S):
    ms1, f1, b1 = bench_matmul(S, HIDDEN, INTERMEDIATE)       # gate_proj
    ms2, f2, b2 = bench_matmul(S, HIDDEN, INTERMEDIATE)       # up_proj
    ms3, f3, b3 = bench_matmul(S, INTERMEDIATE, HIDDEN)       # down_proj
    return ms1 + ms2 + ms3, f1 + f2 + f3, b1 + b2 + b3


def bench_qkvo(S):
    msq, fq, bq = bench_matmul(S, HIDDEN, Q_DIM)
    msk, fk, bk = bench_matmul(S, HIDDEN, KV_DIM)
    msv, fv, bv = bench_matmul(S, HIDDEN, KV_DIM)
    mso, fo, bo = bench_matmul(S, Q_DIM, HIDDEN)
    return msq + msk + msv + mso, fq + fk + fv + fo, bq + bk + bv + bo


def sdpa_supports_gqa():
    try:
        q = torch.randn(1, N_HEADS, 4, HEAD_DIM, device=DEVICE, dtype=DTYPE)
        k = torch.randn(1, N_KV_HEADS, 4, HEAD_DIM, device=DEVICE, dtype=DTYPE)
        v = torch.randn(1, N_KV_HEADS, 4, HEAD_DIM, device=DEVICE, dtype=DTYPE)
        F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        return True
    except TypeError:
        return False


GQA_NATIVE = sdpa_supports_gqa()
GROUP = N_HEADS // N_KV_HEADS


def expand_kv(k, v):
    if GQA_NATIVE:
        return k, v
    return k.repeat_interleave(GROUP, dim=1), v.repeat_interleave(GROUP, dim=1)


def sdpa(q, k, v, is_causal):
    k, v = expand_kv(k, v)
    kwargs = {"enable_gqa": True} if GQA_NATIVE else {}
    return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, **kwargs)


def bench_attention_prefill(S):
    q = torch.randn(1, N_HEADS, S, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    k = torch.randn(1, N_KV_HEADS, S, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    v = torch.randn(1, N_KV_HEADS, S, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    ms = time_block(lambda: sdpa(q, k, v, is_causal=True))
    flops = 2 * (2 * HEAD_DIM * S * S) * N_HEADS
    bytes_ = 2 * (S * Q_DIM + S * KV_DIM + S * KV_DIM + S * Q_DIM)
    return ms, flops, bytes_


def bench_attention_decode(L):
    q = torch.randn(1, N_HEADS, 1, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    k = torch.randn(1, N_KV_HEADS, L, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    v = torch.randn(1, N_KV_HEADS, L, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    ms = time_block(lambda: sdpa(q, k, v, is_causal=False))
    flops = 2 * (2 * HEAD_DIM * 1 * L) * N_HEADS
    bytes_ = 2 * (1 * Q_DIM + L * KV_DIM + L * KV_DIM + 1 * Q_DIM)
    return ms, flops, bytes_


def main():
    if not torch.cuda.is_available():
        print(json.dumps({"error": "CUDA not available"}), file=sys.stdout)
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    results = {"gpu": gpu_name, "torch_version": torch.__version__, "gqa_native": GQA_NATIVE, "rows": []}

    benches = [
        ("FFN (gate+up+down)", bench_ffn),
        ("Attention QKVO proj", bench_qkvo),
        ("Attention score (prefill)", bench_attention_prefill),
        ("Attention score (decode, vs KV-cache len)", bench_attention_decode),
    ]

    for op_name, fn in benches:
        for S in SEQ_LENS:
            ms, flops, bytes_ = fn(S)
            achieved_tflops = flops / (ms / 1000) / 1e12
            achieved_gbs = bytes_ / (ms / 1000) / 1e9
            row = {
                "op": op_name,
                "seq_len": S,
                "elapsed_ms": round(ms, 5),
                "flops": flops,
                "bytes": bytes_,
                "achieved_tflops": round(achieved_tflops, 4),
                "achieved_gbs": round(achieved_gbs, 2),
            }
            results["rows"].append(row)
            print(f"{op_name:<42} S={S:<6} {ms:>10.4f}ms  {achieved_tflops:>8.2f} TFLOPS  {achieved_gbs:>9.1f} GB/s", file=sys.stderr)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()