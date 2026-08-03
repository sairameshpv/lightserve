"""Analytical arithmetic-intensity / roofline model for Llama-3-8B on an L40S.

No live GPU needed — this is pure FLOP/byte counting from the model's
published architecture (bf16 weights, matches what the profiled server
actually ran) against the L40S's published peak specs. Complements the
empirical kernel-trace findings in ../report.md (that report found the
LM-head GEMV is memory-bound at batch=1 with AI ~ a few FLOPs/byte; this
script generalizes that same kind of analysis to attention and FFN across
a range of sequence lengths).

Sources (see report.md / conversation for links):
  L40S: 362 TFLOPS dense BF16, 864 GB/s memory bandwidth (NVIDIA official specs).
  Llama-3-8B: hidden=4096, intermediate=14336, heads=32, kv_heads=8 (GQA),
  head_dim=128, layers=32 (HuggingFace config.json, matches the model actually served).

Memory-traffic model per matmul [S,K]@[K,N]->[S,N] in bf16 (2 bytes/elem):
  bytes = 2 * (S*K + K*N + S*N)   # read input, read weight, write output
This is the standard single-kernel-launch model; it doesn't credit any
cross-kernel fusion (e.g. gate+up fused into one launch), so treat FFN/QKVO
numbers as a conservative (slightly pessimistic on AI) baseline.

Attention score compute (QK^T + softmax@V) is modeled flash-attention style:
Q/K/V/O are read/written once each in bf16, the S x S (or S x L) score
matrix is NEVER materialized to HBM (kept in SRAM within the kernel) --
matches what the profiled server's flash_fwd_splitkv_kernel actually does.
"""
import csv
import json
from pathlib import Path

# ---- L40S hardware (verified via NVIDIA official specs) ----
PEAK_BF16_FLOPS = 362e12  # dense, no sparsity
PEAK_BW_BYTES = 864e9
RIDGE_AI = PEAK_BF16_FLOPS / PEAK_BW_BYTES  # FLOPs/byte where roofline bends

# ---- Llama-3-8B architecture (verified via HF config.json) ----
HIDDEN = 4096
INTERMEDIATE = 14336
N_HEADS = 32
N_KV_HEADS = 8
HEAD_DIM = HIDDEN // N_HEADS  # 128
Q_DIM = N_HEADS * HEAD_DIM      # 4096
KV_DIM = N_KV_HEADS * HEAD_DIM  # 1024
BYTES_PER_ELEM = 2  # bf16

SEQ_LENS = [1, 8, 32, 128, 512, 2048, 8192]


def matmul_flops_bytes(S, K, N):
    flops = 2 * S * K * N
    bytes_ = BYTES_PER_ELEM * (S * K + K * N + S * N)
    return flops, bytes_


def ffn_block(S):
    """gate_proj + up_proj + down_proj (SwiGLU), summed as independent kernels."""
    f1, b1 = matmul_flops_bytes(S, HIDDEN, INTERMEDIATE)       # gate_proj
    f2, b2 = matmul_flops_bytes(S, HIDDEN, INTERMEDIATE)       # up_proj
    f3, b3 = matmul_flops_bytes(S, INTERMEDIATE, HIDDEN)       # down_proj
    return f1 + f2 + f3, b1 + b2 + b3


def qkvo_block(S):
    """Q/K/V/O linear projections (GQA: K,V project to fewer heads than Q)."""
    fq, bq = matmul_flops_bytes(S, HIDDEN, Q_DIM)
    fk, bk = matmul_flops_bytes(S, HIDDEN, KV_DIM)
    fv, bv = matmul_flops_bytes(S, HIDDEN, KV_DIM)
    fo, bo = matmul_flops_bytes(S, Q_DIM, HIDDEN)
    return fq + fk + fv + fo, bq + bk + bv + bo


def attention_score_prefill(S):
    """QK^T + softmax(.)@V for a batched forward pass over S tokens attending
    to each other (causal prefill). Flash-attention-style byte accounting."""
    flops = 2 * (2 * HEAD_DIM * S * S) * N_HEADS  # QK^T and AV, per head, x N_HEADS
    # bytes: read Q (S*Q_DIM), read K,V (S*KV_DIM each), write O (S*Q_DIM) -- bf16
    bytes_ = BYTES_PER_ELEM * (S * Q_DIM + S * KV_DIM + S * KV_DIM + S * Q_DIM)
    return flops, bytes_


def attention_score_decode(L):
    """QK^T + softmax(.)@V for ONE new query token attending to an L-token
    KV-cache (autoregressive decode step)."""
    S = 1
    flops = 2 * (2 * HEAD_DIM * S * L) * N_HEADS
    bytes_ = BYTES_PER_ELEM * (S * Q_DIM + L * KV_DIM + L * KV_DIM + S * Q_DIM)
    return flops, bytes_


def classify(ai):
    return "memory-bound" if ai < RIDGE_AI else "compute-bound"


def row(op, S, flops, bytes_):
    ai = flops / bytes_
    attainable = min(PEAK_BF16_FLOPS, ai * PEAK_BW_BYTES)
    return {
        "op": op,
        "seq_len": S,
        "flops": flops,
        "bytes": bytes_,
        "arithmetic_intensity": round(ai, 4),
        "attainable_tflops": round(attainable / 1e12, 4),
        "classification": classify(ai),
    }


def main():
    rows = []
    for S in SEQ_LENS:
        f, b = ffn_block(S)
        rows.append(row("FFN (gate+up+down)", S, f, b))

        f, b = qkvo_block(S)
        rows.append(row("Attention QKVO proj", S, f, b))

        f, b = attention_score_prefill(S)
        rows.append(row("Attention score (prefill, QK^T+AV)", S, f, b))

        f, b = attention_score_decode(S)  # S here = KV-cache length L
        rows.append(row("Attention score (decode, 1 query vs cache)", S, f, b))

    out_dir = Path(__file__).parent
    with (out_dir / "roofline_data.csv").open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with (out_dir / "roofline_data.json").open("w") as fp:
        json.dump(
            {
                "l40s_peak_bf16_tflops": PEAK_BF16_FLOPS / 1e12,
                "l40s_peak_bw_gbs": PEAK_BW_BYTES / 1e9,
                "ridge_point_ai": round(RIDGE_AI, 2),
                "rows": rows,
            },
            fp,
            indent=2,
        )

    print(f"Ridge point (compute-bound above, memory-bound below): {RIDGE_AI:.1f} FLOPs/byte\n")
    header = f"{'op':<42}{'S':>7}{'AI (FLOP/B)':>14}{'attain TFLOPS':>16}{'class':>16}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['op']:<42}{r['seq_len']:>7}{r['arithmetic_intensity']:>14.2f}"
              f"{r['attainable_tflops']:>16.2f}{r['classification']:>16}")


if __name__ == "__main__":
    main()
