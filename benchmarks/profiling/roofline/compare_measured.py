"""Merge the empirical measurements from measure_roofline.py against the
analytical roofline model from compute_roofline.py, op-by-op, seq_len-by-
seq_len. Answers: how close does the real L40S get to its own roofline?

Run after both roofline_data.json (analytical) and roofline_measured.json
(scp'd back from the instance -- see README.md "Empirical validation") exist:

    python3 compare_measured.py
"""
import json
from pathlib import Path

OUT_DIR = Path(__file__).parent

# measured op names -> analytical op names (measure_roofline.py's benches
# list uses shorter labels than compute_roofline.py's)
OP_ALIAS = {
    "FFN (gate+up+down)": "FFN (gate+up+down)",
    "Attention QKVO proj": "Attention QKVO proj",
    "Attention score (prefill)": "Attention score (prefill, QK^T+AV)",
    "Attention score (decode, vs KV-cache len)": "Attention score (decode, 1 query vs cache)",
}


def main():
    analytical = json.loads((OUT_DIR / "roofline_data.json").read_text())
    measured = json.loads((OUT_DIR / "roofline_measured.json").read_text())

    theory_by_key = {(r["op"], r["seq_len"]): r for r in analytical["rows"]}

    merged = []
    for m in measured["rows"]:
        key = (OP_ALIAS[m["op"]], m["seq_len"])
        t = theory_by_key.get(key)
        if t is None:
            continue
        efficiency = m["achieved_tflops"] / t["attainable_tflops"] if t["attainable_tflops"] else float("nan")
        merged.append({
            "op": t["op"],
            "seq_len": m["seq_len"],
            "arithmetic_intensity": t["arithmetic_intensity"],
            "classification": t["classification"],
            "attainable_tflops": t["attainable_tflops"],
            "achieved_tflops": m["achieved_tflops"],
            "achieved_gbs": m["achieved_gbs"],
            "roofline_efficiency": round(efficiency, 3),
        })

    (OUT_DIR / "roofline_comparison.json").write_text(json.dumps({
        "gpu": measured["gpu"],
        "torch_version": measured["torch_version"],
        "l40s_peak_bf16_tflops": analytical["l40s_peak_bf16_tflops"],
        "l40s_peak_bw_gbs": analytical["l40s_peak_bw_gbs"],
        "ridge_point_ai": analytical["ridge_point_ai"],
        "rows": merged,
    }, indent=2))

    header = f"{'op':<42}{'S':>7}{'AI':>10}{'attain TFLOPS':>15}{'achv TFLOPS':>14}{'achv GB/s':>12}{'eff':>8}"
    print(header)
    print("-" * len(header))
    for r in merged:
        print(f"{r['op']:<42}{r['seq_len']:>7}{r['arithmetic_intensity']:>10.1f}"
              f"{r['attainable_tflops']:>15.2f}{r['achieved_tflops']:>14.2f}"
              f"{r['achieved_gbs']:>12.1f}{r['roofline_efficiency']:>8.1%}")

    max_ai = max(r["arithmetic_intensity"] for r in merged)
    peak_achieved = max(r["achieved_tflops"] for r in merged)
    print(f"\nMax achieved TFLOPS across all ops/shapes: {peak_achieved:.1f} "
          f"({peak_achieved/analytical['l40s_peak_bf16_tflops']:.0%} of the {analytical['l40s_peak_bf16_tflops']:.0f} TFLOPS spec)")


if __name__ == "__main__":
    main()