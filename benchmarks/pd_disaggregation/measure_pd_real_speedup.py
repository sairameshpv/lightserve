"""Real 2-GPU P/D disaggregation benchmark -- the actual "does hardware
isolation help" measurement, per ~/.claude/plans/floating-squishing-sonnet.md.
Drives two already-running server/pd_role_server.py processes (one
--role prefill, one --role decode, on two separate L40S nodes) over real
HTTP, plus a "monolithic" baseline (one plain LLMEngine, in-process, no
network at all) for comparison -- both on the real Llama-3-8B checkpoint.

Two stages:

Stage 1 (correctness + latency breakdown): a handful of real tokenized
prompts (reused from benchmarks/speculative_decoding/prompts_tokenized.
jsonl), through the real prefill-node -> HTTP -> decode-node path,
checked byte-identical against a dense reference (same bar every prior
correctness script in this repo uses) -- proves the new code this stage
actually adds (wire serialization, the background _DecodeWorker) doesn't
change output, since model/pd_disaggregation.py's own mechanism was
already proven correct in-process (benchmarks/pd_disaggregation/
verify_pd_correctness.py). Also reports the first honest latency
breakdown: prefill compute time (server-reported) vs. prefill network
overhead (client round-trip minus that), and the decode leg's own
round-trip plus its per-token gaps (pulled from the decode node's
/itl_log, the same mechanism stage 2 uses for its headline number).

Stage 2 (the actual contention benchmark): reuses benchmarks/
chunked_prefill/measure_itl.py's own workload shape (settle a decode
population, then inject one long prefill) and its build_decode_prompts/
build_prefill_prompt/run_mixed_workload directly for the MONOLITHIC
condition -- that function IS already exactly "one box, settle-then-
inject, report baseline vs. disrupted decode ITL". The DISAGGREGATED
condition mirrors the same baseline/disrupted split, but orchestrated
over real HTTP: settle N requests onto the decode node (each through a
real prefill()+resume_decode() round trip, not an in-process shortcut),
capture a clean baseline window on the decode node's own /itl_log, then
inject the long prefill on the *other* node entirely and capture the
disrupted window the same way. The decode-ITL comparison between the two
conditions is the real number this whole exercise exists for.

Content is synthetic random token ids for stage 2 (same reasoning
measure_itl.py's own module docstring gives: this is testing scheduling/
contention behavior, not correctness/quality, so real text carries no
more meaning here than random ids do) -- stage 1 uses real tokenized
text specifically because it's a correctness check.

IMPORTANT -- run this from one of the two GPU nodes (needs model/
pd_disaggregation.py's classes importable for unpickling wire payloads,
and stage 1 loads its own copy of the real checkpoint for the dense
reference -- see stage1()'s own docstring on the resulting VRAM
footprint if run on the same box as a role-server).

Setup (once, before this): both nodes running
    sudo .venv/bin/python3 -m server.pd_role_server --role prefill --port 8100 --num-gpu-blocks N
    sudo .venv/bin/python3 -m server.pd_role_server --role decode  --port 8100 --num-gpu-blocks N

Run:
    sudo .venv/bin/python3 -m benchmarks.pd_disaggregation.measure_pd_real_speedup \\
        --prefill-host <node0-ip-or-localhost> --decode-host <node1-internal-ip>
"""
import argparse
import json
import os
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from model.llm_engine import GenerationOutput
from model.pd_disaggregation import PrefillHandoff
from server.pd_role_server import pack, unpack

PROMPTS_PATH = Path(__file__).parent.parent / "speculative_decoding" / "prompts_tokenized.jsonl"
STAGE1_NUM_PROMPTS = 3  # kept small -- see module docstring on the extra VRAM a 2nd loaded checkpoint costs

# Same shape as benchmarks/chunked_prefill/measure_itl.py's own defaults,
# just smaller -- this is a first real cross-machine run, not yet a full
# sweep (see this stage's own plan file on "smoke-test first").
NUM_DECODE_REQUESTS = 4
DECODE_PROMPT_LEN = 16
DECODE_MAX_TOKENS = 60
PREFILL_LEN = 2048
PREFILL_MAX_TOKENS = 1
SETTLE_SECONDS = 3.0
BASELINE_WINDOW_SECONDS = 2.0
BLOCK_SIZE = 16

_HF_HUB_DIR = os.path.expanduser("~/.cache/huggingface/hub")


def _find_snapshot_dir(model_repo_dir_name):
    snapshots_dir = os.path.join(_HF_HUB_DIR, model_repo_dir_name, "snapshots")
    if not os.path.isdir(snapshots_dir):
        return None
    for name in os.listdir(snapshots_dir):
        candidate = os.path.join(snapshots_dir, name)
        if os.path.exists(os.path.join(candidate, "config.json")):
            return candidate
    return None


def reference_generate(weights, config, prompt, max_tokens):
    """Same dense re-run-from-scratch-every-step reference every
    verify_*_correctness.py script in this repo uses -- inlined here
    rather than imported, same reasoning as those.
    """
    from model.minimal_llama import reference_llama_forward
    tokens = list(prompt)
    generated = []
    for _ in range(max_tokens):
        input_ids = torch.tensor([tokens], device="cuda")
        logits = reference_llama_forward(weights, config, input_ids, causal=True)
        next_tok = int(logits[0, -1].argmax().item())
        tokens.append(next_tok)
        generated.append(next_tok)
    return generated


def http_post(url: str, data: bytes, content_type: str, timeout: float = 300) -> bytes:
    req = urllib.request.Request(url, data=data, headers={"Content-Type": content_type}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def http_get(url: str, timeout: float = 30) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()


def http_post_json(url: str, obj: dict, timeout: float = 300) -> dict:
    raw = http_post(url, json.dumps(obj).encode("utf-8"), "application/json", timeout=timeout)
    return json.loads(raw)


def do_prefill(prefill_base: str, prompt_token_ids: list, max_tokens: int, request_id: str = None):
    """One real HTTP round trip to the prefill node. Returns
    (result, prefill_seconds, network_seconds) -- result is either a
    PrefillHandoff (forward it to /resume_decode next) or a
    GenerationOutput (finished during prefill, see model/
    pd_disaggregation.py's prefill() docstring -- nothing left to do).
    """
    body = {"prompt_token_ids": prompt_token_ids, "max_tokens": max_tokens}
    if request_id is not None:
        body["request_id"] = request_id
    t0 = time.perf_counter()
    raw = http_post(f"{prefill_base}/prefill", json.dumps(body).encode("utf-8"), "application/json")
    roundtrip_seconds = time.perf_counter() - t0
    unpacked = unpack(raw)
    prefill_seconds = unpacked["prefill_seconds"]
    return unpacked["result"], prefill_seconds, roundtrip_seconds - prefill_seconds


def do_resume_decode(decode_base: str, handoff: PrefillHandoff, timeout: float = 300) -> dict:
    """One real HTTP round trip to the decode node -- blocks until this
    request finishes entirely (see server/pd_role_server.py's module
    docstring on why /resume_decode has this shape)."""
    raw = http_post(f"{decode_base}/resume_decode", pack(handoff), "application/octet-stream", timeout=timeout)
    return json.loads(raw)


def summarize_records(records: list) -> dict:
    """Same core arithmetic as measure_itl.py's summarize_itl (baseline/
    disrupted mean + disrupted max), without that function's chunk_size/
    prefill_ttft fields, which don't apply to this benchmark's own
    condition axis (monolithic vs. disaggregated, not a chunk-size sweep).
    """
    baseline = [r["itl_seconds"] for r in records if r["phase"] == "baseline"]
    disrupted = [r["itl_seconds"] for r in records if r["phase"] == "disrupted"]
    return {
        "baseline_itl_ms_mean": statistics.mean(baseline) * 1000 if baseline else 0.0,
        "disrupted_itl_ms_mean": statistics.mean(disrupted) * 1000 if disrupted else 0.0,
        "disrupted_itl_ms_max": max(disrupted) * 1000 if disrupted else 0.0,
        "num_baseline_samples": len(baseline),
        "num_disrupted_samples": len(disrupted),
    }


def stage1(prefill_base: str, decode_base: str, target_config, target_weights) -> bool:
    """Correctness + latency breakdown. Loads its own copy of the real
    checkpoint's weights for the dense reference (target_config/
    target_weights, passed in from main() -- already loaded there) --
    if this script runs on the same box as a role-server, that's a
    *second* independent ~16GB copy of the 8B weights resident at once,
    on top of whatever KV cache that role-server's engine reserved.
    Kept to STAGE1_NUM_PROMPTS=3 partly for this reason -- this stage's
    job is proving the new wire-serialization/background-worker code
    path preserves correctness, not a large-scale correctness sweep
    (verify_pd_correctness.py already did that in-process).
    """
    records = []
    with PROMPTS_PATH.open() as f:
        for line in f:
            line = line.strip()
            if line and len(records) < STAGE1_NUM_PROMPTS:
                records.append(json.loads(line))
    if not records:
        raise SystemExit(f"No prompts found at {PROMPTS_PATH}.")

    all_match = True
    for r in records:
        http_post_json(f"{decode_base}/reset_itl_log", {})

        result, prefill_seconds, prefill_network_seconds = do_prefill(
            prefill_base, r["prompt"], r["max_tokens"], request_id=r["id"],
        )

        if isinstance(result, PrefillHandoff):
            t0 = time.perf_counter()
            decode_result = do_resume_decode(decode_base, result)
            decode_roundtrip_seconds = time.perf_counter() - t0
            output_token_ids = decode_result["output_token_ids"]
            finish_reason = decode_result["finish_reason"]
        elif isinstance(result, GenerationOutput):
            decode_roundtrip_seconds = 0.0
            output_token_ids = result.output_token_ids
            finish_reason = result.finish_reason
        else:
            raise TypeError(f"unexpected /prefill result type: {type(result)}")

        expected = reference_generate(target_weights, target_config, r["prompt"], max_tokens=r["max_tokens"])
        ok = output_token_ids == expected
        all_match &= ok

        itl_log = json.loads(http_get(f"{decode_base}/itl_log"))["records"]
        itl_gaps_ms = []
        if len(itl_log) > 1:
            times = sorted(rec["t"] for rec in itl_log)
            itl_gaps_ms = [(b - a) * 1000 for a, b in zip(times, times[1:])]

        status = "MATCH" if ok else "MISMATCH"
        itl_mean_str = f"{statistics.mean(itl_gaps_ms):.1f}ms" if itl_gaps_ms else "n/a"
        print(f"[{status}] {r['id']} (finish={finish_reason}): "
              f"prefill={prefill_seconds*1000:.1f}ms (network {prefill_network_seconds*1000:.1f}ms), "
              f"decode_roundtrip={decode_roundtrip_seconds*1000:.1f}ms, "
              f"decode_itl_mean={itl_mean_str}")
        if not ok:
            print(f"  got:      {output_token_ids}")
            print(f"  expected: {expected}")

    print("STAGE 1: MATCH -- disaggregated (real HTTP) output agrees with the dense reference on every prompt."
          if all_match else "STAGE 1: MISMATCH -- see above.")
    return all_match


def run_monolithic(target_config, target_weights, num_gpu_blocks: int, args: argparse.Namespace) -> dict:
    from engine.config import CacheConfig, SchedulerConfig
    from model.llm_engine import LLMEngine
    from benchmarks.chunked_prefill.measure_itl import build_decode_prompts, build_prefill_prompt, run_mixed_workload

    # One cache_config, shared by both the warmup engine and the real one
    # below -- int8_kv applies identically to each, same as every other
    # CacheConfig field here.
    cache_config = CacheConfig(block_size=args.block_size, num_gpu_blocks=num_gpu_blocks,
                                int8_kv=args.int8_kv)
    scheduler_config = SchedulerConfig(max_num_seqs=args.num_decode_requests + 1, max_num_batched_tokens=2048)

    decode_prompts = build_decode_prompts(args.num_decode_requests, args.decode_prompt_len, seed=0)
    prefill_prompt = build_prefill_prompt(args.prefill_len, seed=1)

    # Warmup: Triton's @triton.autotune compiles kernels for a given batch
    # shape on FIRST use -- this can take seconds and would otherwise land
    # inside the timed baseline window (measure_itl.py's own warmup() exists
    # for exactly this reason). Build a throwaway engine, run the same shape
    # untimed with tiny settle_steps/max_tokens, then discard it.
    warmup_engine = LLMEngine(cache_config, scheduler_config, target_config, weights=target_weights, device="cuda")
    run_mixed_workload(warmup_engine, decode_prompts, decode_max_tokens=3, prefill_prompt=prefill_prompt,
                        prefill_max_tokens=1, settle_steps=2)
    del warmup_engine

    engine = LLMEngine(cache_config, scheduler_config, target_config, weights=target_weights, device="cuda")
    itl_records, _step_records, _prefill_ttft = run_mixed_workload(
        engine, decode_prompts, args.decode_max_tokens, prefill_prompt, args.prefill_max_tokens,
        settle_steps=10,
    )
    return summarize_records(itl_records)


def warmup_disaggregated(prefill_base: str, decode_base: str, args: argparse.Namespace) -> None:
    """Same reasoning as run_monolithic's warmup step, but for the two
    long-lived role-server processes: Triton compiles kernels for a given
    batch shape on first use, and that cost must not land inside a
    captured window. Sends one real round trip at each shape stage 2
    actually uses -- the decode population's batch size/prompt length,
    and the injected prefill's length -- with small max_tokens so it's
    fast; only the shape needs to match, not the full duration.
    """
    from benchmarks.chunked_prefill.measure_itl import build_decode_prompts, build_prefill_prompt

    warmup_decode_prompts = build_decode_prompts(args.num_decode_requests, args.decode_prompt_len, seed=2)
    warmup_prefill_prompt = build_prefill_prompt(args.prefill_len, seed=3)

    def prefill_and_resume(prompt, max_tokens):
        result, _ps, _ns = do_prefill(prefill_base, prompt, max_tokens)
        if isinstance(result, PrefillHandoff):
            do_resume_decode(decode_base, result)

    with ThreadPoolExecutor(max_workers=args.num_decode_requests) as ex:
        list(ex.map(lambda p: prefill_and_resume(p, 3), warmup_decode_prompts))

    prefill_and_resume(warmup_prefill_prompt, 1)
    http_post_json(f"{decode_base}/reset_itl_log", {})  # drop warmup's own token records


def run_disaggregated(prefill_base: str, decode_base: str, args: argparse.Namespace) -> dict:
    """Same settle-then-inject shape as run_mixed_workload, orchestrated
    over real HTTP instead of one in-process engine's own step() loop.
    Settling uses wall-clock sleeps rather than an exact step count
    (measure_itl.py's own mechanism) since this script doesn't control
    the decode node's step loop directly -- both are just "reach genuine
    steady state before the timed window starts", the exact mechanism
    doesn't need to match as long as each is sufficient for its own side.
    """
    from benchmarks.chunked_prefill.measure_itl import build_decode_prompts, build_prefill_prompt

    warmup_disaggregated(prefill_base, decode_base, args)

    decode_prompts = build_decode_prompts(args.num_decode_requests, args.decode_prompt_len, seed=0)
    prefill_prompt = build_prefill_prompt(args.prefill_len, seed=1)

    # Real prefill()+resume_decode() round trip per decode-workload
    # prompt, matching what a real system does to populate "already
    # decoding" traffic -- not an in-process shortcut. Concurrent across
    # the group (ThreadPoolExecutor): the prefill node handling several
    # of these together is fine for this untimed setup phase (only the
    # *injected* prefill needs to be alone on that node, and only during
    # the timed window below -- see module docstring).
    def prefill_and_get_handoff(prompt):
        result, _prefill_seconds, _network_seconds = do_prefill(prefill_base, prompt, args.decode_max_tokens)
        return result if isinstance(result, PrefillHandoff) else None

    with ThreadPoolExecutor(max_workers=args.num_decode_requests) as ex:
        handoffs = list(ex.map(prefill_and_get_handoff, decode_prompts))
    handoffs = [h for h in handoffs if h is not None]

    # Fire every /resume_decode concurrently and do NOT wait for
    # completion here -- "settling" means getting them all running
    # together on the decode node, not waiting for them to finish.
    executor = ThreadPoolExecutor(max_workers=len(handoffs) + 1)
    settle_futures = [executor.submit(do_resume_decode, decode_base, h) for h in handoffs]

    time.sleep(args.settle_seconds)  # let them reach steady-state, several real decode steps' worth

    # Clean baseline window: reset, wait, capture -- entirely before the
    # injected prefill exists anywhere.
    http_post_json(f"{decode_base}/reset_itl_log", {})
    time.sleep(args.baseline_window_seconds)
    baseline_records = json.loads(http_get(f"{decode_base}/itl_log"))["records"]
    for rec in baseline_records:
        rec["phase"] = "baseline"

    # Inject the long prefill, alone on the prefill node -- then hand it
    # to the decode node, which is now also running the settled decode
    # population concurrently. This moment is what's being measured.
    http_post_json(f"{decode_base}/reset_itl_log", {})
    injected_result, _prefill_seconds, _network_seconds = do_prefill(
        prefill_base, prefill_prompt, args.prefill_max_tokens,
    )
    if isinstance(injected_result, PrefillHandoff):
        executor.submit(do_resume_decode, decode_base, injected_result).result(timeout=120)

    for f in settle_futures:
        f.result(timeout=120)  # wait for the settled population to finish naturally, same as run_mixed_workload's own termination condition
    executor.shutdown()

    disrupted_records = json.loads(http_get(f"{decode_base}/itl_log"))["records"]
    for rec in disrupted_records:
        rec["phase"] = "disrupted"

    # itl_log entries carry a raw "t" timestamp, not "itl_seconds" --
    # convert to consecutive gaps per request, same shape
    # measure_itl.py's own itl_records use, so summarize_records() can
    # treat both conditions identically.
    def to_gaps(records):
        by_request: dict = {}
        for rec in records:
            by_request.setdefault(rec["request_id"], []).append(rec)
        gaps = []
        for rid, recs in by_request.items():
            recs.sort(key=lambda r: r["token_index"])
            times = [r["t"] for r in recs]
            for a, b in zip(times, times[1:]):
                gaps.append({"phase": recs[0]["phase"], "itl_seconds": b - a})
        return gaps

    all_gaps = to_gaps(baseline_records) + to_gaps(disrupted_records)
    return summarize_records(all_gaps)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefill-host", required=True)
    ap.add_argument("--prefill-port", type=int, default=8100)
    ap.add_argument("--decode-host", required=True)
    ap.add_argument("--decode-port", type=int, default=8100)
    ap.add_argument("--stage", choices=["1", "2", "both"], default="both")
    ap.add_argument("--num-decode-requests", type=int, default=NUM_DECODE_REQUESTS)
    ap.add_argument("--decode-prompt-len", type=int, default=DECODE_PROMPT_LEN)
    ap.add_argument("--decode-max-tokens", type=int, default=DECODE_MAX_TOKENS)
    ap.add_argument("--prefill-len", type=int, default=PREFILL_LEN)
    ap.add_argument("--prefill-max-tokens", type=int, default=PREFILL_MAX_TOKENS)
    ap.add_argument("--settle-seconds", type=float, default=SETTLE_SECONDS)
    ap.add_argument("--baseline-window-seconds", type=float, default=BASELINE_WINDOW_SECONDS)
    ap.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    ap.add_argument("--num-gpu-blocks", type=int, required=True,
                     help="For this script's own monolithic-condition engine; the two role-servers "
                          "size their own separately when started.")
    ap.add_argument("--int8-kv", action="store_true",
                     help="int8 KV cache for this script's own monolithic-condition engine (see "
                          "engine/config.py's CacheConfig.int8_kv). The disaggregated condition's "
                          "role-servers choose this independently on their own command line "
                          "(server/pd_role_server.py's own --int8-kv) -- this flag only covers "
                          "run_monolithic's engine.")
    args = ap.parse_args()

    prefill_base = f"http://{args.prefill_host}:{args.prefill_port}"
    decode_base = f"http://{args.decode_host}:{args.decode_port}"

    for name, base in [("prefill", prefill_base), ("decode", decode_base)]:
        health = json.loads(http_get(f"{base}/health"))
        print(f"{name} node health: {health}")

    checkpoint_dir = _find_snapshot_dir("models--meta-llama--Meta-Llama-3-8B-Instruct")
    if checkpoint_dir is None:
        raise SystemExit(f"Real Llama-3-8B-Instruct checkpoint not found under {_HF_HUB_DIR}.")
    from model.hf_loader import load_hf_checkpoint
    target_config, target_weights = load_hf_checkpoint(checkpoint_dir, device="cuda")

    if args.stage in ("1", "both"):
        ok = stage1(prefill_base, decode_base, target_config, target_weights)
        if not ok and args.stage == "both":
            raise SystemExit("Stage 1 failed -- not proceeding to stage 2's timing measurement.")

    if args.stage in ("2", "both"):
        num_gpu_blocks = args.num_gpu_blocks
        monolithic = run_monolithic(target_config, target_weights, num_gpu_blocks, args)
        print(f"STAGE 2 [monolithic]:    {monolithic}")
        disaggregated = run_disaggregated(prefill_base, decode_base, args)
        print(f"STAGE 2 [disaggregated]: {disaggregated}")


if __name__ == "__main__":
    main()
