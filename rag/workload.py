"""Every model call the demo makes, as JSONL: a realistic workload for the A/B tool.

One line per call: engine, model, the conversation sent, the reply's size, sampling settings and
timing, in the same format whatever the engine (vLLM now, lightserve later). `to_prompts` turns a
log into the A/B tool's prompt file, so the demo's real traffic can be replayed as load:

    python -m rag.workload runs/rag/workload.jsonl runs/rag/prompts.jsonl
    python benchmarks/ab/ab_run.py ... --target vllm --prompts runs/rag/prompts.jsonl
"""
import json
import threading
import time
from pathlib import Path


class WorkloadLog:
    """Append-only; safe to call from the 2 x N runs' threads at once."""

    def __init__(self, path):
        self.path, self.lock = Path(path), threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, engine: str, model: str, messages: list, reply: dict, temperature: float, top_p: float,
               seed, max_tokens: int):
        """`reply` is llm.chat's result. t_start = when the call began (its end minus its time)."""
        row = {"t_start": time.time() - reply["seconds"], "engine": engine, "model": model,
               "temperature": temperature, "top_p": top_p, "seed": seed, "max_tokens": max_tokens,
               "prompt_tokens": reply.get("prompt_tokens"), "completion_tokens": reply.get("completion_tokens"),
               "reply_chars": len(reply["text"]), "seconds": reply["seconds"], "messages": messages}
        line = json.dumps(row) + "\n"
        with self.lock, self.path.open("a") as f:
            f.write(line)


def llama3_prompt(messages: list) -> str:
    """The Llama-3 chat template as text, ending where the assistant's reply starts. No
    <|begin_of_text|>: the engine's tokenizer adds it to every prompt."""
    turns = "".join(f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n{m['content'].strip()}<|eot_id|>"
                    for m in messages)
    return turns + "<|start_header_id|>assistant<|end_header_id|>\n\n"


def to_prompts(log_path, out_path) -> int:
    """The log -> the A/B tool's prompt file ({"prompt", "max_tokens"} per line), in call order;
    max_tokens = the real reply's length, so a replay asks for the same amount of work."""
    rows = sorted((json.loads(line) for line in Path(log_path).open() if line.strip()), key=lambda r: r["t_start"])
    with Path(out_path).open("w") as f:
        for r in rows:
            f.write(json.dumps({"prompt": llama3_prompt(r["messages"]),
                                "max_tokens": r["completion_tokens"] or r["max_tokens"]}) + "\n")
    return len(rows)


if __name__ == "__main__":
    import sys
    print(f"{to_prompts(sys.argv[1], sys.argv[2])} prompts written to {sys.argv[2]}")
