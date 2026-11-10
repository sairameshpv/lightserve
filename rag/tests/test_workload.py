import json
import threading

from rag.workload import WorkloadLog, llama3_prompt, to_prompts

MESSAGES = [{"role": "system", "content": "You answer questions."},
            {"role": "user", "content": "Question: What was Apple's revenue growth in 2019?"}]
REPLY = {"text": "Answer: -2% | Net sales fell 2%.", "prompt_tokens": 1200, "completion_tokens": 14, "seconds": 0.8}


def test_a_record_keeps_what_a_replay_and_a_comparison_need(tmp_path):
    log = WorkloadLog(tmp_path / "w.jsonl")
    log.record(engine="vllm", model="v3", messages=MESSAGES, reply=REPLY, temperature=0.7, top_p=0.95, seed=3,
               max_tokens=512)
    (row,) = [json.loads(line) for line in (tmp_path / "w.jsonl").open()]
    assert {k: row[k] for k in ("engine", "model", "temperature", "top_p", "seed", "max_tokens", "prompt_tokens",
                                "completion_tokens", "seconds")} == {
        "engine": "vllm", "model": "v3", "temperature": 0.7, "top_p": 0.95, "seed": 3, "max_tokens": 512,
        "prompt_tokens": 1200, "completion_tokens": 14, "seconds": 0.8}
    assert row["messages"] == MESSAGES and row["reply_chars"] == len(REPLY["text"]) and row["t_start"] > 0


def test_records_from_many_threads_stay_whole_lines(tmp_path):
    log = WorkloadLog(tmp_path / "w.jsonl")
    threads = [threading.Thread(target=lambda: [log.record(engine="vllm", model="m", messages=MESSAGES, reply=REPLY,
                                                           temperature=0.7, top_p=0.95, seed=0, max_tokens=512)
                                                for _ in range(20)]) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len([json.loads(line) for line in (tmp_path / "w.jsonl").open()]) == 200


def test_llama3_prompt_is_the_chat_template_without_the_start_token():
    """The engine's tokenizer adds <|begin_of_text|> itself, so the text leaves it out."""
    assert llama3_prompt(MESSAGES) == (
        "<|start_header_id|>system<|end_header_id|>\n\nYou answer questions.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\nQuestion: What was Apple's revenue growth in 2019?<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n")


def test_to_prompts_writes_the_ab_tools_format_in_time_order(tmp_path):
    log = WorkloadLog(tmp_path / "w.jsonl")
    for seed, tokens in ((1, 30), (0, 14)):
        log.record(engine="vllm", model="v3", messages=MESSAGES, reply=dict(REPLY, completion_tokens=tokens),
                   temperature=0.7, top_p=0.95, seed=seed, max_tokens=512)
    n = to_prompts(tmp_path / "w.jsonl", tmp_path / "prompts.jsonl")
    rows = [json.loads(line) for line in (tmp_path / "prompts.jsonl").open()]
    assert n == 2 and [r["max_tokens"] for r in rows] == [30, 14]      # the real reply's length, in call order
    assert rows[0]["prompt"] == llama3_prompt(MESSAGES) and set(rows[0]) == {"prompt", "max_tokens"}
