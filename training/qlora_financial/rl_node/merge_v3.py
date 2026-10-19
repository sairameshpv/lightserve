"""Merge v3 (LoRA r16) into bf16 Llama-3-8B-Instruct; check greedy answers match."""
import json, torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
BASE, OUT = "meta-llama/Meta-Llama-3-8B-Instruct", "/home/ubuntu/rl/v3_merged"
test = [json.loads(l) for l in open("/home/ubuntu/v3_run/data_v3/test.jsonl")]
picked = {}
for l in open("/home/ubuntu/v3_run/outputs_v3/samples.jsonl"):
    r = json.loads(l)
    if r["source"] not in picked and "Step" in test[r["id"]]["messages"][2]["content"] + r["source"][:0]:
        picked[r["source"]] = r
picked = list(picked.values())[:3]
for r in picked: assert r["gold"] in test[r["id"]]["messages"][2]["content"], r["id"]
tok = AutoTokenizer.from_pretrained(BASE)
model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda"), "/home/ubuntu/rl/v3_adapter")
def answers():
    out = []
    for r in picked:
        ids = tok.apply_chat_template(test[r["id"]]["messages"][:2], add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")
        g = model.generate(**ids, max_new_tokens=300, do_sample=False)
        out.append(tok.decode(g[0, ids["input_ids"].shape[1]:], skip_special_tokens=True))
    return out
before = answers()
model = model.merge_and_unload()
after = answers()
for r, b, a in zip(picked, before, after):
    print(f"--- id {r['id']} {r['source']} gold={r['gold']} v3-eval-greedy={r['greedy']}\n  identical={a == b}\n  merged: {a[-160:]!r}")
model.save_pretrained(OUT, safe_serialization=True); tok.save_pretrained(OUT)
print("saved", OUT)
