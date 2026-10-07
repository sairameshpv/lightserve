"""Before/after measurements for continued pretraining, on text the model never trains on:
1. financial reading: loss on fixed check-pile pages (Corpus-Prep val_*.bin);
2. general reading (forgetting check): loss on the WikiText-2 test set;
3. still a chat model?: answers to a few fixed questions, read by eye.
Loss = average surprise per predicted token (lower = predicts the text better); perplexity = e^loss.
"""

import math


def mean_loss(model, pages) -> float:
    """Average next-token loss over all pages (equal length, so a plain mean of page losses)."""
    import torch
    model.eval()
    losses = []
    with torch.no_grad():
        for page in pages:
            ids = torch.as_tensor(page["input_ids"]).unsqueeze(0).to(model.device)
            losses.append(model(input_ids=ids, labels=ids).loss.item())
    return sum(losses) / len(losses)


def wikitext_pages(tokenizer, page_len: int = 4096):
    """WikiText-2 (raw) test set, joined and cut into non-overlapping pages of page_len tokens."""
    import numpy as np
    from datasets import load_dataset
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = np.asarray(tokenizer(text, add_special_tokens=False)["input_ids"], dtype=np.int64)
    return [{"input_ids": ids[s:s + page_len]} for s in range(0, len(ids) - page_len + 1, page_len)]


def report(name: str, loss: float) -> dict:
    return {"set": name, "loss": round(loss, 4), "perplexity": round(math.exp(loss), 3)}


QUESTIONS = [  # 2 financial, 3 general: is it still a helpful chat model after reading?
    "What is the difference between operating income and net income?",
    "A company's revenue grew from $120 million to $150 million. What was the growth rate?",
    "What is the capital of Australia?",
    "Write a haiku about autumn.",
    "Give me three tips for a job interview.",
]


def answers(model, tokenizer, max_new_tokens: int = 120) -> list:
    """Greedy answers (always the most likely next word), so before and after compare directly."""
    out = []
    for q in QUESTIONS:
        ids = tokenizer.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                            return_tensors="pt", return_dict=True).to(model.device)
        gen = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
        out.append({"question": q, "answer": tokenizer.decode(gen[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)})
    return out


def main():
    import argparse, json
    from pathlib import Path
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from training.cpt_financial.pretrain import MODEL, PageDataset
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", help="trained LoRA add-on folder ('after'); leave out for 'before'")
    ap.add_argument("--corpus-dir", default="/home/ubuntu/corpus")
    ap.add_argument("--check-pages", type=int, default=200)
    ap.add_argument("--out", required=True, help="results JSON file")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    check = PageDataset(sorted(Path(args.corpus_dir).glob("val_*.bin")), 4096, args.check_pages, seed=1)
    results = {"adapter": args.adapter,
               "reading": [report("check pile (10-K)", mean_loss(model, check)),
                           report("WikiText-2 test", mean_loss(model, wikitext_pages(tok)))],
               "answers": answers(model, tok)}
    print(json.dumps(results["reading"]))
    Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
