"""Fold the reading add-on (LoRA from pretrain.py) into the model's own weights and save a complete
bf16 model, so a fresh fine-tune (qlora_financial/train.py --model <this folder>) can start from it.
A new LoRA can't be trained on top of another LoRA that is still attached; merged, it's just a model.

    python3 -m training.cpt_financial.merge --adapter training/cpt_financial/outputs/run/adapter \
        --out /home/ubuntu/cpt_merged
"""

import argparse


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True, help="the trained reading add-on folder")
    ap.add_argument("--out", required=True, help="where to save the merged model (~16 GB)")
    args = ap.parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from training.cpt_financial.pretrain import MODEL
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()  # weights += LoRA's B x A
    model.save_pretrained(args.out)
    AutoTokenizer.from_pretrained(MODEL).save_pretrained(args.out)  # train.py loads the tokenizer from --model
    print(f"merged {args.adapter} into {MODEL} -> {args.out}")


if __name__ == "__main__":
    main()
