"""train.py's start-from-the-champion option, on a tiny made-up Llama (CPU, no download, no TRL)."""
import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

from training.qlora_financial.train import load_init_adapter, parse_args


def tiny(targets=("q_proj", "v_proj")):
    torch.manual_seed(0)
    base = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                        num_attention_heads=2, num_key_value_heads=2))
    return get_peft_model(base, LoraConfig(r=4, target_modules=list(targets), task_type="CAUSAL_LM"))


def lora(model):
    return {k: v for k, v in model.state_dict().items() if "lora_" in k}


def test_load_init_adapter_starts_from_the_saved_weights(tmp_path):
    champion = tiny()
    for p in lora(champion).values():
        p.copy_(torch.randn_like(p))  # a fresh adapter's B is all zeros: make it look trained
    champion.save_pretrained(tmp_path)
    fresh = tiny()
    assert any(not torch.equal(v, lora(champion)[k]) for k, v in lora(fresh).items())
    load_init_adapter(fresh, tmp_path)
    assert all(torch.equal(v, lora(champion)[k]) for k, v in lora(fresh).items())


def test_load_init_adapter_refuses_an_adapter_on_other_layers(tmp_path):
    tiny(targets=("q_proj",)).save_pretrained(tmp_path)
    with pytest.raises(ValueError, match="not the same layers"):
        load_init_adapter(tiny(), tmp_path)


def test_without_the_new_options_nothing_changes(monkeypatch):
    monkeypatch.setattr("sys.argv", ["train"])
    args = parse_args()
    assert args.init_adapter is None and args.max_steps == -1
