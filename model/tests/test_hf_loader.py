"""Correctness tests for model/hf_loader.py: does it map a real HF Llama
checkpoint's tensor names/shapes onto this repo's LlamaConfig/LlamaWeights
correctly? Two tiers:

  - CPU-only, always runs, no download needed: a tiny synthetic HF-shaped
    checkpoint (written with safetensors.torch.save_file, not a real
    model) checks the name-mapping/transpose logic directly -- tied and
    untied embeddings, a GQA shape (num_key_value_heads < num_attention_
    heads), the n_layers-truncation override, and the sharded
    model.safetensors.index.json layout large real checkpoints use.
  - requires_cuda, skipped unless a real checkpoint is present on disk:
    loads it for real and checks the forward pass runs and is stable.
"""
import json
import os

import pytest
import torch
from safetensors.torch import save_file

from model.hf_loader import load_hf_checkpoint
from model.minimal_llama import reference_llama_forward

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the real-checkpoint smoke test needs a CUDA GPU"
)

# The real Llama-3-8B-Instruct checkpoint lives in the L40S's HF hub cache
# (already downloaded for the vLLM/SGLang benchmark containers -- see
# ~/.claude/plans/agile-rolling-gray.md's Context section). Resolved by
# glob, not a hardcoded snapshot hash, so a re-download under a new commit
# hash doesn't break this.
_LLAMA3_8B_HUB_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct"
)


def _find_real_checkpoint_dir():
    snapshots_dir = os.path.join(_LLAMA3_8B_HUB_DIR, "snapshots")
    if not os.path.isdir(snapshots_dir):
        return None
    for name in os.listdir(snapshots_dir):
        candidate = os.path.join(snapshots_dir, name)
        if os.path.exists(os.path.join(candidate, "config.json")):
            return candidate
    return None


_REAL_CHECKPOINT_DIR = _find_real_checkpoint_dir()


def _write_fake_checkpoint(tmp_path, *, hidden_size, intermediate_size, n_layers, n_heads,
                            num_key_value_heads, head_dim, vocab_size, tie_word_embeddings):
    hf_config = {
        "hidden_size": hidden_size, "intermediate_size": intermediate_size,
        "num_hidden_layers": n_layers, "num_attention_heads": n_heads,
        "num_key_value_heads": num_key_value_heads, "head_dim": head_dim,
        "vocab_size": vocab_size, "max_position_embeddings": 128,
        "rope_theta": 500000.0, "rms_norm_eps": 1e-5,
        "tie_word_embeddings": tie_word_embeddings,
    }
    (tmp_path / "config.json").write_text(json.dumps(hf_config))

    qkv_dim, kv_dim = n_heads * head_dim, num_key_value_heads * head_dim
    tensors = {"model.embed_tokens.weight": torch.randn(vocab_size, hidden_size)}
    for i in range(n_layers):
        p = f"model.layers.{i}."
        tensors[p + "input_layernorm.weight"] = torch.randn(hidden_size)
        tensors[p + "self_attn.q_proj.weight"] = torch.randn(qkv_dim, hidden_size)
        tensors[p + "self_attn.k_proj.weight"] = torch.randn(kv_dim, hidden_size)
        tensors[p + "self_attn.v_proj.weight"] = torch.randn(kv_dim, hidden_size)
        tensors[p + "self_attn.o_proj.weight"] = torch.randn(hidden_size, qkv_dim)
        tensors[p + "post_attention_layernorm.weight"] = torch.randn(hidden_size)
        tensors[p + "mlp.gate_proj.weight"] = torch.randn(intermediate_size, hidden_size)
        tensors[p + "mlp.up_proj.weight"] = torch.randn(intermediate_size, hidden_size)
        tensors[p + "mlp.down_proj.weight"] = torch.randn(hidden_size, intermediate_size)
    tensors["model.norm.weight"] = torch.randn(hidden_size)
    if not tie_word_embeddings:
        tensors["lm_head.weight"] = torch.randn(vocab_size, hidden_size)

    save_file(tensors, str(tmp_path / "model.safetensors"))
    return hf_config, tensors


@pytest.mark.parametrize("tie_word_embeddings", [True, False])
def test_maps_names_and_transposes_correctly(tmp_path, tie_word_embeddings):
    # n_heads=4/num_key_value_heads=2 -- exercises GQA name-mapping too.
    hf_config, raw = _write_fake_checkpoint(
        tmp_path, hidden_size=32, intermediate_size=48, n_layers=2, n_heads=4,
        num_key_value_heads=2, head_dim=8, vocab_size=64, tie_word_embeddings=tie_word_embeddings,
    )

    config, weights = load_hf_checkpoint(str(tmp_path), device="cpu")

    assert config.vocab_size == hf_config["vocab_size"]
    assert config.hidden_size == hf_config["hidden_size"]
    assert config.intermediate_size == hf_config["intermediate_size"]
    assert config.n_layers == hf_config["num_hidden_layers"]
    assert config.n_heads == hf_config["num_attention_heads"]
    assert config.num_kv_heads == hf_config["num_key_value_heads"]
    assert config.head_dim == hf_config["head_dim"]
    assert config.dtype == raw["model.embed_tokens.weight"].dtype  # ground truth over config.json's torch_dtype

    assert torch.equal(weights.embed_tokens, raw["model.embed_tokens.weight"])
    assert torch.equal(weights.norm_weight, raw["model.norm.weight"])
    if tie_word_embeddings:
        assert torch.equal(weights.lm_head, raw["model.embed_tokens.weight"].t())
    else:
        assert torch.equal(weights.lm_head, raw["lm_head.weight"].t())

    for i, layer in enumerate(weights.layers):
        p = f"model.layers.{i}."
        assert torch.equal(layer.input_layernorm_weight, raw[p + "input_layernorm.weight"])
        assert torch.equal(layer.q_proj, raw[p + "self_attn.q_proj.weight"].t())
        assert torch.equal(layer.k_proj, raw[p + "self_attn.k_proj.weight"].t())
        assert torch.equal(layer.v_proj, raw[p + "self_attn.v_proj.weight"].t())
        assert torch.equal(layer.o_proj, raw[p + "self_attn.o_proj.weight"].t())
        assert torch.equal(layer.post_attention_layernorm_weight, raw[p + "post_attention_layernorm.weight"])
        assert torch.equal(layer.gate_proj, raw[p + "mlp.gate_proj.weight"].t())
        assert torch.equal(layer.up_proj, raw[p + "mlp.up_proj.weight"].t())
        assert torch.equal(layer.down_proj, raw[p + "mlp.down_proj.weight"].t())


def test_n_layers_override_truncates(tmp_path):
    _write_fake_checkpoint(
        tmp_path, hidden_size=16, intermediate_size=24, n_layers=4, n_heads=2,
        num_key_value_heads=2, head_dim=8, vocab_size=32, tie_word_embeddings=True,
    )
    config, weights = load_hf_checkpoint(str(tmp_path), device="cpu", n_layers=2)
    assert config.n_layers == 2
    assert len(weights.layers) == 2


def test_sharded_index_checkpoint(tmp_path):
    """Same tensors as the untied-embeddings case above, but split across
    two shard files with a model.safetensors.index.json -- the other
    on-disk layout real (multi-shard) checkpoints use.
    """
    hf_config, raw = _write_fake_checkpoint(
        tmp_path, hidden_size=16, intermediate_size=24, n_layers=2, n_heads=2,
        num_key_value_heads=2, head_dim=8, vocab_size=32, tie_word_embeddings=False,
    )
    (tmp_path / "model.safetensors").unlink()  # replace the single-file layout with shards

    names = list(raw.keys())
    half = len(names) // 2
    shard_names = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    weight_map = {}
    for shard_name, chunk in zip(shard_names, [names[:half], names[half:]]):
        save_file({n: raw[n] for n in chunk}, str(tmp_path / shard_name))
        weight_map.update({n: shard_name for n in chunk})
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))

    config, weights = load_hf_checkpoint(str(tmp_path), device="cpu")
    assert torch.equal(weights.embed_tokens, raw["model.embed_tokens.weight"])
    assert torch.equal(weights.lm_head, raw["lm_head.weight"].t())


def _greedy_generate(weights, config, prompt_ids, num_new_tokens):
    tokens = list(prompt_ids)
    for _ in range(num_new_tokens):
        input_ids = torch.tensor([tokens], device="cuda")
        logits = reference_llama_forward(weights, config, input_ids)
        assert torch.isfinite(logits).all()
        tokens.append(int(logits[0, -1].argmax().item()))
    return tokens


@requires_cuda
@pytest.mark.skipif(_REAL_CHECKPOINT_DIR is None, reason="real Llama-3-8B-Instruct checkpoint not found on disk")
def test_real_checkpoint_smoke():
    """Loads the real Llama-3-8B-Instruct checkpoint and greedy-generates a
    few tokens with reference_llama_forward (dense, no KV cache -- see its
    own docstring) from an arbitrary real token-id prompt. Tokenizer-free
    per the master plan's assumption 5, so this makes no "correct English"
    claim -- it's a systems check: does the loaded checkpoint forward-pass
    without NaN/Inf, and is the greedy continuation stable across two
    independent runs from scratch.
    """
    config, weights = load_hf_checkpoint(_REAL_CHECKPOINT_DIR, device="cuda")
    assert (config.n_layers, config.hidden_size, config.n_heads, config.num_kv_heads) == (32, 4096, 32, 8)

    prompt_ids = [128000, 9906, 1917, 11, 420, 374, 264, 1296]  # arbitrary ids within vocab_size, not tokenized text
    continuation_a = _greedy_generate(weights, config, prompt_ids, num_new_tokens=3)
    continuation_b = _greedy_generate(weights, config, prompt_ids, num_new_tokens=3)
    assert continuation_a == continuation_b
