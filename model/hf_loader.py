"""Loads a real HF Llama checkpoint (config.json + safetensors) into this
repo's own `LlamaConfig`/`LlamaWeights` shape (model/minimal_llama.py), so
llama_forward/reference_llama_forward can run on real weights instead of
init_weights's random ones. See model/tests/test_hf_loader.py.
"""
import json
import os

from safetensors import safe_open

from model.minimal_llama import LayerWeights, LlamaConfig, LlamaWeights


def _config_from_hf(hf_config: dict, n_layers: int = None) -> LlamaConfig:
    """Maps HF config.json fields to LlamaConfig. Read dynamically, not
    hardcoded -- every real checkpoint's config.json is the source of
    truth for its own shape.
    """
    hidden_size = hf_config["hidden_size"]
    n_heads = hf_config["num_attention_heads"]
    return LlamaConfig(
        vocab_size=hf_config["vocab_size"],
        hidden_size=hidden_size,
        intermediate_size=hf_config["intermediate_size"],
        n_layers=n_layers if n_layers is not None else hf_config["num_hidden_layers"],
        n_heads=n_heads,
        head_dim=hf_config.get("head_dim", hidden_size // n_heads),
        max_seq_len=hf_config["max_position_embeddings"],
        num_kv_heads=hf_config.get("num_key_value_heads"),  # None -> MHA, see __post_init__
        rope_theta=hf_config.get("rope_theta", 500000.0),
        rms_eps=hf_config.get("rms_norm_eps", 1e-5),
    )


class _TensorGetter:
    """Reads named tensors out of a checkpoint directory holding either a
    single `model.safetensors` or a `model.safetensors.index.json` +
    sharded `model-NNNNN-of-MMMMM.safetensors` files. Shard file handles
    are opened lazily and cached, closed on __exit__.
    """
    def __init__(self, path: str, device: str):
        self._path = path
        self._device = device
        self._handles = {}  # shard filename -> open safe_open handle

        index_path = os.path.join(path, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                self._weight_map = json.load(f)["weight_map"]  # tensor name -> shard filename
        else:
            self._weight_map = None  # single-file checkpoint, every name lives in model.safetensors

    def _handle_for(self, filename: str):
        if filename not in self._handles:
            self._handles[filename] = safe_open(
                os.path.join(self._path, filename), framework="pt", device=self._device,
            )
        return self._handles[filename]

    def __call__(self, name: str):
        filename = self._weight_map[name] if self._weight_map is not None else "model.safetensors"
        return self._handle_for(filename).get_tensor(name)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        pass  # safe_open handles have no explicit close; dropping the refs is enough


def load_hf_checkpoint(path: str, device: str = "cuda", n_layers: int = None):
    """Loads a real Llama checkpoint into this repo's (LlamaConfig,
    LlamaWeights) shape. `path` is a directory containing config.json and
    either model.safetensors or model.safetensors.index.json + shards (the
    standard HF snapshot layout -- resolving a ~/.cache/huggingface hub
    path to its snapshot dir is the caller's job, not this function's).
    `n_layers` truncates the layer count (for fast local iteration without
    reading every shard); omit it to load the real full depth.
    """
    with open(os.path.join(path, "config.json")) as f:
        hf_config = json.load(f)
    config = _config_from_hf(hf_config, n_layers=n_layers)

    with _TensorGetter(path, device) as get:
        # [vocab, hidden] -- no transpose, F.embedding wants HF's native
        # layout, same as init_weights already produces.
        embed_tokens = get("model.embed_tokens.weight")

        layers = []
        for i in range(config.n_layers):
            prefix = f"model.layers.{i}."
            # 2D weights transposed to this repo's [in, out] convention
            # (see minimal_llama.py's module docstring); 1D norm weights
            # as-is.
            layers.append(LayerWeights(
                input_layernorm_weight=get(prefix + "input_layernorm.weight"),
                q_proj=get(prefix + "self_attn.q_proj.weight").t().contiguous(),
                k_proj=get(prefix + "self_attn.k_proj.weight").t().contiguous(),
                v_proj=get(prefix + "self_attn.v_proj.weight").t().contiguous(),
                o_proj=get(prefix + "self_attn.o_proj.weight").t().contiguous(),
                post_attention_layernorm_weight=get(prefix + "post_attention_layernorm.weight"),
                gate_proj=get(prefix + "mlp.gate_proj.weight").t().contiguous(),
                up_proj=get(prefix + "mlp.up_proj.weight").t().contiguous(),
                down_proj=get(prefix + "mlp.down_proj.weight").t().contiguous(),
            ))

        norm_weight = get("model.norm.weight")
        if hf_config.get("tie_word_embeddings", False):
            # No lm_head.weight tensor in the checkpoint at all -- reuse
            # the input embedding, transposed the same way a real lm_head
            # would be.
            lm_head = embed_tokens.t().contiguous()
        else:
            lm_head = get("lm_head.weight").t().contiguous()

    # Ground truth over config.json's declared (and sometimes stale/absent)
    # torch_dtype string.
    config.dtype = embed_tokens.dtype

    weights = LlamaWeights(embed_tokens=embed_tokens, layers=layers, norm_weight=norm_weight, lm_head=lm_head)
    return config, weights
