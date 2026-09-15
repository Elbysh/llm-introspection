"""
Single source of truth for everything that differs between models the
experiment scripts can run against. Every script resolves a `--model` CLI
argument through this registry instead of hardcoding a repo id, decoder-layer
attribute path, saved-vector directory, or layer count.

meta-llama/Llama-3.1-8B-Instruct: plain causal LM, 32 layers, decoder stack
at model.model.layers, loaded via AutoModelForCausalLM/AutoTokenizer.

Qwen/Qwen3.8-27B: NOT a plain causal LM -- it is a native vision-language
model (architectures: ["Qwen3_5ForConditionalGeneration"], model_type
"qwen3_5"), 64 layers, hidden_size 5120, with a hybrid 3:1 linear-attention
(Gated DeltaNet) / full-attention stack, all sharing one Qwen3_5DecoderLayer
class whose forward() returns a bare hidden_states tensor (not a tuple like
LlamaDecoderLayer). Its decoder layers live at
model.model.language_model.layers, not model.model.layers. Loaded via
Qwen3_5ForConditionalGeneration + AutoProcessor (confirmed against
transformers' modeling_qwen3_5.py on GitHub main; the vision tower is
irrelevant here since every experiment in this repo is text-only).
"""

from dataclasses import dataclass
from typing import Callable

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, PreTrainedModel


@dataclass
class ModelSpec:
    key: str
    repo_id: str
    vector_dir: str
    num_layers: int
    get_layers: Callable[[PreTrainedModel], "torch.nn.ModuleList"]


MODEL_REGISTRY = {
    "llama": ModelSpec(
        key="llama",
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        vector_dir="saved_vectors/llama",
        num_layers=32,
        get_layers=lambda model: model.model.layers,
    ),
    "qwen": ModelSpec(
        key="qwen",
        repo_id="Qwen/Qwen3.8-27B",
        vector_dir="saved_vectors/qwen",
        num_layers=64,
        get_layers=lambda model: model.model.language_model.layers,
    ),
}


def resolve_spec(model_key: str) -> ModelSpec:
    try:
        return MODEL_REGISTRY[model_key]
    except KeyError as exc:
        raise KeyError(
            f"Unknown model key {model_key!r}; choices are {sorted(MODEL_REGISTRY)}"
        ) from exc


def default_layer_pool(spec: ModelSpec, step: int = 1) -> list:
    """Layer indices 0..num_layers-1 stepped by `step`. Pass step=2 to match
    calibration.py's coarser Llama sweep (16 of 32 layers); the Qwen
    equivalent (step=2 over 64 layers) then also samples 32 layers, keeping
    sweep density comparable across models rather than silently doubling it."""
    return list(range(0, spec.num_layers, step))


def load_model_and_tokenizer(model_key: str, device_map: str = "auto"):
    """Loads the model + a tokenizer-like object for the given model key.

    Returns (model, tokenizer, spec). `tokenizer` always exposes
    .apply_chat_template / .encode / .decode / __call__ the same way for
    both models, so every existing call site (chat templating,
    tokenizer.encode("YES")[0] for logit lookups, etc.) works unchanged
    regardless of --model: for qwen this is AutoProcessor(...).tokenizer,
    not the full multimodal processor.
    """
    spec = resolve_spec(model_key)

    if spec.key == "llama":
        tokenizer = AutoTokenizer.from_pretrained(spec.repo_id)
        model = AutoModelForCausalLM.from_pretrained(
            spec.repo_id, torch_dtype=torch.bfloat16, device_map=device_map
        )
    elif spec.key == "qwen":
        try:
            from transformers import Qwen3_5ForConditionalGeneration
        except ImportError as exc:
            raise ImportError(
                "This transformers install doesn't expose "
                "Qwen3_5ForConditionalGeneration -- Qwen/Qwen3.8-27B needs a "
                "transformers release with qwen3_5 support merged. Check "
                "`python -c \"from transformers import "
                "Qwen3_5ForConditionalGeneration\"` before running any real job."
            ) from exc
        processor = AutoProcessor.from_pretrained(spec.repo_id)
        tokenizer = processor.tokenizer
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            spec.repo_id, torch_dtype=torch.bfloat16, device_map=device_map
        )
    else:
        raise KeyError(f"No loader configured for model key {spec.key!r}")

    model.eval()
    return model, tokenizer, spec


def build_inputs(tokenizer, formatted_prompt: str, device):
    """tokenizer(...).to(device), with a defensive drop of token_type_ids --
    harmless no-op for a plain-tokenizer/Llama call, but matches the
    official Qwen3-VL usage pattern (image-capable processors can emit
    token_type_ids that plain generate() doesn't accept) for the qwen path,
    where `tokenizer` is a processor's .tokenizer."""
    inputs = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False)
    inputs.pop("token_type_ids", None)
    return inputs.to(device)
