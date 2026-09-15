from compute_concept_vector_utils import compute_concept_vector
from inject_concept_vector import inject_concept_vector
from model_registry import MODEL_REGISTRY, default_layer_pool, load_model_and_tokenizer, resolve_spec
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse
import numpy as np
from pathlib import Path
def sweep_all_layers_and_coefficients(model, tokenizer, model_name, datasets, layer_range, save_dir):
    """Sweep all layers to compute concept vectors for all concepts in all datasets"""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    for dataset_name in datasets:
        # print(f"\n{'='*80}")
        # print(f"Processing dataset: {dataset_name}")
        # print(f"{'='*80}")
        # Compute concept vectors for all layers
        for layer_idx in layer_range:
            # print(f"\n{'='*60}")
            # print(f"LAYER {layer_idx}")
            # print(f"{'='*60}")
            steering_vectors = compute_concept_vector(model, tokenizer, dataset_name, layer_idx)
            for concept_name, (vec_last, vec_avg) in steering_vectors.items():
                # print(f"\nConcept: {concept_name}")
                # Process both vec_last and vec_avg
                for vec_type, steering_vector in [("last", vec_last), ("avg", vec_avg)]:
                        # print(f"DEBUG: shape of steering vector is {steering_vector.shape}")
                        # Save all vectors with metadata
                        filename = f"{concept_name}_{layer_idx}_{vec_type}.pt"
                        filepath = save_path / filename
                        save_data = {
                            'vector': steering_vector,
                            'model_name': model_name,
                            'concept_name': concept_name,
                            'layer': layer_idx,
                            'vec_type': vec_type
                        }
                        torch.save(save_data, filepath)
def main():
    parser = argparse.ArgumentParser(description="Sweep layers and coefficients for concept vector injection")
    parser.add_argument("--model", type=str, default="llama", choices=sorted(MODEL_REGISTRY),
                       help="Which registered model to compute vectors for (code/utils/model_registry.py)")
    parser.add_argument("--datasets", type=str, nargs="+",
                       default=["simple_data", "complex_data"],
                       help="Datasets to process")
    parser.add_argument("--layer_range", type=int, nargs="+", default=None,
                       help="Layer indices to sweep (default: every layer of the selected model)")
    parser.add_argument("--save_dir", type=str, default=None,
                       help="Directory to save vectors (default: the selected model's registered vector_dir)")
    parser.add_argument("--device", type=str, choices=["auto", "mps", "cuda", "cpu"], default="auto",
                       help="'mps'/'cpu' force single-device loading for local dev (llama only -- not "
                            "large enough to hold a 27B model). 'auto'/'cuda' load via device_map='auto' "
                            "(accelerate-sharded across GPUs), required for Qwen/Qwen3.8-27B.")

    args = parser.parse_args()
    spec = resolve_spec(args.model)
    layer_range = args.layer_range if args.layer_range is not None else default_layer_pool(spec)
    save_dir = args.save_dir if args.save_dir is not None else spec.vector_dir

    print(f"Loading model: {spec.repo_id}")

    if args.device in ("mps", "cpu"):
        if spec.key != "llama":
            raise ValueError(
                f"--device {args.device} is a local-dev, single-device path only supported for "
                f"--model llama; {spec.repo_id} needs device_map='auto' sharding "
                f"(use --device auto or --device cuda)."
            )
        device = torch.device(args.device)
        if args.device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available in this PyTorch environment")
        # float16 is supported by MPS and uses substantially less memory than float32.
        model_dtype = torch.float16 if device.type == "mps" else None
        load_kwargs = {"torch_dtype": model_dtype} if model_dtype is not None else {}
        model = AutoModelForCausalLM.from_pretrained(spec.repo_id, **load_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(spec.repo_id)
        model.to(device)
        model.eval()
        print(f"Model loaded on {device}")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available in this PyTorch environment")
        model, tokenizer, _ = load_model_and_tokenizer(args.model, device_map="auto")
        print("Model loaded (device_map='auto')")

    sweep_all_layers_and_coefficients(model, tokenizer, spec.repo_id, args.datasets, layer_range, save_dir)

if __name__ == "__main__":
    main()
