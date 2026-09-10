#!/usr/bin/env python3
"""
Build and save random direction vectors, the control family for the concept vectors.

Concept vectors come from save_vectors.py and are not handled here. Files use the same
layout and schema as save_vectors.py (<save_dir>/<name>_<layer>_<vec_type>.pt), so they
are drop-in for the existing experiments:

    python code/experiments/position_detection.py --concept random_s0 --strengths 1 2 3 4 5

Two seeding schemes are available. `blake2b` (default) hashes (seed, family, layer, sample)
so every draw is independent. `linear` reproduces the fixed-random bank of Experiment 0,
whose seeds are `seed + layer * layer_stride + sample` and whose vectors are the CPU
Gaussian draws normalised exactly (no epsilon), as in
code/experiment_0_calibration/prepare_material.py:materialize_direction. Regenerating the
Experiment 0 bank as files, with `random_sI_L_avg.pt` holding `fixed_random__block_L__000I`:

    python code/utils/save_random_vectors.py --seed_scheme linear --seed 2026091001 \
        --num_samples 10 --layer_range $(seq 0 31)
"""

import argparse
import hashlib
from pathlib import Path

import torch

DEFAULT_HIDDEN_DIM = 4096

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAVE_DIR = _REPO_ROOT / "data" / "saved_vectors" / "llama"


DEFAULT_LAYER_STRIDE = 100_000


def derive_seed(base_seed, *args):
    """Deterministic sub-seed, so each (family, layer, sample) draws independently."""
    key = "|".join(str(p) for p in (base_seed, *args)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(key, digest_size=7).digest(), "big")


def linear_seed(base_seed, layer_idx, sample, layer_stride=DEFAULT_LAYER_STRIDE):
    """Experiment 0's fixed-random seed: base + layer * stride + sample."""
    return int(base_seed) + int(layer_idx) * int(layer_stride) + int(sample)


def unit_normalize(vector, epsilon=1e-6):
    """L2-normalise in float32 (bfloat16 normalisation is ~0.3% off unit).

    Experiment 0 divides by the exact norm; pass epsilon=0.0 to match it bit for bit.
    """
    v = vector.to(torch.float32)
    return (v / (v.norm(p=2) + epsilon)).to(vector.dtype)


def make_random_direction(hidden_dim, seed, epsilon=1e-6):
    """Unit-norm isotropic direction, matching the concept vectors' dimension and norm."""
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    return unit_normalize(
        torch.randn(hidden_dim, generator=gen, dtype=torch.float32), epsilon=epsilon
    )


def infer_hidden_dim(save_dir, fallback=DEFAULT_HIDDEN_DIM):
    """Read the hidden dim off a vector already in save_dir, so dimensions can't mismatch."""
    for path in sorted(Path(save_dir).glob("*.pt")):
        try:
            data = torch.load(path, weights_only=False)
        except Exception:
            continue
        vector = data.get("vector") if isinstance(data, dict) else data
        if isinstance(vector, torch.Tensor) and vector.numel() > 1:
            return vector.reshape(-1).numel(), path.name
    return fallback, None


def save_random_directions(hidden_dim, layer_range, save_dir, seed, num_samples,
                           model_name, vec_type, seed_scheme="blake2b",
                           layer_stride=DEFAULT_LAYER_STRIDE):
    """Generate and save one random direction per (layer, sample)."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    # Experiment 0 normalises by the exact norm, so the epsilon has to go for the
    # regenerated bank to equal the vectors its s(l, v) were measured on.
    epsilon = 0.0 if seed_scheme == "linear" else 1e-6

    written = []
    for layer_idx in layer_range:
        for i in range(num_samples):
            name = f"random_s{i}"
            if seed_scheme == "linear":
                draw_seed = linear_seed(seed, layer_idx, i, layer_stride)
            else:
                draw_seed = derive_seed(seed, "random", layer_idx, i)
            vector = make_random_direction(hidden_dim, draw_seed, epsilon=epsilon)
            filename = f"{name}_{layer_idx}_{vec_type}.pt"
            payload = {
                "vector": vector,
                "model_name": model_name,
                "concept_name": name,
                "layer": layer_idx,
                "vec_type": vec_type,
                "family": "random",
                "seed": seed,
                "seed_scheme": seed_scheme,
                "draw_seed": draw_seed,
                "sample": i,
                "hidden_dim": hidden_dim,
            }
            if seed_scheme == "linear":
                # `layer` is Experiment 0's decoder_block_index; its concept files are
                # named by hidden_state_index = block + 1, so record both explicitly.
                payload["decoder_block_index"] = layer_idx
                payload["hidden_state_index"] = layer_idx + 1
                payload["experiment_0_direction_id"] = (
                    "fixed_random__block_{:02d}__{:04d}".format(layer_idx, i)
                )
            torch.save(payload, save_path / filename)
            written.append(filename)

    return written


def main():
    parser = argparse.ArgumentParser(description="Save random direction control vectors")
    parser.add_argument("--layer_range", type=int, nargs="+", default=list(range(32)),
                        help="Layer indices to generate (default: 0-31)")
    parser.add_argument("--save_dir", type=str, default=str(DEFAULT_SAVE_DIR),
                        help=f"Directory to save vectors (default: {DEFAULT_SAVE_DIR})")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base seed; per-layer seeds are derived from it")
    parser.add_argument("--seed_scheme", type=str, default="blake2b",
                        choices=["blake2b", "linear"],
                        help="blake2b: hash (seed, layer, sample). linear: Experiment 0's "
                             "seed + layer * --layer_stride + sample, normalised exactly")
    parser.add_argument("--layer_stride", type=int, default=DEFAULT_LAYER_STRIDE,
                        help="Per-layer seed stride for --seed_scheme linear")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Independent directions per layer (random_s0 ... random_sN)")
    parser.add_argument("--hidden_dim", type=int, default=None,
                        help="Hidden dimension (default: inferred from vectors in --save_dir)")
    parser.add_argument("--vec_type", type=str, default="avg",
                        help="Suffix matching the experiments' --vec_type")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Model name recorded in the saved metadata")
    args = parser.parse_args()

    if args.hidden_dim is not None:
        hidden_dim, source = args.hidden_dim, "--hidden_dim"
    else:
        hidden_dim, ref = infer_hidden_dim(args.save_dir)
        source = f"inferred from {ref}" if ref else f"default (no vectors in {args.save_dir})"

    print(f"Hidden dim:  {hidden_dim} ({source})")
    print(f"Layers:      {args.layer_range}")
    print(f"Seed:        {args.seed} ({args.seed_scheme})")
    print(f"Save dir:    {args.save_dir}")

    written = save_random_directions(
        hidden_dim=hidden_dim,
        layer_range=args.layer_range,
        save_dir=args.save_dir,
        seed=args.seed,
        num_samples=args.num_samples,
        model_name=args.model,
        vec_type=args.vec_type,
        seed_scheme=args.seed_scheme,
        layer_stride=args.layer_stride,
    )

    names = sorted({f.rsplit("_", 2)[0] for f in written})
    print(f"\nWrote {len(written)} files under {len(names)} names: {names}")
    print(f"Use e.g.: --concept {names[0]}")


if __name__ == "__main__":
    main()
