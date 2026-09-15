#!/bin/bash
# Experiment 0 for Qwen/Qwen3.8-27B: the calibration Experiment 1 consumes.
#
# The Llama pipeline splits this across jobs/experiment_0_calibration/*.sbatch with
# afterok dependencies. Here the three stages share one allocation instead: the model
# is 52 GB of weights, so paying the load twice costs more than the plan stage saves
# by not holding a GPU.
#
#   1. save_random_vectors      the persisted fixed-random bank of doc 3.4
#   2. prepare_concept_vectors  concept directions at all 64 hidden-state indices
#   3. prepare_material_plan    contexts, admissible token spans, direction bank
#   4. run_experiment_0         natural activations, projections, scales, bootstrap
#
# Qwen3.8-27B needs the full 80 GB card: the 3g.40gb MIG slice of prod40 cannot hold
# bf16 weights of this size. prod80 caps memory at 120G (MaxMemPerNode=122880).
#SBATCH --job-name=exp0-qwen38
#SBATCH --partition=prod80
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --output=logs/experiment_0-qwen38-%j.out
#SBATCH --error=logs/experiment_0-qwen38-%j.err

set -euo pipefail

CONFIG=${1:-configs/experiment_0_calibration/qwen38_27b_full.yaml}
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export PYTHONPATH="${PWD}/code${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1

# The fixed-random directions are read from disk, not redrawn, so that Experiment 1
# can rebuild the exact vector each scale was estimated on. data/saved_vectors/llama
# ships that bank for the main model; a new model has none, and prepare_material_plan
# fails on the first missing random_s0_0_avg.pt. The `linear` scheme and the seed and
# stride below are what prepare_material.py:materialize_direction reproduces, and
# they must stay equal to the config's directions.fixed_random block.
echo "=== Stage 1/4: fixed-random direction bank ==="
if [[ -f data/saved_vectors/qwen38_27b/random_s0_0_avg.pt ]]; then
  echo "Fixed-random bank already present, skipping"
else
  # --hidden_dim is left to be inferred from a concept vector in the same directory
  # when one exists, which is the point: the bank cannot disagree with it.
  python code/utils/save_random_vectors.py \
    --seed_scheme linear \
    --seed 2026091001 \
    --layer_stride 100000 \
    --num_samples 10 \
    --hidden_dim 5120 \
    --vec_type avg \
    --model Qwen/Qwen3.8-27B \
    --save_dir data/saved_vectors/qwen38_27b \
    --layer_range $(seq 0 63)
fi

echo "=== Stage 2/4: concept vectors ==="
python -m experiment_0_calibration.prepare_concept_vectors --config "${CONFIG}"

echo "=== Stage 3/4: material plan ==="
python -m experiment_0_calibration.prepare_material_plan --config "${CONFIG}"

echo "=== Stage 4/4: calibration ==="
python -m experiment_0_calibration.run_experiment_0 --config "${CONFIG}"
