#!/bin/bash
# Experiment 0 for Qwen/Qwen3.8-27B: the calibration Experiment 1 consumes.
#
# The Llama pipeline splits this across jobs/experiment_0_calibration/*.sbatch with
# afterok dependencies. Here the three stages share one allocation instead: the model
# is 52 GB of weights, so paying the load twice costs more than the plan stage saves
# by not holding a GPU.
#
#   1. prepare_concept_vectors  concept directions at all 65 hidden-state indices
#   2. prepare_material_plan    contexts, admissible token spans, direction bank
#   3. run_experiment_0         natural activations, projections, scales, bootstrap
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

echo "=== Stage 1/3: concept vectors ==="
python -m experiment_0_calibration.prepare_concept_vectors --config "${CONFIG}"

echo "=== Stage 2/3: material plan ==="
python -m experiment_0_calibration.prepare_material_plan --config "${CONFIG}"

echo "=== Stage 3/3: calibration ==="
python -m experiment_0_calibration.run_experiment_0 --config "${CONFIG}"
