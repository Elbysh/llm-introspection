#!/bin/bash
# Experiment 1 on Qwen/Qwen3.8-27B: 2AFC localization, alpha- and z-matched
# psychometric curves. Section 3.1 of docs/livrables/cadrage-experiments.md lists
# this model as the secondary one, to be run once the protocol is validated on
# Llama-3.1-8B-Instruct; jobs/experiment1.sbatch is that Llama run.
#
# Consumes the Qwen Experiment 0 calibration: the direction bank and the scales
# s(l, v) it estimated. Produce it first with
#   sbatch jobs/experiment_0_calibration_qwen38_27b.sbatch
# and pass its paths.output_dir as the third argument.
#
# Qwen3.8-27B needs the full 80 GB card: 52 GB of bf16 weights do not fit the
# 3g.40gb MIG slice that jobs/experiment1.sbatch uses for the 8B model. prod80 caps
# memory at 120G (MaxMemPerNode=122880).
#SBATCH --job-name=experiment1-qwen38
#SBATCH --partition=prod80
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --output=logs/experiment1-qwen38-%j.out
#SBATCH --error=logs/experiment1-qwen38-%j.err

set -euo pipefail

CONFIG=${1:-configs/experiment_0_calibration/qwen38_27b_full.yaml}
RUN_NAME=${2:-qwen38-main}
CALIBRATION_DIR=${3:-results/experiment_0_calibration_qwen38_27b}
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export HF_HUB_OFFLINE=1

# Experiment 0 calibrates all 64 decoder blocks. The full sweep over them is far
# beyond one job, so pick the blocks the analysis needs. These are the Llama job's
# 3/16/28 rescaled to twice the depth: the same ~10%, 50% and 87% of the stack.
# All three are linear_attention blocks under Qwen3.5's 4-block hybrid pattern
# (full attention sits at every index = 3 mod 4), so block type does not vary with
# depth across the three. The last block is excluded on purpose: no attention layer
# follows it, so a perturbation at sentence tokens cannot reach the answer token.
LAYERS=${LAYERS:-"6 32 56"}

# The two grids are not comparable numerically and neither transfers from the Llama
# job unchanged: alpha is a raw activation norm, and Qwen's differ from Llama's. The
# defaults below are set from this model's own calibrated scales, printed by
#   python code/analysis/summarize_calibration_scales.py \
#       --calibration_dir results/experiment_0_calibration_qwen38_27b --layers 6 32 56
# Re-check them against that table before trusting a sweep.
ALPHA_DOSES=${ALPHA_DOSES:-"0.25 0.5 1 2 4 8 16 32 64 128"}
Z_DOSES=${Z_DOSES:-"0.01 0.02 0.04 0.08 0.16 0.32 0.64 1.28 2.56 5.12 10.24 20.48"}

python code/experiments/experiment1_psychometrics.py \
  --calibration_config "${CONFIG}" \
  --calibration_dir "${CALIBRATION_DIR}" \
  --run_name "${RUN_NAME}" \
  --layers ${LAYERS} \
  --families concept random noise dropout \
  --matchings alpha z \
  --concepts Dust Satellites recursion shutdown \
  --num_pairs 5 \
  --num_random 3 \
  --num_noise 2 \
  --num_dropout 2 \
  --label_orders AB BA \
  --alpha_doses ${ALPHA_DOSES} \
  --z_doses ${Z_DOSES} \
  --contamination_trials 24 \
  --no_progress
