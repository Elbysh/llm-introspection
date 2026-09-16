#!/bin/bash
# Experiment 1 on Qwen/Qwen3.8-27B, z-matched half only, against the calibration
# recomputed in the behavioural 2AFC context. Companion of
# jobs/experiment1_z_recalibrated.sbatch, which does the same on the main Llama model.
#
# Why. The z results of the earlier Qwen sweeps are not interpretable, for two
# independent reasons documented in
# docs/livrables/calibration-contexte-probleme-et-correctif.md:
#
#   1. s(l, v) was estimated on isolated sentences, whose first token sits at
#      position 0 -- a position Experiment 1 never perturbs. Fixed by
#      configs/experiment_0_calibration/qwen38_27b_2afc.yaml.
#   2. The z grid stopped at 20.48, below every control's threshold at every live
#      block. Independent of the first, and it survives the recalibration.
#
# NOTE, and it differs from Llama. Qwen never had the first defect at Llama's
# magnitude: its isolated-context s_SD/s_MAD was already 0.93-1.50, against 15-2360
# on Llama. Recalibrating still moves the scales -- concept at block 3 goes from
# 2.779 to 1.227 -- but the concept/generic ratio only falls from 15.8 to 12.6 at
# block 3 and 8.5 to 4.9 at block 6, where on Llama it collapsed to 1.5-3.0. On this
# model the concept directions' larger natural scale is mostly a real geometric
# property, not a sink artefact, so a shared z grid still has to span it.
#
# Blocks. Only the eight the alpha sweep covers: the section 8.5 coverage rule needs
# each block's own alpha threshold, and running z where no alpha counterpart exists
# would leave nothing to check against. It also keeps the run directly comparable to
# results/experiment1/qwen38_all_layers. All 64 blocks are out of budget anyway: at
# ~2.7 passes/s this model would need ~52 h.
#
# Qwen3.8-27B needs the full 80 GB card: 52 GB of bf16 weights do not fit a 3g.40gb
# MIG slice. prod80 caps memory at 120G.
#SBATCH --job-name=experiment1-z2afc-qwen
#SBATCH --partition=prod80
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --time=05:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --output=logs/experiment1-z2afc-qwen-%j.out
#SBATCH --error=logs/experiment1-z2afc-qwen-%j.err

set -euo pipefail

CONFIG=${1:-configs/experiment_0_calibration/qwen38_27b_2afc.yaml}
RUN_NAME=${2:-z2afc-qwen}
CALIBRATION_DIR=${3:-results/experiment_0_calibration_qwen38_27b_2afc}
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export HF_HUB_OFFLINE=1

# trials.csv is written only after the final trial, so a crash loses the whole run.
# Submitted in block groups and merged with code/analysis/merge_experiment1_runs.py,
# which needs every contributing run to share the z grid -- hence one grid below.
#
#   LAYERS="3 6 8"   sbatch ... '' z2afc-qwen-g0
#   LAYERS="10 12 16" ...       '' z2afc-qwen-g1
#   LAYERS="32 56"   ...        '' z2afc-qwen-g2
#
# 11 conditions x 18 doses x 40 trials = 7920 passes per block, ~49 min at 2.7
# passes/s, so ~2.5 h for a three-block group.
LAYERS=${LAYERS:-"3 6 8"}

# From
#   python code/analysis/plan_z_grid.py \
#       --calibration_dir results/experiment_0_calibration_qwen38_27b_2afc \
#       --alpha_trials results/experiment1/qwen38_all_layers/trials.csv \
#       --layers 3 6 8 10 12 16 32 56 --alpha_low 0.0625 --alpha_high 64
# which reports 0 uncovered (block, family) cells. The alpha grid it must match is
# this model's own, 0.0625 to 64, not Llama's 0.25 to 128.
#
# Binding constraints:
#   z_max >= 749  block 3 scrambled/noise, whose alpha curves sit at or past the top
#                 of the alpha grid, so the z grid must deliver at least 64 against
#                 s = 0.085 before a flat curve means anything.
#   z_min <= 0.0097  block 56 concept, s = 6.449, to reach the alpha grid's floor.
#
# The deep blocks drive the bottom of the ladder and the shallow ones its top; that
# is the price of one grid shared by eight blocks whose scales span 75-fold.
Z_DOSES=${Z_DOSES:-"0.0078125 0.015625 0.03125 0.0625 0.125 0.25 0.5 1 2 4 8 16 32 64 128 256 512 1024"}

python code/experiments/experiment1_psychometrics.py \
  --calibration_config "${CONFIG}" \
  --calibration_dir "${CALIBRATION_DIR}" \
  --run_name "${RUN_NAME}" \
  --layers ${LAYERS} \
  --families ${FAMILIES:-concept random noise dropout} \
  --matchings z \
  --concepts Dust Satellites recursion shutdown \
  --num_pairs 5 \
  --num_random 3 \
  --num_noise 2 \
  --num_dropout 2 \
  --label_orders AB BA \
  --z_doses ${Z_DOSES} \
  --contamination_trials 24 \
  --no_progress
