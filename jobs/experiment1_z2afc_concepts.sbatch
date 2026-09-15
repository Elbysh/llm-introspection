#!/bin/bash
# Experiment 1, z-matched, CONCEPT FAMILY ONLY, for the concepts the main sweep left out.
#
# Why this job exists. Doc de cadrage section 7.3 (Experience 3) asks for at least five
# concepts and the 32 blocks. jobs/experiment1_z_recalibrated.sbatch swept four --
# Dust, Satellites, recursion, shutdown -- because line 99 names them, not because
# anything else was missing: results/experiment_0_calibration_2afc carries s(l,v) for
# all ten concepts at all 32 blocks, and data/saved_vectors/llama/ carries their
# vectors at blocks 0-32, versioned. Only the forward passes are absent.
#
# So this sweeps the remaining concepts on the SAME grid, the SAME calibration and the
# SAME 32 blocks, and the result is pooled with results/experiment1/z2afc_all at
# analysis time rather than merged into it:
#
#   python code/analysis/experiment3_variability.py \
#       --run_dir results/experiment1/z2afc_all \
#       --run_dir results/experiment1/z2afc-concepts-a \
#       --run_dir results/experiment1/z2afc-concepts-b \
#       --matching z --out_dir results/experiment3/llama_z_10concepts
#
# --families concept only. The random, noise and dropout cells already exist in
# z2afc_all for every block, and re-running them would put two rows under each pairing
# key of doc 5.8, where experiment3_variability.py:paired_s requires exactly two per
# (target A, target B) pair and silently drops any key carrying four. Do not add
# --num_random here to extend the random family: --num_random takes the FIRST n
# calibrated ids, so it cannot select 0003-0004 without re-running 0000-0002. Extending
# the random family needs an id-selection flag in experiment1_psychometrics.py first.
#
# The z grid must stay byte-identical to the one in experiment1_z_recalibrated.sbatch:
# pooling curves fitted on different dose grids is meaningless, and the analysis keys
# its dense (direction, layer, phrase) design on a shared dose set.
#
# Cost: 1 condition x 18 doses x 40 trials (5 pairs x 2 orders x 2 label orders x 2
# targets) = 720 passes per block, 23040 per concept over 32 blocks. Measured rate on
# this slice is ~39 passes/s (jobs 8255-8258: 63380 trials in 27 min), so ~10 min of
# compute per concept plus model load. Three concepts per job is ~35 min.
#
# An 8B model in bf16 fits the 3g.40gb MIG slice; prod40 caps memory at 60G.
#SBATCH --job-name=experiment1-z2afc-concepts
#SBATCH --partition=prod40
#SBATCH --gres=gpu:nvidia_a100_3g.40gb:1
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --output=logs/experiment1-z2afc-concepts-%j.out
#SBATCH --error=logs/experiment1-z2afc-concepts-%j.err

set -euo pipefail

CONFIG=${1:-configs/experiment_0_calibration/llama_2afc_full.yaml}
RUN_NAME=${2:-z2afc-concepts}
CALIBRATION_DIR=${3:-results/experiment_0_calibration_2afc}
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export HF_HUB_OFFLINE=1

# trials.csv is written only after the final trial, so a crash loses the whole run.
# Split the concepts across jobs rather than the blocks: every concept needs all 32
# blocks for the layer component, and two jobs is what the queue's QOS cap runs at once.
#
#   CONCEPTS="betrayal"                                  ... '' z2afc-betrayal
#   CONCEPTS="appreciation betrayal fibonacci_numbers"   ... '' z2afc-concepts-a
#   CONCEPTS="Illusions Origami Trumpets"                ... '' z2afc-concepts-b
CONCEPTS=${CONCEPTS:-"betrayal"}
LAYERS=${LAYERS:-"0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31"}

# Identical to experiment1_z_recalibrated.sbatch. See that file for why these 18 doses
# and why the extremes are deliberately out of range at some blocks.
Z_DOSES=${Z_DOSES:-"0.0625 0.125 0.25 0.5 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192"}

echo "Concepts: ${CONCEPTS}"
echo "Layers:   ${LAYERS}"

python code/experiments/experiment1_psychometrics.py \
  --calibration_config "${CONFIG}" \
  --calibration_dir "${CALIBRATION_DIR}" \
  --run_name "${RUN_NAME}" \
  --layers ${LAYERS} \
  --families concept \
  --matchings z \
  --concepts ${CONCEPTS} \
  --num_pairs 5 \
  --label_orders AB BA \
  --z_doses ${Z_DOSES} \
  --contamination_trials 24 \
  --no_progress
