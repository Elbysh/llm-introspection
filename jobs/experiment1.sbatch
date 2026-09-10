#!/bin/bash
# Experiment 1: 2AFC localization, alpha- and z-matched psychometric curves.
# Needs the direction bank and the calibration of the same config to exist first:
#   PYTHONPATH=code python -m calibration.directions     --config <config>
#   PYTHONPATH=code python -m calibration.compute_scales --config <config>
#SBATCH --job-name=experiment1
#SBATCH --partition=prod40
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --output=logs/experiment1-%j.out
#SBATCH --error=logs/experiment1-%j.err

set -euo pipefail

CONFIG=${1:-configs/calibration/full.yaml}
RUN_NAME=${2:-main}
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export HF_HUB_OFFLINE=1

# The two grids are not comparable numerically. Concept directions have a natural
# scale of 5-200 at these layers while random and noise directions sit near 1, so a
# single z grid has to span three decades to bracket every family's transition.
python code/experiments/experiment1_psychometrics.py \
  --calibration_config "${CONFIG}" \
  --run_name "${RUN_NAME}" \
  --families concept random noise dropout \
  --matchings alpha z \
  --num_pairs 5 \
  --num_random 3 \
  --num_noise 2 \
  --num_dropout 2 \
  --label_orders AB BA \
  --alpha_doses 0.25 0.5 1 2 4 8 16 32 64 128 \
  --z_doses 0.01 0.02 0.04 0.08 0.16 0.32 0.64 1.28 2.56 5.12 10.24 20.48 \
  --contamination_trials 24 \
  --no_progress
