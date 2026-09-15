#!/bin/bash
# Experiment 1: 2AFC localization, alpha- and z-matched psychometric curves.
# Consumes the Experiment 0 calibration: the direction bank and the scales
# s(l, v) it estimated. The committed development calibration is the default;
# for a fresh one run, in order:
#   python -m experiment_0_calibration.prepare_concept_vectors --config <config>
#   python -m experiment_0_calibration.prepare_material_plan   --config <config>
#   python -m experiment_0_calibration.run_experiment_0        --config <config>
# and pass its paths.output_dir as the third argument.
#SBATCH --job-name=experiment1
#SBATCH --partition=prod40
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --output=logs/experiment1-%j.out
#SBATCH --error=logs/experiment1-%j.err

set -euo pipefail

CONFIG=${1:-configs/experiment_0_calibration/development_full.yaml}
RUN_NAME=${2:-main}
CALIBRATION_DIR=${3:-results/experiment_0_calibration}
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export HF_HUB_OFFLINE=1

# Experiment 0 calibrates all 32 decoder blocks. The full sweep over them is far
# beyond one job, so pick the blocks the analysis needs.
LAYERS=${LAYERS:-"3 16 28"}

# The two grids are not comparable numerically. At these blocks a concept direction
# has a natural scale of 20-200 while random and noise directions sit near 1-3, so a
# single z grid has to span three decades to bracket every family's transition.
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
  --alpha_doses 0.25 0.5 1 2 4 8 16 32 64 128 \
  --z_doses 0.01 0.02 0.04 0.08 0.16 0.32 0.64 1.28 2.56 5.12 10.24 20.48 \
  --contamination_trials 24 \
  --no_progress
