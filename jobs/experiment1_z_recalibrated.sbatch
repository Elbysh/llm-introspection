#!/bin/bash
# Experiment 1 on Llama-3.1-8B-Instruct, z-matched half only, against the
# calibration recomputed in the behavioural 2AFC context.
#
# Why this job exists rather than a rerun of jobs/experiment1.sbatch. The z results
# of the full32_all sweep are not interpretable, for two independent reasons, both
# in docs/livrables/calibration-contexte-probleme-et-correctif.md:
#
#   1. s(l, v) was estimated on isolated sentences, whose first token sits at
#      position 0 -- the attention-sink slot Llama fills with an activation two to
#      three orders of magnitude above every other token, and a position Experiment 1
#      never perturbs. z = alpha / s(l, v), so every z dose inherited the defect.
#      Fixed by configs/experiment_0_calibration/llama_2afc_full.yaml, which
#      calibrates on the exact 2AFC prompt.
#   2. The z grid stopped at 20.48, which with correct scales tops out near alpha 5,
#      below every family's threshold at every live block. That defect is independent
#      of the first and survives the recalibration (section 8.5).
#
# The alpha half of full32_all invokes no calibrated scale and is untouched, so this
# runs MATCHINGS=z only and the two halves are read together.
#
# The grid below comes from
#   python code/analysis/plan_z_grid.py \
#       --calibration_dir results/experiment_0_calibration_2afc \
#       --alpha_trials results/experiment1/full32_all/trials.csv
# which applies the section 8.5 rule: the top dose must deliver, for EVERY family at
# EVERY block, more raw amplitude than that family's own threshold on the alpha grid.
# Re-run it and re-read its coverage table before changing anything here.
#
# An 8B model in bf16 fits the 3g.40gb MIG slice; prod40 caps memory at 60G.
#SBATCH --job-name=experiment1-z2afc
#SBATCH --partition=prod40
#SBATCH --gres=gpu:nvidia_a100_3g.40gb:1
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --output=logs/experiment1-z2afc-%j.out
#SBATCH --error=logs/experiment1-z2afc-%j.err

set -euo pipefail

CONFIG=${1:-configs/experiment_0_calibration/llama_2afc_full.yaml}
RUN_NAME=${2:-z2afc}
CALIBRATION_DIR=${3:-results/experiment_0_calibration_2afc}
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export HF_HUB_OFFLINE=1

# trials.csv is written only after the final trial, so a crash loses the whole run.
# The sweep is therefore submitted in layer groups and merged afterwards with
# code/analysis/merge_experiment1_runs.py, which needs every contributing run to
# share the z grid -- hence one grid below rather than one per group.
#
#   LAYERS="0 1 2 3 4 5 6 7"      sbatch ... jobs/experiment1_z_recalibrated.sbatch '' z2afc-g0
#   LAYERS="8 9 10 11 12 13 14 15"  ...                                             '' z2afc-g1
#   LAYERS="16 17 18 19 20 21 22 23" ...                                            '' z2afc-g2
#   LAYERS="24 25 26 27 28 29 30 31" ...                                            '' z2afc-g3
#
# 11 conditions x 18 doses x 40 trials (5 pairs x 2 orders x 2 label orders x 2
# targets) = 7920 passes per block, so 63360 per group: ~40 min at the 28 passes/s
# this model reaches on a 3g.40gb slice.
LAYERS=${LAYERS:-"0 1 2 3 4 5 6 7"}

# The recalibrated scales are smooth in depth and no longer split by family:
#
#   block          0        3        9       13       16       24       31
#   concept   0.0302   0.0733   0.1898   0.2770   0.3234   0.4868   2.8430
#   random    0.0142   0.0395   0.0846   0.0924   0.1266   0.2551   0.7505
#
# so the concept/generic ratio is 1.5-3.8 instead of 56-95, and ONE grid can bracket
# the four families at every block -- which the isolated-sentence calibration made
# impossible. What the grid still has to span is depth: s grows 200-fold from block 0
# to block 31, and that, not the family, is what sets its extent.
#
# Binding constraints, both from the coverage table of plan_z_grid.py:
#   z_max >= 6372  block 1 fixed-random, which the alpha grid never brought to 75%,
#                  so the z grid has to deliver at least the alpha grid's own top
#                  (128) against s = 0.0201 before its flat curve means anything.
#   z_min <= 0.088 block 31 concept, s = 2.843, to reach the alpha grid's floor 0.25.
#
# 18 doses at a factor of two, and the table reports 0 uncovered (block, family)
# cells. For contrast, the grid of the full32_all sweep stopped at z = 20.48, which
# with these scales tops out near alpha 0.4 at block 1: it would measure nothing.
#
# The extreme doses are deliberately out of range for most blocks -- z = 8192 at
# block 31 asks for alpha 23000 -- because a single shared grid pays for its extent
# at both ends. Doses producing non-finite activations are an operating limit to
# document, per doc 5.5; `finite` is recorded per trial.
Z_DOSES=${Z_DOSES:-"0.0625 0.125 0.25 0.5 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192"}

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
