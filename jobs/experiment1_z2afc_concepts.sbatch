#!/bin/bash
# Experiment 1, completing the panel: the concepts and fixed-random directions the
# earlier sweeps left out, on both matchings, over all 32 blocks.
#
# Why. Doc de cadrage 5.7 fixes the number of fixed-random directions WITH the number of
# conceptual directions, and 7.3 asks Experiment 3 for at least five concepts over the 32
# blocks. The sweeps to date carry four concepts and three random directions, so they
# satisfy neither. Nothing was missing but the forward passes:
# results/experiment_0_calibration_2afc has s(l,v) for all ten concepts and all ten
# random directions at every block, and data/saved_vectors/llama/ has the vectors,
# versioned. The four were simply what the earlier job named.
#
# The four are also a biased sample, chosen once the ten-concept run was in hand: they
# rank 1, 5, 6 and 10 of the ten at block 3, so the panel holds both extremes and its
# dispersion sits at the 86.7th percentile of all 210 four-subsets. Its concept mean is
# correspondingly low -- 0.578 against the ten's 0.629 -- because Dust, a near-null,
# carries a quarter of the weight instead of a tenth. At block 3 that understates the
# concept-minus-random gap by 0.071 against 0.121, so completing the panel is not
# bookkeeping for Experiment 3: it corrects Experiment 1's own effect size.
#
# Both matchings, deliberately. The z arm pools with results/experiment1/z2afc_all, which
# shares this calibration. The alpha arm pools with results/experiment1/full32_all, whose
# calibration is the superseded one -- which does not matter, because under
# matching == "alpha" the amplitude IS the grid value and calibration.scale() is never
# consulted (experiment1_psychometrics.py:852). That invariance was checked, not assumed:
# results/experiment1/main, from a third and older pipeline, agrees with full32_all to
# three decimals on the four shared concepts.
#
# One family per job. Re-running a direction another sweep already covered would put four
# rows under each pairing key of doc 5.8, where paired_s expects two and DROPS the key
# silently rather than failing -- data loss that looks like nothing. So job A adds only
# the six unswept concepts and job B only random 0003-0009, via --random_ids, which
# exists because --num_random can take only a prefix.
#
# Both dose grids must stay byte-identical to the sweeps they pool with: curves fitted on
# different grids cannot be compared, and experiment3_variability.py keys its dense
# (direction, layer, phrase) design on a shared dose set.
#
# Cost at the measured 39 passes/s (jobs 8255-8258): 28 doses x 40 trials x 32 blocks =
# 35840 passes per direction. Job A, 6 concepts, ~92 min; job B, 7 random, ~107 min.
#
# An 8B model in bf16 fits the 3g.40gb MIG slice; prod40 caps memory at 60G. prod80 is
# free of MIG but was scheduling four hours later than prod40 when this was written --
# check `sbatch --test-only` against both before assuming either.
#SBATCH --job-name=experiment1-panel
#SBATCH --partition=prod40
#SBATCH --gres=gpu:nvidia_a100_3g.40gb:1
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --output=logs/experiment1-panel-%j.out
#SBATCH --error=logs/experiment1-panel-%j.err

set -euo pipefail

CONFIG=${1:-configs/experiment_0_calibration/llama_2afc_full.yaml}
RUN_NAME=${2:-panel}
CALIBRATION_DIR=${3:-results/experiment_0_calibration_2afc}
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export HF_HUB_OFFLINE=1

# trials.csv is written only after the final trial, so a crash loses the whole run.
# Split by family rather than by block: every direction needs all 32 blocks for the
# layer component, and two jobs is what the queue runs at once.
#
#   FAMILIES=concept CONCEPTS="appreciation betrayal fibonacci_numbers Illusions Origami Trumpets" \
#     sbatch jobs/experiment1_z2afc_concepts.sbatch '' panel-concepts
#   FAMILIES=random RANDOM_IDS="0003 0004 0005 0006 0007 0008 0009" \
#     sbatch jobs/experiment1_z2afc_concepts.sbatch '' panel-random
FAMILIES=${FAMILIES:-concept}
CONCEPTS=${CONCEPTS:-"appreciation betrayal fibonacci_numbers Illusions Origami Trumpets"}
RANDOM_IDS=${RANDOM_IDS:-""}
LAYERS=${LAYERS:-"0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31"}
MATCHINGS=${MATCHINGS:-"alpha z"}

# Identical to full32_all (alpha) and experiment1_z_recalibrated.sbatch (z). See the
# latter for why these 18 z doses and why their extremes are out of range at some blocks.
ALPHA_DOSES=${ALPHA_DOSES:-"0.25 0.5 1 2 4 8 16 32 64 128"}
Z_DOSES=${Z_DOSES:-"0.0625 0.125 0.25 0.5 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192"}

EXTRA=()
if [ -n "${RANDOM_IDS}" ]; then EXTRA+=(--random_ids ${RANDOM_IDS}); fi
if [ "${FAMILIES}" = "concept" ]; then EXTRA+=(--concepts ${CONCEPTS}); fi

echo "Families:  ${FAMILIES}"
echo "Concepts:  ${CONCEPTS}"
echo "RandomIDs: ${RANDOM_IDS}"
echo "Matchings: ${MATCHINGS}"

python code/experiments/experiment1_psychometrics.py \
  --calibration_config "${CONFIG}" \
  --calibration_dir "${CALIBRATION_DIR}" \
  --run_name "${RUN_NAME}" \
  --layers ${LAYERS} \
  --families ${FAMILIES} \
  --matchings ${MATCHINGS} \
  --num_pairs 5 \
  --label_orders AB BA \
  --alpha_doses ${ALPHA_DOSES} \
  --z_doses ${Z_DOSES} \
  --contamination_trials 24 \
  --no_progress \
  "${EXTRA[@]}"
