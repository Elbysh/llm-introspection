#!/bin/bash
# Experiment 4: detection and task degradation, on meta-llama/Llama-3.1-8B-Instruct.
# Section 8 of docs/livrables/cadrage-experiments.md.
#
# The model is never asked about the intervention here. It classifies a sentence
# while the same perturbation runs, so that Experiment 1's detection curve can be
# read against the dose range where the model still does semantic work (H4).
#
# Consumes the Experiment 0 calibration: the direction bank and the scales s(l, v).
# The committed development calibration is the default; for a fresh one run, in order:
#   python -m experiment_0_calibration.prepare_concept_vectors --config <config>
#   python -m experiment_0_calibration.prepare_material_plan   --config <config>
#   python -m experiment_0_calibration.run_experiment_0        --config <config>
# and pass its paths.output_dir as the third argument.
#
# The join against Experiment 1 (doc 8.5 step 7 and doc 8.7) is deliberately NOT done
# here. It needs no GPU, and `results/` is gitignored and per-worktree, so hardcoding
# a path to a detection run would break on any other checkout. Do it afterwards:
#
#   python code/experiments/experiment4_task_degradation.py \
#       --reanalyze results/experiment4/main \
#       --experiment1_summary <the Experiment 1 run directory>
#
# That rewrites summary.json with detection_vs_performance, the per-family
# correlation and the performance-matched bins, and draws the scatter.
#
# An 8B model fits the 3g.40gb MIG slice that jobs/experiment1.sbatch uses, so prod80
# is not a memory requirement here -- it is a throughput one. The full card runs this
# sweep in roughly half the time of the slice, and the sweep writes trials.csv only
# after its last trial, so shortening the window in which a crash loses everything is
# worth the larger partition.
#
# Host memory stays at the 60G of the Llama Experiment 1 job rather than prod80's
# 120G cap: 16 GB of bf16 weights need no more, and a request the node cannot spare
# alongside the small-partition jobs only makes this one harder to backfill.
#
# The wall time is a little under twice the measured cost. Experiment 1 ran at ~28
# forward passes/s on the 40 GB slice, which puts this sweep's 436560 passes at ~4.3 h
# there and less on the full card. Do not trim the margin: trials.csv is written once,
# at the end, so a job that hits the wall loses every trial it ran.
#SBATCH --job-name=experiment4
#SBATCH --partition=prod80
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --output=logs/experiment4-%j.out
#SBATCH --error=logs/experiment4-%j.err

set -euo pipefail

# --allow_calibration_mismatch below is needed for the committed development
# calibration and is provenance-only, not a numerical licence: that run was launched
# from an uncommitted tree, so its manifest records a config sha256 no committed
# version of the file hashes to. DirectionBank still rebuilds every direction from the
# config's seeds and refuses any whose norm differs from the one the scales were
# estimated on, so a real drift fails at preflight. Check the log for the line
# "Directions:  N rebuilt and matched to their calibrated norms".
CONFIG=${1:-configs/experiment_0_calibration/development_full.yaml}
RUN_NAME=${2:-main}
CALIBRATION_DIR=${3:-results/experiment_0_calibration}
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export HF_HUB_OFFLINE=1

# Doc 8.4 reuses Experiment 1's blocks. Its Llama sweep covered all 32, but block 31
# is dropped here: its concept directions need data/saved_vectors/llama/*_32_*.pt,
# which is not versioned, and a regenerated copy does not reproduce the norm the
# calibration recorded -- a concept vector is a bf16 difference of activation means.
# DirectionBank refuses it, correctly, and does so at preflight before the weights are
# loaded. Nothing is lost for the detection side: no attention layer follows block 31,
# so a perturbation at sentence tokens cannot reach the answer token and Experiment 1
# returns chance there by construction.
LAYERS=${LAYERS:-"$(seq -s' ' 0 30)"}

# Doc 8.4 also requires the same alpha and z values as Experiment 1. These are the two
# grids of its Llama full-32 sweep, copied rather than re-derived: a cell of this
# experiment only joins onto a detection cell if the dose matches exactly.
ALPHA_DOSES=${ALPHA_DOSES:-"0.25 0.5 1 2 4 8 16 32 64 128"}
Z_DOSES=${Z_DOSES:-"0.01 0.02 0.04 0.08 0.16 0.32 0.64 1.28 2.56 5.12 10.24 20.48"}

# The evaluated concepts are the five of data/dataset/complex_data.json: only that
# file carries the labelled positive and negative examples a classification task needs.
#
# The injected concepts are drawn from Dust and Satellites. Both are in the concept set
# Experiment 1 swept, so the concept family joins onto the same directions; and both
# come from simple_data, so they share no sentence with the classification corpus. The
# evaluated concept is never the injected one, which doc 8.3 requires so that a
# targeted injection cannot hand the model its answer, and which also keeps doc 3.3's
# construction/evaluation separation with nothing left to check by hand.
EVAL_CONCEPTS=${EVAL_CONCEPTS:-"appreciation betrayal fibonacci_numbers recursion shutdown"}
INJECTED_POOL=${INJECTED_POOL:-"Dust Satellites"}

# 4 positive and 4 negative sentences per concept, in both letter mappings: 80 items.
# With 3 fixed-random directions, 2 noise and 2 dropout realizations per block, that
# is 436480 perturbed passes over the 31 blocks and the 22 doses.
python code/experiments/experiment4_task_degradation.py \
  --calibration_config "${CONFIG}" \
  --calibration_dir "${CALIBRATION_DIR}" \
  --allow_calibration_mismatch \
  --run_name "${RUN_NAME}" \
  --layers ${LAYERS} \
  --families ${FAMILIES:-concept random noise dropout} \
  --matchings ${MATCHINGS:-alpha z} \
  --eval_concepts ${EVAL_CONCEPTS} \
  --injected_concept_pool ${INJECTED_POOL} \
  --num_injected_concepts 1 \
  --num_sentences_per_class 4 \
  --mappings XY YX \
  --num_random 3 \
  --num_noise 2 \
  --num_dropout 2 \
  --alpha_doses ${ALPHA_DOSES} \
  --z_doses ${Z_DOSES} \
  --js_per_cell 1 \
  --no_progress
