#!/bin/bash
# Experiment 5 end to end in a single prod40 allocation: plan, run, analyze.
#
# The three stages are strictly sequential and share one node, so running them as
# one job costs one QOS slot instead of three chained ones, and removes the
# dependency chain entirely -- no queued stage can be silently scancelled while an
# earlier one runs. 'set -e' stops the job at the first failing stage, which gives
# the same all-or-nothing behaviour the afterok chain gave.
#
# Sizing is the union of the three stages: the run stage dominates at 12 h, and
# the two CPU-only stages add an hour each. prod40 caps memory at 60G
# (MaxMemPerNode=61440) and walltime at 24 h, and allocates its MIG slice
# implicitly -- a bare --gres=gpu:1 is rejected at submission.
#
# The per-stage scripts in this directory are kept for rerunning a single stage,
# for example analysis alone after a completed run.
#SBATCH --job-name=exp5-all
#SBATCH --partition=prod40
#SBATCH --time=14:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --output=logs/experiment_5-all-%j.out
#SBATCH --error=logs/experiment_5-all-%j.err

set -euo pipefail
CONFIG=${1:?configuration path required}
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export HF_HUB_OFFLINE=1
export PYTHONPATH="${PWD}/code${PYTHONPATH:+:${PYTHONPATH}}"

echo "=== [1/3] prepare_material_plan $(date -Is) ==="
python -m experiment_5_text_induction.prepare_material_plan --config "${CONFIG}"

echo "=== [2/3] run_experiment_5 $(date -Is) ==="
python -m experiment_5_text_induction.run_experiment_5 --config "${CONFIG}"

echo "=== [3/3] analyze_experiment_5 $(date -Is) ==="
python -m experiment_5_text_induction.analyze_experiment_5 --config "${CONFIG}"

echo "=== done $(date -Is) ==="
