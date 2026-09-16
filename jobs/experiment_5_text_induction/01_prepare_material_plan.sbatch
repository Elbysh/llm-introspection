#!/bin/bash
# Freeze reviewed texts, prompts, counterbalancing and calibrated coefficients.
# CPU-only, so it runs on prod10 rather than taking a large-partition slot: the
# QOS 'normal' 2-running-job cap counts prod40 and prod80 only. prod10 allocates a
# 1g.10gb slice implicitly and caps memory at 15G (MaxMemPerNode=15360) and CPUs at
# 4 per GPU -- asking for more is rejected at submission.
#
# Only the tokenizer loads here. The model itself cannot run on prod10: 16.06 GB of
# bf16 weights do not fit a 10 GB slice, and the 15G host cap rules out CPU offload,
# which is why 02_run_experiment_5.sbatch stays on prod40.
#SBATCH --job-name=exp5-plan
#SBATCH --partition=prod10
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=15G
#SBATCH --output=logs/experiment_5-plan-%j.out
#SBATCH --error=logs/experiment_5-plan-%j.err

set -euo pipefail
CONFIG=${1:?configuration path required}
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
# Only the tokenizer is needed here, but it still resolves through the hub.
export HF_HUB_OFFLINE=1
export PYTHONPATH="${PWD}/code${PYTHONPATH:+:${PYTHONPATH}}"
python -m experiment_5_text_induction.prepare_material_plan --config "${CONFIG}"
