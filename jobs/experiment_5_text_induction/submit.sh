#!/bin/bash
# Submit Experiment 5 as one prod40 job running plan -> run -> analyze in
# sequence. This cluster has no cpu_short and no gpua100 partition, only
# interactive10, prod10, prod40 and prod80.
#
# Pass --staged to submit the three stages as separate afterok-chained jobs
# instead, which is useful when a stage needs different resources or a rerun.
# Staged, the two CPU-only stages go to prod10 and only the GPU run takes a
# prod40 slot; the single combined job necessarily keeps all three on prod40,
# since one job cannot span partitions. The run stage cannot go to prod10 either
# way: 16.06 GB of bf16 weights do not fit its 10 GB slice, and its 15G host cap
# rules out CPU offload.
set -euo pipefail

STAGED=0
if [[ ${1-} == "--staged" ]]; then
  STAGED=1
  shift
fi
CONFIG=${1:?usage: bash jobs/experiment_5_text_induction/submit.sh [--staged] <configuration.yaml>}
test -f "${CONFIG}"

# The sbatch scripts source .venv from the submission directory. In this worktree
# .venv is a symlink to the main checkout's environment; either way, fail here
# rather than inside the first job's log.
if [[ ! -e .venv/bin/activate ]]; then
  echo "Missing .venv in $(pwd); link or create it before submitting." >&2
  exit 2
fi

# QOS 'normal' allows at most 2 RUNNING jobs across prod40 and prod80 combined.
# Over that, sbatch refuses new submissions outright, and a job already queued is
# scancelled without a log when it would have become the third.
RUNNING=$(squeue -u "${USER}" -h -t RUNNING -p prod40,prod80 -o "%i" | wc -l)
if (( RUNNING >= 2 )); then
  echo "Already ${RUNNING} running jobs on prod40/prod80 (QOS limit 2)." >&2
  echo "Wait for a slot to free before submitting Experiment 5." >&2
  exit 3
fi

mkdir -p logs

if (( STAGED )); then
  PLAN_ID=$(sbatch --parsable jobs/experiment_5_text_induction/01_prepare_material_plan.sbatch "${CONFIG}")
  RUN_ID=$(sbatch --parsable --dependency="afterok:${PLAN_ID}" jobs/experiment_5_text_induction/02_run_experiment_5.sbatch "${CONFIG}")
  ANALYSIS_ID=$(sbatch --parsable --dependency="afterok:${RUN_ID}" jobs/experiment_5_text_induction/03_analyze_experiment_5.sbatch "${CONFIG}")
  echo "Experiment 5 plan: ${PLAN_ID}"
  echo "Experiment 5 run: ${RUN_ID} (afterok:${PLAN_ID})"
  echo "Experiment 5 analysis: ${ANALYSIS_ID} (afterok:${RUN_ID})"
else
  JOB_ID=$(sbatch --parsable jobs/experiment_5_text_induction/run_all_experiment_5.sbatch "${CONFIG}")
  echo "Experiment 5 (plan -> run -> analyze): ${JOB_ID}"
  echo "Logs: logs/experiment_5-all-${JOB_ID}.out"
fi
