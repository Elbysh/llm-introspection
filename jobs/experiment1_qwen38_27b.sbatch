#!/bin/bash
# Experiment 1 on Qwen/Qwen3.8-27B: 2AFC localization, alpha- and z-matched
# psychometric curves. Section 3.1 of docs/livrables/cadrage-experiments.md lists
# this model as the secondary one, to be run once the protocol is validated on
# Llama-3.1-8B-Instruct; jobs/experiment1.sbatch is that Llama run.
#
# Consumes the Qwen Experiment 0 calibration: the direction bank and the scales
# s(l, v) it estimated. Produce it first with
#   sbatch jobs/experiment_0_calibration_qwen38_27b.sbatch
# and pass its paths.output_dir as the third argument.
#
# Qwen3.8-27B needs the full 80 GB card: 52 GB of bf16 weights do not fit the
# 3g.40gb MIG slice that jobs/experiment1.sbatch uses for the 8B model. prod80 caps
# memory at 120G (MaxMemPerNode=122880).
#SBATCH --job-name=experiment1-qwen38
#SBATCH --partition=prod80
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --output=logs/experiment1-qwen38-%j.out
#SBATCH --error=logs/experiment1-qwen38-%j.err

set -euo pipefail

CONFIG=${1:-configs/experiment_0_calibration/qwen38_27b_full.yaml}
RUN_NAME=${2:-qwen38-main}
CALIBRATION_DIR=${3:-results/experiment_0_calibration_qwen38_27b}
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export HF_HUB_OFFLINE=1

# Experiment 0 calibrates all 64 decoder blocks. The full sweep over them is far
# beyond one job, so pick the blocks the analysis needs. Scaling the Llama job's
# 3/16/28 to twice the depth would give 6/32/56, but the full-32 Llama sweep has
# since shown where the signal lives: blocks 0-13 of 32, with 14-31 returning a flat
# zero (docs/livrables/experiment1-full32-findings.md, section 1.6). Two of three
# rescaled blocks would land in the dead band. So the default samples the band that
# was live on Llama more densely and keeps the deep blocks as built-in nulls:
#
#   3, 6, 16   the first ~25% of the stack, where Llama's effect was confined
#   32, 56     half and seven eighths of the way up, expected null
#
# Whether Qwen's live band sits at the same relative depth is exactly the open
# question; the deep blocks are what answers it either way. The last block is
# excluded on purpose: no attention layer follows it, so a perturbation at sentence
# tokens cannot reach the answer token, and it returns chance by construction.
#
# At 2.7 forward passes/s this costs ~1 h per block: 11 conditions x 22 doses x 40
# trials = 9680 passes each. Five blocks is ~5 h against the 20 h wall.
LAYERS=${LAYERS:-"3 6 16 32 56"}

# The two grids are not comparable numerically, and the alpha one does not transfer
# from the Llama job: alpha is a raw activation norm, and Qwen's are far smaller.
# Its Experiment 0 gives, as median s(l, v) over blocks 3/6/16/32/56,
#
#   concept       2.8  2.0  3.2  2.7  17.4     (Llama, blocks 3/16/28: 167 / 52 / 20)
#   fixed_random  0.18 0.23 0.50 0.92  2.75    (Llama:                 3.3 / 1.5 / 1.4)
#   renewed_noise 0.14 0.24 0.54 1.01  2.73
#
# so the Llama grid's first dose, 0.25, already sits above every generic direction's
# scale at the shallow blocks: it would miss the whole transition. The ladder below
# is a factor of two from below the smallest family median to above the largest,
# and is what this prints:
#   python code/analysis/summarize_calibration_scales.py \
#       --calibration_dir results/experiment_0_calibration_qwen38_27b \
#       --layers 3 6 16 32 56
# extended by one step at the top so the concept family at block 56 (median 17.4,
# max 27.9) still has headroom to saturate. Re-check it against that table if the
# calibration is ever rerun.
ALPHA_DOSES=${ALPHA_DOSES:-"0.0625 0.125 0.25 0.5 1 2 4 8 16 32 64"}
# z is dimensionless, so the Llama grid carries over unchanged.
Z_DOSES=${Z_DOSES:-"0.01 0.02 0.04 0.08 0.16 0.32 0.64 1.28 2.56 5.12 10.24 20.48"}

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
  --alpha_doses ${ALPHA_DOSES} \
  --z_doses ${Z_DOSES} \
  --contamination_trials 24 \
  --no_progress
