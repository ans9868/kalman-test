#!/bin/bash
#SBATCH --job-name=kalman-infer
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/kalman-test/logs/infer_%j.out
#SBATCH --error=/scratch/%u/kalman-test/logs/infer_%j.err
# Usage:
#   # one session, 5-min mid-window:
#   SESSIONS=TES_sResp_M05_20240729 sbatch scripts/run_infer.sbatch
#
#   # all five default sessions:
#   sbatch scripts/run_infer.sbatch
#
#   # full session (slower, bigger output):
#   WINDOW_MIN=full sbatch scripts/run_infer.sbatch
#
# Env-var knobs (override defaults):
#   SESSIONS    — space-separated list (default: all 5 chronic Mishi M05)
#   WINDOW_MIN  — minutes (default 5), or "full" for the whole session
#   OUT_ROOT    — output dir (default $SCRATCH/kalman-test/outputs/chronic_mishi_v2)
#   CKPT        — checkpoint (default: paper's Path B 75K λ=3.0 flagship)

REPO=$SCRATCH/kalman-test
source $REPO/scripts/_env_prelude.sh

mkdir -p $REPO/logs

# defaults
CKPT=${CKPT:-/scratch/pl2820/ray_results/benchmarks/finetune_xyz_75k_joint_lambda_3_0_7741664/checkpoints/finetune_best_valmse.pt}
OUT_ROOT=${OUT_ROOT:-$REPO/outputs/chronic_mishi_v2}
WINDOW_MIN=${WINDOW_MIN:-5}
SESSIONS=${SESSIONS:-TES_sResp_M05_20240729 TES_sResp_M05_20240730 TES_sResp_M05_20240731 TES_sResp_M05_20241014 TES_sResp_M05_20241015}

mkdir -p "$OUT_ROOT"

echo "[infer] sessions: $SESSIONS"
echo "[infer] ckpt:     $CKPT"
echo "[infer] out_root: $OUT_ROOT"
echo "[infer] window:   $WINDOW_MIN"

cd $REPO

if [ "$WINDOW_MIN" = "full" ]; then
    python infer_chronic_mishi_window.py \
        --ckpt "$CKPT" \
        --out_root "$OUT_ROOT" \
        --sessions $SESSIONS \
        --full_session
else
    python infer_chronic_mishi_window.py \
        --ckpt "$CKPT" \
        --out_root "$OUT_ROOT" \
        --sessions $SESSIONS \
        --window_min "$WINDOW_MIN"
fi

echo "[infer] DONE."
