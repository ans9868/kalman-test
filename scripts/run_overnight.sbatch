#!/bin/bash
#SBATCH --job-name=kalman-overnight
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --partition=h100_tandon
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=berkesencan1@gmail.com
#SBATCH --output=/scratch/%u/kalman-test/logs/overnight_%j.out
#SBATCH --error=/scratch/%u/kalman-test/logs/overnight_%j.err
# End-to-end overnight: inference for every chronic Mishi session that
# doesn't yet have a predictions.npz, then Kalman + viz + plot_stability
# for every session with a fresh predictions.npz.
#
# Idempotent — infer_chronic_mishi_window.py skips sessions where
# predictions.npz already exists, and the Kalman/viz steps are cheap
# to redo if needed.
#
# Usage:
#   # default = all 5 chronic Mishi sessions (skips ones already done):
#   sbatch scripts/run_overnight.sbatch
#
#   # explicit subset:
#   SESSIONS="TES_sResp_M05_20240730 TES_sResp_M05_20240731" sbatch scripts/run_overnight.sbatch
#
# Env-var knobs:
#   SESSIONS    — space-separated PIDs (default: all 5 chronic Mishi)
#   OUT_ROOT    — output dir (default $SCRATCH/kalman-test/outputs/chronic_mishi_v2)
#   WINDOW_MIN  — minutes (default 5) or "full"
#   CKPT        — checkpoint (default: Path B 75K λ=3.0 flagship)
#   KALMAN_ARGS — extra flags forwarded to kalman_refine_predictions.py

REPO=$SCRATCH/kalman-test
source $REPO/scripts/_env_prelude.sh

mkdir -p $REPO/logs

CKPT=${CKPT:-/scratch/pl2820/ray_results/benchmarks/finetune_xyz_75k_joint_lambda_3_0_7741664/checkpoints/finetune_best_valmse.pt}
OUT_ROOT=${OUT_ROOT:-$REPO/outputs/chronic_mishi_v2}
WINDOW_MIN=${WINDOW_MIN:-5}
SESSIONS=${SESSIONS:-TES_sResp_M05_20240729 TES_sResp_M05_20240730 TES_sResp_M05_20240731 TES_sResp_M05_20241014 TES_sResp_M05_20241015}
KALMAN_ARGS=${KALMAN_ARGS:-}

mkdir -p "$OUT_ROOT"

echo "[overnight] sessions: $SESSIONS"
echo "[overnight] out_root: $OUT_ROOT"
echo "[overnight] window:   $WINDOW_MIN min"
echo "[overnight] ckpt:     $CKPT"

cd $REPO

# ─── stage 1: inference ───
echo
echo "============================================================"
echo "[overnight] STAGE 1: inference"
echo "============================================================"
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

# ─── stage 2: per-session Kalman + viz + stability ───
echo
echo "============================================================"
echo "[overnight] STAGE 2: Kalman + viz + plot_stability"
echo "============================================================"

n_processed=0
n_failed=0
for sess in $SESSIONS; do
    npz=$OUT_ROOT/$sess/predictions.npz
    if [ ! -f "$npz" ]; then
        echo "[overnight] [skip] $sess — no predictions.npz produced"
        n_failed=$((n_failed + 1))
        continue
    fi
    echo
    echo "--- $sess ---"

    python kalman_refine_predictions.py "$npz" $KALMAN_ARGS \
        || { echo "[overnight] kalman_refine FAILED for $sess"; n_failed=$((n_failed + 1)); continue; }

    python viz_session_inference_timechunk.py \
        --root "$OUT_ROOT" --pid "$sess" \
        --pred_key pred_xyz_gauss \
        || { echo "[overnight] viz raw FAILED for $sess"; n_failed=$((n_failed + 1)); continue; }

    python viz_session_inference_timechunk.py \
        --root "$OUT_ROOT" --pid "$sess" \
        --in_name predictions_kalman.npz \
        --pred_key pred_xyz_kalman_smooth \
        || { echo "[overnight] viz kalman FAILED for $sess"; n_failed=$((n_failed + 1)); continue; }

    python plot_stability.py "$OUT_ROOT/$sess/predictions_kalman.npz" \
        || { echo "[overnight] plot_stability FAILED for $sess"; n_failed=$((n_failed + 1)); continue; }

    n_processed=$((n_processed + 1))
done

echo
echo "============================================================"
echo "[overnight] SUMMARY"
echo "============================================================"
echo "[overnight] processed: $n_processed sessions"
echo "[overnight] failures:  $n_failed sessions"
echo "[overnight] outputs under: $OUT_ROOT/<session>/"
for sess in $SESSIONS; do
    if [ -d "$OUT_ROOT/$sess" ]; then
        echo "  $sess:"
        ls -1 "$OUT_ROOT/$sess" 2>/dev/null | sed 's/^/    /'
    fi
done

if [ $n_failed -gt 0 ]; then
    exit 1
fi
echo "[overnight] DONE."
