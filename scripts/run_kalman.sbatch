#!/bin/bash
#SBATCH --job-name=kalman-refine
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/%u/kalman-test/logs/kalman_%j.out
#SBATCH --error=/scratch/%u/kalman-test/logs/kalman_%j.err
# Usage:
#   # Kalman-refine every session under outputs/chronic_mishi_v2 that has a
#   # predictions.npz, then render raw + Kalman heatmaps for each.
#   sbatch scripts/run_kalman.sbatch
#
#   # Just one session:
#   SESSIONS=TES_sResp_M05_20240729 sbatch scripts/run_kalman.sbatch
#
# Env-var knobs:
#   OUT_ROOT    — input/output dir   (default $SCRATCH/kalman-test/outputs/chronic_mishi_v2)
#   SESSIONS    — space-separated PIDs; default = every dir under OUT_ROOT
#   KALMAN_ARGS — extra flags to kalman_refine_predictions.py (e.g. "--Q 400")

REPO=$SCRATCH/kalman-test
source $REPO/scripts/_env_prelude.sh

mkdir -p $REPO/logs

OUT_ROOT=${OUT_ROOT:-$REPO/outputs/chronic_mishi_v2}
KALMAN_ARGS=${KALMAN_ARGS:-}

if [ -z "${SESSIONS:-}" ]; then
    SESSIONS=$(ls -1 "$OUT_ROOT" 2>/dev/null | tr '\n' ' ')
fi

echo "[kalman] OUT_ROOT=$OUT_ROOT"
echo "[kalman] SESSIONS=$SESSIONS"
echo "[kalman] KALMAN_ARGS=$KALMAN_ARGS"

cd $REPO

n_done=0
for sess in $SESSIONS; do
    npz=$OUT_ROOT/$sess/predictions.npz
    if [ ! -f "$npz" ]; then
        echo "[kalman] skip $sess (no $npz)"
        continue
    fi
    echo
    echo "=== $sess ==="

    # 1) Kalman refinement
    python kalman_refine_predictions.py "$npz" $KALMAN_ARGS

    # 2) Two heatmaps: raw chunk-level and Kalman-smoothed trajectory
    python viz_session_inference_timechunk.py \
        --root "$OUT_ROOT" --pid "$sess" \
        --pred_key pred_xyz_gauss

    python viz_session_inference_timechunk.py \
        --root "$OUT_ROOT" --pid "$sess" \
        --in_name predictions_kalman.npz \
        --pred_key pred_xyz_kalman_smooth

    n_done=$((n_done + 1))
done

echo
echo "[kalman] DONE — processed $n_done sessions."
