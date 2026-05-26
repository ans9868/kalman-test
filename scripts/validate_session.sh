#!/bin/bash
# Fast eyeball check on a session's predictions.npz. Prints shape, NaN/Inf
# counts, sigma + position ranges, raw outlier rate vs per-channel median,
# and brain-region coverage. Useful before committing to a full overnight
# batch — if anything's weird, we catch it on session 1.
#
# Usage:
#   bash scripts/validate_session.sh <path/to/predictions.npz>
#   bash scripts/validate_session.sh           # default = first session under OUT_ROOT
set -e

REPO=$SCRATCH/kalman-test
source $REPO/scripts/_env_prelude.sh

OUT_ROOT=${OUT_ROOT:-$REPO/outputs/chronic_mishi_v2}

NPZ=${1:-}
if [ -z "$NPZ" ]; then
    NPZ=$(ls -1 $OUT_ROOT/*/predictions.npz 2>/dev/null | head -1)
fi
if [ -z "$NPZ" ] || [ ! -f "$NPZ" ]; then
    echo "no predictions.npz found (looked at: $1, $OUT_ROOT/*/predictions.npz)"
    exit 1
fi

echo "[validate] $NPZ"
python <<PY
import numpy as np

d = np.load("$NPZ", allow_pickle=True)
y     = d["pred_xyz_gauss"]
sigma = d["pred_sigma_gauss"]
n_ch, T, _ = y.shape
pid = str(d["pid"])
chunk_dur = float(d["chunk_dur_sec"])

print(f"  pid:       {pid}")
print(f"  shape:     pred {y.shape} {y.dtype}  sigma {sigma.shape} {sigma.dtype}")
print(f"  duration:  {T} chunks × {chunk_dur}s = {T*chunk_dur/60:.1f} min")

# Health: NaN, Inf, all-zero?
nan_y = int(np.isnan(y).sum()); inf_y = int(np.isinf(y).sum())
nan_s = int(np.isnan(sigma).sum()); inf_s = int(np.isinf(sigma).sum())
print(f"  NaN/Inf:   pred NaN={nan_y} Inf={inf_y}   sigma NaN={nan_s} Inf={inf_s}")
if nan_y or inf_y or nan_s or inf_s:
    print("  ** WARNING: non-finite values present **")

# Position ranges per axis
print(f"  pred AP:   [{y[...,0].min():.0f}, {y[...,0].max():.0f}] µm  (Allen typical 0–13200)")
print(f"  pred DV:   [{y[...,1].min():.0f}, {y[...,1].max():.0f}] µm  (Allen typical 0–8000)")
print(f"  pred ML:   [{y[...,2].min():.0f}, {y[...,2].max():.0f}] µm  (Allen midline 5739; left hem <5739)")

# Sigma sanity
print(f"  sigma AP:  median {np.median(sigma[...,0]):.0f}, p90 {np.percentile(sigma[...,0],90):.0f} µm")
print(f"  sigma DV:  median {np.median(sigma[...,1]):.0f}, p90 {np.percentile(sigma[...,1],90):.0f} µm")
print(f"  sigma ML:  median {np.median(sigma[...,2]):.0f}, p90 {np.percentile(sigma[...,2],90):.0f} µm")

# Outlier rate: per-channel L2 distance from per-channel median
med = np.median(y, axis=1, keepdims=True)
dist = np.linalg.norm(y - med, axis=2)
for thr in (500, 1000, 2000):
    rate = float((dist > thr).mean())
    print(f"  outlier rate >{thr}µm from per-channel median:  {rate:.2%}")

# Cross-channel spread (does the probe look like a probe?)
per_ch_mean = y.mean(axis=1)
ap_span = per_ch_mean[:,0].max() - per_ch_mean[:,0].min()
dv_span = per_ch_mean[:,1].max() - per_ch_mean[:,1].min()
ml_span = per_ch_mean[:,2].max() - per_ch_mean[:,2].min()
print(f"  per-channel-mean span:  AP={ap_span:.0f}µm DV={dv_span:.0f}µm ML={ml_span:.0f}µm")
print(f"  (NP2.0 shank is ~2880µm tip-to-base — the dominant span should match)")

# Metadata pass-through
for k in ("raw_std","scale_factor","win_start_sec","win_end_sec","fs_target"):
    if k in d.files:
        print(f"  {k}: {float(d[k]):.4g}")

print()
print("[validate] looks reasonable" if (nan_y+inf_y+nan_s+inf_s)==0 and dist.mean()<5000 else
      "[validate] **CHECK ABOVE**")
PY
