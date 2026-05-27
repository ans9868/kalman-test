# Scripts Reference

What each file does and how to invoke it. All scripts assume you've sourced
`scripts/_env_prelude.sh` (which activates the conda env + redirects HOME to
scratch). The sbatch scripts source it themselves.

## scripts/_env_prelude.sh

Sourceable bash. Sets up everything needed before running any of our Python
scripts on torch:

- `HOME`, `XDG_CACHE_HOME`, `XDG_CONFIG_HOME`, `CONDARC`, `TMPDIR`,
  `MPLCONFIGDIR` → `$SCRATCH/kalman_scratch/...` (avoid home quota)
- `PYTHONNOUSERSITE=True` (don't read `~/.local/lib/...`)
- `CONDA_PKGS_DIRS`, `PIP_CACHE_DIR` → `$SCRATCH`
- Activates `$SCRATCH/conda_storage/kalman` conda env

**Note**: does NOT `set -u`. Conda's MKL activate hook trips on it.

## scripts/run_infer.sbatch

GPU inference for one or more chronic Mishi sessions.

```bash
# default args: 5 sessions × 5-min window, h100_tandon, 2h walltime
sbatch scripts/run_infer.sbatch

# override sessions
SESSIONS=TES_sResp_M05_20240729 sbatch scripts/run_infer.sbatch

# full session instead of 5-min window
WINDOW_MIN=full sbatch scripts/run_infer.sbatch
```

Writes per-session `predictions.npz` into
`$SCRATCH/kalman-test/outputs/chronic_mishi_v2/<SESSION>/`.

Idempotent — `infer_chronic_mishi_window.py` skips sessions where the npz
already exists.

## scripts/run_kalman.sbatch

CPU job: per-session Kalman refinement + viz heatmaps (both raw and Kalman).
Loops over every session under `OUT_ROOT` with a `predictions.npz`.

```bash
sbatch scripts/run_kalman.sbatch
# or for a subset:
SESSIONS="TES_sResp_M05_20240729 TES_sResp_M05_20240730" sbatch scripts/run_kalman.sbatch
# tune Kalman params:
KALMAN_ARGS="--Q 400 --R_min 100" sbatch scripts/run_kalman.sbatch
```

Note: runs on `h100_tandon` even though it doesn't need a GPU — partitioned
that way for priority-account access. Could be moved to a CPU partition if
queue is hostile.

## scripts/run_overnight.sbatch

The fire-and-forget combo: inference + Kalman + viz + plot_stability for
every chronic Mishi session, all in one allocation.

```bash
sbatch scripts/run_overnight.sbatch
# or subset, custom params
SESSIONS="TES_sResp_M05_20240730 TES_sResp_M05_20240731" KALMAN_ARGS="--Q 200" \
  sbatch scripts/run_overnight.sbatch
```

8h walltime. Emails `berkesencan1@gmail.com` on END / FAIL / TIME_LIMIT.

Both stages are idempotent. If session 1 already has predictions.npz, stage 1
skips it; stage 2 always re-runs Kalman + viz + plots (cheap).

## scripts/validate_session.sh

CPU sanity check on a `predictions.npz`. Reports:

- Shape + dtype
- NaN / Inf counts
- Per-axis AP/DV/ML position ranges (with Allen typical bounds)
- Per-axis sigma median + p90
- Outlier rate at 500/1000/2000 µm thresholds (vs per-channel median)
- Per-channel-mean span (should match NP2.0 shank length ~2880 µm)

```bash
# defaults to first predictions.npz under OUT_ROOT
bash scripts/validate_session.sh

# or specify
bash scripts/validate_session.sh $SCRATCH/kalman-test/outputs/chronic_mishi_v2/TES_sResp_M05_20240729/predictions.npz
```

Run this BEFORE submitting the overnight sbatch — catches data-format
surprises on session 1 instead of after spending another queue cycle.

# Python scripts (called from sbatch + manually)

## kalman_refine_predictions.py

Per-channel per-axis 1D constant-position Kalman filter + RTS smoother.

```bash
python kalman_refine_predictions.py <input.npz> \
    [-o <output.npz>] \
    [--Q 100] [--R_min 50] [--R_max 3000] [--scale_R 1.0] \
    [--stability_K 10] [--outlier_um 1000] [--trim_frac 0.1]
```

Input: `predictions.npz` from `infer_chronic_mishi_window.py` (needs
`pred_xyz_gauss`, `pred_sigma_gauss`, `channel_axial_um`, `channel_ids`,
`pid`, `chunk_dur_sec`).

Output: `<input_stem>_kalman.npz` next to input + sidecar `.json` summary.

**Default Kalman config:**
- `Q = 100` µm² per chunk (≈10 µm/√chunk RMS process noise — slow drift)
- `R_min = 50, R_max = 3000` µm (clamps the head's per-chunk sigma)
- `scale_R = 1.0` (trust head's sigma at face value)
- `stability_K = 10` chunks (≈30 s at 3 s/chunk required below threshold)
- `outlier_um = 1000` µm

See dev-notes §8 of `Kalman-Temporal-Refinement-Context+Setup-TODO.md` for
the rationale on these defaults.

## viz_session_inference_timechunk.py

Channel × time-chunk predicted-region heatmap. Voxel-looks up each (ch, chunk)
predicted xyz into the Allen CCF, colors by region.

```bash
# raw heatmap
python viz_session_inference_timechunk.py \
    --root $SCRATCH/kalman-test/outputs/chronic_mishi_v2 \
    --pid TES_sResp_M05_20240729 \
    --pred_key pred_xyz_gauss

# Kalman heatmap
python viz_session_inference_timechunk.py \
    --root $SCRATCH/kalman-test/outputs/chronic_mishi_v2 \
    --pid TES_sResp_M05_20240729 \
    --in_name predictions_kalman.npz \
    --pred_key pred_xyz_kalman_smooth
```

Output filename suffixes `pred_key` (e.g. `channel_timechunk_gauss.png` vs
`channel_timechunk_kalman_smooth.png`) so raw and Kalman render side-by-side
without overwriting.

Falls back to "UNK" if `channel_acronyms` is missing (chronic Mishi has no
histology GT) and skips the GT-strip annotation.

## plot_stability.py

Per-session stability curve + aggregation comparison table.

```bash
python plot_stability.py <predictions_kalman.npz>
```

Outputs next to input:
- `stability_curve.png` — distance-to-final (median + IQR + p90) over time
- `aggregation_table.txt` — text table comparing 7 aggregation methods'
  disagreement with the Kalman smoother's mean estimate

## _make_mock_predictions.py

Synthetic data generator for offline testing. Generates a fake `predictions.npz`
matching the chronic Mishi schema (384 ch × 100 chunks × 3 axes, with 2%
outliers seeded). Used to validate the post-processing pipeline before real
predictions exist.

```bash
python _make_mock_predictions.py [output.npz]
```

# Output data contracts

## predictions.npz (from inference)

```
pred_xyz_gauss      (n_ch, n_chunks, 3)  float32  µm CCF (left hem decode)
pred_sigma_gauss    (n_ch, n_chunks, 3)  float32  µm per axis
channel_axial_um    (n_ch,)              float32
channel_ids         (n_ch,)              int64
pid                 ()                   object   session name string
chunk_dur_sec       ()                   float64  3.0
fs_target           ()                   float64  1250
win_start_sec, win_end_sec               float64
raw_std, scale_factor                    float64
```

## predictions_kalman.npz (from Kalman script)

All keys above, plus:

```
pred_xyz_kalman_filter    (n_ch, n_chunks, 3)  online filter trajectory
pred_xyz_kalman_smooth    (n_ch, n_chunks, 3)  RTS-smoothed trajectory
kalman_cov_filter         (n_ch, n_chunks, 3)
kalman_cov_smooth         (n_ch, n_chunks, 3)
kalman_R_used             (n_ch, n_chunks, 3)  clamped observation variance
pred_xyz_kalman_final     (n_ch, 3)            filter at last chunk
pred_xyz_smooth_final     (n_ch, 3)            smoother at last chunk
pred_xyz_smooth_mean      (n_ch, 3)            mean of smoother over time
pred_xyz_unweighted_mean  (n_ch, 3)
pred_xyz_weighted_mean    (n_ch, 3)            inverse-variance weighted
pred_xyz_median           (n_ch, 3)
pred_xyz_trimmed_mean     (n_ch, 3)
distance_to_final         (n_ch, n_chunks)
time_to_stable_100um      (n_ch,)              int32, -1 if never stable
time_to_stable_200um      (n_ch,)              int32
outlier_distance_raw      (n_ch, n_chunks)
outlier_distance_kalman   (n_ch, n_chunks)
kalman_params             ()                   JSON string of run params
```
