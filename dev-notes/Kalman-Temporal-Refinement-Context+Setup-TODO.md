# Kalman Temporal Refinement Context + Setup TODO

## 0. Current Slack Context

Subhrajit asked whether I tried the Kalman filter. I said I got pulled into other work but would look at it and report back. We agreed that showing results on Thursday would be good.

Important message from Subhrajit:

- Main model path:
  `/scratch/pl2820/ray_results/benchmarks/finetune_xyz_75k_joint_lambda_3_0_7741664/`

- Files sent:
  - `eval_regression_probe.py`
  - `infer_chronic_mishi_window.py`
  - `viz_session_inference_timechunk.py`

- Data paths mentioned:
  - `/scratch/mc10168/mishi/`
  - `/scratch/mkp6112/Monkey/`

Current blocker:
- I currently do **not** have access to the monkey session(s), or at least the path is not readable from my account/environment.

---

# 1. Main Goal

The goal is **not** to retrain the SSL backbone or redesign the model head yet.

The immediate goal is:

> Add a lightweight downstream temporal localization layer using a Kalman filter or Kalman smoother over chunk-level AP/DV/ML predictions.

The idea is to improve inference-time localization by replacing or comparing the current confidence-weighted temporal averaging with a temporal state-space model.

Current model output per chunk:

```text
3-second LFP chunk
    → SSL encoder / Wav2Vec2 backbone
    → Gaussian localization head
    → predicted AP/DV/ML coordinate
    → predicted per-axis uncertainty / sigma
```

Proposed downstream refinement:

```text
chunk-level AP/DV/ML + sigma
    → Kalman filter / smoother over time per channel
    → stabilized coordinate estimate per channel
    → optional Kabsch spatial refinement across probe channels
    → final probe trajectory / heatmap / stability metrics
```

---

# 2. Why Kalman Makes Sense Here

The current head is local and memoryless:

```text
chunk_t embedding → μ_t, σ²_t
```

It does not know:

- what previous predictions were
- what future predictions will be
- whether a chunk is an outlier relative to the trajectory
- whether the predicted coordinate has stabilized
- whether the model is flickering between regions
- whether the predicted uncertainty is calibrated over time

The current uncertainty estimate is chunk-local. A Kalman smoother can convert this into trajectory-level uncertainty.

Core intuition:

```text
The model gives noisy observations:
    y_t = predicted AP/DV/ML from chunk t

The hidden true state is:
    x_t = true channel location

For acute recordings:
    x_t ≈ x_{t-1}

For chronic recordings:
    x_t ≈ x_{t-1} + slow drift
```

So a simple constant-position or slow-drift Kalman filter is physically aligned with the problem.

---

# 3. Expected Contribution

This should be framed as:

> Better temporal localization / temporal post-processing for existing chunk-level predictions.

Not:

> We need to redesign the head.

The initial contribution is low-risk because it can use existing saved predictions.

Deliverables:

1. Reproduce current inference output.
2. Implement Kalman filter / smoother over time.
3. Compare:
   - current confidence-weighted average
   - unweighted mean
   - median / robust mean if easy
   - Kalman filter
   - Kalman smoother
4. Compare orderings:
   - current: weighted average → Kabsch
   - proposed default: Kalman → Kabsch
   - optional: Kabsch per window → Kalman
5. Add time-to-stability curves:
   - after 3 seconds
   - after 30 seconds
   - after 1 minute
   - after 5 minutes
   - after 10 minutes
6. Report whether Kalman reduces extreme temporal outliers.

---

# 4. Hypothesis

Primary hypothesis:

> Kalman smoothing should reduce temporal extremes in AP/DV/ML predictions because the head’s confidence estimates are chunk-local and may not be calibrated enough to identify outliers.

Secondary hypothesis:

> Even if final RMSE improves modestly, Kalman will provide a useful “time-to-stability” estimate, which is more meaningful for online or real-time localization than final averaged RMSE alone.

Expected gain:

- Mean RMSE: maybe modest, possibly 0–8%.
- Worst-probe / worst-session behavior: possibly larger gain.
- Outlier reduction: likely more visible.
- Time-to-stability: likely the most useful new result.

---

# 5. Current Scripts

## 5.1 `infer_chronic_mishi_window.py`

Purpose:
- Runs the trained Path B 75K λ=3.0 model on chronic Mishi M05 NP2.0 sessions.
- Reads `.lfp` memmap files.
- Takes either a 5-minute mid-session window or the full session.
- Slices data into 3-second chunks.
- Runs backbone + Gaussian head.
- Saves `predictions.npz`.

Default sessions:

```python
SESSIONS = [
    "TES_sResp_M05_20240729",
    "TES_sResp_M05_20240730",
    "TES_sResp_M05_20240731",
    "TES_sResp_M05_20241014",
    "TES_sResp_M05_20241015",
]
```

Default checkpoint:

```text
/scratch/pl2820/ray_results/benchmarks/finetune_xyz_75k_joint_lambda_3_0_7741664/checkpoints/finetune_best_valmse.pt
```

Default output root:

```text
/scratch/pl2820/ray_results/eval_v2/chronic_mishi_v2
```

Important output keys in `predictions.npz`:

```text
pred_xyz_gauss      shape: (384, n_chunks, 3)
pred_sigma_gauss    shape: (384, n_chunks, 3)
channel_axial_um    shape: (384,)
channel_ids         shape: (384,)
pid                 session name
chunk_dur_sec       usually 3.0
fs_target           1250
win_start_sec
win_end_sec
raw_std
scale_factor
```

Command to run one session:

```bash
python infer_chronic_mishi_window.py \
  --ckpt /scratch/pl2820/ray_results/benchmarks/finetune_xyz_75k_joint_lambda_3_0_7741664/checkpoints/finetune_best_valmse.pt \
  --out_root /scratch/pl2820/ray_results/eval_v2/chronic_mishi_v2 \
  --sessions TES_sResp_M05_20240729 \
  --window_min 5
```

Command to run all default sessions:

```bash
python infer_chronic_mishi_window.py \
  --ckpt /scratch/pl2820/ray_results/benchmarks/finetune_xyz_75k_joint_lambda_3_0_7741664/checkpoints/finetune_best_valmse.pt \
  --out_root /scratch/pl2820/ray_results/eval_v2/chronic_mishi_v2 \
  --window_min 5
```

Full session option:

```bash
python infer_chronic_mishi_window.py \
  --ckpt /scratch/pl2820/ray_results/benchmarks/finetune_xyz_75k_joint_lambda_3_0_7741664/checkpoints/finetune_best_valmse.pt \
  --out_root /scratch/pl2820/ray_results/eval_v2/chronic_mishi_v2_full \
  --full_session
```

Warning:
- Full-session mode may be much larger and slower.
- Start with a 5-minute window first.

---

## 5.2 `viz_session_inference_timechunk.py`

Purpose:
- Renders channel × time-chunk predicted-region heatmaps.
- It voxel-lookups predicted AP/DV/ML coordinates into Allen CCF regions.
- It colors each cell by predicted region.
- It expects a `predictions.npz`.

Default hardcoded input root:

```text
/scratch/pl2820/ray_results/eval_v2/session_inference
```

Important issue:
- This visualization script appears originally written for IBL session inference output.
- It expects keys like:

```text
pred_xyz_gauss
channel_acronyms
channel_axial_um
pid
chunk_dur_sec
```

But `infer_chronic_mishi_window.py` does **not** save `channel_acronyms`.

So this visualization script may fail on chronic Mishi outputs unless modified.

Needed fix:
- Add fallback behavior when `channel_acronyms` is missing.
- For chronic Mishi data, there is no histology ground-truth region label.
- The left GT strip should either be hidden, blank, or replaced with channel index / axial depth.

Suggested modification:
- Add `--root` argument so we can point it at chronic output root.
- Add logic:

```python
if "channel_acronyms" in d:
    ch_acrs = ...
else:
    ch_acrs = np.array(["UNK"] * n_ch)
    has_gt_acronyms = False
```

Also:
- If no ground truth acronyms exist, skip `annotate_gt_strip`.
- Rename left strip label from “GT” to “Channel index / no GT”.

Potential command after patch:

```bash
python viz_session_inference_timechunk.py \
  --root /scratch/pl2820/ray_results/eval_v2/chronic_mishi_v2 \
  --all
```

Or for one session:

```bash
python viz_session_inference_timechunk.py \
  --root /scratch/pl2820/ray_results/eval_v2/chronic_mishi_v2 \
  --pid TES_sResp_M05_20240729
```

---

## 5.3 `eval_regression_probe.py`

Purpose:
- Evaluates frozen embeddings with linear regression probes and Gaussian probes.
- Trains:
  - MSE linear probe: `nn.Linear(768, 3)`
  - Gaussian linear probe: `nn.Linear(768, 6)`, split into `(μ, log σ²)`
  - optional hemisphere head
- Saves raw predictions to `<out>/predictions/*.npz`.
- Reports coordinate errors, Gaussian calibration, and region lookup accuracy.

Relevant ideas for this Kalman work:
- The Gaussian probe produces both coordinate predictions and sigma.
- The calibration section checks whether predicted sigma corresponds to actual residuals.
- This is useful because the Kalman filter can use sigma as observation noise.
- If sigma is poorly calibrated, Kalman may still help if we clamp or rescale observation noise.

Potential useful output keys:
- `pred_xyz_gauss`
- `pred_sigma_gauss`
- `true_xyz`
- `fine_id`
- `meta_str`

This script is more relevant for IBL-style evaluation with ground truth. For chronic Mishi, there is likely no true histology coordinate available, so metrics should focus on stability and consistency unless ground truth exists.

---

# 6. Access Checks

Before doing anything, verify access.

## 6.1 Check model checkpoint

```bash
ls -lah /scratch/pl2820/ray_results/benchmarks/finetune_xyz_75k_joint_lambda_3_0_7741664/
ls -lah /scratch/pl2820/ray_results/benchmarks/finetune_xyz_75k_joint_lambda_3_0_7741664/checkpoints/
```

Expected checkpoint:

```text
finetune_best_valmse.pt
```

## 6.2 Check Mishi data path

```bash
ls -lah /scratch/mc10168/mishi/
ls -lah /scratch/mc10168/mishi/5DAYS/
```

Expected sessions:

```text
TES_sResp_M05_20240729
TES_sResp_M05_20240730
TES_sResp_M05_20240731
TES_sResp_M05_20241014
TES_sResp_M05_20241015
```

Check for `.lfp` files:

```bash
find /scratch/mc10168/mishi/5DAYS -maxdepth 3 -name "*.lfp"
```

If permission denied:
- Need Subhrajit or Lawrence to grant path access.
- Ask for either:
  - group permission update
  - copied `.lfp` files
  - saved `predictions.npz` files
  - or access to `/scratch/mkp6112/Monkey/`

## 6.3 Check Python environment

Need imports:

```text
torch
numpy
matplotlib
nrrd
iblatlas
Alphabrain_staging modules
```

Check:

```bash
python - <<'PY'
import torch, numpy, matplotlib
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
PY
```

Check Alphabrain import path:

```bash
ls -lah /scratch/pl2820/Alphabrain_staging
ls -lah /scratch/pl2820/Alphabrain_staging/Lfp2vec_benchmarks
```

---

# 7. Kalman Implementation Plan

Create a new script:

```text
kalman_refine_predictions.py
```

Input:
- A `predictions.npz` file from `infer_chronic_mishi_window.py`.

Required keys:

```text
pred_xyz_gauss      (n_ch, n_chunks, 3)
pred_sigma_gauss    (n_ch, n_chunks, 3)
channel_axial_um
channel_ids
pid
chunk_dur_sec
```

Output:
- A new `predictions_kalman.npz` or updated file with extra arrays:

```text
pred_xyz_weighted_mean      (n_ch, 3)
pred_xyz_median             (n_ch, 3)
pred_xyz_kalman_filter      (n_ch, n_chunks, 3)
pred_xyz_kalman_smooth      (n_ch, n_chunks, 3)
pred_xyz_kalman_final       (n_ch, 3)
kalman_cov_filter           (n_ch, n_chunks, 3)
kalman_cov_smooth           (n_ch, n_chunks, 3)
time_to_stable_100um        (n_ch,)
time_to_stable_200um        (n_ch,)
```

Optional:
- region labels per method after Allen lookup.

---

# 8. Simple Kalman Model

For each channel and each axis independently:

State:

```text
x_t = true coordinate at chunk t
```

Observation:

```text
y_t = model-predicted coordinate at chunk t
```

Constant-position dynamics:

```text
x_t = x_{t-1} + process_noise
y_t = x_t + observation_noise
```

Parameters:

```text
Q = process noise variance
R_t = observation noise variance from model sigma
```

Observation noise:

```text
R_t = sigma_t^2
```

But because model sigma may be poorly calibrated, use:

```text
R_t = clamp((scale_R * sigma_t)^2, R_min, R_max)
```

Potential defaults:

```text
scale_R = 1.0
R_min = 25^2 or 50^2 um^2
R_max = 3000^2 or 5000^2 um^2
Q = 1^2 to 25^2 um^2 per 3-sec step for acute
Q = larger for chronic drift
```

Start simple:

```text
Q = 10^2 um^2
R_min = 50^2 um^2
R_max = 3000^2 um^2
```

Then tune.

---

# 9. Kalman First vs Kabsch First

Default hypothesis:

```text
Kalman first → Kabsch second
```

Reason:
- Chunk-level predictions have temporal extremes.
- Stabilize each channel over time first.
- Then force the stabilized per-channel estimates to respect probe geometry.

Default pipeline:

```text
AP/DV/ML per chunk + sigma
    → Kalman smoother per channel
    → stable coordinate estimate per channel
    → Kabsch refinement across channels
```

Also test:

```text
Kabsch per time window → Kalman smoother
```

Reason to test both:
- Kabsch first may help if per-chunk probe clouds are coherent enough.
- But Kabsch first may also be unstable if individual time chunks contain many outliers.

Experimental order ablation:

```text
A. current baseline: confidence-weighted average → Kabsch
B. Kalman → Kabsch
C. Kabsch per-window → Kalman
D. maybe robust Kalman → Kabsch
```

---

# 10. Time-to-Stability Metric

This may be the most useful result.

Question:

```text
How long does the model need to listen before localization stabilizes?
```

Compute online estimate at each time chunk:

```text
estimate_t = Kalman filtered estimate after chunks 1...t
```

Compare to final estimate:

```text
final_estimate = Kalman smoothed estimate using all chunks
```

For each channel:

```text
distance_t = ||estimate_t - final_estimate||_2
```

Define time to stability:

```text
first t such that distance_t < 100 um and remains below 100 um for K chunks
first t such that distance_t < 200 um and remains below 200 um for K chunks
```

Suggested K:

```text
K = 10 chunks = 30 seconds
```

Report:
- median time-to-stability across channels
- IQR across channels
- per-session stability curves
- percentage of channels stable by 30 sec, 1 min, 5 min

For sessions without ground truth:
- Use distance to final smoothed estimate.
- Also show region-label stability over time.

For sessions with ground truth:
- Use RMSE vs recording duration.

---

# 11. Metrics to Report

## 11.1 Chronic Mishi, likely no ground truth

Report:

```text
1. Distance-to-final estimate vs time
2. Time-to-stability at 100 um and 200 um
3. Outlier rate before vs after Kalman
4. Region-label flicker before vs after Kalman
5. Visual heatmaps:
   - raw chunk-level predictions
   - Kalman-smoothed predictions
6. Cross-day consistency:
   - July 29/30/31
   - October 14/15
   - July vs October shift
```

Outlier definition ideas:

```text
chunk is outlier if ||pred_t - median_channel_pred|| > 1000 um
or if ||pred_t - kalman_smooth_t|| > 1000 um
```

Region flicker:

```text
number of region transitions over time per channel
```

## 11.2 IBL / data with ground truth

Report:

```text
1. RMSE before Kabsch
2. RMSE after Kabsch
3. weighted average vs Kalman
4. worst-probe RMSE
5. time-to-RMSE threshold
6. calibration of sigma before/after scaling
```

---

# 12. Potential Quick Win

Even before implementing full Kalman, compute:

```text
weighted mean vs median vs trimmed mean
```

If median beats weighted mean, that strongly supports the outlier argument.

Baselines:

```text
Method 1: unweighted mean
Method 2: inverse-variance weighted mean
Method 3: median over chunks
Method 4: trimmed mean, e.g. remove top/bottom 10% per axis
Method 5: Kalman filter
Method 6: Kalman smoother
```

This helps establish that the current aggregation may be sensitive to bad confident predictions.

---

# 13. Important Script Compatibility Issues

## Issue 1: Visualization root is hardcoded

`viz_session_inference_timechunk.py` uses:

```python
SESSION_ROOT = Path("/scratch/pl2820/ray_results/eval_v2/session_inference")
```

But chronic script writes to:

```python
DEFAULT_OUT_ROOT = "/scratch/pl2820/ray_results/eval_v2/chronic_mishi_v2"
```

Fix:
- Add `--root` argument to visualization script.

## Issue 2: Missing `channel_acronyms`

`viz_session_inference_timechunk.py` expects:

```python
ch_acrs = np.array([str(a) for a in d["channel_acronyms"].tolist()])
```

But chronic script does not save `channel_acronyms`.

Fix:
- Use fallback `UNK` labels.
- Or remove GT strip for chronic data.

## Issue 3: Need to visualize Kalman outputs

The current viz script only visualizes:

```python
pred = d["pred_xyz_gauss"]
```

Add an argument:

```bash
--pred_key pred_xyz_gauss
--pred_key pred_xyz_kalman_smooth
```

Then render raw vs Kalman-smoothed heatmaps from the same script.

Possible command:

```bash
python viz_session_inference_timechunk.py \
  --root /scratch/pl2820/ray_results/eval_v2/chronic_mishi_v2 \
  --pid TES_sResp_M05_20240729 \
  --pred_key pred_xyz_kalman_smooth
```

---

# 14. Suggested File Outputs

For each session directory:

```text
/scratch/pl2820/ray_results/eval_v2/chronic_mishi_v2/<SESSION>/
    predictions.npz
    predictions_kalman.npz
    channel_timechunk_raw.png
    channel_timechunk_kalman.png
    stability_curve.png
    outlier_summary.json
```

Where `outlier_summary.json` contains:

```json
{
  "session": "...",
  "n_channels": 384,
  "n_chunks": 100,
  "chunk_dur_sec": 3.0,
  "median_time_to_stable_100um_sec": "...",
  "median_time_to_stable_200um_sec": "...",
  "raw_outlier_rate": "...",
  "kalman_outlier_rate": "...",
  "region_flicker_raw": "...",
  "region_flicker_kalman": "..."
}
```

---

# 15. Minimal Work Plan Before Thursday

## Step 1: Confirm access

Run:

```bash
ls -lah /scratch/mc10168/mishi/5DAYS/
find /scratch/mc10168/mishi/5DAYS -maxdepth 3 -name "*.lfp"
```

If access fails, ask for either:
- path permission
- copied `.lfp` files
- existing `predictions.npz`
- or access to `/scratch/mkp6112/Monkey/`

## Step 2: Confirm checkpoint

```bash
ls -lah /scratch/pl2820/ray_results/benchmarks/finetune_xyz_75k_joint_lambda_3_0_7741664/checkpoints/
```

## Step 3: Run inference for one 5-minute session

```bash
python infer_chronic_mishi_window.py \
  --ckpt /scratch/pl2820/ray_results/benchmarks/finetune_xyz_75k_joint_lambda_3_0_7741664/checkpoints/finetune_best_valmse.pt \
  --out_root /scratch/pl2820/ray_results/eval_v2/chronic_mishi_v2 \
  --sessions TES_sResp_M05_20240729 \
  --window_min 5
```

## Step 4: Inspect output

```bash
python - <<'PY'
import numpy as np
p = "/scratch/pl2820/ray_results/eval_v2/chronic_mishi_v2/TES_sResp_M05_20240729/predictions.npz"
d = np.load(p, allow_pickle=True)
print(d.files)
for k in d.files:
    arr = d[k]
    try:
        print(k, arr.shape, arr.dtype)
    except Exception:
        print(k, arr)
PY
```

## Step 5: Implement Kalman script

New file:

```text
kalman_refine_predictions.py
```

First version:
- read one predictions file
- run independent per-channel/per-axis Kalman filter
- run RTS smoother if time
- save `predictions_kalman.npz`

## Step 6: Add basic stability metrics

Produce:
- `stability_summary.json`
- `stability_curve.png`

## Step 7: Patch visualization

Modify `viz_session_inference_timechunk.py`:
- add `--root`
- add `--pred_key`
- handle missing `channel_acronyms`

## Step 8: Render raw vs Kalman

```bash
python viz_session_inference_timechunk.py \
  --root /scratch/pl2820/ray_results/eval_v2/chronic_mishi_v2 \
  --pid TES_sResp_M05_20240729 \
  --pred_key pred_xyz_gauss

python viz_session_inference_timechunk.py \
  --root /scratch/pl2820/ray_results/eval_v2/chronic_mishi_v2 \
  --pid TES_sResp_M05_20240729 \
  --pred_key pred_xyz_kalman_smooth
```

## Step 9: Run all five sessions if one works

```bash
python infer_chronic_mishi_window.py \
  --ckpt /scratch/pl2820/ray_results/benchmarks/finetune_xyz_75k_joint_lambda_3_0_7741664/checkpoints/finetune_best_valmse.pt \
  --out_root /scratch/pl2820/ray_results/eval_v2/chronic_mishi_v2 \
  --window_min 5
```

Then run Kalman over all session directories.

---

# 16. Message to Send if Access Is Still Blocked

```text
Quick update: I started setting up the Kalman temporal refinement, but I still do not seem to have access to the Mishi / monkey session paths. Could you either add my permissions for `/scratch/mc10168/mishi/5DAYS` and/or `/scratch/mkp6112/Monkey/`, or send me an existing `predictions.npz` output from one session?

For the first pass I only need the chunk-level model outputs:
- pred_xyz_gauss, shape (channels, chunks, 3)
- pred_sigma_gauss, shape (channels, chunks, 3)
- channel_axial_um
- chunk duration / session metadata

With that I can test Kalman vs current confidence-weighted aggregation without rerunning the full model.
```

---

# 17. TODO

## Access / Environment

- [ ] Check access to `/scratch/mc10168/mishi/`
- [ ] Check access to `/scratch/mc10168/mishi/5DAYS/`
- [ ] Check access to `/scratch/mkp6112/Monkey/`
- [ ] Check checkpoint exists under `/scratch/pl2820/ray_results/benchmarks/finetune_xyz_75k_joint_lambda_3_0_7741664/checkpoints/`
- [ ] Confirm Python environment has `torch`, `numpy`, `matplotlib`, `nrrd`, and Alphabrain modules
- [ ] If access fails, message Subhrajit/Lawrence for permissions or existing `predictions.npz`

## Run Existing Inference

- [ ] Run `infer_chronic_mishi_window.py` on one 5-minute session
- [ ] Inspect `predictions.npz` keys and shapes
- [ ] Run all five sessions once one session works

## Patch Visualization

- [ ] Add `--root` argument to `viz_session_inference_timechunk.py`
- [ ] Add `--pred_key` argument so we can visualize raw vs Kalman predictions
- [ ] Add fallback when `channel_acronyms` is missing
- [ ] Generate raw channel × time heatmap for one session
- [ ] Generate raw heatmaps for all five sessions

## Implement Kalman

- [ ] Create `kalman_refine_predictions.py`
- [ ] Load `pred_xyz_gauss` and `pred_sigma_gauss`
- [ ] Implement per-channel, per-axis constant-position Kalman filter
- [ ] Add sigma clipping / rescaling parameters
- [ ] Implement RTS smoother if time
- [ ] Save `pred_xyz_kalman_filter`
- [ ] Save `pred_xyz_kalman_smooth`
- [ ] Save `pred_xyz_kalman_final`
- [ ] Save Kalman uncertainty / covariance estimates

## Metrics

- [ ] Compute current inverse-variance weighted mean
- [ ] Compute unweighted mean
- [ ] Compute median / trimmed mean
- [ ] Compute Kalman final estimate
- [ ] Compute outlier rate before vs after Kalman
- [ ] Compute region-label flicker before vs after Kalman
- [ ] Compute time-to-stability at 100 µm
- [ ] Compute time-to-stability at 200 µm
- [ ] Plot distance-to-final-estimate vs time
- [ ] Plot percentage of channels stable vs time

## Kabsch Order Ablation

- [ ] Baseline: weighted average → Kabsch
- [ ] Proposed: Kalman → Kabsch
- [ ] Optional: Kabsch per-window → Kalman
- [ ] Compare visual trajectory stability
- [ ] Compare RMSE if ground truth is available
- [ ] Compare stability if ground truth is not available

## Thursday Deliverable

- [ ] One figure showing raw vs Kalman heatmap
- [ ] One stability curve
- [ ] One table comparing aggregation methods
- [ ] One short summary:
  - Did Kalman reduce outliers?
  - Did it stabilize region labels faster?
  - How long until localization stabilizes?
  - Does Kalman-first look better than current averaging?

---

# 18. One-Sentence Pitch

The current pipeline uses chunk-local uncertainty and confidence-weighted averaging, but the localization problem is temporally structured: electrode position should be stable or slowly drifting. A Kalman smoother can use the existing AP/DV/ML predictions and predicted variance as noisy observations, suppress temporal outliers, and estimate how much recording time is needed before localization becomes stable.

