# Spike-Based Probe Motion Validation + Kalman Temporal Refinement TODO

## 0. High-Level Idea

This project connects two complementary technologies:

1. **Geometry-Aware Self-Supervised Electrophysiology Representation Learning**
   - This paper predicts 3D Allen CCF coordinates from raw single-channel LFP.
   - Input: 3-second LFP chunks from one channel.
   - Output: predicted AP/DV/ML coordinate plus uncertainty.
   - Goal: localize electrode recording sites without relying only on post-hoc histology.

2. **End-to-End Spike Localization and Drift Correction Leveraging Structure From Motion**
   - This paper predicts 3D spike source coordinates for neural units/spikes and jointly estimates drift.
   - Input: multi-channel spike waveforms plus probe geometry.
   - Output: spike/unit x, y, z positions relative to the probe, plus drift-corrected trajectories.
   - Goal: improve spike localization and drift correction by coupling localization and motion estimation.

The new idea:

> Use the LFP-based coordinate predictor to estimate where channels are in Allen CCF space, then use Kalman filtering/smoothing to stabilize those predictions over time, and finally use spike localization / drift correction as an independent dynamic validation signal for how the probe moves through brain tissue.

This creates a stronger validation story than only comparing to histology-derived CCF labels.

---

# 1. Core Motivation

Current evaluation mostly depends on post-hoc histology:

```text
recording session
    ↓
animal sacrificed / tissue processed
    ↓
probe track reconstructed
    ↓
registered to Allen CCF
    ↓
channel coordinates used as ground truth
```

That is valuable, but limited:

- It is post-hoc, not online.
- It is destructive.
- It may contain registration error.
- It gives a static anatomical reconstruction.
- It does not fully explain within-session or day-to-day probe drift.
- It does not directly capture how neural units move relative to the probe over time.

The LFP paper already argues that neural signals contain anatomical information that can predict AP/DV/ML coordinates.

The spike localization paper adds a second dynamic signal:

```text
spike waveforms
    ↓
unit/source x, y, z localization relative to probe
    ↓
drift correction / motion field
    ↓
trajectory of units over time
```

Therefore, we can supplement static CCF evaluation with dynamic consistency:

```text
Does the LFP-predicted probe trajectory move in a way that agrees with spike-localized unit motion?
```

---

# 2. Important Terminology

## Allen CCF

Allen CCF = Allen Common Coordinate Framework.

This is the 3D atlas coordinate system used for mouse brain localization.

Coordinates:

```text
AP = anterior/posterior
DV = dorsal/ventral
ML = medial/lateral
```

The LFP paper predicts AP/DV/ML coordinates in micrometers.

## Histology-derived CCF ground truth

This is the anatomical reference coordinate obtained after recording by reconstructing the probe track in brain tissue and registering it to the Allen CCF.

This often requires post-hoc tissue processing, which generally means the animal is sacrificed and the brain is dissected / imaged / registered.

## Spike localization

Spike localization estimates the 3D position of each spike source relative to the probe.

The spike localization paper’s method predicts:

```text
x, y, z source coordinates
```

from spike waveforms and probe geometry.

This is not directly Allen CCF unless aligned to atlas space. It is usually a probe-relative coordinate system.

## Drift correction

Drift correction estimates how the tissue/probe relationship changes over time.

In Neuropixels recordings, apparent unit positions can shift because the probe and brain tissue move relative to each other.

---

# 3. Key Hypothesis

Main hypothesis:

> The Kalman-smoothed LFP coordinate trajectory should become more physically plausible over time, and its inferred channel/probe motion should agree with independent spike-based estimates of unit drift.

More simply:

```text
LFP model says:
    channel location / region assignment shifts over time

Spike localization says:
    units/spike clouds shift relative to probe over time

If both agree:
    the LFP model may be tracking real probe/tissue motion, not just noisy coordinate predictions
```

This is important because there may be no histology-aligned ground truth for chronic Mishi recordings.

So instead of only asking:

```text
How close is the LFP prediction to histology CCF?
```

we can also ask:

```text
Does the predicted probe movement agree with spike-based movement through the recording?
```

---

# 4. New Contribution Framing

The original Kalman pitch was:

> Improve downstream temporal aggregation of LFP-derived AP/DV/ML predictions.

The expanded pitch is:

> Build a dynamic localization validation framework that combines LFP-derived atlas localization, Kalman temporal smoothing, and spike-based motion estimates.

This is stronger because it creates a bridge between:

```text
LFP-based anatomical localization
```

and:

```text
spike-based unit localization / drift tracking
```

The result is a multi-evidence localization framework.

---

# 5. Why This Matters

The LFP coordinate predictor is trying to infer:

```text
Where is this recording channel in brain atlas space?
```

The spike localization system is trying to infer:

```text
Where are the spike sources / units relative to the probe, and how do they drift over time?
```

If we combine them, we may get:

```text
Where is the probe in the brain?
How stable is that estimate?
How does the probe or tissue move during a session?
How does the inferred movement compare to spike-derived drift?
How much recording time is needed before the localization stabilizes?
```

This could make the method useful for chronic recordings, where histology is unavailable during the experiment and probe drift can make traditional static localization unreliable.

---

# 6. Relationship Between the Two Papers

## Paper 1: Geometry-Aware Self-Supervised Electrophysiology Representation Learning

This is the atlas localization side.

Pipeline:

```text
single-channel 3-second LFP chunk
    ↓
audio SSL backbone / Wav2Vec2
    ↓
768-dimensional embedding
    ↓
Gaussian localization head
    ↓
AP/DV/ML coordinate + uncertainty
    ↓
confidence-weighted averaging
    ↓
Kabsch probe refinement
```

Important output:

```text
pred_xyz_gauss      shape: (channels, time_chunks, 3)
pred_sigma_gauss    shape: (channels, time_chunks, 3)
```

This is the model that gives us chunk-level anatomical coordinate estimates.

## Paper 2: End-to-End Spike Localization and Drift Correction Leveraging Structure From Motion

This is the spike/unit motion side.

Pipeline:

```text
detected spikes
    ↓
10-nearest-channel waveform tensor
    ↓
probe-geometry-aware transformer
    ↓
spike source x/y/z prediction
    ↓
DREDge drift estimation
    ↓
drift-corrected spike localization images
```

Important outputs:

```text
spike localizations over time
drift field over time
corrected spike positions
spatial histograms / localization images
pairwise NCC over time
```

This method gives an independent estimate of motion and spatial consistency.

---

# 7. Central Integration Idea

The LFP method predicts where the probe channels are in atlas coordinates.

The spike method predicts where neural units/spikes are relative to the probe and how they move over time.

If we can align the two, we can reverse-engineer probe movement.

Conceptual flow:

```text
LFP coordinate predictions:
    channel c at time t → AP/DV/ML estimate

Spike localization:
    spike/unit u at time t → x/y/z relative to probe

Drift correction:
    unit/source motion over time → tissue/probe displacement estimate

Compare:
    LFP-inferred probe/channel movement
    vs.
    spike-inferred tissue/probe movement
```

This can supplement ground truth.

---

# 8. Important Caveat

Do **not** claim spike localization is a replacement for histology ground truth.

Better framing:

```text
Histology-derived Allen CCF:
    anatomical reference

LFP model:
    signal-derived atlas coordinate predictor

Spike localization:
    independent dynamic physiological consistency signal

Kalman filter:
    temporal probabilistic smoother for chunk-level predictions
```

So the claim is:

> Spike localization can supplement CCF evaluation by providing an independent dynamic check on whether predicted channel/probe movement is physically plausible.

Not:

> Spike localization is the new ground truth.

---

# 9. Possible Strong Claim

Careful version:

> Geometry-Aware Self-Supervised Electrophysiology Representation Learning provides an LFP-based AP/DV/ML atlas coordinate readout for recording channels, while End-to-End Spike Localization and Drift Correction provides a spike-based estimate of unit/source movement relative to the probe. Combining them could allow us to evaluate not only static coordinate error against histology-derived Allen CCF labels, but also dynamic consistency: whether the inferred probe trajectory moves through space in agreement with spike-derived drift.

More ambitious version:

> This is a path toward online probe localization without requiring immediate histology: LFP provides an anatomical coordinate estimate, Kalman filtering stabilizes it over time, and spike localization provides an independent movement signal that can be used to validate or refine the inferred probe trajectory.

---

# 10. Why Kalman Still Matters

The LFP model’s uncertainty is local:

```text
chunk_t → AP/DV/ML + sigma
```

It does not know previous/future predictions.

A Kalman filter/smoother adds temporal structure:

```text
true channel location should not teleport between 3-second chunks
```

Kalman uses:

```text
observation:
    y_t = model-predicted AP/DV/ML

observation noise:
    R_t = predicted sigma_t^2

latent state:
    x_t = true channel AP/DV/ML

dynamics:
    x_t = x_{t-1} + small drift
```

This converts the model’s chunk-local uncertainty into trajectory-level uncertainty.

Then spike localization can be used to ask whether the resulting trajectory is plausible.

---

# 11. Proposed Full Pipeline

## Stage A: Generate LFP coordinate predictions

Use the existing LFP coordinate model.

Input:

```text
continuous LFP recording
```

Output:

```text
pred_xyz_gauss      (n_channels, n_chunks, 3)
pred_sigma_gauss    (n_channels, n_chunks, 3)
channel_axial_um
chunk_dur_sec
```

For each channel and time chunk:

```text
channel c, time chunk t → AP/DV/ML + sigma
```

---

## Stage B: Kalman smooth over time

For each channel independently:

```text
AP/DV/ML time series
    ↓
Kalman filter
    ↓
Kalman smoother
    ↓
stable AP/DV/ML trajectory
```

Output:

```text
pred_xyz_kalman_filter      (n_channels, n_chunks, 3)
pred_xyz_kalman_smooth      (n_channels, n_chunks, 3)
pred_xyz_kalman_final       (n_channels, 3)
kalman_cov_filter           (n_channels, n_chunks, 3)
kalman_cov_smooth           (n_channels, n_chunks, 3)
```

---

## Stage C: Kabsch / probe geometry refinement

After Kalman smoothing:

```text
stable coordinate estimate per channel
    ↓
Kabsch / line fitting / probe geometry constraint
    ↓
physically plausible probe trajectory
```

Default order:

```text
Kalman first → Kabsch second
```

Reason:

```text
first remove temporal outliers
then enforce probe geometry
```

Also test:

```text
Kabsch per window → Kalman
```

---

## Stage D: Spike localization / drift correction

Run the spike localization pipeline.

Input:

```text
raw AP-band data or detected spikes
10-nearest-channel waveforms
probe geometry
spike times
peak channel
```

Output:

```text
spike x/y/z source coordinates
drift field over time
drift-corrected spike positions
per-time-bin localization histograms
pairwise NCC matrix
```

---

## Stage E: Dynamic comparison

Compare the LFP-derived probe/channel trajectory with spike-derived unit/source motion.

Potential comparisons:

```text
LFP-predicted channel movement over time
vs.
spike-derived drift field over time
```

```text
LFP-predicted region boundary shift
vs.
shift in spike cloud depth / unit localization
```

```text
Kalman uncertainty decreasing over time
vs.
spike localization consistency increasing over time
```

---

# 12. Concrete Metrics

## 12.1 LFP-only metrics

These can be computed without spike data or histology.

```text
1. distance-to-final estimate vs time
2. time-to-stability at 100 µm
3. time-to-stability at 200 µm
4. outlier rate before vs after Kalman
5. region-label flicker before vs after Kalman
6. percentage of channels stable by 30 sec / 1 min / 5 min
7. within-session trajectory smoothness
8. day-to-day channel trajectory consistency
```

## 12.2 Spike-only metrics

From the spike localization paper:

```text
1. pairwise NCC of localization images
2. spatial entropy
3. drift trace smoothness
4. number of good units after sorting
5. good-to-total cluster fraction
6. total cluster count
```

## 12.3 LFP-spike agreement metrics

These are the new combined metrics.

### Metric 1: Drift correlation

Question:

```text
Does LFP-predicted movement agree with spike-derived drift?
```

Compute:

```text
LFP shift over time:
    Δ_LFP(t) = median_channel(pred_xyz_kalman_smooth[:, t, :] - pred_xyz_kalman_smooth[:, 0, :])

Spike shift over time:
    Δ_spike(t) = DREDge drift field or SLN+DREDge motion estimate

Compare:
    correlation(Δ_LFP(t), Δ_spike(t))
```

Possible axes:

```text
AP
DV
ML
probe-depth direction
```

Most likely useful axis:

```text
probe-depth / axial direction
```

### Metric 2: Directional agreement

Question:

```text
When spike localization says tissue moved up/down, does LFP localization imply the same direction?
```

Compute sign agreement:

```text
sign(Δ_LFP_depth(t)) == sign(Δ_spike_depth(t))
```

Report:

```text
percentage of time bins with matching motion direction
```

### Metric 3: Magnitude agreement

Question:

```text
Are the movement magnitudes similar?
```

Compute:

```text
MAE between normalized drift traces
RMSE between drift traces
slope from linear regression Δ_LFP ~ Δ_spike
```

### Metric 4: Region-boundary consistency

Question:

```text
When LFP-predicted region assignments shift across channels, do spike-derived unit locations shift in the same depth direction?
```

Example:

```text
LFP heatmap shows boundary moves from channel 120 to channel 135
Spike localization shows unit cloud shifts by ~15 channels / ~225 µm
```

Metric:

```text
boundary displacement agreement
```

### Metric 5: Time-to-certainty vs spike stability

Question:

```text
Does the LFP estimate stabilize when spike localization images become stable?
```

Compare:

```text
LFP Kalman uncertainty over time
vs.
pairwise NCC / localization image stability over time
```

### Metric 6: Cross-day consistency

For chronic recordings:

```text
July 29 → July 30 → July 31
October 14 → October 15
July → October
```

Questions:

```text
Are within-week changes smaller than across-month changes?
Does spike-derived drift show the same pattern?
Does LFP-derived channel region assignment shift similarly?
```

---

# 13. Possible Reverse-Engineering of Probe Location

The ambitious idea:

> Spike localization gives unit/source coordinates relative to the probe. If units can be tracked over time and their drift is estimated, then the inverse motion field can help infer how the probe moved relative to brain tissue.

Conceptually:

```text
spike/unit positions are observed relative to probe
unit population should be stable in brain tissue
apparent movement reflects probe/tissue drift
therefore:
    unit motion relative to probe can imply probe motion relative to tissue
```

If we already have an LFP-predicted CCF coordinate map:

```text
channel c → AP/DV/ML
```

then spike drift can help refine:

```text
probe trajectory through CCF over time
```

Potential reverse-engineering logic:

```text
1. Use LFP model to get initial CCF coordinate for each channel.
2. Use spike localization to estimate how unit/source positions shift relative to probe.
3. Interpret the opposite of the unit drift as probe/tissue displacement.
4. Apply that displacement to the LFP-derived channel coordinates.
5. Track the probe/channel trajectory over time in atlas space.
```

In formula-like terms:

```text
CCF_channel(c, t)
    ≈ CCF_channel_initial(c)
      + inferred_probe_motion_from_spikes(t)
      + Kalman_correction(t)
```

This is speculative but very interesting.

---

# 14. What We Need From the Data

## For LFP side

Need:

```text
predictions.npz
```

with:

```text
pred_xyz_gauss
pred_sigma_gauss
channel_axial_um
chunk_dur_sec
channel_ids
session id
```

Optional:

```text
channel_acronyms
true_xyz
histology-derived channel coordinates
```

## For spike side

Need:

```text
detected spike times
peak channel per spike
waveforms or tPCA-denoised waveforms
channel geometry
monopolar localizations
DREDge drift field
SLN predicted localizations if available
```

Ideal output from spike pipeline:

```text
spike_times
spike_depths
spike_x
spike_y
spike_z
corrected_spike_x
corrected_spike_y
corrected_spike_z
drift_x_t
drift_y_t
time_bins
```

## For combined analysis

Need shared session identity:

```text
same recording session
same probe
same channel geometry
same time base
same chunk duration or conversion between time bins
```

LFP chunks are probably every 3 seconds.

Spike drift may be estimated at 1-second bins or another bin size.

Need to align:

```text
LFP chunk t
    ↔
spike drift bins within same time interval
```

---

# 15. Implementation Plan

## Step 1: Get LFP predictions working

Run existing LFP inference.

Expected output:

```text
pred_xyz_gauss      (384, n_chunks, 3)
pred_sigma_gauss    (384, n_chunks, 3)
```

Save:

```text
predictions.npz
```

## Step 2: Run Kalman smoother

Create:

```text
kalman_refine_predictions.py
```

Outputs:

```text
pred_xyz_kalman_smooth
pred_xyz_kalman_final
kalman_uncertainty
time_to_stability
```

## Step 3: Produce LFP stability figures

Figures:

```text
raw channel × time predicted-region heatmap
Kalman-smoothed channel × time predicted-region heatmap
distance-to-final curve
percentage stable over time
outlier rate before/after Kalman
```

## Step 4: Run or obtain spike localization outputs

Options:

```text
A. Run End-to-End Spike Localization and Drift Correction pipeline
B. Use existing saved SLN+DREDge outputs
C. Use monopolar + DREDge baseline outputs if SLN is not available
```

Minimum needed for first pass:

```text
drift_y(t)
```

Better:

```text
drift_x(t)
drift_y(t)
corrected spike positions
per-bin histograms
pairwise NCC
```

## Step 5: Align time axes

Convert LFP chunks to time:

```text
lfp_time_t = t * chunk_dur_sec
```

Convert spike drift bins to same time base:

```text
spike_drift_time
```

Aggregate spike drift into LFP chunk bins:

```text
for each 3-second LFP chunk:
    use mean/median spike drift over that 3-second window
```

## Step 6: Compare movement traces

Compute:

```text
LFP channel trajectory shift over time
Spike drift trace over time
Correlation
Direction agreement
Magnitude agreement
```

## Step 7: Compare stability

Compute:

```text
LFP time-to-stability
Spike localization image NCC over time
Drift trace stability
```

Question:

```text
Does LFP become stable at the same time as spike localization becomes stable?
```

---

# 16. Possible Figures

## Figure 1: Combined pipeline diagram

```text
LFP stream → SSL coordinate model → AP/DV/ML + sigma → Kalman smoother → CCF probe trajectory

Spike stream → SLN / monopolar localizer → DREDge → unit/source motion field

Compare:
    LFP-derived probe movement
    spike-derived drift / unit movement
```

## Figure 2: Raw vs Kalman LFP heatmap

Panels:

```text
A. raw predicted region heatmap
B. Kalman-smoothed predicted region heatmap
C. difference / outlier removed map
```

## Figure 3: Time-to-stability

```text
x-axis: recording time
y-axis: distance to final estimate
lines: median / IQR across channels
thresholds: 100 µm, 200 µm
```

## Figure 4: LFP vs spike drift trace

```text
x-axis: time
y-axis: displacement
line 1: LFP-derived displacement
line 2: spike-derived DREDge displacement
```

## Figure 5: Agreement scatter

```text
x-axis: spike-derived drift
y-axis: LFP-derived shift
metric: correlation / slope / R²
```

## Figure 6: Cross-day chronic trajectory

Rows:

```text
Jul 29
Jul 30
Jul 31
Oct 14
Oct 15
```

Columns:

```text
LFP region heatmap
Kalman-smoothed heatmap
spike drift trace
combined inferred probe movement
```

---

# 17. Technical Risks

## Risk 1: Coordinate systems may not match directly

LFP model predicts Allen CCF coordinates.

Spike localization predicts probe-relative coordinates.

Need alignment:

```text
probe-relative coordinate → channel axial/depth coordinate → approximate CCF direction
```

At first, compare along probe-depth axis, not full AP/DV/ML.

## Risk 2: Spike drift is tissue-relative, not exactly probe motion

Spike localization tracks apparent unit movement relative to the probe.

This may reflect:

```text
brain tissue motion
probe motion
recording instability
unit dropout / appearance
sorting artifacts
```

So be careful.

Use language:

```text
spike-derived drift / tissue-probe relative motion
```

not:

```text
absolute probe movement
```

unless validated.

## Risk 3: No shared session between LFP outputs and spike localization outputs

Need the same recording.

If the spike paper outputs are on different datasets, this becomes a conceptual extension, not an immediate experiment.

## Risk 4: No histology ground truth for chronic Mishi

That is okay.

For chronic Mishi, use:

```text
dynamic consistency
within-session stability
cross-day reproducibility
LFP-spike drift agreement
```

rather than CCF RMSE.

## Risk 5: Kalman may oversmooth real drift

Need tune process noise Q.

If Q too low:

```text
real drift gets flattened
```

If Q too high:

```text
outliers leak through
```

Test multiple Q values:

```text
Q = 1^2, 5^2, 10^2, 25^2, 50^2 µm² per chunk
```

---

# 18. Suggested Analysis Order

## First milestone: LFP-only Kalman

Goal:

```text
Show raw vs Kalman stability on chronic sessions.
```

Tasks:

```text
1. Run LFP inference.
2. Run Kalman smoothing.
3. Generate raw and smoothed heatmaps.
4. Compute time-to-stability.
5. Compute outlier rate.
```

## Second milestone: spike drift comparison

Goal:

```text
Use spike drift trace as independent motion signal.
```

Tasks:

```text
1. Obtain DREDge or SLN+DREDge drift outputs.
2. Align time bins with LFP chunks.
3. Compare LFP shift vs spike drift.
4. Plot correlation and agreement.
```

## Third milestone: probe movement inference

Goal:

```text
Estimate probe/channel trajectory over time.
```

Tasks:

```text
1. Use LFP model for initial atlas coordinate.
2. Use Kalman for temporal stabilization.
3. Use spike-derived motion as dynamic correction or validation.
4. Produce probe trajectory over time.
```

---

# 19. Possible Naming

Possible project names:

```text
Temporal Probe Localization
Dynamic CCF Localization
LFP-Spike Probe Tracking
Kalman-Refined Signal-Based Probe Localization
Spike-Validated LFP Localization
Dynamic Probe Tracking from LFP and Spikes
```

Best working title:

```text
Spike-Validated Temporal Refinement of LFP-Based Electrode Localization
```

Alternative:

```text
Dynamic Probe Localization by Combining LFP-Based CCF Prediction with Spike-Based Drift Tracking
```

---

# 20. Message / Pitch to Collaborators

```text
I think there is a bigger opportunity here than just smoothing the LFP predictions. The LFP paper gives us chunk-level AP/DV/ML coordinate predictions from raw LFP, but evaluation is still mostly tied to static histology-derived CCF coordinates. The spike localization paper gives us a complementary dynamic signal: estimated 3D spike/unit locations and drift over time.

My proposal is to first add a Kalman smoother over the LFP-derived AP/DV/ML predictions, using the model’s predicted sigma as observation noise. That should suppress chunk-level outliers and give us a time-to-stability estimate.

Then, for chronic recordings where histology is unavailable or incomplete, we can supplement static CCF evaluation with a dynamic consistency metric: does the Kalman-smoothed LFP probe trajectory agree with spike-derived drift or unit movement from SLN+DREDge / DREDge?

In other words, histology remains the anatomical reference, but spike localization gives us an independent movement signal. If the LFP-inferred probe movement and spike-derived drift agree, that strengthens the case that the model is tracking real probe/tissue motion rather than just producing noisy coordinate estimates.
```

---

# 21. Concrete TODO

## Access / Data

- [ ] Confirm access to chronic Mishi LFP data.
- [ ] Confirm access to monkey / Neuropixels sessions.
- [ ] Confirm whether spike data exists for the same sessions as the LFP coordinate predictions.
- [ ] Ask whether SLN+DREDge outputs already exist.
- [ ] Ask whether DREDge drift traces already exist.
- [ ] Ask whether monopolar localizations already exist.
- [ ] Ask whether detected spikes / waveforms are available.
- [ ] Ask whether there is any histology-derived CCF for these chronic sessions.

## LFP Inference

- [ ] Run `infer_chronic_mishi_window.py` on one 5-minute session.
- [ ] Confirm `pred_xyz_gauss` shape.
- [ ] Confirm `pred_sigma_gauss` shape.
- [ ] Run full session if 5-minute test works.
- [ ] Run all available chronic days.

## Kalman Smoothing

- [ ] Implement constant-position Kalman filter.
- [ ] Implement RTS smoother.
- [ ] Use `pred_sigma_gauss` as observation noise.
- [ ] Add sigma clipping.
- [ ] Add process-noise sweep.
- [ ] Save filtered and smoothed AP/DV/ML trajectories.
- [ ] Compute final stable coordinate estimate per channel.
- [ ] Compute uncertainty over time.
- [ ] Compare Kalman vs weighted mean vs median vs trimmed mean.

## LFP Visualization

- [ ] Patch heatmap script to accept `--root`.
- [ ] Patch heatmap script to accept `--pred_key`.
- [ ] Patch heatmap script to handle missing `channel_acronyms`.
- [ ] Render raw predicted-region heatmap.
- [ ] Render Kalman-smoothed predicted-region heatmap.
- [ ] Render difference/outlier map.
- [ ] Render channel-wise stability curve.
- [ ] Render session-level stability summary.

## Spike Localization / Drift

- [ ] Locate spike localization code or outputs.
- [ ] Determine whether SLN model is available.
- [ ] Determine whether DREDge outputs are saved.
- [ ] Extract spike drift trace over time.
- [ ] Extract spike x/y/z localizations if possible.
- [ ] Extract corrected spike x/y/z localizations if possible.
- [ ] Extract pairwise NCC / entropy if available.
- [ ] Run SLN+DREDge if outputs are not available and data exists.

## Time Alignment

- [ ] Convert LFP chunk index to seconds.
- [ ] Convert spike drift bins to seconds.
- [ ] Aggregate spike drift into 3-second windows.
- [ ] Match each LFP chunk to spike drift estimate.
- [ ] Verify session start time alignment.
- [ ] Handle missing spike bins or low-spike-count windows.

## LFP-Spike Agreement

- [ ] Compute LFP-derived displacement trace.
- [ ] Compute spike-derived drift trace.
- [ ] Compare along probe-depth axis.
- [ ] Compute correlation.
- [ ] Compute direction agreement.
- [ ] Compute magnitude agreement.
- [ ] Compute lagged correlation in case one signal is delayed.
- [ ] Compare within-session stability.
- [ ] Compare cross-day consistency.

## Probe Motion Inference

- [ ] Define initial probe trajectory from Kalman-smoothed LFP CCF coordinates.
- [ ] Define spike-derived drift as relative tissue/probe displacement.
- [ ] Apply spike-derived displacement to LFP trajectory as a dynamic correction candidate.
- [ ] Compare corrected vs uncorrected trajectories.
- [ ] Visualize inferred probe movement in CCF space.
- [ ] Report whether movement is physically plausible.

## Evaluation Without Histology

- [ ] Report time-to-stability.
- [ ] Report region-label flicker.
- [ ] Report LFP-spike drift agreement.
- [ ] Report cross-day reproducibility.
- [ ] Report within-week vs across-month shift.
- [ ] Report spatial smoothness of probe trajectory.
- [ ] Report whether Kalman reduces extreme jumps.

## Evaluation With Histology, If Available

- [ ] Compute RMSE raw vs weighted mean vs Kalman.
- [ ] Compute RMSE before Kabsch.
- [ ] Compute RMSE after Kabsch.
- [ ] Compute worst-probe improvement.
- [ ] Compute outlier reduction.
- [ ] Compute time-to-RMSE threshold.
- [ ] Check whether spike-informed correction improves CCF error.

---

# 22. First Minimal Experiment

If time is short, do this first:

```text
1. Take one chronic session with existing LFP predictions.
2. Run Kalman smoothing over pred_xyz_gauss.
3. Plot raw vs Kalman predicted-region heatmap.
4. Compute time-to-stability.
5. If spike drift exists, overlay spike-derived drift trace against LFP-derived displacement.
```

This gives a first proof of concept.

---

# 23. First Strong Figure

Best first figure:

```text
Panel A: raw LFP predicted-region heatmap
Panel B: Kalman-smoothed LFP predicted-region heatmap
Panel C: distance-to-final estimate vs time
Panel D: spike-derived drift trace vs LFP-derived displacement trace
```

Caption idea:

```text
Kalman smoothing suppresses chunk-level extremes in LFP-derived AP/DV/ML predictions and yields a stable channel trajectory. Spike-derived drift provides an independent physiological motion signal; agreement between the two suggests the LFP model is tracking real tissue/probe movement rather than only producing noisy coordinate estimates.
```

---

# 24. Final One-Sentence Summary

Geometry-Aware Self-Supervised Electrophysiology Representation Learning provides the first LFP-based AP/DV/ML atlas coordinate readout from raw neural signals, while End-to-End Spike Localization and Drift Correction provides a spike-based x/y/z motion signal for neural units; combining them with Kalman smoothing could turn static post-hoc localization into dynamic, spike-validated probe tracking over time.

