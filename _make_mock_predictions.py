"""Generate a synthetic predictions.npz matching the chronic Mishi schema.

Used to test kalman_refine_predictions.py before real data is available.
Underscore prefix marks this as a dev/testing helper, not a pipeline script.

Usage:
    python _make_mock_predictions.py [output.npz]
"""
import sys
import numpy as np

N_CH = 384
N_CHUNKS = 100        # 5 min @ 3 s/chunk
CHUNK_DUR = 3.0
SR = 1250
rng = np.random.default_rng(42)

# Per-channel "true" CCF position: ramp AP from 4000→9000 µm along the probe,
# DV around 3000 µm, LR around 4000 µm (left of Allen midline 5739).
true_ap = np.linspace(4000.0, 9000.0, N_CH, dtype=np.float32)
true_dv = np.full(N_CH, 3000.0, dtype=np.float32) + rng.normal(0, 50, N_CH).astype(np.float32)
true_ml = np.full(N_CH, 4000.0, dtype=np.float32) + rng.normal(0, 100, N_CH).astype(np.float32)
true_xyz_ch = np.stack([true_ap, true_dv, true_ml], axis=1)            # (n_ch, 3)

# Small drift over time so a constant-position Kalman is mildly stressed
drift = rng.normal(0, 5, (N_CH, N_CHUNKS, 3)).astype(np.float32).cumsum(axis=1)
true_xyz = true_xyz_ch[:, None, :] + drift                              # (n_ch, n_chunks, 3)

# Observation noise: per-axis sigma roughly in [100, 800] µm with structure
sigma = rng.uniform(100.0, 800.0, (N_CH, N_CHUNKS, 3)).astype(np.float32)

# Gaussian noise scaled by sigma
noise = (rng.standard_normal((N_CH, N_CHUNKS, 3)).astype(np.float32) * sigma)

# Sprinkle 2% outlier chunks: huge offsets the head doesn't flag in sigma
outlier_mask = (rng.random((N_CH, N_CHUNKS, 3)) < 0.02)
outlier_kick = (rng.standard_normal((N_CH, N_CHUNKS, 3)).astype(np.float32) * 2500.0)
noise += outlier_mask * outlier_kick

pred = true_xyz + noise

# NP2.0 single-shank axial: 192 rows × 2 cols, 15 µm row pitch
rows = np.arange(N_CH // 2, dtype=np.float32) * 15.0
axial = np.repeat(rows, 2)

out = sys.argv[1] if len(sys.argv) > 1 else "mock_predictions.npz"
np.savez_compressed(
    out,
    pred_xyz_gauss=pred.astype(np.float32),
    pred_sigma_gauss=sigma,
    channel_axial_um=axial,
    channel_ids=np.arange(N_CH, dtype=np.int64),
    pid=np.array("MOCK_SESSION", dtype=object),
    chunk_dur_sec=np.float64(CHUNK_DUR),
    fs_target=np.float64(SR),
    win_start_sec=np.float64(0.0),
    win_end_sec=np.float64(N_CHUNKS * CHUNK_DUR),
    raw_std=np.float64(100.0),
    scale_factor=np.float64(8.9e-7),
)
print(f"wrote {out}")
print(f"  pred:  {pred.shape} {pred.dtype}   range AP[{pred[..., 0].min():.0f},{pred[..., 0].max():.0f}]")
print(f"  sigma: {sigma.shape} {sigma.dtype}  range [{sigma.min():.0f},{sigma.max():.0f}]")
print(f"  outliers seeded: {int(outlier_mask.sum())} cells "
      f"({outlier_mask.mean():.2%} of {N_CH * N_CHUNKS * 3} (ch,chunk,axis) slots)")
