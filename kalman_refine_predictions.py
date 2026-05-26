#!/usr/bin/env python3
"""Kalman temporal refinement of chunk-level AP/DV/ML predictions.

Reads a `predictions.npz` produced by `infer_chronic_mishi_window.py`. For
each (channel, axis) independently, runs a 1D constant-position Kalman
filter + RTS smoother where the observation noise is the head's per-chunk
sigma (clamped to avoid pathological values from a poorly-calibrated head).

Also computes the four baseline aggregations the paper / dev-notes call
for, and per-channel time-to-stability vs the smoother's final estimate.

Output `predictions_kalman.npz` keys are documented in the README block
inside this script and in dev-notes section 7.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# core math
# ---------------------------------------------------------------------------

def kalman_filter_rts(y, sigma, Q, R_min_sq, R_max_sq, scale_R):
    """Per (channel, axis) 1D constant-position Kalman filter + RTS smoother.

    Vectorized across the leading (n_ch, ..., 3) dims so the only Python-level
    loops are over the time axis. For 384 channels × ~100 chunks × 3 axes this
    runs in well under a second.

    Model:
        x_t = x_{t-1} + w_t,    w_t ~ N(0, Q)
        y_t = x_t + v_t,        v_t ~ N(0, R_t)
        R_t = clip((scale_R * sigma_t)^2, R_min_sq, R_max_sq)

    Args:
        y:         (n_ch, n_chunks, 3) observations in µm
        sigma:     (n_ch, n_chunks, 3) per-axis sigma in µm
        Q:         scalar process-noise variance in µm² per chunk
        R_min_sq:  scalar lower clamp on observation variance, µm²
        R_max_sq:  scalar upper clamp on observation variance, µm²
        scale_R:   scalar multiplier on input sigma before squaring

    Returns:
        x_filter:  (n_ch, n_chunks, 3) filter mean
        P_filter:  (n_ch, n_chunks, 3) filter variance
        x_smooth:  (n_ch, n_chunks, 3) RTS smoother mean
        P_smooth:  (n_ch, n_chunks, 3) RTS smoother variance
        R_used:    (n_ch, n_chunks, 3) clamped observation variance
    """
    _, T, _ = y.shape
    R_used = np.clip((scale_R * sigma) ** 2, R_min_sq, R_max_sq).astype(np.float64)

    x_filter = np.zeros_like(y, dtype=np.float64)
    P_filter = np.zeros_like(y, dtype=np.float64)

    # Initialize with the first observation. P_0 = R_0 captures the fact that
    # our first belief is exactly as confident as the first sigma allows.
    x_filter[:, 0, :] = y[:, 0, :]
    P_filter[:, 0, :] = R_used[:, 0, :]

    # forward pass
    for t in range(1, T):
        x_pred = x_filter[:, t - 1, :]
        P_pred = P_filter[:, t - 1, :] + Q
        K = P_pred / (P_pred + R_used[:, t, :])
        x_filter[:, t, :] = x_pred + K * (y[:, t, :] - x_pred)
        P_filter[:, t, :] = (1.0 - K) * P_pred

    # RTS backward pass
    x_smooth = x_filter.copy()
    P_smooth = P_filter.copy()
    for t in range(T - 2, -1, -1):
        P_pred_next = P_filter[:, t, :] + Q
        C = P_filter[:, t, :] / P_pred_next
        x_smooth[:, t, :] = x_filter[:, t, :] + C * (
            x_smooth[:, t + 1, :] - x_filter[:, t, :]
        )
        P_smooth[:, t, :] = P_filter[:, t, :] + C ** 2 * (
            P_smooth[:, t + 1, :] - P_pred_next
        )

    return x_filter, P_filter, x_smooth, P_smooth, R_used


def aggregate_baselines(y, sigma, trim_frac=0.1):
    """Per-channel reference aggregations for comparison with Kalman output.

    Returns dict[name -> (n_ch, 3) float32]:
        unweighted_mean — plain mean over chunks
        weighted_mean   — inverse-variance weighted mean (paper's current default)
        median          — per-axis median over chunks
        trimmed_mean    — drop top/bottom `trim_frac` per (channel, axis) then mean
    """
    _, T, _ = y.shape
    inv_var = 1.0 / np.clip(sigma ** 2, 1e-6, None)

    unweighted = y.mean(axis=1)
    weighted = (y * inv_var).sum(axis=1) / inv_var.sum(axis=1)
    median = np.median(y, axis=1)

    k = int(np.floor(trim_frac * T))
    if k > 0 and 2 * k < T:
        sorted_y = np.sort(y, axis=1)
        trimmed = sorted_y[:, k:T - k, :].mean(axis=1)
    else:
        trimmed = unweighted.copy()

    return {
        "unweighted_mean": unweighted.astype(np.float32),
        "weighted_mean":   weighted.astype(np.float32),
        "median":          median.astype(np.float32),
        "trimmed_mean":    trimmed.astype(np.float32),
    }


def time_to_stable(distance, threshold, K):
    """First chunk index where `distance < threshold` for K consecutive chunks.

    Args:
        distance:  (n_ch, n_chunks) float — typically ||filter_est_t - final||
        threshold: scalar µm
        K:         consecutive chunks required (default in main: 10 = 30 s)

    Returns:
        (n_ch,) int32 — first stable chunk index, or -1 if never stable.
    """
    n_ch, T = distance.shape
    out = np.full(n_ch, -1, dtype=np.int32)
    if T < K:
        return out

    below = distance < threshold  # (n_ch, T) bool
    # Rolling sum of width K via K shifted sums. Each entry of `window_sum`
    # is the count of True in `below[ch, t:t+K]`.
    window_sum = np.zeros((n_ch, T - K + 1), dtype=np.int32)
    for k in range(K):
        window_sum += below[:, k:T - K + 1 + k].astype(np.int32)
    stable = window_sum == K  # (n_ch, T-K+1)

    any_stable = stable.any(axis=1)
    # argmax over booleans returns the index of the first True
    out[any_stable] = np.argmax(stable[any_stable], axis=1).astype(np.int32)
    return out


def outlier_rate(y, reference, threshold_um):
    """Fraction of (channel, chunk) predictions farther than threshold_um from reference.

    Args:
        y:         (n_ch, T, 3) predictions
        reference: (n_ch, 3) or (n_ch, T, 3) reference position(s)
        threshold_um: scalar µm

    Returns:
        (rate, dist) where rate is scalar in [0, 1] and dist is (n_ch, T) µm.
    """
    if reference.ndim == 2:
        reference = reference[:, None, :]
    dist = np.linalg.norm(y - reference, axis=-1)
    return float((dist > threshold_um).mean()), dist


# ---------------------------------------------------------------------------
# I/O + CLI
# ---------------------------------------------------------------------------

def load_predictions(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _fmt_optional_seconds(arr_idx, chunk_dur):
    """Format median of stable times nicely; 'never' if no channel stable."""
    valid = arr_idx[arr_idx >= 0]
    if len(valid) == 0:
        return "never"
    return f"{float(np.median(valid.astype(np.float32) * chunk_dur)):.1f}s"


def main():
    ap = argparse.ArgumentParser(
        description="Kalman temporal refinement of chunk-level AP/DV/ML predictions",
    )
    ap.add_argument("input", help="Path to predictions.npz from infer_chronic_mishi_window.py")
    ap.add_argument("-o", "--output", default=None,
                    help="Output npz (default: predictions_kalman.npz next to input)")
    ap.add_argument("--Q", type=float, default=100.0,
                    help="Process-noise variance µm² per chunk (default 10² = 100; "
                         "raise for chronic drift)")
    ap.add_argument("--R_min", type=float, default=50.0,
                    help="Lower clamp on observation sigma in µm (default 50)")
    ap.add_argument("--R_max", type=float, default=3000.0,
                    help="Upper clamp on observation sigma in µm (default 3000)")
    ap.add_argument("--scale_R", type=float, default=1.0,
                    help="Multiplier on input sigma before squaring (default 1.0)")
    ap.add_argument("--stability_K", type=int, default=10,
                    help="Chunks required below threshold to call a channel stable "
                         "(default 10 ≈ 30s at 3s/chunk)")
    ap.add_argument("--outlier_um", type=float, default=1000.0,
                    help="Outlier distance threshold in µm (default 1000)")
    ap.add_argument("--trim_frac", type=float, default=0.1,
                    help="Trim fraction per tail for trimmed mean (default 0.10)")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path.with_stem(in_path.stem + "_kalman")
    print(f"[kalman] in : {in_path}")
    print(f"[kalman] out: {out_path}")

    data = load_predictions(in_path)
    y     = np.asarray(data["pred_xyz_gauss"],   dtype=np.float32)
    sigma = np.asarray(data["pred_sigma_gauss"], dtype=np.float32)
    if y.ndim != 3 or y.shape != sigma.shape:
        raise SystemExit(f"shape mismatch: y {y.shape} vs sigma {sigma.shape}")
    n_ch, T, _ = y.shape
    chunk_dur = float(data.get("chunk_dur_sec", 3.0))
    pid = str(data.get("pid", "unknown"))
    print(f"[kalman]   pid={pid}  (n_ch={n_ch}, n_chunks={T}, 3)  "
          f"chunk_dur={chunk_dur}s  → {T * chunk_dur / 60:.1f} min")

    # Kalman + RTS
    print(f"\n[kalman] running filter+smoother: "
          f"Q={args.Q:.0f} µm²  R∈[{args.R_min:.0f},{args.R_max:.0f}] µm  scale_R={args.scale_R}")
    x_filter, P_filter, x_smooth, P_smooth, R_used = kalman_filter_rts(
        y, sigma, args.Q, args.R_min ** 2, args.R_max ** 2, args.scale_R,
    )

    # Baselines
    print("[kalman] computing baseline aggregations…")
    baselines = aggregate_baselines(y, sigma, trim_frac=args.trim_frac)

    # Final per-channel estimates from each method
    kalman_final = x_filter[:, -1, :]         # online: filter at last chunk
    smooth_final = x_smooth[:, -1, :]         # offline: smoother at last chunk (= filter[T-1])
    smooth_mean  = x_smooth.mean(axis=1)      # offline robust average of smoothed trajectory

    # Distance from the *online* filter trajectory to the offline-mean target.
    # This is what "time-to-stability" asks: at chunk t, how far is your online
    # estimate from the value you would converge to given the entire window?
    final_target = smooth_mean
    distance_to_final = np.linalg.norm(
        x_filter - final_target[:, None, :], axis=-1,
    )

    t100 = time_to_stable(distance_to_final, 100.0, args.stability_K)
    t200 = time_to_stable(distance_to_final, 200.0, args.stability_K)

    print(f"\n[kalman] time-to-stability (K={args.stability_K} chunks = "
          f"{args.stability_K * chunk_dur:.0f}s):")
    print(f"  100µm: {int((t100 >= 0).sum())}/{n_ch} ch stable, "
          f"median {_fmt_optional_seconds(t100, chunk_dur)}")
    print(f"  200µm: {int((t200 >= 0).sum())}/{n_ch} ch stable, "
          f"median {_fmt_optional_seconds(t200, chunk_dur)}")

    # Outlier rates
    raw_or, raw_dist = outlier_rate(y, baselines["median"], args.outlier_um)
    kal_or, kal_dist = outlier_rate(y, x_smooth, args.outlier_um)
    print(f"\n[kalman] outlier rate (>{args.outlier_um:.0f}µm from reference):")
    print(f"  raw vs per-channel median:    {raw_or:7.4%}")
    print(f"  raw vs Kalman-smoothed track: {kal_or:7.4%}")

    # ---- save ----
    print(f"\n[kalman] saving {out_path} …")
    np.savez_compressed(
        out_path,
        # Kalman trajectories
        pred_xyz_kalman_filter=x_filter.astype(np.float32),
        pred_xyz_kalman_smooth=x_smooth.astype(np.float32),
        kalman_cov_filter=P_filter.astype(np.float32),
        kalman_cov_smooth=P_smooth.astype(np.float32),
        kalman_R_used=R_used.astype(np.float32),
        # Per-channel final estimates
        pred_xyz_kalman_final=kalman_final.astype(np.float32),
        pred_xyz_smooth_final=smooth_final.astype(np.float32),
        pred_xyz_smooth_mean=smooth_mean.astype(np.float32),
        # Baseline aggregations
        pred_xyz_unweighted_mean=baselines["unweighted_mean"],
        pred_xyz_weighted_mean=baselines["weighted_mean"],
        pred_xyz_median=baselines["median"],
        pred_xyz_trimmed_mean=baselines["trimmed_mean"],
        # Stability + outlier metrics
        distance_to_final=distance_to_final.astype(np.float32),
        time_to_stable_100um=t100,
        time_to_stable_200um=t200,
        outlier_distance_raw=raw_dist.astype(np.float32),
        outlier_distance_kalman=kal_dist.astype(np.float32),
        # Pass-through of the original inputs so viz can read either file
        pred_xyz_gauss=y,
        pred_sigma_gauss=sigma,
        channel_axial_um=data.get("channel_axial_um"),
        channel_ids=data.get("channel_ids"),
        pid=data.get("pid"),
        chunk_dur_sec=data.get("chunk_dur_sec"),
        win_start_sec=data.get("win_start_sec"),
        win_end_sec=data.get("win_end_sec"),
        # Run params
        kalman_params=np.array(json.dumps({
            "Q": args.Q,
            "R_min": args.R_min,
            "R_max": args.R_max,
            "scale_R": args.scale_R,
            "stability_K": args.stability_K,
            "outlier_um": args.outlier_um,
            "trim_frac": args.trim_frac,
        }), dtype=object),
    )

    # Sidecar JSON for quick eyeballing without np.load
    summary = {
        "session": pid,
        "n_channels": int(n_ch),
        "n_chunks": int(T),
        "chunk_dur_sec": chunk_dur,
        "duration_min": T * chunk_dur / 60,
        "median_time_to_stable_100um_sec": (
            float(np.median(t100[t100 >= 0].astype(np.float32) * chunk_dur))
            if (t100 >= 0).any() else None
        ),
        "median_time_to_stable_200um_sec": (
            float(np.median(t200[t200 >= 0].astype(np.float32) * chunk_dur))
            if (t200 >= 0).any() else None
        ),
        "frac_channels_stable_100um": float((t100 >= 0).mean()),
        "frac_channels_stable_200um": float((t200 >= 0).mean()),
        "raw_outlier_rate": raw_or,
        "kalman_outlier_rate": kal_or,
        "kalman_params": {
            "Q": args.Q, "R_min": args.R_min, "R_max": args.R_max,
            "scale_R": args.scale_R, "stability_K": args.stability_K,
            "outlier_um": args.outlier_um, "trim_frac": args.trim_frac,
        },
    }
    json_path = out_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[kalman] sidecar {json_path}")
    print("[kalman] done.")


if __name__ == "__main__":
    main()
