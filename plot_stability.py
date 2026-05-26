#!/usr/bin/env python3
"""Plot per-session stability curves and aggregation comparison table.

Reads a `predictions_kalman.npz` produced by `kalman_refine_predictions.py`.

Generates two artifacts next to the input npz:
  - stability_curve.png   — distance(filter_t, final) percentiles vs time
  - aggregation_table.txt — pretty text table comparing aggregation methods
                            (mean spread / disagreement with the Kalman smoother)

For chronic Mishi we have no ground truth, so all metrics are method-agnostic:
either distance-to-Kalman-smoothed-final (a strong "best estimate" target) or
spread of per-channel predictions across methods.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def stability_curve(distance_to_final, chunk_dur, out_path,
                    thresholds=(100, 200), pid="(unknown)"):
    """Plot distance_to_final percentiles over time + threshold lines.

    Args:
        distance_to_final: (n_ch, n_chunks) µm
        chunk_dur:         seconds per chunk
        out_path:          PNG path
        thresholds:        list of µm thresholds to draw horizontal lines for
    """
    n_ch, T = distance_to_final.shape
    t = np.arange(T) * chunk_dur

    pct = {
        "median (50%)": np.median(distance_to_final, axis=0),
        "25–75% IQR":   (np.percentile(distance_to_final, 25, axis=0),
                          np.percentile(distance_to_final, 75, axis=0)),
        "90th":         np.percentile(distance_to_final, 90, axis=0),
    }

    fig, ax = plt.subplots(figsize=(12, 5))

    # IQR band
    lo, hi = pct["25–75% IQR"]
    ax.fill_between(t, lo, hi, alpha=0.20, color="C0", label="25–75% across channels")
    # median + p90 lines
    ax.plot(t, pct["median (50%)"], color="C0", lw=2, label="median across channels")
    ax.plot(t, pct["90th"], color="C0", lw=1, ls="--", alpha=0.7, label="90th pct")

    # threshold guide lines
    for thr in thresholds:
        ax.axhline(thr, color="grey", lw=0.8, ls=":")
        ax.text(t[-1] * 1.005, thr, f"{thr}µm",
                color="grey", fontsize=9, va="center")

    ax.set_xlim(0, t[-1] if T else 1)
    ax.set_ylim(0, max(pct["90th"].max() * 1.05, max(thresholds) * 1.5, 100))
    ax.set_xlabel("Time since first chunk (s)")
    ax.set_ylabel("‖filter_t − Kalman_smoothed_final‖ (µm)")
    ax.set_title(f"Online-filter stability vs smoother final  |  PID {pid}  |  "
                 f"{n_ch} channels × {T} chunks ({chunk_dur:.0f}s each)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def aggregation_table(d, out_path):
    """Compare the per-channel aggregations against the Kalman smoothed final.

    For each method, report:
      - mean / median Euclidean distance from the Kalman smoother's per-channel
        smoothed_mean (treated as the strongest offline estimate)
      - mean per-axis bias
      - per-axis spread across channels
    """
    smoother_target = d["pred_xyz_smooth_mean"]   # (n_ch, 3)

    methods = [
        ("unweighted_mean", d["pred_xyz_unweighted_mean"]),
        ("weighted_mean",   d["pred_xyz_weighted_mean"]),
        ("median",          d["pred_xyz_median"]),
        ("trimmed_mean",    d["pred_xyz_trimmed_mean"]),
        ("kalman_filter_T", d["pred_xyz_kalman_final"]),
        ("kalman_smooth_T", d["pred_xyz_smooth_final"]),
        ("kalman_smooth_mean", smoother_target),  # the target itself, sanity
    ]

    lines = []
    lines.append(f"PID: {str(d['pid'])}")
    lines.append(f"  n_channels: {smoother_target.shape[0]}")
    lines.append(f"  n_chunks:   {d['pred_xyz_kalman_filter'].shape[1]}")
    lines.append(f"  chunk_dur:  {float(d['chunk_dur_sec'])}s")
    lines.append("")
    lines.append("Per-channel disagreement vs kalman_smooth_mean (treated as offline best estimate):")
    lines.append("")
    header = (
        f"  {'method':<22} | {'mean ‖Δ‖':>9} {'med ‖Δ‖':>9} | "
        f"{'bias AP':>8} {'bias DV':>8} {'bias ML':>8} | "
        f"{'std AP':>7} {'std DV':>7} {'std ML':>7}    (all µm)"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for name, est in methods:
        diff = est - smoother_target
        euc = np.linalg.norm(diff, axis=1)
        bias = diff.mean(axis=0)
        spread = est.std(axis=0)
        lines.append(
            f"  {name:<22} | {euc.mean():9.1f} {np.median(euc):9.1f} | "
            f"{bias[0]:+8.1f} {bias[1]:+8.1f} {bias[2]:+8.1f} | "
            f"{spread[0]:7.1f} {spread[1]:7.1f} {spread[2]:7.1f}"
        )

    text = "\n".join(lines) + "\n"
    Path(out_path).write_text(text)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="predictions_kalman.npz from kalman_refine_predictions.py")
    ap.add_argument("--curve_out", default=None,
                    help="Output PNG for stability curve (default: stability_curve.png next to input)")
    ap.add_argument("--table_out", default=None,
                    help="Output TXT for aggregation table (default: aggregation_table.txt next to input)")
    args = ap.parse_args()

    in_path = Path(args.input)
    d = np.load(in_path, allow_pickle=True)
    chunk_dur = float(d["chunk_dur_sec"])
    pid = str(d["pid"])

    curve_out = Path(args.curve_out) if args.curve_out else in_path.with_name("stability_curve.png")
    table_out = Path(args.table_out) if args.table_out else in_path.with_name("aggregation_table.txt")

    print(f"[plot] in:    {in_path}")
    print(f"[plot] curve: {curve_out}")
    print(f"[plot] table: {table_out}")

    stability_curve(d["distance_to_final"], chunk_dur, curve_out, pid=pid)
    print(f"[plot] wrote {curve_out}")

    table = aggregation_table(d, table_out)
    print(f"[plot] wrote {table_out}")
    print()
    print(table)


if __name__ == "__main__":
    main()
