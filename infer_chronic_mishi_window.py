"""Chronic Mishi M05 — same model, same probe, same time-window across 5 days.

Question: is Path B 75K λ=3.0 (joint-finetuned flagship) stable when run on the
SAME chronic NP2.0 probe across 5 different recording days (3 days in July
2024 + 2 days in October 2024, ~2.5 months apart)?

For each session:
  - Read 384-channel destriped LFP from .lfp memmap
  - Take ONE contiguous mid-session window (default 5 min)
  - Calibrate raw_std → 8.9e-5 (matches IBL preprocessing)
  - Slice into 3-sec chunks (default 100 chunks = 5 min @ 3-sec each)
  - Run encoder + Gaussian head → (n_ch, n_chunks, 3) µm-CCF predictions
  - Save predictions.npz in the format viz_session_inference_timechunk reads

CPU-friendly when small (~3 min total over 5 sessions), GPU recommended.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ALPHABRAIN_ROOT = "/scratch/pl2820/Alphabrain_staging"
sys.path.insert(0, ALPHABRAIN_ROOT)
sys.path.insert(0, os.path.join(ALPHABRAIN_ROOT, "Lfp2vec_benchmarks"))

MISHI_ROOT = "/scratch/mc10168/mishi/5DAYS"
SESSIONS = [
    "TES_sResp_M05_20240729",
    "TES_sResp_M05_20240730",
    "TES_sResp_M05_20240731",
    "TES_sResp_M05_20241014",
    "TES_sResp_M05_20241015",
]
DEFAULT_CKPT = ("/scratch/pl2820/ray_results/benchmarks/"
                "finetune_xyz_75k_joint_lambda_3_0_7741664/"
                "checkpoints/finetune_best_valmse.pt")
DEFAULT_OUT_ROOT = "/scratch/pl2820/ray_results/eval_v2/chronic_mishi_v2"

CHUNK_SEC = 3.0
SR_LFP = 1250                      # mishi data already at 1250 Hz
SAMPLES_PER_CHUNK = int(CHUNK_SEC * SR_LFP)  # 3750
N_CH = 384


class GaussianProbe(nn.Module):
    def __init__(self, in_dim=768, out_dim=3):
        super().__init__()
        self.mu = nn.Linear(in_dim, out_dim)
        self.log_var = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.mu(x), self.log_var(x)


def log(msg):
    print(f"[chronic_v2] {msg}", flush=True)


def load_lfp_memmap(lfp_path, n_lfp_channels=N_CH, n_saved_channels=385):
    size = os.path.getsize(lfp_path)
    total_int16 = size // 2
    assert total_int16 % n_saved_channels == 0
    n_samples = total_int16 // n_saved_channels
    mm = np.memmap(lfp_path, dtype=np.int16, mode="r",
                   shape=(n_samples, n_saved_channels))
    return mm[:, :n_lfp_channels], n_samples


def denormalize_xyz(pred_norm, stats):
    """Map normalized [0, 1] → CCF µm using stored min/max + LR-fold inverse."""
    ap = pred_norm[..., 0] * (stats["ap_max"] - stats["ap_min"]) + stats["ap_min"]
    dv = pred_norm[..., 1] * (stats["dv_max"] - stats["dv_min"]) + stats["dv_min"]
    d = pred_norm[..., 2] * (stats["d_max"] - stats["d_min"]) + stats["d_min"]
    lr = stats["midline"] - d
    return np.stack([ap, dv, lr], axis=-1)


def denormalize_sigma(sigma_norm, stats):
    """σ in normalized space → CCF µm. Each axis independently scaled."""
    sa = sigma_norm[..., 0] * (stats["ap_max"] - stats["ap_min"])
    sd = sigma_norm[..., 1] * (stats["dv_max"] - stats["dv_min"])
    sl = sigma_norm[..., 2] * (stats["d_max"] - stats["d_min"])
    return np.stack([sa, sd, sl], axis=-1)


def load_path_b(ckpt_path, device):
    log(f"Loading Path B ckpt: {os.path.basename(ckpt_path)}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    log(f"  ep {ckpt.get('epoch', '?')}  val_mse {ckpt.get('val_mse', '?'):.4f}")
    from backbones import get_backbone
    bb = get_backbone(cfg.get("backbone", "wav2vec2"), cfg).to(device)
    bb.load_state_dict(ckpt["backbone_state_dict"])
    bb.eval()
    gauss = GaussianProbe(768, 3).to(device)
    gauss.load_state_dict(ckpt["gauss_head_state_dict"])
    gauss.eval()
    return bb, gauss, ckpt["normalizer_stats"]


def axial_um_np2(n_ch=N_CH):
    """NP2.0 single-shank axial position per channel.
    Single shank = 2 columns × 192 rows, 15 µm row pitch.
    Channel order: pairs (col 0, col 1) per row → flatten."""
    rows = np.arange(n_ch // 2, dtype=np.float32) * 15.0
    return np.repeat(rows, 2)   # (n_ch,) — col0 row0, col1 row0, col0 row1, …


def process_session(session, bb, gauss, stats, device, args):
    from utils.preprocessing import preprocess_batch

    t0 = time.time()
    sess_dir = os.path.join(MISHI_ROOT, session)
    lfp_path = next((os.path.join(sess_dir, f) for f in os.listdir(sess_dir)
                     if f.endswith(".lfp")), None)
    assert lfp_path is not None, f"no .lfp in {sess_dir}"

    lfp, n_samples = load_lfp_memmap(lfp_path)
    duration_s = n_samples / SR_LFP
    log(f"  {session}: {duration_s/60:.1f} min total ({n_samples} samples)")

    if args.full_session:
        # Full session: take everything, drop the trailing partial chunk.
        n_chunks = n_samples // SAMPLES_PER_CHUNK
        actual_window = n_chunks * SAMPLES_PER_CHUNK
        win_start = 0
        win_end = actual_window
    else:
        # Window: take args.window_min minutes from middle of session.
        window_samples_total = int(args.window_min * 60 * SR_LFP)
        n_chunks = window_samples_total // SAMPLES_PER_CHUNK
        actual_window = n_chunks * SAMPLES_PER_CHUNK
        mid = n_samples // 2
        win_start = max(0, mid - actual_window // 2)
        win_end = win_start + actual_window
    log(f"  Window: [{win_start/SR_LFP:.0f}s, {win_end/SR_LFP:.0f}s] "
        f"= {n_chunks} chunks × {CHUNK_SEC}s")

    # Calibration: use the chosen window itself for std (representative).
    cal = lfp[win_start:win_end, :].astype(np.float32)
    raw_std = float(cal.std())
    target_std = 8.9e-5
    scale_factor = target_std / max(raw_std, 1e-12)
    log(f"  Calibration: raw_std={raw_std:.1f} → scale={scale_factor:.2e}")

    # Pull window into RAM (small enough: 5 min × 384 ch × 1250 Hz × 2B = 720 MB)
    wave_window = cal * scale_factor   # (T, 384) float32, T = actual_window
    wave_window = wave_window.T        # (384, T)

    # Pre-allocate (384, n_chunks, 3)
    pred_mu = np.zeros((N_CH, n_chunks, 3), dtype=np.float32)
    pred_sigma = np.zeros((N_CH, n_chunks, 3), dtype=np.float32)

    # Loop chunks (outer). Each call: batch = 384 channels.
    for ci in range(n_chunks):
        ts = ci * SAMPLES_PER_CHUNK
        chunk = wave_window[:, ts:ts + SAMPLES_PER_CHUNK]  # (384, 3750)
        x = torch.from_numpy(chunk).float().to(device)
        with torch.no_grad():
            xp = preprocess_batch(x, bb.needs_resampling, bb.needs_normalization)
            emb = bb.encode(xp)                           # (384, 768)
            mu_norm, log_var = gauss(emb)                  # (384, 3) each
            sigma_norm = torch.exp(0.5 * log_var)
        pred_mu[:, ci, :] = mu_norm.cpu().numpy()
        pred_sigma[:, ci, :] = sigma_norm.cpu().numpy()
        if (ci + 1) % 20 == 0:
            log(f"    chunk {ci+1}/{n_chunks}  "
                f"({(time.time()-t0):.0f}s elapsed)")

    # Denormalize → CCF µm
    mu_um = denormalize_xyz(pred_mu.reshape(-1, 3), stats).reshape(N_CH, n_chunks, 3)
    sigma_um = denormalize_sigma(pred_sigma.reshape(-1, 3),
                                  stats).reshape(N_CH, n_chunks, 3)
    log(f"  Done: {time.time()-t0:.0f}s for {n_chunks} chunks × {N_CH} ch")

    return {
        "pred_xyz_gauss":   mu_um.astype(np.float32),     # (n_ch, n_chunks, 3)
        "pred_sigma_gauss": sigma_um.astype(np.float32),
        "channel_axial_um": axial_um_np2(N_CH),
        "channel_ids":      np.arange(N_CH, dtype=np.int64),
        "pid":              np.array(session, dtype=object),
        "chunk_dur_sec":    np.float64(CHUNK_SEC),
        "fs_target":        np.float64(SR_LFP),
        "win_start_sec":    np.float64(win_start / SR_LFP),
        "win_end_sec":      np.float64(win_end / SR_LFP),
        "raw_std":          np.float64(raw_std),
        "scale_factor":     np.float64(scale_factor),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--out_root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--sessions", nargs="+", default=SESSIONS)
    ap.add_argument("--window_min", type=float, default=5.0,
                    help="Window length per session in MINUTES (default 5).")
    ap.add_argument("--full_session", action="store_true",
                    help="Run inference over the ENTIRE session "
                         "(overrides --window_min).")
    args = ap.parse_args()

    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log(f"device: {device}")

    bb, gauss, stats = load_path_b(args.ckpt, device)

    for session in args.sessions:
        log(f"\n=== {session} ===")
        out_dir = Path(args.out_root) / session
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "predictions.npz"
        if out_path.exists():
            log(f"  [skip] {out_path} exists")
            continue
        result = process_session(session, bb, gauss, stats, device, args)
        np.savez_compressed(out_path, **result)
        log(f"  Saved {out_path}")

    log("\nDone. Run viz_session_inference_timechunk.py per session.")


if __name__ == "__main__":
    main()
