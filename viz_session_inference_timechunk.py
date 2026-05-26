#!/usr/bin/env python3
"""Channel × time-chunk predicted-region heatmap for one IBL probe insertion.

Adapter of teammate's plot_probe_channel_timechunk_regions.py for our
continuous-time session inference output. Reads:

    /scratch/pl2820/ray_results/eval_v2/session_inference/<pid>/predictions.npz

with keys:
    pred_xyz_gauss   (n_ch, n_chunks, 3) µm CCF
    channel_xyz_true (n_ch, 3)           µm CCF
    channel_acronyms (n_ch,) <U…
    channel_axial_um (n_ch,)             µm along probe shaft

Each (ch, chunk) predicted xyz is voxel-looked-up in Allen annotation_25
(int-truncation, 25 µm voxel pitch) → sid → acronym → color (Allen
structure-tree color_hex_triplet, with parent-walk fallback for sids that
aren't directly colored).

Layout: GT-acronym strip on the left, prediction grid on the right.
No LFP envelope panel and no stim ribbon (IBL has no Allen visual stims).
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nrrd
import numpy as np
from matplotlib.colors import ListedColormap, to_rgba

ALPHABRAIN = Path("/scratch/pl2820/Alphabrain_staging")
ANNOTATION_PATH = ALPHABRAIN / "data/allen_ccf/annotation_25.nrrd"
STRUCTURE_TREE_PATH = ALPHABRAIN / "data/allen_ccf/structure_tree.json"
SESSION_ROOT = Path("/scratch/pl2820/ray_results/eval_v2/session_inference")


def build_atlas_lookup():
    """Returns (annotation, parent, acr_to_color, sid_to_acr)."""
    annotation, _ = nrrd.read(str(ANNOTATION_PATH))
    with open(STRUCTURE_TREE_PATH) as f:
        tree = json.load(f)
    parent = {n["id"]: n.get("parent_structure_id") for n in tree}
    acr_to_color = {}
    sid_to_acr = {}
    for n in tree:
        a = n.get("acronym")
        c = n.get("color_hex_triplet")
        if a and c:
            acr_to_color[a] = "#" + c.upper()
        if a:
            sid_to_acr[int(n["id"])] = a
    return annotation, parent, acr_to_color, sid_to_acr


def relabel_xyz(pred_xyz_flat, annotation, parent, sid_to_acr, acr_to_id):
    """Voxel-lookup with parent-walk fallback. Returns int32 acr_to_id labels.

    pred_xyz_flat: (N, 3) µm CCF
    Out: (N,) int32, -1 for OOB or unmapped.
    """
    voxels = np.floor(pred_xyz_flat / 25.0).astype(np.int64)
    shape = np.array(annotation.shape)
    in_bounds = np.all((voxels >= 0) & (voxels < shape), axis=1)
    out = np.full(len(pred_xyz_flat), -1, dtype=np.int32)
    if not in_bounds.any():
        return out
    uniq, inverse = np.unique(voxels[in_bounds], axis=0, return_inverse=True)
    sids = annotation[uniq[:, 0], uniq[:, 1], uniq[:, 2]]

    cache = {}

    def resolve(sid):
        sid = int(sid)
        if sid in cache:
            return cache[sid]
        cur = sid
        ans = -1
        # Walk up the structure tree until we hit a sid with an acronym we
        # know AND that acronym is in our color map.
        while cur and cur != 0:
            acr = sid_to_acr.get(cur)
            if acr is not None and acr in acr_to_id:
                ans = acr_to_id[acr]
                break
            cur = parent.get(cur)
            if cur is None:
                break
        cache[sid] = ans
        return ans

    uniq_labels = np.array([resolve(s) for s in sids], dtype=np.int32)
    out[in_bounds] = uniq_labels[inverse]
    return out


def build_label_map(channel_acronyms, sid_to_acr):
    """Make sure every acronym we encounter (channel GT + every Allen acr) gets
    an integer label. Returns acr_to_id, id_to_acr."""
    all_acrs = set(sid_to_acr.values())
    all_acrs.update(map(str, channel_acronyms.tolist()))
    all_acrs.discard("")
    all_acrs.discard("nan")
    all_acrs = sorted(all_acrs)
    acr_to_id = {a: i for i, a in enumerate(all_acrs)}
    id_to_acr = {i: a for a, i in acr_to_id.items()}
    return acr_to_id, id_to_acr


def build_cmap(id_to_acr, acr_to_color, fallback="#888888"):
    n = max(id_to_acr.keys()) + 1
    cmap = ListedColormap([
        to_rgba(acr_to_color.get(id_to_acr.get(i, ""), fallback))
        for i in range(n)
    ])
    cmap.set_under("#000000")
    return cmap


def annotate_gt_strip(ax, gt_per_channel, id_to_acr, min_run=2):
    n_ch = len(gt_per_channel)
    boundaries = [0]
    for i in range(1, n_ch):
        if gt_per_channel[i] != gt_per_channel[i - 1]:
            boundaries.append(i)
    boundaries.append(n_ch)
    for s, e in zip(boundaries[:-1], boundaries[1:]):
        if e - s < min_run:
            continue
        cid = int(gt_per_channel[s])
        label = "?" if cid < 0 else id_to_acr.get(cid, f"id{cid}")
        mid = (s + e - 1) / 2.0
        ax.text(0, mid, label, color="black", fontsize=7,
                ha="center", va="center", rotation=0,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none",
                          boxstyle="round,pad=0.15"))


def annotate_pred_bands(ax, grid, id_to_acr, min_run=20):
    """Label dominant horizontal stripes in the prediction grid."""
    n_ch, n_chunks = grid.shape
    seen = {}
    for ch in range(n_ch):
        row = grid[ch]
        valid = row[row >= 0]
        if not len(valid):
            continue
        vals, counts = np.unique(valid, return_counts=True)
        mj = int(vals[counts.argmax()])
        if mj < 0:
            continue
        mask = row == mj
        edges = np.diff(np.r_[0, mask.astype(np.int8), 0])
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0]
        if not len(starts):
            continue
        lengths = ends - starts
        best = lengths.argmax()
        if lengths[best] < max(min_run, n_chunks // 200):
            continue
        prev = seen.get(mj)
        if prev is None or lengths[best] > prev[2]:
            seen[mj] = (ch, (starts[best] + ends[best] - 1) / 2.0,
                        int(lengths[best]))
    for cid, (ch, mid, _l) in seen.items():
        ax.text(mid, ch, id_to_acr.get(cid, f"id{cid}"),
                color="black", fontsize=10, ha="center", va="center",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none",
                          boxstyle="round,pad=0.15"))


def render(npz_path, out_path):
    d = np.load(npz_path, allow_pickle=True)
    pred = d["pred_xyz_gauss"]                      # (n_ch, n_chunks, 3)
    ch_acrs = np.array([str(a) for a in d["channel_acronyms"].tolist()])
    axial = d["channel_axial_um"].astype(np.float32)
    pid = str(d["pid"])
    chunk_dur = float(d["chunk_dur_sec"])

    n_ch, n_chunks, _ = pred.shape
    print(f"  pid={pid}  n_ch={n_ch}  n_chunks={n_chunks}  "
          f"chunk_dur={chunk_dur:.1f}s  → {n_chunks * chunk_dur / 60:.1f} min")

    # Sort channels surface→deep using axial position (high axial = base).
    # IBL probes are inserted top-down so the TIP (low axial) is DEEPEST in
    # brain. We want surface (shallowest brain tissue) at top of plot, so
    # we sort by axial DESCENDING (high → top). Channels at higher axial
    # are the base of the probe = closest to brain surface.
    order = np.argsort(-axial, kind="stable")
    pred = pred[order]
    ch_acrs = ch_acrs[order]
    axial = axial[order]

    # Atlas + colors
    annotation, parent, acr_to_color, sid_to_acr = build_atlas_lookup()
    acr_to_id, id_to_acr = build_label_map(ch_acrs, sid_to_acr)
    cmap = build_cmap(id_to_acr, acr_to_color)

    # Voxel lookup for predictions (vectorized over all (ch, chunk) pairs)
    flat = pred.reshape(-1, 3)
    labels = relabel_xyz(flat, annotation, parent, sid_to_acr, acr_to_id)
    grid = labels.reshape(n_ch, n_chunks)

    # GT strip from per-channel acronym (no voxel lookup needed)
    gt_strip = np.array(
        [acr_to_id.get(a, -1) for a in ch_acrs], dtype=np.int32
    )

    # ----- figure -----
    # Native data is ~4.5:1 (1741 chunks × 384 channels). Slightly wider than
    # that so per-channel rows are readable. Drop bbox_inches='tight' below
    # so the saved PNG keeps the figsize aspect.
    fig_w = 24
    fig_h = 9
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 32], wspace=0.012)
    ax_gt = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[0, 1], sharey=ax_gt)

    vmax = max(cmap.N - 0.5, 0.5)
    ax.imshow(grid, aspect="auto", interpolation="nearest", cmap=cmap,
              vmin=-0.5, vmax=vmax,
              extent=(0, n_chunks * chunk_dur / 60, n_ch, 0))
    ax.set_xlim(0, n_chunks * chunk_dur / 60)
    ax.set_ylim(n_ch, 0)
    ax.set_xlabel("Time (min)", fontsize=11)
    ax.tick_params(axis="y", labelleft=False, left=False)

    n_unmapped = int((labels < 0).sum())
    title = (f"Continuous session inference  |  PID {pid}  |  "
             f"{n_ch} channels × {n_chunks} chunks ({chunk_dur:.0f}s each, "
             f"{n_chunks * chunk_dur / 60:.1f} min total)")
    if n_unmapped:
        title += f"  |  {n_unmapped:,} cells unmapped (outside Allen)"
    ax.set_title(title, fontsize=11)

    ax_gt.imshow(gt_strip.reshape(-1, 1), aspect="auto",
                 interpolation="nearest", cmap=cmap,
                 vmin=-0.5, vmax=vmax, extent=(0, 1, n_ch, 0))
    ax_gt.set_xticks([])
    step = max(1, n_ch // 25)
    ax_gt.set_yticks(np.arange(0.5, n_ch, step))
    ax_gt.set_yticklabels(
        [f"ch{order[i]}" for i in range(0, n_ch, step)], fontsize=7,
    )
    ax_gt.set_ylabel("Channel (surface → tip)", fontsize=11)
    ax_gt.set_ylim(n_ch, 0)
    annotate_gt_strip(ax_gt, gt_strip, id_to_acr, min_run=2)
    annotate_pred_bands(ax, grid, id_to_acr)

    fig.subplots_adjust(left=0.04, right=0.99, top=0.94, bottom=0.07)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", default=None,
                    help="Insertion PID. Default = render every PID under "
                         "session_inference/.")
    ap.add_argument("--all", action="store_true",
                    help="Render all PIDs under session_inference/")
    args = ap.parse_args()

    if args.pid:
        targets = [SESSION_ROOT / args.pid]
    else:
        targets = sorted([p for p in SESSION_ROOT.iterdir() if p.is_dir()])
        if not args.all and len(targets) > 1:
            print(f"[!] {len(targets)} PIDs found under {SESSION_ROOT}. "
                  f"Pass --all to render every one, or --pid <id> for one.")
            for p in targets:
                print(f"    {p.name}")
            return 1

    n = 0
    for tgt in targets:
        npz = tgt / "predictions.npz"
        if not npz.exists():
            print(f"[skip] no predictions.npz under {tgt.name}")
            continue
        out = tgt / "channel_timechunk.png"
        print(f"\n=== {tgt.name} ===")
        render(npz, out)
        n += 1
    print(f"\n[+] rendered {n} session(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
