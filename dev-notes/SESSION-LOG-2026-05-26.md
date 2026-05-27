# Session Log — 2026-05-26

Chronological record of what got done in the first working session on this
project. Captures decisions, dead ends, and rationale so neither of us has
to re-derive any of it.

## What we started with

- `eval_regression_probe.py`, `infer_chronic_mishi_window.py`,
  `viz_session_inference_timechunk.py` — three scripts dropped in by Subhrajit
- `Kalman-Temporal-Refinement-Context+Setup-TODO.md` — your scoping doc with
  state-space, ordering ablation, time-to-stability metric, and the
  "no chronic-Mishi access" blocker
- The Geometry-Aware Self-Supervised paper PDF
- A clean GitHub repo `ans9868/kalman-test`, cloned to local Mac

## Pre-flight reading

Read paper (10 pages incl. refs), dev notes (981 lines), and all three scripts
(1481 + 233 + 298 lines). Produced summary reports for each. Key takeaways:

- Paper §3.5 "Inference-Time Refinement" is exactly the seam your Kalman work
  targets: replace the inverse-variance temporal average with a Kalman smoother.
- `infer_chronic_mishi_window.py` produces a clean `predictions.npz` with
  `pred_xyz_gauss (384, n_chunks, 3)` + per-axis sigma in µm — drop-in input
  to the Kalman script.
- `viz_session_inference_timechunk.py` needs three patches per dev-notes §13:
  `--root` arg, `--pred_key` arg, fallback for missing `channel_acronyms`.
- `eval_regression_probe.py` is only relevant as a reference for σ semantics
  and calibration — we don't run it.

## Cluster bootstrap (in order)

1. **SSH ControlMaster** — torch uses Microsoft device-code auth which my Bash
   tool can't drive interactively. Solved by adding `ControlMaster auto` +
   `ControlPath ~/.ssh/cm-%r@%h:%p` + `ControlPersist 8h` to `~/.ssh/config`.
   User runs `ssh torch` once in a terminal, authenticates via browser, all my
   subsequent `ssh torch <cmd>` calls multiplex over that connection.

2. **Cluster probe** — confirmed access to:
   - `/scratch/mc10168/mishi/5DAYS/` (5 chronic sessions present ✓)
   - `/scratch/pl2820/ray_results/.../finetune_best_valmse.pt` (checkpoint ✓)
   - `/scratch/pl2820/Alphabrain_staging/` (code + Allen CCF files ✓)
   - `/scratch/mkp6112/Monkey/` was inaccessible, but we don't need it.

3. **Conda env build** at `$SCRATCH/conda_storage/kalman/`. Took 4 tries
   (see `CLUSTER-SETUP.md` for the full saga). Winning recipe:
   - Full HOME redirect to scratch (so quota-fragile `~/.cache` writes go to
     scratch instead)
   - `--override-channels` (system `.condarc` adds `pkgs/main` + `pkgs/r` which
     now require TOS acceptance for org accounts)
   - `--prefix $SCRATCH/conda_storage/kalman` (not in `~/.conda/envs/`)

4. **transformers downgrade** — initial install pulled transformers 5.9.0
   (latest), which needs `torch>=2.6` for CVE-2025-32434, but we have 2.5.1.
   Plus the 4→5 major bump might break Alphabrain's wav2vec2 wrapping. Pinned
   to `transformers~=4.46.0`. Verified backbone instantiates and checkpoint
   loads (`epoch=46 val_mse=0.008`, 95.1M params).

## Code written (in order)

5. **`kalman_refine_predictions.py`** — pure-numpy 1D constant-position Kalman
   filter + RTS smoother, vectorized across (channel, axis), plus the four
   baseline aggregations (unweighted mean / inverse-variance weighted mean /
   median / trimmed mean). Time-to-100µm and time-to-200µm stability metrics.
   Outlier rate.

6. **viz patches** — `--root` / `--pred_key` / `--in_name` flags; fallback
   `ch_acrs = ["UNK"] * n_ch` when `channel_acronyms` missing; skip
   `annotate_gt_strip` in that case. Output filename suffixed with `pred_key`
   so raw + Kalman heatmaps don't overwrite each other.

7. **`plot_stability.py`** — per-session stability curve (filter-to-final
   distance percentiles over time) + aggregation comparison table.

8. **Slurm sbatch scripts** — `run_infer.sbatch`, `run_kalman.sbatch`,
   `_env_prelude.sh` for shared scratch-redirect + conda activation.

9. **Local mock-data test** — wrote `_make_mock_predictions.py` (synthetic
   384×100×3 with 2% outliers seeded) and ran the full Kalman script against
   it. Outputs sane: filter variance reduction 600µm → 100µm, smoother → 30µm.
   Recovered true positions within ~10 µm. No NaN.

10. **End-to-end test on torch** with the mock npz. Caught two bugs:
    - `Path.with_name()` replaces the whole filename instead of suffixing →
      fixed with `with_stem(stem + "_kalman")`.
    - `_env_prelude.sh` had `set -u` which tripped conda's MKL activation
      hook (`MKL_INTERFACE_LAYER: unbound variable`) — dropped `-u`.
    All 7 expected outputs produced under
    `$SCRATCH/kalman-test/outputs/mock/MOCK_SESSION/`.

## The queue saga

11. Submitted inference for session 1 on `a100_tandon` with account
    `torch_pr_60_tandon_advanced` (the default Slurm account from CLAUDE.md
    reference). Job pending. `a100_tandon` has only 2 mixed nodes (9 drained)
    → throttled.

12. Switched to `h100_tandon` (same account) — 15 mixed nodes. Still pending
    REASON=Priority. Inspected `sshare`: our `_advanced` account has
    FairShare=0.051 because we'd been heavy users (RawUsage 144k).

13. Switched account to `torch_pr_60_tandon_priority` (FairShare=0.101,
    zero recent usage). Still pending REASON=Priority — relatively low FS even
    after the switch.

14. User suggested trying account `torch_pr_60_general` (FairShare=0.133). It
    rejected the submit: that account is **CPU-only**, can't submit to GPU
    partitions. Reverted.

15. Switched partition to `l40s_public` (no account barrier, 66 mixed nodes
    visible). Job pending for an hour with no progress. Investigated:
    - Did a single `sinfo` + `squeue` cluster-wide partition scan
    - `l40s_public` had **822 real pending jobs ahead of us** (worst possible)
    - `h100_tandon` had only 21 real ahead, 60 free GPU slots
    - Tried generic `h100`/`h200` partitions — account-rejected (Tandon-only)
    - Plus: of `l40s_public`'s 66 "mixed" nodes, 63 were `mixed-` (draining).
      Only 4 actually accept new jobs.

16. Settled on `h100_tandon` + `torch_pr_60_tandon_priority` as the realistic
    best. Job 9662539 currently queued there. Slurm hasn't computed a
    `--start` estimate yet; per the queue snapshot we expect 60-90 min.

## Hard-won lessons (already saved to Claude memory)

- **Don't spam squeue** — admins flag it; use `sacct` for completed-job state
  and a single `sinfo` for partition state. One call, never in a loop.
- **NYU home quota is tight** — always redirect HOME + XDG + caches to scratch
  via `_env_prelude.sh`. Conda + pip write things you don't expect.
- **Sync via git push/pull only** — never rsync, never commit on torch. Torch
  doesn't have GitHub creds for `ans9868`.

## End-of-day state

- Job 9662539: pending on h100_tandon, priority ~12,070, ETA tonight.
- All scripts at origin/main `e28e57d`. Both local repo and torch's clone in sync.
- Post-inference pipeline 100% validated on torch with mock data.
- Plan locked in: let session 1 run overnight → validate in morning → fire
  `run_overnight.sbatch` for sessions 2-5 → assemble Thursday deliverable.
- Pending decision: GCP fallback if torch keeps queueing. Drafted Slack DM to
  Subhrajit asking permission to scp the `.lfp` files + checkpoint to a
  personal GCP instance. NYU data policy says it's Low Risk and exporting is
  fine; lab social/courtesy norm says ask anyway.

## Time spent

Roughly:
- 30 min reading paper + dev notes + scripts
- 30 min cluster bootstrap (ControlMaster, env, transformers fix)
- 90 min writing scripts + local tests
- 60 min queue saga + cluster discovery
- 30 min docs (this file and siblings)

Most of the elapsed wall time was Slurm waits and the conda install. Actual
coding+thinking was ~2-3 hours.
