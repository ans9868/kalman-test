# Cluster Setup — NYU torch state reference

What lives where on `login.torch.hpc.nyu.edu` (user `ans9868`). Static reference;
update if/when things change.

## Filesystem layout on torch

| Path | Purpose | Notes |
|---|---|---|
| `/scratch/ans9868/kalman-test/` | This repo, cloned | `git pull` from origin/main |
| `/scratch/ans9868/conda_storage/kalman/` | Conda env (~6.5 GB) | python 3.11 + pytorch 2.5.1 + transformers 4.46.3 etc. |
| `/scratch/ans9868/conda_storage/.pkgs/` | Conda package cache | ~7 GB; persistent across env rebuilds |
| `/scratch/ans9868/kalman_scratch/` | Runtime redirect tree | HOME/cache/config/tmp/matplotlib all symlinked here during script runs |
| `/scratch/ans9868/pip-cache/` | pip download cache | |
| `/scratch/ans9868/kalman-test/outputs/` | Inference + Kalman outputs | gitignored |
| `/scratch/ans9868/kalman-test/logs/` | Slurm job logs (`infer_*.out`, `overnight_*.out`) | gitignored |

External (read-only) paths we depend on:

| Path | Purpose |
|---|---|
| `/scratch/mc10168/mishi/5DAYS/` | Mishi M05 chronic recordings, 5 sessions |
| `/scratch/pl2820/ray_results/benchmarks/finetune_xyz_75k_joint_lambda_3_0_7741664/checkpoints/finetune_best_valmse.pt` | Path B 75K λ=3.0 model checkpoint |
| `/scratch/pl2820/Alphabrain_staging/` | Lawrence's codebase — `backbones/`, `utils/preprocessing.py` imported by `infer_chronic_mishi_window.py` |
| `/scratch/pl2820/Alphabrain_staging/data/allen_ccf/` | `annotation_25.nrrd`, `structure_tree.json` for the viz script |

## Conda env build (already done, reproducible if needed)

Why it took 4 tries on 2026-05-26:
1. Conda TOS plugin tried to write `~/.cache/conda-anaconda-tos/...` → `OSError: Disk quota exceeded` (NYU home quota is *tight*).
2. `CONDA_NO_PLUGINS=true` disabled libmamba (which is itself a plugin in conda 25.5.1) → solver missing.
3. `--override-channels` needed because system condarc adds `defaults` channel which requires TOS acceptance (`pkgs/main` + `pkgs/r`).
4. **Final winning recipe** = HOME redirect to scratch (so TOS plugin writes there) + `--override-channels`.

Reproduce from scratch:

```bash
export PYTHONNOUSERSITE=True
export SCR_BASE=$SCRATCH/kalman_scratch
mkdir -p $SCR_BASE/{home,cache,config,conda,tmp,matplotlib}
export HOME=$SCR_BASE/home
export XDG_CACHE_HOME=$SCR_BASE/cache
export XDG_CONFIG_HOME=$SCR_BASE/config
export CONDARC=$SCR_BASE/conda/.condarc
export TMPDIR=$SCR_BASE/tmp
export MPLCONFIGDIR=$SCR_BASE/matplotlib
export CONDA_PKGS_DIRS=$SCRATCH/conda_storage/.pkgs
export PIP_CACHE_DIR=$SCRATCH/pip-cache

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh

conda create -y --prefix $SCRATCH/conda_storage/kalman \
    --override-channels \
    -c pytorch -c nvidia -c conda-forge \
    python=3.11 pytorch torchaudio pytorch-cuda=12.4

conda activate $SCRATCH/conda_storage/kalman
pip install --no-user numpy scipy matplotlib scikit-learn pandas pyyaml pynrrd
# IMPORTANT: pin transformers to 4.x — 5.x requires torch>=2.6 (CVE-2025-32434)
pip install --no-user "transformers~=4.46.0"
```

All this is encapsulated for *runtime* by `scripts/_env_prelude.sh`. Source it
from any interactive session or sbatch job to activate the env with the
quota-safe HOME redirect already in place.

## Slurm accounts available to us

| Account | FairShare | Recent usage | Partitions allowed | Notes |
|---|---|---|---|---|
| `torch_pr_60_tandon_advanced` | 0.051 | 144k (heavy) | Tandon partitions | Throttled. Use `_priority` instead. |
| **`torch_pr_60_tandon_priority`** | **0.101** | 0 | **Tandon partitions** | **Current default in `scripts/run_*.sbatch`** |
| `torch_pr_60_general` | 0.133 | 26k | **CPU-only partitions** | Higher FairShare but rejects GPU jobs |
| `users` | 1.0 | 0 | minimal | No GPU access |

## GPU partitions evaluated 2026-05-26 evening

Snapshot of competing-jobs-ahead-of-us when we were at priority 12,069.
"Real" excludes Dependency / QOSMax* / JobArray-blocked entries.

| Partition | GPU | Free nodes | Real jobs ahead | Verdict |
|---|---|---|---|---|
| `l40s_public` | l40s (48 GB) | 30 + 37 draining | **822** | catastrophic — initial choice, abandoned |
| `a100_tandon` | a100 (40 GB) | 2 (29 drained) | 59 + 9 dep-blocked | bad — drains kill it |
| `h100_tandon` | h100 (80 GB) | 15 | **21** | **current choice** (job 9662539) |
| `h200_tandon` | h200 (141 GB) | 22 | 143 | overkill mem + hotter queue |
| `h200_public` | h200 | 22 | 308 | hottest after l40s |
| `h100` (generic) | h100 | 15 | 2 | great math but ACCOUNT-REJECTED for `_tandon_priority` |
| `h100_plus`, `h200_plus`, `a100_plus` | various | various | 0 | account-rejected (lab-funded) |

**Submit rule**: with `_tandon_priority` account, we can ONLY use `*_tandon`
partitions. Generic and lab partitions reject. CPU-only `_general` rejects
GPU jobs. So `h100_tandon` is the realistic optimum.

## SSH ControlMaster (this is local-machine state)

Per-Mac, not committed to git. Sets up once, persists 8h. See `RESUME.md` for
the config block. The `torch` host alias was already in `~/.ssh/config` before
this project — we just added 3 lines (`ControlMaster auto`, `ControlPath`,
`ControlPersist 8h`).

## Outstanding cluster-side state at end-of-day 2026-05-26

- Job `9662539` queued on `h100_tandon`, account `torch_pr_60_tandon_priority`,
  walltime 2h, requesting 1 GPU. Pending at priority ~12,070 with REASON=Priority.
  Expected to run overnight; will produce `outputs/chronic_mishi_v2/TES_sResp_M05_20240729/predictions.npz`.
- All scripts up to date (origin/main HEAD = `e28e57d` "add overnight sbatch + per-session validation helper").
- No active monitors (Claude session ends; cluster job continues independently).

## Things to not re-discover the hard way

- NYU home quota is tight (~5-10 GB? — see `feedback_torch_hpc_scratch_redirection`
  memory). Anything that writes to `$HOME/.cache`, `$HOME/.local`, `$HOME/.conda`
  will silently fail with "disk full". **Always** source `_env_prelude.sh`.
- `squeue` is slow under cluster load; admins flag spammers. **One** call,
  never in a loop. Use `sacct` (database-backed) for completed-job state.
- `chmod +x` on torch leaves uncommitted permission diffs that block `git pull`.
  Don't chmod files that git tracks (or commit the +x bit upfront from local).
- Conda's MKL activation hook references `MKL_INTERFACE_LAYER` without a default
  → fails under `set -u`. Don't use `set -u` in scripts that source conda activate.
- Sync only via git push from local → `git pull` on torch. Never rsync, never
  commit on torch (no GitHub creds there).
