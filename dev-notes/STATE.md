# Live State Snapshot

Updated 2026-05-26 ~22:00 ET. Single page that says "where we are right now".

## Current Slurm jobs

| Job ID | Script | State | Partition | Account | Sessions |
|---|---|---|---|---|---|
| **9662539** | `run_infer.sbatch` | PENDING (Priority) | `h100_tandon` | `torch_pr_60_tandon_priority` | TES_sResp_M05_20240729 |

Priority ~12,070. ETA tonight (60-90 min from submission once fair-share clears).

## Code state

| Repo | HEAD | Branch |
|---|---|---|
| local `/Volumes/CrucialX6/Home/projects/kalman-test` | `e28e57d` | `main`, clean |
| torch `/scratch/ans9868/kalman-test` | `e28e57d` | `main`, clean |
| origin (GitHub) | `e28e57d` | `main` |

All three in sync. Working tree clean.

## Conda env

`$SCRATCH/conda_storage/kalman` exists, ~6.5 GB.
- python 3.11.15
- torch 2.5.1 (CUDA 12.4)
- transformers 4.46.3 (pinned)
- numpy 2.4.6, scipy 1.17.1, matplotlib 3.10.9, pandas 3.0.3, sklearn 1.8.0
- pynrrd 1.1.3, pyyaml

Activate via: `source $SCRATCH/kalman-test/scripts/_env_prelude.sh`

## Outputs

Real predictions: not yet (job pending).

Mock test outputs (validation): `$SCRATCH/kalman-test/outputs/mock/MOCK_SESSION/`
contains 7 files demonstrating the full Kalman → viz → plot pipeline works
end-to-end on torch.

## Memory entries persisted (Claude's persistent memory, not git)

- `torch-hpc-ssh-controlmaster` (implied — not yet written; covered ad hoc)
- `torch-hpc-scratch-redirection` ✓
- `kalman-test-git-sync-workflow` ✓
- `torch-hpc-slurm-query-restraint` ✓

## Decisions

| Question | Decision | Why |
|---|---|---|
| Which GPU partition? | `h100_tandon` | Only Tandon partitions accept our `_priority` account; `h100_tandon` has 15 free nodes, 21 real competitors — best math we can access |
| Which account? | `torch_pr_60_tandon_priority` | FairShare 0.101 (2× higher than `_advanced` at 0.051); `_general` is CPU-only |
| Run 1 session first or all 5? | 1 first (Option A) | Bug surface — catch issues on cheap case |
| Stay on torch or move to GCP? | torch tonight, decide GCP in morning | NYU policy allows export but ask Subhrajit first |
| Sync via git or rsync? | git only | rsync caused identical-content double-commit divergence on 2026-05-26 |
| Kalman params? | defaults (Q=100, R∈[50,3000], scale_R=1) | tune after seeing session 1 output |

## Open questions / pending actions

- **Slack DM to Subhrajit** about GCP — drafted, ready to send tomorrow.
- **Kalman parameter tuning** — wait for real session 1 output before any change.
- **Cross-day stability story** (July vs October Mishi comparison) — defer to
  Thursday-day-of work once all 5 sessions have outputs.
