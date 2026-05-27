# Resume — Pickup Instructions

State at end of 2026-05-26 evening. Everything below assumes you've fresh-cloned
this repo on a new machine and are starting from zero local state.

## 1. Local setup (new machine)

```bash
git clone https://github.com/ans9868/kalman-test
cd kalman-test
# Optional local venv for testing Kalman script with mock data:
python3 -m venv .venv
source .venv/bin/activate
pip install numpy scipy matplotlib pynrrd
```

## 2. SSH to torch

You'll need to re-establish SSH ControlMaster (it expires after 8h idle and is
machine-local). Your `~/.ssh/config` block for `torch` should include:

```
Host torch
    HostName login.torch.hpc.nyu.edu
    User ans9868
    ServerAliveInterval 60
    ServerAliveCountMax 10
    ConnectTimeout 60
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 8h
```

Then:

```bash
ssh torch    # one device-code auth, leave terminal open
```

## 3. Check session 1 (job 9662539) status

```bash
# State + exit code from the database (no live scheduler hit)
ssh torch 'sacct -j 9662539 -X --format=State,ExitCode,Elapsed,Submit,Start,End'

# Did it produce output?
ssh torch 'ls -la $SCRATCH/kalman-test/outputs/chronic_mishi_v2/TES_sResp_M05_20240729/ 2>/dev/null'
```

Expected if it completed: `State=COMPLETED ExitCode=0:0` and a `predictions.npz`
of a few MB in the session dir.

## 4. Validate session 1

```bash
ssh torch 'cd $SCRATCH/kalman-test && bash scripts/validate_session.sh'
```

Reads the first `predictions.npz` it finds; prints shape, NaN/Inf counts, per-axis
position + sigma ranges, outlier rate, per-channel-mean span (should be ≈2880µm
to match NP2.0 shank length). If anything is weird, **stop here** and tune
Kalman params before committing to overnight.

## 5. Run the post-processing on session 1

```bash
ssh torch 'cd $SCRATCH/kalman-test && bash -c "
  source scripts/_env_prelude.sh
  python kalman_refine_predictions.py outputs/chronic_mishi_v2/TES_sResp_M05_20240729/predictions.npz
  python viz_session_inference_timechunk.py --root outputs/chronic_mishi_v2 --pid TES_sResp_M05_20240729 --pred_key pred_xyz_gauss
  python viz_session_inference_timechunk.py --root outputs/chronic_mishi_v2 --pid TES_sResp_M05_20240729 --in_name predictions_kalman.npz --pred_key pred_xyz_kalman_smooth
  python plot_stability.py outputs/chronic_mishi_v2/TES_sResp_M05_20240729/predictions_kalman.npz
"'
```

Outputs in `$SCRATCH/kalman-test/outputs/chronic_mishi_v2/TES_sResp_M05_20240729/`:

- `predictions_kalman.npz`
- `predictions_kalman.json` (summary metrics)
- `channel_timechunk_gauss.png` (raw heatmap)
- `channel_timechunk_kalman_smooth.png` (Kalman heatmap)
- `stability_curve.png`
- `aggregation_table.txt`

Pull figures down to look at them:

```bash
mkdir -p outputs/chronic_mishi_v2/TES_sResp_M05_20240729
scp torch:'$SCRATCH/kalman-test/outputs/chronic_mishi_v2/TES_sResp_M05_20240729/*.png' \
    torch:'$SCRATCH/kalman-test/outputs/chronic_mishi_v2/TES_sResp_M05_20240729/*.json' \
    torch:'$SCRATCH/kalman-test/outputs/chronic_mishi_v2/TES_sResp_M05_20240729/*.txt' \
    outputs/chronic_mishi_v2/TES_sResp_M05_20240729/
```

## 6. If session 1 looks good — submit overnight for sessions 2–5

```bash
ssh torch 'cd $SCRATCH/kalman-test && sbatch scripts/run_overnight.sbatch'
```

The overnight sbatch is idempotent: it sees session 1's predictions.npz exists
and skips it, then runs inference + Kalman + viz + stability plots for sessions
2-5 in one allocation. Emails `berkesencan1@gmail.com` on END/FAIL/TIME_LIMIT.

## 7. Tuning Kalman parameters (if needed)

Defaults: `--Q 100 --R_min 50 --R_max 3000 --scale_R 1.0 --stability_K 10`.

If after viewing session 1 outputs you want different behavior:

- **Smoother trajectories** → raise `--R_min` (e.g. 100) or lower `--Q` (e.g. 25)
- **Faster response to real movement** → raise `--Q` (e.g. 400)
- **More aggressive outlier suppression** → raise `--R_max` (e.g. 5000)
- **Trust head's sigma less** → raise `--scale_R` (e.g. 2.0)

Pass via `KALMAN_ARGS` env var to the overnight sbatch:

```bash
ssh torch 'cd $SCRATCH/kalman-test && KALMAN_ARGS="--Q 400 --R_min 100" sbatch scripts/run_overnight.sbatch'
```

## 8. Pending decisions

- **Slack DM to Subhrajit about GCP**: pending. Worth sending even if you stay on torch — gets explicit permission cached for any future cluster-jammed evening.
- **Run kalman + viz on session 1 alone first** (Option A) vs **just fire run_overnight for all 5** (Option B): we chose A. Pivot to B if validation is clean and you want to skip a step.
