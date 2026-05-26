# Shared scratch-redirection + conda-env activation for every job/script.
# Source this from sbatch scripts and interactive sessions to keep $HOME
# writes off the quota-limited NYU torch home.
#
#   source $SCRATCH/kalman-test/scripts/_env_prelude.sh
#
# Refer to memory: torch-hpc-scratch-redirection.

set -e
# NB: do NOT set -u. Conda's MKL activation hook references MKL_INTERFACE_LAYER
# unconditionally and would error here. The shell scripts we run handle their
# own undefined-var safety where it matters.

# ─── scratch-redirect everything that might write to $HOME ───
export PYTHONNOUSERSITE=True
export SCR_BASE=${SCR_BASE:-$SCRATCH/kalman_scratch}
mkdir -p "$SCR_BASE"/{home,cache,config,conda,tmp,matplotlib}
export HOME=$SCR_BASE/home
export XDG_CACHE_HOME=$SCR_BASE/cache
export XDG_CONFIG_HOME=$SCR_BASE/config
export CONDARC=$SCR_BASE/conda/.condarc
export TMPDIR=$SCR_BASE/tmp
export MPLCONFIGDIR=$SCR_BASE/matplotlib
export CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS:-$SCRATCH/conda_storage/.pkgs}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$SCRATCH/pip-cache}

# ─── activate conda env ───
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate ${KALMAN_ENV_PREFIX:-$SCRATCH/conda_storage/kalman}

# Sanity log
echo "[prelude] HOME=$HOME"
echo "[prelude] CONDA_PREFIX=${CONDA_PREFIX:-(unset)}"
echo "[prelude] python=$(which python)  $(python --version 2>&1)"
