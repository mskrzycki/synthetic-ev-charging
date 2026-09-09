#!/bin/bash
#SBATCH --job-name=ev_cost_benchmark
#SBATCH --error=/home/iroshanij/Jobs/SynDaGen/ev_cost_benchmark_%j.err
#SBATCH --output=/home/iroshanij/Jobs/SynDaGen/ev_cost_benchmark_%j.out
#SBATCH --partition=sintef
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a30:1
#SBATCH --time=24:00:00

set -euo pipefail

echo "Job started on $(hostname)"
echo "Start time: $(date)"

# ---- paths on the server ----
PROJECT_DIR="${PROJECT_DIR:-/home/iroshanij/SynDaGen/synthetic-ev-charging}"
TABDDPM_REPO="${TABDDPM_REPO:-/home/iroshanij/SynDaGen/tab-ddpm}"
JOB_DIR="${JOB_DIR:-/home/iroshanij/Jobs/SynDaGen}"

export TABDDPM_REPO
export PYTHONPATH="${TABDDPM_REPO}:${TABDDPM_REPO}/scripts:${PYTHONPATH:-}"

mkdir -p "${JOB_DIR}" "${PROJECT_DIR}/logs" "${PROJECT_DIR}/computational_cost"
cd "${PROJECT_DIR}"

# Keep joblib/loky from creating fragile multiprocessing semaphores on the cluster.
export JOBLIB_TEMP_FOLDER="${PROJECT_DIR}/computational_cost/joblib_tmp"
export JOBLIB_MULTIPROCESSING=0
export LOKY_MAX_CPU_COUNT=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
mkdir -p "${JOBLIB_TEMP_FOLDER}"

# ---- conda/mamba environment ----
# This creates the environment with mamba, then activates it through conda.
# mkl=2024.0 is pinned because newer MKL/Intel runtime combinations can cause
# PyTorch import errors such as: undefined symbol: iJIT_NotifyEvent.
CONDA_ROOT="${CONDA_ROOT:-/opt/miniforge3}"
CONDA_ENV="${CONDA_ENV:-ev-charging-cost}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/home/iroshanij/.conda/envs/${CONDA_ENV}}"
CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/home/iroshanij/.conda/pkgs}"
CONDA_ENVS_DIRS="${CONDA_ENVS_DIRS:-/home/iroshanij/.conda/envs}"
MAMBA_EXE="${MAMBA_EXE:-${CONDA_ROOT}/bin/mamba}"
CONDARC="${PROJECT_DIR}/.condarc.slurm"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/home/iroshanij/.mamba}"
MAMBA_PKGS_DIRS="${MAMBA_PKGS_DIRS:-${CONDA_PKGS_DIRS}}"
export CONDA_PKGS_DIRS CONDA_ENVS_DIRS CONDARC MAMBA_ROOT_PREFIX MAMBA_PKGS_DIRS
mkdir -p "${CONDA_PKGS_DIRS}" "${CONDA_ENVS_DIRS}" "$(dirname "${CONDA_ENV_PREFIX}")" "${MAMBA_ROOT_PREFIX}"

cat > "${CONDARC}" <<EOF
pkgs_dirs:
  - ${CONDA_PKGS_DIRS}
envs_dirs:
  - ${CONDA_ENVS_DIRS}
channels:
  - pytorch
  - nvidia
  - conda-forge
channel_priority: flexible
EOF

if [ ! -x "${MAMBA_EXE}" ]; then
  if command -v mamba >/dev/null 2>&1; then
    MAMBA_EXE="$(command -v mamba)"
  else
    echo "mamba was not found. Set MAMBA_ROOT or MAMBA_EXE before submitting the job."
    exit 1
  fi
fi

if [ ! -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]; then
  echo "Cannot find conda.sh at ${CONDA_ROOT}/etc/profile.d/conda.sh"
  echo "Set CONDA_ROOT before submitting the job."
  exit 1
fi

source "${CONDA_ROOT}/etc/profile.d/conda.sh"

if [ ! -d "${CONDA_ENV_PREFIX}" ]; then
  echo "Creating conda environment with mamba: ${CONDA_ENV_PREFIX}"
  "${MAMBA_EXE}" create -y -p "${CONDA_ENV_PREFIX}" \
    -c pytorch -c nvidia -c conda-forge \
    python=3.10 pip "numpy<2" pandas scikit-learn matplotlib tqdm psutil tomli mkl=2024.0 intel-openmp \
    pytorch pytorch-cuda=11.8
fi

conda activate "${CONDA_ENV_PREFIX}"

# Avoid cluster-level Intel/OpenMP libraries taking precedence over the env.
unset LD_PRELOAD || true

python -m pip install --upgrade pip
python -m pip install "numpy<2" "sdv>=1.0" "ctgan>=0.7" "rdt>=1.3" tomli
python -m pip install --upgrade --force-reinstall "numpy<2"

python - <<'PY'
import torch
print("Torch import check:", torch.__version__)
PY

# The original TabDDPM requirements can pin old packages, including rdt==0.6.4,
# which is incompatible with Python 3.10. The adapter script includes small
# compatibility shims for the legacy helper imports, so do not install the
# TabDDPM requirements unless you explicitly need to reproduce that old env.
INSTALL_TABDDPM_REQUIREMENTS="${INSTALL_TABDDPM_REQUIREMENTS:-0}"
if [ "${INSTALL_TABDDPM_REQUIREMENTS}" = "1" ] && [ -f "${TABDDPM_REPO}/requirements.txt" ]; then
  python -m pip install -r "${TABDDPM_REPO}/requirements.txt"
else
  echo "Skipping TabDDPM requirements.txt install. Set INSTALL_TABDDPM_REQUIREMENTS=1 to enable it."
fi

echo "Using Python: $(command -v python)"
python --version
python - <<'PY'
import platform
print("Platform:", platform.platform())
try:
    import torch
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("Torch check failed:", exc)
PY

export FORCE_RETRAIN="${FORCE_RETRAIN:-1}"
export CTGAN_DEPTHS="${CTGAN_DEPTHS:-2}"
export CTGAN_HEADS="${CTGAN_HEADS:-8}"

COST_INCLUDE_TRAINING="${COST_INCLUDE_TRAINING:-1}"
COST_MODELS="${COST_MODELS:-diffusion ctgan tabddpm}"
COST_TRAINING_MODELS="${COST_TRAINING_MODELS:-${COST_MODELS}}"
COST_GENERATION_MODELS="${COST_GENERATION_MODELS:-${COST_MODELS}}"
TRAINING_ARGS=()
if [ "${COST_INCLUDE_TRAINING}" = "1" ]; then
  TRAINING_ARGS=(--include-training)
fi

python scripts/measure_computational_cost.py \
  --models ${COST_MODELS} \
  --training-models ${COST_TRAINING_MODELS} \
  --generation-models ${COST_GENERATION_MODELS} \
  "${TRAINING_ARGS[@]}" \
  --generation-scope full \
  --seeds 406 100 200 300 400 \
  --output-dir computational_cost

if [ -f computational_cost/failures.log ]; then
  echo ""
  echo "Failure details from computational_cost/failures.log:"
  cat computational_cost/failures.log
fi

echo "End time: $(date)"
