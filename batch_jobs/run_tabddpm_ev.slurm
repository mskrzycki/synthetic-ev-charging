#!/bin/bash
#SBATCH --output /home/iroshanij/jobs/TabDDPM_EV/tabddpm_ev-%j.out
#SBATCH --job-name tabddpm_ev
#SBATCH --partition sintef
#SBATCH --ntasks 1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8
#SBATCH --gres gpu:a30:1
#SBATCH --time 02-00:00:00

set -euo pipefail

echo "Job started on $(hostname)"
echo "Start time: $(date)"

# ---- paths: edit these on the server ----
PROJECT_DIR=/home/iroshanij/SynDaGen/synthetic-ev-charging
TABDDPM_REPO=/home/iroshanij/SynDaGen/tab-ddpm

export TABDDPM_REPO
export PYTHONPATH="${TABDDPM_REPO}:${TABDDPM_REPO}/scripts:${PYTHONPATH:-}"

cd "${PROJECT_DIR}"

# ---- activate environment: edit if needed ----
# Example conda:
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tddpm

# Or, if using venv instead, comment conda lines above and use:
# source .venv/bin/activate

echo "Python: $(which python)"
python --version

# ---- 1. Prepare EV data in TabDDPM format ----
python TabDDPM/scripts/tabddpm_benchmark.py prepare

# ---- 2. Train TabDDPM ----
# Faithful/full setting based on original TabDDPM-style training budget.
python TabDDPM/scripts/tabddpm_benchmark.py train \
  --device cuda \
  --steps 30000 \
  --timesteps 1000 \
  --batch-size 4096

# ---- 3. Generate yearly synthetic data for all benchmark seeds ----
python TabDDPM/scripts/tabddpm_benchmark.py sample-year \
  --device cuda \
  --seeds 406 100 200 300 400 \
  --timesteps 1000 \
  --batch-size 4096

# ---- 4. Recompute paper tables including TabDDPM ----
python scripts/evaluation_tables.py

echo "End time: $(date)"
echo "Job finished successfully"
