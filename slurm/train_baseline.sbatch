#!/bin/bash
#SBATCH --job-name=xview2_unet
#SBATCH --output=/scratch/%u/ml_projects/xview2_damage/logs/train_baseline_%j.out
#SBATCH --error=/scratch/%u/ml_projects/xview2_damage/logs/train_baseline_%j.err
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --account=def-xxxx

set -euo pipefail

module --force purge
module load StdEnv/2023
module load python/3.11
module load gcc
module load cuda

REPO_ROOT="/scratch/${USER}/ml_projects/xview2_damage/repo"
VENV_ROOT="/scratch/${USER}/ml_projects/xview2_damage/.venv"

cd "${REPO_ROOT}"
source "${VENV_ROOT}/bin/activate"

set -a
if [ -f .env ]; then
  source .env
fi
set +a

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

RUN_DIR="${EXPERIMENTS_ROOT}/baseline_unet/${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"

python src/training/train_baseline.py \
  --config configs/baseline_unet_alliance.yaml \
  --data-root "${XVIEW2_DATA_ROOT}" \
  --split-csv "${XVIEW2_MANIFEST_ROOT}/pair_splits.csv" \
  --run-dir "${RUN_DIR}"
