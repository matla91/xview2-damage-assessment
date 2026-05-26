#!/bin/bash
#SBATCH --job-name=xview2_splits
#SBATCH --output=/scratch/%u/ml_projects/xview2_damage/logs/make_splits_%j.out
#SBATCH --error=/scratch/%u/ml_projects/xview2_damage/logs/make_splits_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --account=def-xxxx

set -euo pipefail

module --force purge
module load StdEnv/2023
module load python/3.11
module load gcc

REPO_ROOT="/scratch/${USER}/ml_projects/xview2_damage/repo"
VENV_ROOT="/scratch/${USER}/ml_projects/xview2_damage/.venv"

cd "${REPO_ROOT}"
source "${VENV_ROOT}/bin/activate"

set -a
if [ -f .env ]; then
  source .env
fi
set +a

mkdir -p "${XVIEW2_MANIFEST_ROOT}"

python src/data/make_splits.py \
  --data-root "${XVIEW2_DATA_ROOT}" \
  --output "${XVIEW2_MANIFEST_ROOT}/pair_splits.csv" \
  --train-size 0.70 \
  --val-size 0.15 \
  --test-size 0.15 \
  --seed 42
