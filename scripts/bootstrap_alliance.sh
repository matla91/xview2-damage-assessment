#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/scratch/${USER}/ml_projects/xview2_damage"
REPO_ROOT="${PROJECT_ROOT}/repo"
VENV_ROOT="${PROJECT_ROOT}/.venv"

mkdir -p "${PROJECT_ROOT}"/{datasets/xview2/{raw,processed,manifests},experiments,logs,tmp}
mkdir -p "${REPO_ROOT}"

module --force purge
module load StdEnv/2023
module load python/3.11
module load gcc

cd "${REPO_ROOT}"

python -m venv --system-site-packages "${VENV_ROOT}"
source "${VENV_ROOT}/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

if [ ! -f .env ]; then
  cp configs/paths.example.env .env
  sed -i "s#/scratch/USER#/scratch/${USER}#g" .env
  sed -i "s#/project/def-xxxx/USER#/project/def-xxxx/${USER}#g" .env
  echo "Created .env from configs/paths.example.env. Edit PROJECT_STORAGE before serious runs."
fi

echo "Alliance bootstrap completed."
echo "Repo root: ${REPO_ROOT}"
echo "Venv root: ${VENV_ROOT}"
