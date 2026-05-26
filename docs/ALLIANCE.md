# Alliance Canada setup

This repository is designed so that GitHub only stores code, configuration templates and Slurm launchers. The xBD/xView2 dataset, generated masks, checkpoints and experiment outputs must stay on Alliance storage.

## Recommended cluster layout

```text
/scratch/$USER/ml_projects/xview2_damage/
├── repo/                         # GitHub repository clone
├── .venv/                        # Python virtual environment
├── datasets/
│   └── xview2/
│       ├── raw/                  # working copy of the dataset
│       ├── processed/            # generated data
│       └── manifests/            # pair_splits.csv and future CSVs
├── experiments/                  # run outputs and checkpoints during active work
├── logs/                         # Slurm logs
└── tmp/
```

Persistent storage should be placed under your project allocation:

```text
/project/def-xxxx/$USER/xview2_damage/
├── datasets/xview2/raw/
├── models/
├── results/
└── archives/
```

## First setup on Alliance

Clone the repository into scratch:

```bash
mkdir -p /scratch/$USER/ml_projects/xview2_damage/repo
cd /scratch/$USER/ml_projects/xview2_damage/repo
git clone https://github.com/matla91/xview2-damage-assessment.git .
git checkout alliance-structure
```

Bootstrap the environment:

```bash
bash scripts/bootstrap_alliance.sh
```

Then edit `.env` and replace the placeholder project allocation:

```bash
nano .env
```

Replace all occurrences of:

```text
/project/def-xxxx/$USER
```

with your real project path.

## Dataset import

Expected training folder:

```text
$XVIEW2_DATA_ROOT/
├── images/
├── labels/
└── targets/
```

If the dataset is already in project storage, copy it to scratch:

```bash
source .env
rsync -avhP --info=progress2 "$XVIEW2_PROJECT_RAW/train/" "$XVIEW2_DATA_ROOT/"
```

If the dataset is on your local machine, transfer it to scratch with `rsync` or `scp`, then archive a copy into project storage.

## Run order

1. Check dataset integrity.
2. Generate pair-level train/validation/test splits.
3. Launch the baseline training job.

```bash
sbatch slurm/check_dataset.sbatch
sbatch slurm/make_splits.sbatch
sbatch slurm/train_baseline.sbatch
```

Watch jobs:

```bash
squeue -u $USER
```

Inspect logs:

```bash
ls -lh /scratch/$USER/ml_projects/xview2_damage/logs
tail -f /scratch/$USER/ml_projects/xview2_damage/logs/train_baseline_<job_id>.out
```

## Important rules

- Do not commit the dataset.
- Do not commit checkpoints.
- Do not commit Slurm logs.
- Commit only code, configuration templates, small manifests if useful, documentation and reproducible scripts.
- Keep active experiments on `/scratch`.
- Keep important final models and results in `/project`.
