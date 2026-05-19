# xView2 Damage Assessment

Computer vision project for disaster damage assessment on the **xBD / xView2 Challenge training set**.

The goal is to build a reproducible pipeline that demonstrates that the dataset is accessible, loaded, visualized, split into train/validation/test sets, and used to train a first baseline model.

## Project scope

This repository focuses on:

- exploratory data analysis of the xBD / xView2 Challenge training set;
- clean preparation of pre-disaster and post-disaster image pairs;
- generation of train/validation/test splits by `pair_id`;
- a first post-disaster segmentation baseline using U-Net;
- a future pre/post change-detection prototype using 6-channel input or a Siamese architecture.

## Current dataset status

The local EDA confirmed the following on the Challenge training set:

- 2,799 complete pre/post image pairs;
- 5,598 RGB PNG images;
- 5,598 JSON label files;
- 5,598 target masks;
- 0 incomplete instances;
- 0 unreadable images;
- image size: 1024 x 1024.

The dataset is imbalanced. Post-disaster building annotations are dominated by `no-damage`, while severe classes are much smaller. The target masks are also dominated by background pixels, so plain accuracy is not an appropriate metric.

## Expected local data structure

The dataset itself must not be committed to Git.

Expected local structure:

```text
data/
└── raw/
    └── train/
        ├── images/
        ├── labels/
        └── targets/
```

On Windows, the original local path used during EDA was:

```text
D:\IA\Projets\xView2\train_images_labels_targets\train
```

## Repository structure

```text
xview2-damage-assessment/
├── configs/
│   └── baseline_unet.yaml
├── data/
│   └── README.md
├── notebooks/
│   └── README.md
├── outputs/
│   └── README.md
├── src/
│   ├── data/
│   │   ├── make_splits.py
│   │   └── xbd_dataset.py
│   ├── models/
│   │   └── unet.py
│   ├── training/
│   │   ├── evaluate.py
│   │   └── train_baseline.py
│   └── utils/
│       ├── metrics.py
│       └── visualization.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Baseline

Initial baseline:

```text
post-disaster RGB image -> U-Net -> post-disaster damage mask
```

Target classes:

```text
0 = background
1 = building / no-damage
2 = minor-damage
3 = major-damage
4 = destroyed
```

This baseline is intentionally simple. It validates the full training pipeline before moving to a stronger pre/post change-detection model.

## Planned prototype architecture

After the baseline, the prototype should evolve toward:

```text
pre-disaster RGB + post-disaster RGB -> change-detection model -> damage segmentation mask
```

Candidate models:

- 6-channel U-Net;
- Siamese U-Net;
- DeepLabV3+;
- SegFormer or another transformer-based segmentation model.

## Metrics

Because the target masks are highly imbalanced, the project should prioritize:

- IoU per class;
- mean IoU;
- Dice score;
- macro-F1;
- recall for `major-damage` and `destroyed`.

Avoid using plain pixel accuracy as the main metric.

## Quick start

Install dependencies:

```bash
pip install -r requirements.txt
```

Create train/validation/test splits:

```bash
python src/data/make_splits.py \
  --data-root data/raw/train \
  --output outputs/splits/pair_splits.csv \
  --train-size 0.70 \
  --val-size 0.15 \
  --test-size 0.15
```

Train the first baseline:

```bash
python src/training/train_baseline.py \
  --config configs/baseline_unet.yaml
```

## Current milestone checklist

- [x] Dataset downloaded and extracted.
- [x] Dataset loaded in JupyterLab.
- [x] EDA notebook created.
- [x] Dataset structure verified.
- [x] Pre/post pairs checked.
- [x] Damage class imbalance identified.
- [x] Initial Git repository organized.
- [ ] Train/validation/test CSV committed or generated locally.
- [ ] First baseline trained.
- [ ] First predictions visualized.
- [ ] Technical architecture slide prepared.

## License and data use

The xBD / xView2 dataset is released under its own data license. Do not redistribute the dataset in this repository. Users must download the data from the official xView2 source and respect the data use terms.
