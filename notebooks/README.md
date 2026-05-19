# Notebooks

Recommended notebooks:

```text
01_eda_xbd.ipynb
02_baseline_unet.ipynb
```

## 01_eda_xbd.ipynb

Purpose:

- verify dataset structure;
- count images, labels and masks;
- check pre/post pairing;
- analyze disaster distribution;
- parse JSON annotations;
- analyze damage class imbalance;
- visualize pre/post images, masks and polygons.

## 02_baseline_unet.ipynb

Purpose:

- load train/validation/test splits;
- create a PyTorch Dataset;
- train a first post-disaster U-Net baseline;
- evaluate IoU, Dice and recall for severe classes;
- visualize predicted masks.

The actual notebooks can be added here once executed locally.
