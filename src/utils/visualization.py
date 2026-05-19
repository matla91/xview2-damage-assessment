"""Visualization helpers for xBD images and masks."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


CLASS_NAMES = {
    0: "background",
    1: "no-damage",
    2: "minor-damage",
    3: "major-damage",
    4: "destroyed",
}


def load_rgb(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def load_mask(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"))


def show_pre_post_mask(pre_path: str | Path, post_path: str | Path, mask_path: str | Path) -> None:
    pre = load_rgb(pre_path)
    post = load_rgb(post_path)
    mask = load_mask(mask_path)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(pre)
    plt.title("Pre-disaster")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(post)
    plt.title("Post-disaster")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(mask)
    plt.title("Target mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def show_prediction(image: np.ndarray, target: np.ndarray, prediction: np.ndarray) -> None:
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Input image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(target)
    plt.title("Ground truth")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(prediction)
    plt.title("Prediction")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
