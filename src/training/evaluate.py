"""Evaluation utilities for segmentation baselines."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from src.utils.metrics import confusion_matrix, compute_dice_from_confusion, compute_iou_from_confusion


@torch.no_grad()
def evaluate_model(model: torch.nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> dict:
    model.eval()
    total_cm = torch.zeros((num_classes, num_classes), dtype=torch.long, device=device)

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        logits = model(images)
        preds = logits.argmax(dim=1)
        total_cm += confusion_matrix(preds, masks, num_classes=num_classes).to(device)

    iou = compute_iou_from_confusion(total_cm.cpu())
    dice = compute_dice_from_confusion(total_cm.cpu())

    return {
        "confusion_matrix": total_cm.cpu(),
        "per_class_iou": iou["per_class_iou"],
        "mean_iou": iou["mean_iou"],
        "per_class_dice": dice["per_class_dice"],
        "mean_dice": dice["mean_dice"],
    }
