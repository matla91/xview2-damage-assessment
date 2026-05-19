"""Segmentation metrics for xBD damage masks."""

from __future__ import annotations

import torch


def confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Compute confusion matrix for segmentation predictions."""
    pred = pred.view(-1).long()
    target = target.view(-1).long()
    mask = (target >= 0) & (target < num_classes)
    indices = num_classes * target[mask] + pred[mask]
    cm = torch.bincount(indices, minlength=num_classes**2)
    return cm.reshape(num_classes, num_classes)


def compute_iou_from_confusion(cm: torch.Tensor, eps: float = 1e-7) -> dict:
    """Compute per-class IoU and mean IoU from a confusion matrix."""
    cm = cm.float()
    tp = torch.diag(cm)
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp
    iou = tp / (tp + fp + fn + eps)
    return {
        "per_class_iou": iou,
        "mean_iou": iou.mean(),
    }


def compute_dice_from_confusion(cm: torch.Tensor, eps: float = 1e-7) -> dict:
    """Compute per-class Dice and mean Dice from a confusion matrix."""
    cm = cm.float()
    tp = torch.diag(cm)
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    return {
        "per_class_dice": dice,
        "mean_dice": dice.mean(),
    }
