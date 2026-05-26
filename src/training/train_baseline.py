"""Train the first U-Net baseline on xBD post-disaster masks."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

from src.data.xbd_dataset import XBDDamageDataset
from src.models.unet import SimpleUNet
from src.training.evaluate import evaluate_model


def get_device(config_value: str) -> torch.device:
    if config_value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config_value)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--split-csv", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    if args.data_root is not None:
        cfg["data"]["root"] = str(args.data_root)
    if args.split_csv is not None:
        cfg["data"]["split_csv"] = str(args.split_csv)
    if args.run_dir is not None:
        cfg["outputs"]["run_dir"] = str(args.run_dir)
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    return cfg


def main() -> None:
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    cfg = apply_cli_overrides(cfg, args)

    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)

    device = get_device(cfg["training"].get("device", "auto"))
    print(f"Using device: {device}")

    data_cfg = cfg["data"]
    print(f"Data root: {data_cfg['root']}")
    print(f"Split CSV: {data_cfg['split_csv']}")

    train_dataset = XBDDamageDataset(
        data_root=data_cfg["root"],
        split_csv=data_cfg["split_csv"],
        split="train",
        image_size=data_cfg["image_size"],
        use_pre_image=data_cfg.get("use_pre_image", False),
        use_post_image=data_cfg.get("use_post_image", True),
    )
    val_dataset = XBDDamageDataset(
        data_root=data_cfg["root"],
        split_csv=data_cfg["split_csv"],
        split="val",
        image_size=data_cfg["image_size"],
        use_pre_image=data_cfg.get("use_pre_image", False),
        use_post_image=data_cfg.get("use_post_image", True),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"].get("num_workers", 0),
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"].get("num_workers", 0),
        pin_memory=device.type == "cuda",
    )

    model_cfg = cfg["model"]
    model = SimpleUNet(
        in_channels=model_cfg["in_channels"],
        num_classes=model_cfg["num_classes"],
        base_channels=model_cfg.get("base_channels", 32),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"].get("weight_decay", 0.0),
    )

    run_dir = Path(cfg["outputs"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "resolved_config.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(cfg, file, sort_keys=False)

    best_miou = -1.0
    epochs = cfg["training"]["epochs"]
    num_classes = data_cfg["num_classes"]

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        metrics = evaluate_model(model, val_loader, device, num_classes=num_classes)
        mean_iou = float(metrics["mean_iou"])
        mean_dice = float(metrics["mean_dice"])

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_mIoU={mean_iou:.4f} | "
            f"val_mDice={mean_dice:.4f}"
        )

        if mean_iou > best_miou:
            best_miou = mean_iou
            checkpoint_path = run_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_miou": best_miou,
                    "config": cfg,
                },
                checkpoint_path,
            )
            print(f"Saved best checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
