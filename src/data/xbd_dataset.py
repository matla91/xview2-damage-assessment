"""PyTorch Dataset for xBD / xView2 Challenge training data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class XBDDamageDataset(Dataset):
    """Dataset for post-disaster damage segmentation.

    This first baseline uses only the post-disaster RGB image and the post-disaster
    target mask. It can later be extended to load both pre and post images.
    """

    def __init__(
        self,
        data_root: str | Path,
        split_csv: str | Path,
        split: str,
        image_size: int = 256,
        use_pre_image: bool = False,
        use_post_image: bool = True,
    ) -> None:
        self.data_root = Path(data_root)
        self.images_dir = self.data_root / "images"
        self.targets_dir = self.data_root / "targets"
        self.image_size = image_size
        self.use_pre_image = use_pre_image
        self.use_post_image = use_post_image

        if not self.images_dir.exists() or not self.targets_dir.exists():
            raise FileNotFoundError(
                f"Expected images/ and targets/ under data root: {self.data_root}"
            )

        split_df = pd.read_csv(split_csv)
        self.items = split_df[split_df["split"] == split].reset_index(drop=True)

        if self.items.empty:
            raise ValueError(f"No items found for split={split!r} in {split_csv}")

    def __len__(self) -> int:
        return len(self.items)

    def _image_path(self, pair_id: str, phase: str) -> Path:
        return self.images_dir / f"{pair_id}_{phase}_disaster.png"

    def _target_path(self, pair_id: str, phase: str) -> Path:
        return self.targets_dir / f"{pair_id}_{phase}_disaster_target.png"

    def _load_rgb(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        image = image.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return tensor

    def _load_mask(self, path: Path) -> torch.Tensor:
        mask = Image.open(path).convert("L")
        mask = mask.resize((self.image_size, self.image_size), resample=Image.NEAREST)
        array = np.asarray(mask, dtype=np.int64)
        return torch.from_numpy(array)

    def __getitem__(self, index: int) -> dict:
        row = self.items.iloc[index]
        pair_id = row["pair_id"]

        tensors = []
        if self.use_pre_image:
            tensors.append(self._load_rgb(self._image_path(pair_id, "pre")))
        if self.use_post_image:
            tensors.append(self._load_rgb(self._image_path(pair_id, "post")))

        image = torch.cat(tensors, dim=0)
        mask = self._load_mask(self._target_path(pair_id, "post"))

        return {
            "image": image,
            "mask": mask,
            "pair_id": pair_id,
            "disaster": row["disaster"],
        }
