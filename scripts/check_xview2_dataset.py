"""Check the xBD/xView2 dataset layout before launching jobs on Alliance."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=os.environ.get("XVIEW2_DATA_ROOT"),
        help="Path to the xView2 train folder containing images/, labels/ and targets/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.data_root is None:
        raise ValueError("Provide --data-root or set XVIEW2_DATA_ROOT.")

    data_root = Path(args.data_root).expanduser().resolve()
    images_dir = data_root / "images"
    labels_dir = data_root / "labels"
    targets_dir = data_root / "targets"

    print(f"Dataset root: {data_root}")

    for folder in [images_dir, labels_dir, targets_dir]:
        status = "OK" if folder.exists() else "MISSING"
        print(f"{status:8s} {folder}")
        if not folder.exists():
            raise FileNotFoundError(f"Missing expected folder: {folder}")

    pre_images = sorted(images_dir.glob("*_pre_disaster.png"))
    post_images = sorted(images_dir.glob("*_post_disaster.png"))
    labels = sorted(labels_dir.glob("*.json"))
    targets = sorted(targets_dir.glob("*_target.png"))

    pre_ids = {p.name.replace("_pre_disaster.png", "") for p in pre_images}
    post_ids = {p.name.replace("_post_disaster.png", "") for p in post_images}

    missing_post = sorted(pre_ids - post_ids)
    missing_pre = sorted(post_ids - pre_ids)

    print("\nCounts")
    print(f"  pre images   : {len(pre_images)}")
    print(f"  post images  : {len(post_images)}")
    print(f"  labels json  : {len(labels)}")
    print(f"  target masks : {len(targets)}")
    print(f"  complete pairs: {len(pre_ids & post_ids)}")
    print(f"  missing post : {len(missing_post)}")
    print(f"  missing pre  : {len(missing_pre)}")

    if missing_post[:10]:
        print("\nExamples missing post images:")
        for item in missing_post[:10]:
            print(f"  - {item}")

    if missing_pre[:10]:
        print("\nExamples missing pre images:")
        for item in missing_pre[:10]:
            print(f"  - {item}")

    if missing_post or missing_pre:
        raise RuntimeError("Incomplete pre/post pairs detected.")

    print("\nDataset check passed.")


if __name__ == "__main__":
    main()
