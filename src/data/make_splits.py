"""Create train/validation/test splits for the xBD Challenge training set.

The split is performed at pair_id level, never at individual image level.
This prevents separating pre/post files from the same event across splits.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


FILENAME_RE = re.compile(
    r"^(?P<disaster>.+?)_(?P<event_id>\d+)_(?P<phase>pre|post)_disaster(?:_target)?\.(?P<ext>png|json)$",
    re.IGNORECASE,
)


def parse_filename(path: Path) -> dict:
    match = FILENAME_RE.match(path.name)
    if not match:
        return {"parse_ok": False, "filename": path.name}

    item = match.groupdict()
    pair_id = f"{item['disaster']}_{item['event_id']}"
    return {
        "parse_ok": True,
        "filename": path.name,
        "pair_id": pair_id,
        "disaster": item["disaster"],
        "event_id": item["event_id"],
        "phase": item["phase"].lower(),
        "ext": item["ext"].lower(),
    }


def collect_pairs(data_root: Path) -> pd.DataFrame:
    images_dir = data_root / "images"
    labels_dir = data_root / "labels"
    targets_dir = data_root / "targets"

    for folder in [images_dir, labels_dir, targets_dir]:
        if not folder.exists():
            raise FileNotFoundError(f"Missing expected folder: {folder}")

    rows = []
    for image_path in sorted(images_dir.glob("*.png")):
        parsed = parse_filename(image_path)
        if parsed["parse_ok"]:
            rows.append(parsed)

    files_df = pd.DataFrame(rows)
    if files_df.empty:
        raise RuntimeError(f"No valid image files found in {images_dir}")

    pivot = (
        files_df.groupby(["pair_id", "disaster", "event_id", "phase"])
        .size()
        .reset_index(name="count")
        .pivot_table(
            index=["pair_id", "disaster", "event_id"],
            columns="phase",
            values="count",
            fill_value=0,
            aggfunc="sum",
        )
        .reset_index()
    )
    pivot.columns.name = None

    for phase in ["pre", "post"]:
        if phase not in pivot.columns:
            pivot[phase] = 0

    pivot["has_pre"] = pivot["pre"] == 1
    pivot["has_post"] = pivot["post"] == 1
    pivot["is_complete_pair"] = pivot["has_pre"] & pivot["has_post"]

    return pivot[pivot["is_complete_pair"]].copy()


def make_random_split(
    pairs_df: pd.DataFrame,
    train_size: float,
    val_size: float,
    test_size: float,
    seed: int,
) -> pd.DataFrame:
    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    train_df, temp_df = train_test_split(
        pairs_df,
        train_size=train_size,
        random_state=seed,
        shuffle=True,
    )

    relative_val_size = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=relative_val_size,
        random_state=seed,
        shuffle=True,
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    return pd.concat([train_df, val_df, test_df], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-size", type=float, default=0.70)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pairs_df = collect_pairs(args.data_root)
    split_df = make_random_split(
        pairs_df=pairs_df,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    split_df[["pair_id", "disaster", "event_id", "split"]].to_csv(args.output, index=False)

    print(f"Saved split file to: {args.output}")
    print(split_df["split"].value_counts())


if __name__ == "__main__":
    main()
