"""Phase 2, Step 5: Build stratified progressive training subsets.

Ported unmodified (except the --version choices list) from
0_LogBAK/pre-code_cell_vlm_cl/prepare_progressive_splits.py -- this script is
dataset-agnostic (it only reads `{version}_dataset.output_dir`/
`train_sizes`/`scale_labels`/`stratify_by` from config.yaml and a generic
train.csv/val.csv schema), so it works unchanged for v0.3.0's voxel dataset.

v0.3.0 (--version v03): pooled Allen Institute hiPSC voxel train+val
(2,500 cells), stratified by structure_name (4 classes: LMNB1/TOM20/TUBA1B/
CAAX). See docs/Plan_cell_3dswin_cl_v0-3-0.md, Phase 2.

Usage:
    python prepare_progressive_splits.py --config config.yaml --version v03
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def stratified_subset(pool: pd.DataFrame, n: int, seed: int, stratify_col: str) -> pd.DataFrame:
    """Draw a stratified subset of size n from pool, balanced over stratify_col.

    Falls back to a plain random sample if n >= len(pool) (used for the
    full-pool "2.5k" scale).
    """
    if n >= len(pool):
        return pool.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    n_groups = pool[stratify_col].nunique()
    per_group = max(1, n // n_groups)
    parts = []
    for _, group_df in pool.groupby(stratify_col):
        k = min(per_group, len(group_df))
        parts.append(group_df.sample(n=k, random_state=seed))
    subset = pd.concat(parts, ignore_index=True)

    # Top up/trim to hit the exact target size n (stratified sampling with
    # integer division can undershoot slightly).
    if len(subset) < n:
        remaining = pool.drop(subset.index, errors="ignore")
        extra = remaining.sample(n=min(n - len(subset), len(remaining)), random_state=seed)
        subset = pd.concat([subset, extra], ignore_index=True)
    elif len(subset) > n:
        subset = subset.sample(n=n, random_state=seed)

    return subset.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--version", default="v03", choices=["v03"],
                         help="which config section prefix to use, e.g. 'v03_dataset'")
    args = parser.parse_args()

    cfg = load_config(args.config)[f"{args.version}_dataset"]
    output_dir = Path(cfg["output_dir"])
    seed = cfg["random_seed"]
    stratify_col = cfg["stratify_by"]

    train_df = pd.read_csv(output_dir / "train.csv")
    val_df = pd.read_csv(output_dir / "val.csv")
    pool = pd.concat([train_df, val_df], ignore_index=True)
    print(f"[prepare_progressive_splits] pooled train+val = {len(pool)} rows, "
          f"{pool[stratify_col].nunique()} classes")

    default_scale_labels = {800: "0.8k", 1600: "1.6k", 2500: "2.5k"}
    scale_labels = cfg.get("scale_labels", default_scale_labels)
    for n in cfg["train_sizes"]:
        subset = stratified_subset(pool, n, seed, stratify_col)
        label = scale_labels.get(n, f"{n}")
        out_path = output_dir / f"train_{label}.csv"
        subset.to_csv(out_path, index=False, encoding="utf-8")
        print(f"[prepare_progressive_splits] wrote {out_path} "
              f"({len(subset)} rows, {subset[stratify_col].nunique()} classes)")


if __name__ == "__main__":
    main()
