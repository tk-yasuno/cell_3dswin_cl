"""Phase 1, Step 4: Collect and prepare the v0.3.0 Allen Institute hiPSC 3D
voxel cell dataset.

Downloads single-cell 3D fluorescence crops (quilt3 package
aics/hipsc_single_cell_image_dataset, s3://allencell) for 4 structures
(LMNB1/TOMM20/TUBA1B/TJP1 -- see config.yaml v03_dataset.structures and
docs/Plan_cell_3dswin_cl_v0-3-0.md Decisions for why "CAAX"/"TOM20" from the
original Concept Note were substituted/corrected after inspecting the real
metadata.csv), extracts the [dna, structure] channels (crop_raw's fixed
3-channel order is [dna, membrane, structure] for every cell line --
verified empirically and against the dataset documentation, NOT the
per-FOV ChannelNumber405/ChannelNumberStruct metadata columns, which index
a different, larger channel set and do not apply to crop_raw), resizes to a
fixed voxel shape, caches as float16 .npy, and writes a unified
(voxel_path, structure_name, cell_stage, finding_text, split) CSV schema --
the 3D-voxel analog of prepare_cell_dataset.py (v0.2.0).

Mitotic (cell_stage != 'M0') cells are deliberately oversampled up to
`mitotic_oversample_fraction` of each structure's per_class_target, since
mitosis is naturally rare (~2-13% per structure in this dataset -- verified
empirically), so the anomaly-scoring proxy validation (Phase 5) has enough
abnormal-like samples to compute a meaningful ROC-AUC.

Usage:
    python prepare_voxel_cell_dataset.py --config config.yaml
    python prepare_voxel_cell_dataset.py --config config.yaml --per-class-target 20 --test-per-class 5
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import quilt3
import yaml
from scipy.ndimage import zoom


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Metadata: fetch once from S3 (quilt3), cache locally (a single fetch takes
# ~10 minutes over this environment's network; a local CSV cache avoids
# repeating that on every re-run).
# ---------------------------------------------------------------------------

METADATA_COLUMNS = [
    "CellId", "crop_raw", "crop_seg", "structure_name", "FOVId", "cell_stage",
]

# crop_raw single-cell OME-TIFFs always use this FIXED 3-channel order,
# verified empirically (shape [3, Z, Y, X] for every cell line) and
# confirmed against the dataset's documentation: channel 0 = DNA dye,
# channel 1 = cell membrane dye, channel 2 = the structure protein tagged
# in that cell line. This is NOT the same as the per-FOV
# ChannelNumberStruct/ChannelNumber405/ChannelNumber638 metadata columns
# (those index the much-larger original multi-channel FOV, not the
# already-channel-reduced single-cell crop -- using them against crop_raw
# raises an out-of-bounds IndexError, caught during Phase 1 implementation).
CROP_RAW_DNA_CHANNEL = 0
CROP_RAW_STRUCTURE_CHANNEL = 2


def get_package_and_metadata(cfg: dict):
    pkg = quilt3.Package.browse(cfg["quilt3_package"], registry=cfg["registry"])

    # Shared across config versions (v03, v031, ...) since it's the same
    # underlying quilt3 package/metadata regardless of which structures a
    # given version samples from -- avoids repeating the ~10 minute S3
    # metadata.csv download for every new version.
    shared_cache_path = Path("data/_shared_cache/allen_metadata_raw.csv")
    legacy_v03_cache_path = Path("data/v03_fine_tuning/_cache/allen_metadata_raw.csv")

    if shared_cache_path.exists():
        print(f"[prepare_voxel_cell_dataset] loading cached metadata from {shared_cache_path}")
        meta_df = pd.read_csv(shared_cache_path, usecols=METADATA_COLUMNS)
    elif legacy_v03_cache_path.exists():
        print(f"[prepare_voxel_cell_dataset] reusing existing v0.3.0 metadata cache "
              f"from {legacy_v03_cache_path}")
        shared_cache_path.parent.mkdir(parents=True, exist_ok=True)
        shared_cache_path.write_bytes(legacy_v03_cache_path.read_bytes())
        meta_df = pd.read_csv(shared_cache_path, usecols=METADATA_COLUMNS)
    else:
        print("[prepare_voxel_cell_dataset] fetching metadata.csv from S3 "
              "(first run; this can take ~10 minutes) ...")
        full_df = pkg["metadata.csv"]()
        shared_cache_path.parent.mkdir(parents=True, exist_ok=True)
        full_df.to_csv(shared_cache_path, index=False)
        meta_df = full_df[METADATA_COLUMNS]
    return pkg, meta_df


# ---------------------------------------------------------------------------
# Per-cell voxel fetch + preprocess + cache
# ---------------------------------------------------------------------------

def _fetch_package_path(pkg, package_relative_path: str, local_path: Path) -> None:
    """Resolve e.g. 'crop_raw/abc123_raw.ome.tif' through the quilt3 package
    tree (arbitrary nesting depth) and fetch it to local_path."""
    parts = package_relative_path.split("/")
    node = pkg
    for part in parts:
        node = node[part]
    local_path.parent.mkdir(parents=True, exist_ok=True)
    node.fetch(str(local_path))


def _load_and_preprocess_voxel(tiff_path: Path, voxel_shape: tuple, clip_percentile: tuple) -> np.ndarray:
    from aicsimageio import AICSImage

    reader = AICSImage(str(tiff_path))
    data = reader.get_image_data("CZYX", S=0, T=0)  # [C, Z, Y, X], fixed 3-channel order
    channels = [data[CROP_RAW_DNA_CHANNEL], data[CROP_RAW_STRUCTURE_CHANNEL]]

    processed = []
    for ch in channels:
        ch = ch.astype(np.float32)
        lo, hi = np.percentile(ch, clip_percentile)
        if hi <= lo:
            hi = lo + 1.0
        ch = np.clip(ch, lo, hi)
        ch = (ch - lo) / (hi - lo)
        zoom_factors = [t / s for t, s in zip(voxel_shape, ch.shape)]
        ch_resized = zoom(ch, zoom_factors, order=1)
        processed.append(ch_resized.astype(np.float16))

    return np.stack(processed, axis=0)  # [2, D, H, W]


def fetch_and_cache_voxel(pkg, row: dict, cfg: dict, tmp_dir: Path) -> str | None:
    """Download+preprocess one cell's crop_raw volume; return the cached
    .npy path, or None if the cell could not be processed (corrupted file,
    etc.) -- callers should skip and top up from the sampling pool,
    mirroring v0.2.0's _sample_valid_images."""
    structure = row["structure_name"]
    cell_id = row["CellId"]
    voxel_shape = tuple(cfg["voxel_shape"])
    cache_path = Path(cfg["voxel_cache_dir"]) / structure / f"{cell_id}.npy"
    if cache_path.exists():
        return str(cache_path)

    tmp_tiff = tmp_dir / f"{cell_id}_raw.ome.tif"
    try:
        _fetch_package_path(pkg, row["crop_raw"], tmp_tiff)
        voxel = _load_and_preprocess_voxel(tmp_tiff, voxel_shape, tuple(cfg["intensity_clip_percentile"]))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, voxel)
        return str(cache_path)
    except Exception as exc:  # noqa: BLE001 - log and let the caller resample
        print(f"[prepare_voxel_cell_dataset] WARNING: failed to process CellId={cell_id} "
              f"({structure}): {exc}")
        return None
    finally:
        tmp_tiff.unlink(missing_ok=True)


def sample_and_cache(pool_df: pd.DataFrame, target_n: int, rng: random.Random,
                      pkg, cfg: dict, structure: str, label: str, tmp_dir: Path,
                      max_workers: int = 8) -> list[dict]:
    """Sample up to target_n cells from pool_df, downloading+caching
    concurrently (network I/O bound) until target_n successfully processed
    rows are found or the pool is exhausted (over-sampling + skip-on-failure
    in waves, mirroring v0.2.0's _sample_valid_images for corrupted-source
    robustness, parallelized via ThreadPoolExecutor for wall-clock
    feasibility over many hundreds of S3 downloads)."""
    import concurrent.futures

    indices = list(pool_df.index)
    rng.shuffle(indices)
    collected: list[dict] = []
    skipped = 0
    cursor = 0

    while len(collected) < target_n and cursor < len(indices):
        remaining_needed = target_n - len(collected)
        # Generous buffer since most downloads succeed; re-waves handle any shortfall.
        wave_size = min(len(indices) - cursor, max(remaining_needed + 2, int(remaining_needed * 1.1) + 1))
        wave_indices = indices[cursor:cursor + wave_size]
        cursor += wave_size

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(fetch_and_cache_voxel, pkg, pool_df.loc[idx].to_dict(), cfg, tmp_dir): idx
                for idx in wave_indices
            }
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                voxel_path = future.result()
                if voxel_path is None:
                    skipped += 1
                    continue
                if len(collected) >= target_n:
                    continue  # this wave over-delivered; extra successes are kept on disk but unused
                collected.append({
                    "voxel_path": voxel_path,
                    "structure_name": structure,
                    "cell_stage": pool_df.loc[idx, "cell_stage"],
                })

    if skipped:
        print(f"[prepare_voxel_cell_dataset] {structure}/{label}: skipped {skipped} "
              f"unprocessable cells (out of {cursor} candidates attempted)")
    if len(collected) < target_n:
        print(f"[prepare_voxel_cell_dataset] WARNING: {structure}/{label} only yielded "
              f"{len(collected)}/{target_n} cells")
    return collected


def collect_structure(pkg, meta_df: pd.DataFrame, structure: str, cfg: dict,
                       rng: random.Random, tmp_dir: Path, max_workers: int = 8) -> list[dict]:
    subset = meta_df[meta_df["structure_name"] == structure]
    mitotic_pool = subset[subset["cell_stage"] != "M0"]
    interphase_pool = subset[subset["cell_stage"] == "M0"]

    per_class_target = cfg["per_class_target"]
    mitotic_fraction = cfg.get("mitotic_oversample_fraction", 0.2)
    n_mitotic_target = min(len(mitotic_pool), round(per_class_target * mitotic_fraction))
    n_interphase_target = per_class_target - n_mitotic_target

    print(f"[prepare_voxel_cell_dataset] {structure}: pool total={len(subset)} "
          f"(mitotic available={len(mitotic_pool)}), targeting "
          f"{n_mitotic_target} mitotic + {n_interphase_target} interphase")

    rows = []
    rows += sample_and_cache(mitotic_pool, n_mitotic_target, rng, pkg, cfg, structure, "mitotic",
                              tmp_dir, max_workers=max_workers)
    rows += sample_and_cache(interphase_pool, n_interphase_target, rng, pkg, cfg, structure, "interphase",
                              tmp_dir, max_workers=max_workers)
    return rows


# ---------------------------------------------------------------------------
# Assembly: caption -> split -> write CSVs
# ---------------------------------------------------------------------------

def build_finding_text(structure_display_name: str, cell_stage: str,
                        templates: dict, rng: random.Random) -> str:
    bucket = "interphase" if cell_stage == "M0" else "mitotic"
    choices = templates[bucket]
    template = rng.choice(choices)
    return template.format(structure=structure_display_name, stage=cell_stage)


def assemble_dataframe(all_rows: list[dict], cfg: dict, rng: random.Random) -> pd.DataFrame:
    test_per_class = cfg["test_per_class"]
    templates = cfg["finding_templates"]
    display_names = cfg["structure_display_names"]

    by_structure: dict[str, list[dict]] = {}
    for row in all_rows:
        by_structure.setdefault(row["structure_name"], []).append(row)

    records = []
    for structure, items in by_structure.items():
        rng.shuffle(items)
        test_items = items[:test_per_class]
        trainval_items = items[test_per_class:]
        for split_name, split_items in (("test", test_items), ("trainval", trainval_items)):
            for item in split_items:
                finding_text = build_finding_text(display_names.get(structure, structure),
                                                   item["cell_stage"], templates, rng)
                records.append({
                    "voxel_path": item["voxel_path"],
                    "structure_name": structure,
                    "cell_stage": item["cell_stage"],
                    "finding_text": finding_text,
                    "split": split_name,
                })
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--version", default="v03", choices=["v03", "v031"],
                         help="which config section prefix to use, e.g. 'v03_dataset' or 'v031_dataset'")
    parser.add_argument("--per-class-target", type=int, default=None,
                         help="override {version}_dataset.per_class_target (for quick smoke runs)")
    parser.add_argument("--test-per-class", type=int, default=None,
                         help="override {version}_dataset.test_per_class (for quick smoke runs)")
    parser.add_argument("--structures", nargs="+", default=None,
                         help="override {version}_dataset.structures (for quick smoke runs)")
    parser.add_argument("--max-workers", type=int, default=8,
                         help="concurrent S3 download threads (I/O bound, safe to raise)")
    args = parser.parse_args()

    cfg = load_config(args.config)[f"{args.version}_dataset"]
    if args.per_class_target is not None:
        cfg["per_class_target"] = args.per_class_target
    if args.test_per_class is not None:
        cfg["test_per_class"] = args.test_per_class
    if args.structures is not None:
        cfg["structures"] = args.structures

    rng = random.Random(cfg["random_seed"])
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["voxel_cache_dir"]).mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(cfg["output_dir"]) / "_tmp_tiff"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    pkg, meta_df = get_package_and_metadata(cfg)

    all_rows: list[dict] = []
    for structure in cfg["structures"]:
        print(f"[prepare_voxel_cell_dataset] collecting {structure} ...")
        all_rows += collect_structure(pkg, meta_df, structure, cfg, rng, tmp_dir, max_workers=args.max_workers)

    df = assemble_dataframe(all_rows, cfg, rng)

    output_dir = Path(cfg["output_dir"])
    test_df = df[df["split"] == "test"].drop(columns=["split"]).reset_index(drop=True)
    trainval_df = df[df["split"] == "trainval"].drop(columns=["split"]).reset_index(drop=True)

    # train.csv / val.csv: split the trainval pool 50/50 so
    # prepare_progressive_splits.py's train+val pooling reconstructs the
    # full per-structure trainval pool (same convention as v0.2.0's
    # prepare_cell_dataset.py).
    trainval_df = trainval_df.sample(frac=1.0, random_state=cfg["random_seed"]).reset_index(drop=True)
    half = len(trainval_df) // 2
    train_df = trainval_df.iloc[:half].reset_index(drop=True)
    val_df = trainval_df.iloc[half:].reset_index(drop=True)

    for name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
        out_path = output_dir / f"{name}.csv"
        split_df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"[prepare_voxel_cell_dataset] wrote {out_path} "
              f"({len(split_df)} rows, {split_df['structure_name'].nunique()} structures)")

    n_structures = df["structure_name"].nunique()
    n_abnormal = int((df["cell_stage"] != "M0").sum())
    print(f"[prepare_voxel_cell_dataset] OK: total {len(df)} rows across {n_structures} structures "
          f"(train={len(train_df)}, val={len(val_df)}, test={len(test_df)}, "
          f"abnormal-like/mitotic={n_abnormal})")


if __name__ == "__main__":
    main()
