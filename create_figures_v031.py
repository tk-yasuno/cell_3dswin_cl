"""Phase 6: Result figures for the v0.3.1 3D voxel contrastive learning MVP
(8-structure, 5,000-cell diversity/scale extension of v0.3.0).

Reads the JSON/CSV outputs from evaluate_zeroshot_voxel.py,
evaluate_retrieval_voxel.py, scoring_cell_anomaly_voxel.py, and
retrieve_diagnosis_reason.py (all run with --version v031, across base +
1.6k/3.2k/4.8k/6.4k scales) and renders comparison figures into
paper_cell_3dswin_cl/0_cell_3dswin_cl_v0-3-0/figures_v031/, mirroring
create_figures_v03.py's plotting conventions.

Usage:
    python create_figures_v031.py --config config.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

SCALES = ["base", "voxel_swin_claip_1.6k", "voxel_swin_claip_3.2k", "voxel_swin_claip_4.8k"]
SCALE_LABELS = ["Base\n(random init)", "1.6k", "3.2k", "4.8k"]


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fig_zero_shot_accuracy(eval_dir: Path, out_dir: Path, cfg: dict) -> None:
    rows = []
    for scale, label in zip(SCALES, SCALE_LABELS):
        data = load_json(eval_dir / f"zero_shot_{scale}.json")
        if data:
            rows.append({"scale": label, "accuracy": data["accuracy"]})
    if not rows:
        print("[create_figures_v031] skip fig_zero_shot_accuracy: no data found")
        return
    df = pd.DataFrame(rows)

    plt.figure(figsize=(6, 4))
    ax = sns.barplot(data=df, x="scale", y="accuracy", color="#4C72B0")
    ax.axhline(cfg["v031_evaluation"]["zero_shot"]["chance_accuracy"], color="gray",
               linestyle="--", label="chance (1/8 structures)")
    ax.set_ylabel("Zero-shot top-1 accuracy (voxel test, n=1,600, 8 structures)")
    ax.set_xlabel("Training scale")
    ax.set_title("v0.3.1: Zero-shot 8-way structure classification accuracy")
    ax.legend()
    plt.tight_layout()
    out_path = out_dir / "01_zero_shot_accuracy_by_scale.png"
    plt.savefig(out_path, dpi=cfg["v031_paper"]["figures"]["dpi"])
    plt.close()
    print(f"[create_figures_v031] wrote {out_path}")


def fig_retrieval_recall(eval_dir: Path, out_dir: Path, cfg: dict) -> None:
    rows = []
    for scale, label in zip(SCALES, SCALE_LABELS):
        data = load_json(eval_dir / f"retrieval_{scale}.json")
        if data:
            for direction in ("voxel_to_text", "text_to_voxel"):
                for k_label, recall in data[direction].items():
                    rows.append({"scale": label, "direction": direction, "k": k_label, "recall": recall})
    if not rows:
        print("[create_figures_v031] skip fig_retrieval_recall: no data found")
        return
    df = pd.DataFrame(rows)

    plt.figure(figsize=(7, 4.5))
    ax = sns.barplot(data=df, x="scale", y="recall", hue="k")
    ax.set_ylabel("Recall@k")
    ax.set_xlabel("Training scale")
    ax.set_title("v0.3.1: Voxel<->text retrieval recall (8 structures)")
    plt.tight_layout()
    out_path = out_dir / "02_retrieval_recall_by_scale.png"
    plt.savefig(out_path, dpi=cfg["v031_paper"]["figures"]["dpi"])
    plt.close()
    print(f"[create_figures_v031] wrote {out_path}")


def fig_anomaly_violin(eval_dir: Path, out_dir: Path, cfg: dict, model_key: str) -> None:
    csv_path = eval_dir / f"cell_anomaly_voxel_scores_{model_key}.csv"
    if not csv_path.exists():
        print(f"[create_figures_v031] skip fig_anomaly_violin: {csv_path} not found")
        return
    df = pd.read_csv(csv_path)
    df["group"] = df["proxy_is_abnormal_like"].map({True: "abnormal-like (mitotic)", False: "normal-like (interphase)"})

    plt.figure(figsize=(6, 4))
    ax = sns.violinplot(data=df, x="group", y="cell_anomaly_voxel_score")
    ax.set_title(f"v0.3.1 voxel anomaly score distribution by cell-stage group ({model_key})")
    ax.set_xlabel("")
    plt.tight_layout()
    out_path = out_dir / f"03_anomaly_violin_{model_key}.png"
    plt.savefig(out_path, dpi=cfg["v031_paper"]["figures"]["dpi"])
    plt.close()
    print(f"[create_figures_v031] wrote {out_path}")


def fig_training_loss(models_dir: Path, out_dir: Path, cfg: dict) -> None:
    rows = []
    for scale in ["1.6k", "3.2k", "4.8k"]:
        history_path = models_dir / f"voxel_swin_claip_{scale}" / "training_history.json"
        data = load_json(history_path)
        if data:
            for epoch_rec in data["epochs"]:
                rows.append({"scale": scale, "epoch": epoch_rec["epoch"], "val_loss": epoch_rec["val_loss"]})
    if not rows:
        print("[create_figures_v031] skip fig_training_loss: no training_history.json found")
        return
    df = pd.DataFrame(rows)

    plt.figure(figsize=(6, 4))
    ax = sns.lineplot(data=df, x="epoch", y="val_loss", hue="scale", marker="o")
    ax.set_title("v0.3.1: Validation InfoNCE loss by progressive training scale")
    plt.tight_layout()
    out_path = out_dir / "04_validation_loss_by_scale.png"
    plt.savefig(out_path, dpi=cfg["v031_paper"]["figures"]["dpi"])
    plt.close()
    print(f"[create_figures_v031] wrote {out_path}")


def fig_diagnosis_retrieval(eval_dir: Path, out_dir: Path, cfg: dict) -> None:
    rows = []
    for scale, label in zip(SCALES, SCALE_LABELS):
        data = load_json(eval_dir / f"diagnosis_retrieval_{scale}.json")
        if data and data.get("n_queries", 0) > 0:
            for key, value in data.items():
                if key.startswith("hit_rate@"):
                    rows.append({"scale": label, "k": key, "hit_rate": value})
    if not rows:
        print("[create_figures_v031] skip fig_diagnosis_retrieval: no data found "
              "(or no abnormal-like queries in the test split)")
        return
    df = pd.DataFrame(rows)

    plt.figure(figsize=(6, 4))
    ax = sns.barplot(data=df, x="scale", y="hit_rate", hue="k")
    ax.set_ylabel("Hit rate")
    ax.set_xlabel("Training scale")
    ax.set_title("v0.3.1: Diagnosis-reason retrieval hit-rate@k")
    plt.tight_layout()
    out_path = out_dir / "05_diagnosis_retrieval_hit_rate.png"
    plt.savefig(out_path, dpi=cfg["v031_paper"]["figures"]["dpi"])
    plt.close()
    print(f"[create_figures_v031] wrote {out_path}")


def fig_zero_shot_v03_vs_v031(eval_dir_v03: Path, eval_dir_v031: Path, out_dir: Path, cfg: dict) -> None:
    """New in v0.3.1: direct before/after comparison of v0.3.0 (4 structures,
    3,000 cells) vs. v0.3.1 (8 structures, 8,000 cells) best-scale zero-shot
    accuracy, to visualize the diversity/scale effect. v0.3.1's best
    trained scale is 4.8k (6.4k was intentionally skipped per compute/time
    constraints)."""
    v03_best = load_json(eval_dir_v03 / "zero_shot_voxel_swin_claip_2.5k.json")
    v031_best = load_json(eval_dir_v031 / "zero_shot_voxel_swin_claip_4.8k.json")
    if not v03_best or not v031_best:
        print("[create_figures_v031] skip fig_zero_shot_v03_vs_v031: missing v0.3.0 or v0.3.1 results")
        return
    df = pd.DataFrame([
        {"version": "v0.3.0\n(4 structures, n=3,000)", "accuracy": v03_best["accuracy"],
         "chance": 0.25},
        {"version": "v0.3.1\n(8 structures, n=8,000)", "accuracy": v031_best["accuracy"],
         "chance": 0.125},
    ])
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(data=df, x="version", y="accuracy", color="#55A868")
    for i, row in df.iterrows():
        ax.axhline(row["chance"], color="gray", linestyle=":", xmin=i / len(df), xmax=(i + 1) / len(df))
    ax.set_ylabel("Zero-shot top-1 accuracy (best scale)")
    ax.set_title("Diversity/scale effect: v0.3.0 vs. v0.3.1 best-scale accuracy")
    plt.tight_layout()
    out_path = out_dir / "06_v03_vs_v031_zero_shot.png"
    plt.savefig(out_path, dpi=cfg["v031_paper"]["figures"]["dpi"])
    plt.close()
    print(f"[create_figures_v031] wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    eval_dir = Path(cfg["v031_evaluation"]["output_dir"])
    models_dir = Path(cfg["v031_contrastive_training"]["output"]["base_dir"])
    out_dir = Path(cfg["v031_paper"]["figures_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    sns.set_style("whitegrid")
    fig_zero_shot_accuracy(eval_dir, out_dir, cfg)
    fig_retrieval_recall(eval_dir, out_dir, cfg)
    for model_key in ["base", "voxel_swin_claip_4.8k"]:
        fig_anomaly_violin(eval_dir, out_dir, cfg, model_key)
    fig_training_loss(models_dir, out_dir, cfg)
    fig_diagnosis_retrieval(eval_dir, out_dir, cfg)
    fig_zero_shot_v03_vs_v031(Path(cfg["v03_evaluation"]["output_dir"]), eval_dir, out_dir, cfg)


if __name__ == "__main__":
    main()
