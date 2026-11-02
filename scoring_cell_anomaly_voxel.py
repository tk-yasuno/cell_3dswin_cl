"""Phase 5, Step 13: 3D voxel cell anomaly scoring prototype.

Computes a WinCLIP-style compositional-prompt anomaly score for each test
voxel (cosine similarity to "normal/interphase" prompts minus similarity to
"abnormal/mitotic" prompts), maps it to a 1-5 level via scoring_rules_voxel.yaml,
and validates the scoring prototype against cell_anomaly_proxy_labels_voxel.yaml
using ROC-AUC (NOT a production acceptance criterion -- see
docs/Plan_cell_3dswin_cl_v0-3-0.md, Decisions).

Usage:
    python scoring_cell_anomaly_voxel.py --config config.yaml --model base
    python scoring_cell_anomaly_voxel.py --config config.yaml --model voxel_swin_claip_2.5k
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from evaluate_zeroshot_voxel import load_model
from train_contrastive_voxel import load_config


def bin_score(score: float, bins: list[dict]) -> tuple[int, str]:
    for b in bins:
        if score <= b["max_score"]:
            return b["level"], b["label"]
    return bins[-1]["level"], bins[-1]["label"]


@torch.no_grad()
def text_prototype(model, prompts: list[str], subject: str, device: torch.device) -> torch.Tensor:
    rendered = [p.format(subject=subject) for p in prompts]
    tokenized = model.tokenizer(rendered, return_tensors="pt", padding=True, truncation=True)
    tokenized = {k: v.to(device) for k, v in tokenized.items()}
    embeds = model.encode_text(tokenized["input_ids"], tokenized["attention_mask"])
    return embeds.mean(dim=0, keepdim=True)  # prototype = mean of ensemble prompts


@torch.no_grad()
def voxel_embed(model, voxel_path: str, device: torch.device) -> torch.Tensor:
    voxel = torch.from_numpy(np.load(voxel_path).astype(np.float32)).unsqueeze(0).to(device)
    return model.encode_voxel(voxel)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--version", default="v03", choices=["v03", "v031"])
    parser.add_argument("--model", default="base")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds_cfg = cfg[f"{args.version}_dataset"]
    train_cfg = cfg[f"{args.version}_contrastive_training"]
    scoring_section = cfg[f"{args.version}_evaluation"]["scoring"]
    output_prefix = scoring_section.get("output_prefix", "cell_anomaly_voxel")
    proxy_cfg = scoring_section["proxy_validation"]

    structure_display_names = ds_cfg["structure_display_names"]

    with open(scoring_section["rules_file"], "r", encoding="utf-8") as f:
        scoring_cfg = yaml.safe_load(f)
    with open(ds_cfg["proxy_labels_file"], "r", encoding="utf-8") as f:
        proxy_labels = yaml.safe_load(f)
    positive_key = proxy_cfg.get("positive_label_key", "abnormal_like_stages")
    positive_set = set(proxy_labels[positive_key])
    positive_col = proxy_cfg.get("positive_column_name", "proxy_is_abnormal_like")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, train_cfg, device)

    test_df = pd.read_csv(Path(ds_cfg["output_dir"]) / "test.csv")

    score_field = f"{output_prefix}_score"
    level_field = f"{output_prefix}_level"
    label_field = f"{output_prefix}_label"

    # One shared normal/abnormal prototype per structure ({subject}-conditioned prompts).
    proto_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    results = []
    for row in tqdm(test_df.to_dict("records"), desc=f"scoring [{args.model}]"):
        structure_name = row["structure_name"]
        subject = structure_display_names.get(structure_name, structure_name)
        if structure_name not in proto_cache:
            normal_proto = text_prototype(model, scoring_cfg["state_prompts"]["normal"], subject, device)
            abnormal_proto = text_prototype(model, scoring_cfg["state_prompts"]["abnormal"], subject, device)
            proto_cache[structure_name] = (normal_proto, abnormal_proto)
        normal_proto, abnormal_proto = proto_cache[structure_name]

        v_emb = voxel_embed(model, row["voxel_path"], device)
        sim_normal = (v_emb @ normal_proto.t()).item()
        sim_abnormal = (v_emb @ abnormal_proto.t()).item()
        score = sim_normal - sim_abnormal
        level, label = bin_score(score, scoring_cfg["score_bins"])

        results.append({
            "voxel_path": row["voxel_path"],
            "structure_name": structure_name,
            "cell_stage": row["cell_stage"],
            score_field: score,
            level_field: level,
            label_field: label,
            positive_col: row["cell_stage"] in positive_set,
        })

    results_df = pd.DataFrame(results)
    out_dir = Path(cfg[f"{args.version}_evaluation"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / f"{output_prefix}_scores_{args.model}.csv"
    results_df.to_csv(results_path, index=False)

    # Proxy validation: higher score should correlate with positive_col ==
    # False (i.e. normal/interphase-like). AUC computed with the
    # positive_col bucket as the positive class on the *negated* score
    # (lower score -> more abnormal/mitotic-like).
    y_true = results_df[positive_col].astype(int).values
    n_positive = int(y_true.sum())
    y_score = -results_df[score_field].values

    summary = {"model": args.model, "n": len(results_df), "n_abnormal_like": n_positive}
    if 0 < n_positive < len(results_df):
        auc = roc_auc_score(y_true, y_score)
        threshold = proxy_cfg["prototype_threshold"]
        status = "OK" if auc >= threshold else "BELOW_THRESHOLD"
        summary.update({"proxy_roc_auc": auc, "prototype_threshold": threshold, "status": status})
        print(f"[scoring_cell_anomaly_voxel] model={args.model} proxy_roc_auc={auc:.4f} "
              f"(threshold={threshold}, n_abnormal_like={n_positive}/{len(results_df)}) -> {status}")
    else:
        summary.update({"proxy_roc_auc": None, "status": "INSUFFICIENT_POSITIVE_SAMPLES"})
        print(f"[scoring_cell_anomaly_voxel] model={args.model} WARNING: only {n_positive} "
              f"abnormal-like (mitotic) samples in test set; ROC-AUC not computed "
              f"(see docs/Plan_cell_3dswin_cl_v0-3-0.md Further Considerations, 'mitotic細胞の出現率の低さ')")

    summary_path = out_dir / f"{output_prefix}_proxy_validation_{args.model}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[scoring_cell_anomaly_voxel] wrote {results_path} and {summary_path}")


if __name__ == "__main__":
    main()
