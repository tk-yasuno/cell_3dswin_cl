"""Phase 5, Step 14: "異常診断理由検索" (abnormality diagnosis-reason retrieval)
evaluation -- new in v0.3.0.

For each abnormal-like (mitotic, cell_stage != 'M0') voxel in the test split,
retrieves the nearest-neighbor finding_text from the training corpus
(train+val pool) by voxel-embedding-to-text-embedding cosine similarity, and
reports hit-rate@k: whether the top-k retrieved findings' (structure_name,
mitotic-vs-interphase) match the query's own ground truth. This operationalizes
the Concept Note's "異常診断理由検索" (search for abnormality diagnosis
reasons) output goal within the retrieval-only downstream-evaluation scope
decided for v0.3.0 (see docs/Plan_cell_3dswin_cl_v0-3-0.md, Decisions).

Usage:
    python retrieve_diagnosis_reason.py --config config.yaml --model base
    python retrieve_diagnosis_reason.py --config config.yaml --model voxel_swin_claip_2.5k
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from evaluate_zeroshot_voxel import load_model
from train_contrastive_voxel import load_config


@torch.no_grad()
def embed_voxels(model, df: pd.DataFrame, device: torch.device, batch_size: int = 16) -> torch.Tensor:
    embeds = []
    rows = df.to_dict("records")
    for start in tqdm(range(0, len(rows), batch_size), desc="embedding voxels"):
        batch_rows = rows[start:start + batch_size]
        voxels = torch.stack([torch.from_numpy(np.load(r["voxel_path"]).astype(np.float32))
                               for r in batch_rows], dim=0).to(device)
        embeds.append(model.encode_voxel(voxels).cpu())
    return torch.cat(embeds, dim=0)


@torch.no_grad()
def embed_texts(model, texts: list[str], device: torch.device, batch_size: int = 32) -> torch.Tensor:
    embeds = []
    for start in tqdm(range(0, len(texts), batch_size), desc="embedding corpus text"):
        batch_texts = texts[start:start + batch_size]
        tokenized = model.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        embeds.append(model.encode_text(tokenized["input_ids"], tokenized["attention_mask"]).cpu())
    return torch.cat(embeds, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--version", default="v03", choices=["v03", "v031"])
    parser.add_argument("--model", default="base")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds_cfg = cfg[f"{args.version}_dataset"]
    train_cfg = cfg[f"{args.version}_contrastive_training"]
    top_ks = cfg[f"{args.version}_evaluation"]["diagnosis_retrieval"]["top_ks"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, train_cfg, device)

    output_dir = Path(ds_cfg["output_dir"])
    train_df = pd.read_csv(output_dir / "train.csv")
    val_df = pd.read_csv(output_dir / "val.csv")
    corpus_df = pd.concat([train_df, val_df], ignore_index=True)
    test_df = pd.read_csv(output_dir / "test.csv")

    query_df = test_df[test_df["cell_stage"] != "M0"].reset_index(drop=True)
    if len(query_df) == 0:
        print("[retrieve_diagnosis_reason] WARNING: no abnormal-like (mitotic) cells in the "
              "test split; skipping (see Plan, Further Considerations: 'mitotic細胞の出現率の低さ')")
        result = {"model": args.model, "n_queries": 0, "status": "NO_ABNORMAL_QUERIES"}
    else:
        corpus_embeds = embed_texts(model, corpus_df["finding_text"].tolist(), device)
        query_embeds = embed_voxels(model, query_df, device)

        sim = query_embeds @ corpus_embeds.t()  # [n_query, n_corpus]
        max_k = max(top_ks)
        top_indices = sim.topk(k=min(max_k, sim.size(1)), dim=-1).indices  # [n_query, max_k]

        hits = {k: 0 for k in top_ks}
        for qi in range(len(query_df)):
            q_structure = query_df.iloc[qi]["structure_name"]
            q_mitotic = query_df.iloc[qi]["cell_stage"] != "M0"
            retrieved_idx = top_indices[qi].tolist()
            for k in top_ks:
                topk_idx = retrieved_idx[:k]
                match = any(
                    corpus_df.iloc[ci]["structure_name"] == q_structure
                    and (corpus_df.iloc[ci]["cell_stage"] != "M0") == q_mitotic
                    for ci in topk_idx
                )
                if match:
                    hits[k] += 1

        hit_rates = {f"hit_rate@{k}": hits[k] / len(query_df) for k in top_ks}
        result = {"model": args.model, "n_queries": len(query_df), "n_corpus": len(corpus_df), **hit_rates}
        print(f"[retrieve_diagnosis_reason] model={args.model} n_queries={len(query_df)} {hit_rates}")

    out_dir = Path(cfg[f"{args.version}_evaluation"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"diagnosis_retrieval_{args.model}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[retrieve_diagnosis_reason] wrote {out_path}")


if __name__ == "__main__":
    main()
