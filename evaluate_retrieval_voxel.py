"""Phase 5, Step 11: voxel<->text retrieval evaluation (Recall@1,5,10) for
VoxelSwinCLAIP.

Computes bidirectional retrieval recall on the test split, for a freshly
initialized VoxelSwinCLAIP (--model base) vs. a domain-adapted VoxelSwinCLAIP
LoRA checkpoint (--model voxel_swin_claip_{scale}), following the same
retrieval protocol as v0.1.0/v0.2.0's evaluate_retrieval.py.

Usage:
    python evaluate_retrieval_voxel.py --config config.yaml --model base
    python evaluate_retrieval_voxel.py --config config.yaml --model voxel_swin_claip_2.5k
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
def embed_all(model, df: pd.DataFrame, device: torch.device, batch_size: int = 16):
    voxel_embeds, text_embeds = [], []
    rows = df.to_dict("records")
    for start in tqdm(range(0, len(rows), batch_size), desc="embedding"):
        batch_rows = rows[start:start + batch_size]
        voxels = torch.stack([torch.from_numpy(np.load(r["voxel_path"]).astype(np.float32))
                               for r in batch_rows], dim=0).to(device)
        texts = [r["finding_text"] for r in batch_rows]

        tokenized = model.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        tokenized = {k: v.to(device) for k, v in tokenized.items()}

        v_e = model.encode_voxel(voxels)
        t_e = model.encode_text(tokenized["input_ids"], tokenized["attention_mask"])

        voxel_embeds.append(v_e.cpu())
        text_embeds.append(t_e.cpu())

    return torch.cat(voxel_embeds, dim=0), torch.cat(text_embeds, dim=0)


def recall_at_k(sim: torch.Tensor, ks: list[int]) -> dict:
    """sim[i, j] = similarity of query i to candidate j; ground truth is the
    diagonal (each voxel's own caption / vice versa)."""
    n = sim.size(0)
    ranks = sim.argsort(dim=-1, descending=True)
    targets = torch.arange(n).unsqueeze(1)
    hit_rank = (ranks == targets).float().argmax(dim=-1)
    return {f"R@{k}": (hit_rank < k).float().mean().item() for k in ks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--version", default="v03", choices=["v03", "v031"])
    parser.add_argument("--model", default="base")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds_cfg = cfg[f"{args.version}_dataset"]
    train_cfg = cfg[f"{args.version}_contrastive_training"]
    retrieval_cfg = cfg[f"{args.version}_evaluation"]["retrieval"]
    ks = retrieval_cfg["recall_at_k"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, train_cfg, device)

    test_df = pd.read_csv(Path(ds_cfg["output_dir"]) / "test.csv")
    voxel_embeds, text_embeds = embed_all(model, test_df, device, retrieval_cfg.get("batch_size", 16))

    sim = voxel_embeds @ text_embeds.t()
    v2t = recall_at_k(sim, ks)
    t2v = recall_at_k(sim.t(), ks)

    result = {"model": args.model, "n": len(test_df),
              "voxel_to_text": v2t, "text_to_voxel": t2v}
    print(f"[evaluate_retrieval_voxel] model={args.model} voxel->text={v2t} text->voxel={t2v}")

    out_dir = Path(cfg[f"{args.version}_evaluation"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"retrieval_{args.model}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[evaluate_retrieval_voxel] wrote {out_path}")


if __name__ == "__main__":
    main()
