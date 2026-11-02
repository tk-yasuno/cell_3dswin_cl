"""Phase 5, Step 10: Zero-shot structure classification evaluation for
VoxelSwinCLAIP.

Evaluates 4-way zero-shot classification accuracy (LMNB1/TOM20/TUBA1B/CAAX)
on the test split, for either:
  - a freshly initialized VoxelSwinCLAIP (--model base, random 3D Swin +
    pretrained-but-not-contrastively-trained Bio_ClinicalBERT), used as the
    self-baseline, or
  - a domain-adapted VoxelSwinCLAIP LoRA checkpoint (--model
    voxel_swin_claip_{scale}).

No external literature baseline exists for this architecture/domain (a
random-initialized 3D Swin voxel encoder has no CLIP-style pretrained
zero-shot prior), so only self-baseline comparison is reported -- same
caveat as v0.2.0 (see docs/Plan_cell_3dswin_cl_v0-3-0.md).

Usage:
    python evaluate_zeroshot_voxel.py --config config.yaml --model base
    python evaluate_zeroshot_voxel.py --config config.yaml --model voxel_swin_claip_2.5k
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.models.voxel_swin_claip import VoxelSwinCLAIP, VoxelSwinCLAIPConfig

from train_contrastive_voxel import load_config  # reuse the shared YAML loader


def build_model_config(train_cfg: dict) -> VoxelSwinCLAIPConfig:
    return VoxelSwinCLAIPConfig(
        voxel_img_size=tuple(train_cfg["model"]["voxel_img_size"]),
        in_channels=train_cfg["model"]["in_channels"],
        feature_size=train_cfg["model"]["feature_size"],
        use_checkpoint=train_cfg["model"]["use_checkpoint"],
        lora_r_voxel=train_cfg["lora"]["voxel"]["r"],
        lora_alpha_voxel=train_cfg["lora"]["voxel"]["alpha"],
        lora_dropout_voxel=train_cfg["lora"]["voxel"]["dropout"],
        voxel_target_regex=train_cfg["lora"]["voxel"]["target_regex"],
        bert_model_id=train_cfg["model"]["bert_model_id"],
        lora_r_text=train_cfg["lora"]["text"]["r"],
        lora_alpha_text=train_cfg["lora"]["text"]["alpha"],
        lora_dropout_text=train_cfg["lora"]["text"]["dropout"],
        text_target_modules=tuple(train_cfg["lora"]["text"]["target_modules"]),
        shared_dim=train_cfg["model"]["shared_dim"],
    )


def load_model(model_key: str, train_cfg: dict, device: torch.device) -> VoxelSwinCLAIP:
    model_cfg = build_model_config(train_cfg)
    if model_key == "base":
        model = VoxelSwinCLAIP(model_cfg)
    else:
        adapter_dir = Path(train_cfg["output"]["base_dir"]) / model_key
        model = VoxelSwinCLAIP.from_pretrained_adapter(str(adapter_dir), config=model_cfg)
    return model.to(device).eval()


@torch.no_grad()
def zero_shot_eval(model: VoxelSwinCLAIP, test_csv: str, prompt_template: str,
                    batch_size: int, device: torch.device,
                    structure_display_names: dict) -> dict:
    df = pd.read_csv(test_csv)
    classes = sorted(df["structure_name"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    text_prompts = [prompt_template.format(c=structure_display_names.get(c, c)) for c in classes]
    tokenized = model.tokenizer(text_prompts, return_tensors="pt", padding=True, truncation=True)
    tokenized = {k: v.to(device) for k, v in tokenized.items()}
    text_embeds = model.encode_text(tokenized["input_ids"], tokenized["attention_mask"])

    correct = 0
    total = 0
    rows = df.to_dict("records")
    for start in tqdm(range(0, len(rows), batch_size), desc="zero-shot"):
        batch_rows = rows[start:start + batch_size]
        voxels = torch.stack([torch.from_numpy(np.load(r["voxel_path"]).astype(np.float32))
                               for r in batch_rows], dim=0).to(device)
        labels = torch.tensor([class_to_idx[r["structure_name"]] for r in batch_rows], device=device)

        voxel_embeds = model.encode_voxel(voxels)
        logits = voxel_embeds @ text_embeds.t()
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += len(labels)

    accuracy = correct / total
    return {"accuracy": accuracy, "n": total, "n_classes": len(classes)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--version", default="v03", choices=["v03", "v031"])
    parser.add_argument("--model", default="base", help="'base' or a models/ dir name, e.g. voxel_swin_claip_1.6k")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds_cfg = cfg[f"{args.version}_dataset"]
    train_cfg = cfg[f"{args.version}_contrastive_training"]
    eval_cfg = cfg[f"{args.version}_evaluation"]["zero_shot"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, train_cfg, device)

    test_csv = Path(ds_cfg["output_dir"]) / "test.csv"
    result = zero_shot_eval(model, str(test_csv), eval_cfg["prompt_template"], eval_cfg["batch_size"],
                             device, ds_cfg["structure_display_names"])
    result["model"] = args.model
    print(f"[evaluate_zeroshot_voxel] model={args.model} accuracy={result['accuracy']:.4f} "
          f"(n={result['n']}, classes={result['n_classes']}, chance={eval_cfg['chance_accuracy']:.2f})")

    if args.model == "base":
        print("[evaluate_zeroshot_voxel] no literature sanity range configured for this "
              "architecture/domain (base_accuracy_sanity_range=null); recording self-baseline only")

    out_dir = Path(cfg[f"{args.version}_evaluation"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"zero_shot_{args.model}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[evaluate_zeroshot_voxel] wrote {out_path}")


if __name__ == "__main__":
    main()
