"""Phase 4, Step 8: Contrastive LoRA domain adaptation training loop for
VoxelSwinCLAIP (3D Swin Transformer + Bio_ClinicalBERT).

Trains VoxelSwinCLAIP (src/models/voxel_swin_claip.py) with the symmetric
InfoNCE loss from the original CLIP paper, independently at each progressive
data scale (0.8k/1.6k/2.5k), mirroring v0.1.0/v0.2.0's
train_contrastive_lora.py methodology (0_LogBAK/pre-code_cell_vlm_cl/
train_contrastive_lora.py) but for 3D voxel volumes + ClinicalBERT text
instead of 2D images + CLIP text.

See docs/Plan_cell_3dswin_cl_v0-3-0.md, Phase 4.

Usage:
    python train_contrastive_voxel.py --config config.yaml --scale 0.8k
    python train_contrastive_voxel.py --config config.yaml --scale all
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.models.voxel_swin_claip import VoxelSwinCLAIP, VoxelSwinCLAIPConfig


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class VoxelCaptionDataset(Dataset):
    """Voxel-caption pairs for one progressive-scale CSV (voxel_path, finding_text)."""

    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        voxel = np.load(row["voxel_path"]).astype(np.float32)  # [C, D, H, W], cached float16->float32
        return {"voxel": torch.from_numpy(voxel), "finding_text": row["finding_text"]}


def make_collate_fn(tokenizer):
    def collate(batch):
        voxels = torch.stack([b["voxel"] for b in batch], dim=0)
        texts = [b["finding_text"] for b in batch]
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        return {"pixel_values": voxels, "input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
    return collate


def clip_infonce_loss(logits_per_voxel: torch.Tensor, logits_per_text: torch.Tensor) -> torch.Tensor:
    """Symmetric InfoNCE: average of voxel->text and text->voxel cross-entropy,
    with the diagonal as the positive pair (same formulation as
    0_LogBAK/pre-code_cell_vlm_cl/train_contrastive_lora.py's clip_infonce_loss).
    """
    batch_size = logits_per_voxel.size(0)
    targets = torch.arange(batch_size, device=logits_per_voxel.device)
    loss_v2t = F.cross_entropy(logits_per_voxel, targets)
    loss_t2v = F.cross_entropy(logits_per_text, targets)
    return (loss_v2t + loss_t2v) / 2.0


def train_one_scale(cfg: dict, scale: str, version: str = "v03") -> dict:
    ds_cfg = cfg[f"{version}_dataset"]
    train_cfg = cfg[f"{version}_contrastive_training"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = VoxelSwinCLAIPConfig(
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
    model = VoxelSwinCLAIP(model_cfg).to(device)
    model.train()

    csv_path = Path(ds_cfg["output_dir"]) / f"train_{scale}.csv"
    full_df = pd.read_csv(csv_path)
    split_ratio = train_cfg["data"]["train_val_split"]
    n_train = int(len(full_df) * split_ratio)
    train_df = full_df.iloc[:n_train].reset_index(drop=True)
    val_df = full_df.iloc[n_train:].reset_index(drop=True)

    tmp_dir = Path(ds_cfg["output_dir"]) / "_tmp_splits"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    train_csv = tmp_dir / f"{scale}_train.csv"
    val_csv = tmp_dir / f"{scale}_val.csv"
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    train_ds = VoxelCaptionDataset(str(train_csv))
    val_ds = VoxelCaptionDataset(str(val_csv))
    collate_fn = make_collate_fn(model.tokenizer)

    batch_size = train_cfg["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=train_cfg["training"]["learning_rate"],
        weight_decay=train_cfg["training"]["weight_decay"],
    )
    num_epochs = train_cfg["training"]["num_epochs"]
    total_steps = max(1, len(train_loader) * num_epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    use_bf16 = train_cfg["training"].get("bf16", False) and device.type == "cuda"
    autocast_ctx = (
        (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16))
        if use_bf16 else (lambda: torch.autocast(device_type="cpu", enabled=False))
    )

    print(f"[train_contrastive_voxel][{scale}] n_train={len(train_df)} n_val={len(val_df)} "
          f"batch_size={batch_size} bf16={use_bf16}", flush=True)

    history = {"scale": scale, "n_train": len(train_df), "n_val": len(val_df), "epochs": []}
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(tqdm(train_loader, desc=f"[{scale}] epoch {epoch+1}/{num_epochs} train")):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            with autocast_ctx():
                out = model(pixel_values=batch["pixel_values"], input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"])
                loss = clip_infonce_loss(out["logits_per_voxel"], out["logits_per_text"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                train_cfg["training"]["max_grad_norm"],
            )
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
        train_loss = running_loss / max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"[{scale}] epoch {epoch+1}/{num_epochs} val"):
                batch = {k: v.to(device) for k, v in batch.items()}
                with autocast_ctx():
                    out = model(pixel_values=batch["pixel_values"], input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"])
                    loss = clip_infonce_loss(out["logits_per_voxel"], out["logits_per_text"])
                val_loss += loss.item()
        val_loss /= max(1, len(val_loader))

        peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
        print(f"[train_contrastive_voxel][{scale}] epoch {epoch+1}/{num_epochs} "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} peak_mem_gb={peak_mem_gb:.2f}",
              flush=True)
        history["epochs"].append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})

    history["duration_sec"] = time.time() - start_time

    out_dir = Path(train_cfg["output"]["base_dir"]) / f"{train_cfg['output']['run_prefix']}_{scale}"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_adapter(str(out_dir))
    with open(out_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"[train_contrastive_voxel][{scale}] saved adapter + history to {out_dir}")
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--version", default="v03", choices=["v03", "v031"],
                         help="which config section prefix to use, e.g. 'v03_dataset'")
    parser.add_argument("--scale", default="all",
                         help="a scale label from {version}_dataset.scale_labels, or 'all'")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds_cfg = cfg[f"{args.version}_dataset"]
    all_scales = list(ds_cfg["scale_labels"].values())
    scales = all_scales if args.scale == "all" else [args.scale]
    for scale in scales:
        train_one_scale(cfg, scale, version=args.version)


if __name__ == "__main__":
    main()
