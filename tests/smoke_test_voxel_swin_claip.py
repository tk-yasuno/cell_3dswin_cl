"""Phase 0, Step 2 / Phase 3, Step 7: GPU feasibility smoke test for
VoxelSwinCLAIP.

Runs a single forward+backward step with dummy voxel tensors and dummy
finding-text strings through VoxelSwinCLAIP (3D Swin Transformer + LoRA +
Bio_ClinicalBERT + LoRA) and reports peak CUDA memory, so the training
hyperparameters in config.yaml (batch_size, feature_size, voxel_img_size) can
be confirmed/tuned BEFORE any real data download or training run -- see
docs/Plan_cell_3dswin_cl_v0-3-0.md, Decisions ("batch_size/feature_sizeの
確定値は実測を正とする").

If this OOMs, retry with a smaller feature_size (24 -> 12) or a smaller
voxel_img_size (64 -> 32), per the plan's documented fallback.

Usage:
    python tests/smoke_test_voxel_swin_claip.py
    python tests/smoke_test_voxel_swin_claip.py --batch-size 2 --feature-size 12
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.voxel_swin_claip import VoxelSwinCLAIP, VoxelSwinCLAIPConfig  # noqa: E402


def run_smoke_test(batch_size: int, img_size: int, feature_size: int, in_channels: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke_test] device={device}, batch_size={batch_size}, "
          f"img_size={img_size}^3, feature_size={feature_size}, in_channels={in_channels}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    cfg = VoxelSwinCLAIPConfig(
        voxel_img_size=(img_size, img_size, img_size),
        in_channels=in_channels,
        feature_size=feature_size,
    )

    t0 = time.time()
    model = VoxelSwinCLAIP(cfg).to(device)
    model.train()
    print(f"[smoke_test] model built + moved to device in {time.time() - t0:.1f}s")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[smoke_test] trainable params: {trainable:,} / total params: {total:,}")

    pixel_values = torch.randn(batch_size, in_channels, img_size, img_size, img_size,
                                device=device, dtype=torch.float32)
    dummy_texts = [
        "a 3D confocal fluorescence volume of an interphase human iPS cell showing an intact nuclear envelope",
        "a 3D microscopy volume showing mitochondria undergoing fragmentation during mitosis",
    ] * (batch_size // 2 + 1)
    dummy_texts = dummy_texts[:batch_size]
    tokenized = model.tokenizer(dummy_texts, return_tensors="pt", padding=True, truncation=True)
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    autocast_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    t0 = time.time()
    with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=device.type == "cuda"):
        out = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
        batch_targets = torch.arange(batch_size, device=device)
        loss = (torch.nn.functional.cross_entropy(out["logits_per_voxel"], batch_targets)
                + torch.nn.functional.cross_entropy(out["logits_per_text"], batch_targets)) / 2.0
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    torch.cuda.synchronize() if device.type == "cuda" else None
    print(f"[smoke_test] forward+backward+step completed in {time.time() - t0:.1f}s, loss={loss.item():.4f}")

    if device.type == "cuda":
        peak_bytes = torch.cuda.max_memory_allocated(device)
        peak_gb = peak_bytes / (1024 ** 3)
        print(f"[smoke_test] peak CUDA memory allocated: {peak_gb:.2f} GB")
        budget_gb = 16.0
        if peak_gb >= budget_gb:
            print(f"[smoke_test] WARNING: peak memory {peak_gb:.2f} GB >= {budget_gb:.0f} GB budget. "
                  f"Retry with a smaller --feature-size or --img-size.")
            sys.exit(1)
        else:
            print(f"[smoke_test] OK: peak memory {peak_gb:.2f} GB < {budget_gb:.0f} GB budget.")
    else:
        print("[smoke_test] CUDA not available; ran on CPU only (memory budget not verified).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--feature-size", type=int, default=24)
    parser.add_argument("--in-channels", type=int, default=2)
    args = parser.parse_args()
    run_smoke_test(args.batch_size, args.img_size, args.feature_size, args.in_channels)


if __name__ == "__main__":
    main()
