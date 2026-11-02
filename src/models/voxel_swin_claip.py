"""Phase 3, Step 6: VoxelSwinCLAIP -- 3D Swin Transformer + ClinicalBERT dual encoder.

Replaces v0.2.0's 2D MaterialCLIP (CLIP ViT-L/14-336 image+text dual encoder,
see 0_LogBAK/pre-code_cell_vlm_cl/src/models/clip_lora.py) with a
voxel/text dual encoder for 3D fluorescence microscopy volumes:

  - Voxel tower: monai.networks.nets.SwinUNETR's encoder-only hierarchical 3D
    Swin Transformer (`swinViT`). The UNETR decoder/segmentation head is
    constructed transiently (monai does not expose a decoder-free
    constructor) and discarded immediately after init so it never reaches
    the GPU -- see _build_voxel_backbone().
  - Text tower: transformers.AutoModel Bio_ClinicalBERT
    (emilyalsentzer/Bio_ClinicalBERT), a BERT-base model pretrained on real
    MIMIC-III clinical notes.

Both towers get LoRA adapters (peft) on their attention projections only,
mirroring MaterialCLIP's design, and feed into independent linear projection
heads to a shared embedding space, L2-normalized, with a learnable
logit_scale (CLIP's temperature) for the symmetric InfoNCE loss computed in
train_contrastive_voxel.py.

See docs/Plan_cell_3dswin_cl_v0-3-0.md, Phase 3 / Decisions.
"""
from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModel, AutoTokenizer


@dataclass
class VoxelSwinCLAIPConfig:
    # Voxel (3D Swin) tower
    voxel_img_size: tuple = (64, 64, 64)
    in_channels: int = 2                    # [dna, structure] fluorescence channels
    feature_size: int = 24                  # must be divisible by 12 (monai SwinUNETR constraint)
    use_checkpoint: bool = True             # gradient checkpointing, required to fit 16GB VRAM
    lora_r_voxel: int = 8
    lora_alpha_voxel: int = 16
    lora_dropout_voxel: float = 0.05
    # only the window-attention qkv/proj projections inside each Swin stage
    # are adapted (NOT patch_embed.proj, which is a Conv3d, not a Linear)
    voxel_target_regex: str = r".*\.attn\.(qkv|proj)$"

    # Text (ClinicalBERT) tower
    bert_model_id: str = "emilyalsentzer/Bio_ClinicalBERT"
    lora_r_text: int = 16
    lora_alpha_text: int = 32
    lora_dropout_text: float = 0.05
    text_target_modules: tuple = ("query", "key", "value")

    # Shared embedding space
    shared_dim: int = 512


def _build_voxel_backbone(cfg: VoxelSwinCLAIPConfig):
    """Construct monai's SwinUNETR, keep only the encoder (swinViT), and drop
    the UNETR decoder/segmentation head before it is ever moved to the GPU.

    monai's SwinUNETR does not expose a decoder-free constructor, so the
    decoder blocks are unavoidably allocated on CPU for a moment; they are
    deleted here (and never have parameters placed on GPU), so this has no
    effect on the 16GB VRAM budget -- only Phase 0's smoke test measures the
    real (post-trim) GPU footprint.
    """
    from monai.networks.nets import SwinUNETR

    kwargs = dict(
        in_channels=cfg.in_channels,
        out_channels=cfg.in_channels,  # unused (decoder is discarded); must be a valid int
        feature_size=cfg.feature_size,
        use_checkpoint=cfg.use_checkpoint,
        spatial_dims=3,
    )
    # Older monai (<~1.4) requires `img_size` as a constructor arg; newer
    # releases made SwinUNETR input-size-dynamic and dropped the parameter.
    # Inspect the installed signature so this works across monai versions.
    sig = inspect.signature(SwinUNETR.__init__)
    if "img_size" in sig.parameters:
        kwargs["img_size"] = tuple(cfg.voxel_img_size)

    full_model = SwinUNETR(**kwargs)
    backbone = full_model.swinViT
    del full_model

    # SwinTransformer.forward() returns 5 hierarchical feature maps
    # [x0..x4] with channels [C, 2C, 4C, 8C, 16C] (C=feature_size); the
    # `backbone.num_features` attribute does NOT match this (it is computed
    # as C * 2**(num_layers-1) = 8C, one stage short -- verified empirically
    # against the actual last-stage tensor). Determine the true channel
    # count of the last returned stage with a tiny CPU dummy forward instead
    # of trusting that attribute.
    with torch.no_grad():
        dummy = torch.zeros(1, cfg.in_channels, 32, 32, 32)
        dummy_out = backbone(dummy, normalize=False)
        num_features = dummy_out[-1].shape[1]

    return backbone, num_features


class VoxelSwinCLAIP(torch.nn.Module):
    """3D Swin Transformer (voxel) + ClinicalBERT (text) dual encoder with
    LoRA adapters on both towers.

    Mirrors MaterialCLIP's API (encode_image -> encode_voxel, encode_text,
    forward, save_adapter, from_pretrained_adapter) so the surrounding
    training/evaluation scripts follow the same pattern as v0.1.0/v0.2.0.
    """

    def __init__(self, config: VoxelSwinCLAIPConfig | None = None):
        super().__init__()
        self.config = config or VoxelSwinCLAIPConfig()
        cfg = self.config

        voxel_backbone, voxel_feat_dim = _build_voxel_backbone(cfg)
        voxel_lora_config = LoraConfig(
            r=cfg.lora_r_voxel,
            lora_alpha=cfg.lora_alpha_voxel,
            lora_dropout=cfg.lora_dropout_voxel,
            target_modules=cfg.voxel_target_regex,
            bias="none",
        )
        self.voxel_model = get_peft_model(voxel_backbone, voxel_lora_config)
        self.voxel_projection = torch.nn.Linear(voxel_feat_dim, cfg.shared_dim)

        text_backbone = AutoModel.from_pretrained(cfg.bert_model_id)
        text_lora_config = LoraConfig(
            r=cfg.lora_r_text,
            lora_alpha=cfg.lora_alpha_text,
            lora_dropout=cfg.lora_dropout_text,
            target_modules=list(cfg.text_target_modules),
            bias="none",
        )
        self.text_model = get_peft_model(text_backbone, text_lora_config)
        self.text_projection = torch.nn.Linear(text_backbone.config.hidden_size, cfg.shared_dim)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.bert_model_id)

        # Freeze policy differs by tower, by design: the Finding Encoder
        # (ClinicalBERT) is genuinely pretrained on real clinical text, so
        # LoRA-only adaptation (preserving its pretrained prior) is
        # appropriate, mirroring MaterialCLIP's scheme. The Voxel Encoder
        # has NO pretrained prior (3D Swin is randomly initialized here --
        # see Decisions, "3D Swinの初期化"): empirically, LoRA-only
        # adaptation (rank=8 on just the window-attention qkv/proj, with
        # patch_embed/PatchMerging/norm layers frozen at their random init)
        # left voxel embeddings almost input-invariant (pairwise cosine
        # similarity 0.93-0.99 across distinct structures after 15 epochs),
        # because a small low-rank delta cannot carve a useful
        # representation out of an otherwise-frozen random backbone. The
        # voxel tower is therefore fully fine-tuned (all parameters
        # trainable); the LoRA adapter remains attached (harmless, adds a
        # redundant parallel low-rank path) only so save_adapter/
        # from_pretrained_adapter's I/O format stays uniform across towers.
        for name, param in self.voxel_model.named_parameters():
            param.requires_grad = True
        for name, param in self.text_model.named_parameters():
            param.requires_grad = "lora_" in name

        self.logit_scale = torch.nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

    def encode_voxel(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """pixel_values: [B, in_channels, D, H, W] float tensor."""
        hidden_states = self.voxel_model(pixel_values, normalize=True)
        last_stage = hidden_states[-1]  # [B, C, d, h, w]
        pooled = F.adaptive_avg_pool3d(last_stage, (1, 1, 1)).flatten(1)  # [B, C]
        return F.normalize(self.voxel_projection(pooled), dim=-1)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = out.last_hidden_state[:, 0, :]  # [CLS] pooled representation
        return F.normalize(self.text_projection(cls_token), dim=-1)

    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> dict:
        voxel_embeds = self.encode_voxel(pixel_values)
        text_embeds = self.encode_text(input_ids, attention_mask)
        logit_scale = self.logit_scale.exp()
        logits_per_voxel = logit_scale * voxel_embeds @ text_embeds.t()
        logits_per_text = logits_per_voxel.t()
        return {
            "voxel_embeds": voxel_embeds,
            "text_embeds": text_embeds,
            "logits_per_voxel": logits_per_voxel,
            "logits_per_text": logits_per_text,
        }

    def save_adapter(self, save_dir: str) -> None:
        import json
        import os

        os.makedirs(save_dir, exist_ok=True)
        self.voxel_model.save_pretrained(os.path.join(save_dir, "voxel_lora"))
        self.text_model.save_pretrained(os.path.join(save_dir, "text_lora"))
        self.tokenizer.save_pretrained(os.path.join(save_dir, "tokenizer"))
        torch.save(
            {
                "voxel_projection": self.voxel_projection.state_dict(),
                "text_projection": self.text_projection.state_dict(),
                "logit_scale": self.logit_scale,
            },
            os.path.join(save_dir, "heads.pt"),
        )
        with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(self.config.__dict__, f, indent=2)

    @classmethod
    def from_pretrained_adapter(cls, adapter_dir: str, config: VoxelSwinCLAIPConfig | None = None):
        """Load a previously trained LoRA adapter pair + projection heads."""
        import json
        import os

        if config is None:
            config_path = os.path.join(adapter_dir, "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                config = VoxelSwinCLAIPConfig(**saved)
            else:
                config = VoxelSwinCLAIPConfig()

        instance = cls.__new__(cls)
        torch.nn.Module.__init__(instance)
        instance.config = config

        voxel_backbone, voxel_feat_dim = _build_voxel_backbone(config)
        instance.voxel_model = PeftModel.from_pretrained(voxel_backbone, os.path.join(adapter_dir, "voxel_lora"))
        instance.voxel_projection = torch.nn.Linear(voxel_feat_dim, config.shared_dim)

        text_backbone = AutoModel.from_pretrained(config.bert_model_id)
        instance.text_model = PeftModel.from_pretrained(text_backbone, os.path.join(adapter_dir, "text_lora"))
        instance.text_projection = torch.nn.Linear(text_backbone.config.hidden_size, config.shared_dim)
        instance.tokenizer = AutoTokenizer.from_pretrained(os.path.join(adapter_dir, "tokenizer"))

        heads = torch.load(os.path.join(adapter_dir, "heads.pt"), map_location="cpu", weights_only=False)
        instance.voxel_projection.load_state_dict(heads["voxel_projection"])
        instance.text_projection.load_state_dict(heads["text_projection"])
        instance.logit_scale = torch.nn.Parameter(heads["logit_scale"])

        return instance
