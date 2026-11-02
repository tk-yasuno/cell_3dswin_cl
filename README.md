# Cell 3D Voxel × Text Contrastive Learning (cell_3dswin_cl)

MVP v0.3.0 / v0.3.1: a 3D Swin Transformer voxel encoder + ClinicalBERT finding-text
encoder (`VoxelSwinCLAIP`), contrastively trained (symmetric InfoNCE) on single-cell
3D confocal fluorescence volumes from the Allen Institute's public hiPSC single-cell
image dataset, paired with template-generated structural finding text. This is an
experimental methodology prototype toward future 3D pavement-material microscopy
anomaly diagnosis, extending the 2D CLIP-based `cell_vlm_cl` (v0.1.0/v0.2.0)
contrastive-learning MVPs to 3D voxels. v0.3.1 extends v0.3.0 from 4 to 8 structures
and 3,000 to 8,000 cells to probe diversity/scale generalization.

Full design rationale: [docs/Plan_cell_3dswin_cl_v0-3-0.md](docs/Plan_cell_3dswin_cl_v0-3-0.md).
Full write-up (architecture, dataset, results, lessons learned): [paper_cell_3dswin_cl/0_cell_3dswin_cl_v0-3-0/methodology.tex](paper_cell_3dswin_cl/0_cell_3dswin_cl_v0-3-0/methodology.tex).
Japanese operational notes: [README_of_cell_3dswin_cl.md](README_of_cell_3dswin_cl.md).

## Setup

```powershell
# Python 3.12, CUDA 12.4 (RTX 4060 Ti 16GB or similar)
python -m venv .venv-cell3dswin
.\.venv-cell3dswin\Scripts\pip.exe install -r requirements.txt
# requirements.txt alone installs CPU-only torch; overwrite with the CUDA build last
.\.venv-cell3dswin\Scripts\pip.exe install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

`monai`/`aicsimageio`/`quilt3` can trigger slow pip dependency resolution; installing
them individually (e.g. `pip install monai==1.5.2 --no-deps`) is recommended.

## Pipeline

All scripts accept `--version {v03,v031}` to select the config section (`config.yaml`)
for the 4-structure/3,000-cell (v0.3.0) or 8-structure/8,000-cell (v0.3.1) experiment.

| Step | Command | Purpose |
| --- | --- | --- |
| 0 | `python tests/smoke_test_voxel_swin_claip.py` | Confirm GPU VRAM budget before any download/training |
| 1 | `python prepare_voxel_cell_dataset.py --config config.yaml --version {v03,v031}` | Download/preprocess single-cell 3D volumes (quilt3, Allen Institute) |
| 2 | `python prepare_progressive_splits.py --config config.yaml --version {v03,v031}` | Build stratified progressive training subsets |
| 3 | `python train_contrastive_voxel.py --config config.yaml --version {v03,v031} --scale all` | Train `VoxelSwinCLAIP` at each progressive scale |
| 4 | `python evaluate_zeroshot_voxel.py / evaluate_retrieval_voxel.py / scoring_cell_anomaly_voxel.py / retrieve_diagnosis_reason.py --config config.yaml --version {v03,v031} --model {base,voxel_swin_claip_*}` | Zero-shot classification, retrieval, anomaly scoring, diagnosis-reason retrieval |
| 5 | `python create_figures_v03.py` / `create_figures_v031.py` | Generate comparison figures |

## Results

### v0.3.0 (voxel test split, n=500, 4 structures)

| Model | Zero-shot accuracy | Anomaly ROC-AUC | Diagnosis-retrieval hit-rate |
| --- | --- | --- | --- |
| base | 25.0% (chance) | 0.486 | 0.0% |
| 2.5k (best) | **48.6%** | 0.628 (1.6k) | 29.8% (1.6k) |

Notable finding: LoRA-only adaptation of the randomly-initialized 3D Swin voxel
backbone collapsed to near-input-invariant embeddings; switching to full
fine-tuning of the voxel tower (keeping LoRA only for the pretrained ClinicalBERT
text tower) resolved this.

### v0.3.1 (voxel test split, n=1,600, 8 structures, 8,000 cells; 1.6k/3.2k/4.8k scales, 6.4k not run)

| Model | Zero-shot accuracy (chance=12.5%) | Anomaly ROC-AUC | Diagnosis-retrieval hit-rate |
| --- | --- | --- | --- |
| base | 7.6% | 0.518 | 9.0% |
| 4.8k (best trained) | 12.5% (exact chance) | 0.604 | 11.4% |

**Key diversity/generalization finding**: under the identical full-fine-tuning
architecture that resolved v0.3.0's collapse, doubling structural diversity (4→8)
and scaling data (3,000→8,000 cells) caused the **same voxel-embedding collapse to
recur** at every trained scale (zero-shot accuracy pinned exactly at chance),
despite InfoNCE training loss falling even lower than in v0.3.0. The text encoder
remained well-separated by structure throughout, localizing the collapse to the
Voxel Encoder. This shows that a fix validated at one diversity/scale regime does
not automatically generalize to a larger one — downstream-metric verification is
required at each new regime, not just training-loss monitoring. See the paper's
"Extension: Structural Diversity and Dataset Scale (v0.3.1)" section for full
diagnostics and discussion.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
