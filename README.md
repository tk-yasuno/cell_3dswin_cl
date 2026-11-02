# Cell 3D Voxel × Text Contrastive Learning (cell_3dswin_cl)

MVP v0.3.0: a 3D Swin Transformer voxel encoder + ClinicalBERT finding-text encoder
(`VoxelSwinCLAIP`), contrastively trained (LoRA + symmetric InfoNCE) on single-cell
3D confocal fluorescence volumes from the Allen Institute's public hiPSC single-cell
image dataset, paired with template-generated structural finding text. This is an
experimental methodology prototype toward future 3D pavement-material microscopy
anomaly diagnosis, extending the 2D CLIP-based `cell_vlm_cl` (v0.1.0/v0.2.0)
contrastive-learning MVPs to 3D voxels.

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

| Step | Command | Purpose |
| --- | --- | --- |
| 0 | `python tests/smoke_test_voxel_swin_claip.py` | Confirm GPU VRAM budget before any download/training |
| 1 | `python prepare_voxel_cell_dataset.py --config config.yaml` | Download/preprocess single-cell 3D volumes (quilt3, Allen Institute) |
| 2 | `python prepare_progressive_splits.py --config config.yaml --version v03` | Build stratified progressive training subsets |
| 3 | `python train_contrastive_voxel.py --config config.yaml --scale all` | Train `VoxelSwinCLAIP` at each progressive scale |
| 4 | `python evaluate_zeroshot_voxel.py / evaluate_retrieval_voxel.py / scoring_cell_anomaly_voxel.py / retrieve_diagnosis_reason.py --config config.yaml --model {base,voxel_swin_claip_*}` | Zero-shot classification, retrieval, anomaly scoring, diagnosis-reason retrieval |
| 5 | `python create_figures_v03.py --config config.yaml` | Generate comparison figures |

## Key v0.3.0 results (voxel test split, n=500, 4 structures)

| Model | Zero-shot accuracy | Anomaly ROC-AUC | Diagnosis-retrieval hit-rate |
| --- | --- | --- | --- |
| base | 25.0% (chance) | 0.486 | 0.0% |
| 2.5k (best) | **48.6%** | 0.628 (1.6k) | 29.8% (1.6k) |

Notable finding: LoRA-only adaptation of the randomly-initialized 3D Swin voxel
backbone collapsed to near-input-invariant embeddings; switching to full
fine-tuning of the voxel tower (keeping LoRA only for the pretrained ClinicalBERT
text tower) resolved this. See the paper's Discussion/Lessons sections for details.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
