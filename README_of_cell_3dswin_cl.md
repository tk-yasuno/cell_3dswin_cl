# Cell 3D Voxel x Text Contrastive Learning (MVP v0.3.0)

3D confocal fluorescence microscopy volumes (Allen Institute hiPSC single-cell image dataset) paired with template-generated structural finding text, contrastively trained with a 3D Swin Transformer voxel encoder + ClinicalBERT finding encoder (`VoxelSwinCLAIP`), as an experimental methodology for domain-adapting 3D cell morphology → text diagnosis toward future pavement-material 3D microscopy anomaly diagnosis. Builds on v0.1.0 (DTD, 2D CLIP+LoRA) and v0.2.0 (`cell_vlm_cl`, 2D cell microscopy, 2D CLIP+LoRA). Full design rationale: [docs/Plan_cell_3dswin_cl_v0-3-0.md](docs/Plan_cell_3dswin_cl_v0-3-0.md). Full write-up: [paper_cell_3dswin_cl/0_cell_3dswin_cl_v0-3-0/methodology.tex](paper_cell_3dswin_cl/0_cell_3dswin_cl_v0-3-0/methodology.tex).

## セットアップ

```powershell
# Python 3.12, CUDA 12.4 (RTX 4060 Ti 16GB)
python -m venv .venv-cell3dswin
.\.venv-cell3dswin\Scripts\pip.exe install -r requirements.txt
# requirements.txt だけでは CPU版 torch が入るため、CUDA版を最後に上書きインストール
.\.venv-cell3dswin\Scripts\pip.exe install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

monai/aicsimageio/quilt3 のインストールはpipの依存解決が重いため、`requirements.txt`一括ではなく個別インストール（`monai==1.5.2 --no-deps` 等）を推奨。詳細は[docs/Plan_cell_3dswin_cl_v0-3-0.md](docs/Plan_cell_3dswin_cl_v0-3-0.md)参照。

## パイプライン（実行順）

| Phase | コマンド | 役割 |
| ----- | -------- | ---- |
| 0 | `python tests/smoke_test_voxel_swin_claip.py` | 16GB VRAM実現可能性の実測（batch_size確定） |
| 1 | `python prepare_voxel_cell_dataset.py --config config.yaml` | Allen Institute hiPSC 4構造体(LMNB1/TOMM20/TUBA1B/TJP1)×750件をquilt3経由で収集・前処理 → `data/v03_fine_tuning/{train,val,test}.csv` |
| 2 | `python prepare_progressive_splits.py --config config.yaml --version v03` | train+valプールから0.8k/1.6k/2.5kの層化累積サブセットを作成 |
| 4 | `python train_contrastive_voxel.py --config config.yaml --scale all` | VoxelSwinCLAIP（3D Swin + ClinicalBERT）を各スケール独立に学習 → `models/voxel_swin_claip_{scale}/` |
| 5 | `python evaluate_zeroshot_voxel.py --config config.yaml --model {base,voxel_swin_claip_*}` | 4-way構造体ゼロショット分類精度 |
| 5 | `python evaluate_retrieval_voxel.py --config config.yaml --model {base,voxel_swin_claip_*}` | voxel↔text検索Recall@1,5,10 |
| 5 | `python scoring_cell_anomaly_voxel.py --config config.yaml --model {base,voxel_swin_claip_*}` | mitotic/interphase異常スコアリング、ROC-AUC検証 |
| 5 | `python retrieve_diagnosis_reason.py --config config.yaml --model {base,voxel_swin_claip_*}` | 「異常診断理由検索」hit-rate@1,5（v0.3.0新規） |
| 6 | `python create_figures_v03.py --config config.yaml` | 評価結果から比較図5枚を生成 → `paper_cell_3dswin_cl/0_cell_3dswin_cl_v0-3-0/figures/` |

## データセット（4構造体, 計3,000セル = 750セル×4構造体）

| 構造体 | structure_name | 表示名 | 件数(mitotic/interphase) |
| ------ | -------------- | ------ | ------------------------ |
| 核膜(ラミンB1) | `LMNB1` | nuclear envelope (lamin B1) | 150/600 |
| ミトコンドリア | `TOMM20` | mitochondria (TOM20) | 150/600 |
| 微小管 | `TUBA1B` | microtubules (alpha-tubulin) | 150/600 |
| タイトジャンクション | `TJP1` | tight junctions (ZO-1) | 150/600 |

test split は125セル/構造体（計500セル）、train+valプールは625セル/構造体（計2,500セル）で0.8k/1.6k/2.5kのprogressive subsetを構築。Concept Note候補の「CAAX」（細胞膜）はこのquilt3パッケージに実在しないため`TJP1`に、「TOM20」は正式HGNC記号`TOMM20`に実装時訂正（詳細は論文Section 5参照）。

## 実験結果（voxel test split, n=500, 4構造体）

### ゼロショット構造体分類精度（chance=25%）

| モデル | Accuracy |
| ------ | -------- |
| base (random init) | 25.0% |
| voxel_swin_claip_0.8k | 29.0% |
| voxel_swin_claip_1.6k | 27.0% |
| voxel_swin_claip_2.5k | **48.6%** |

### 異常スコアリング（cell_stage M0=interphase vs mitotic, n_abnormal=114/500, 閾値0.70）

| モデル | ROC-AUC | 判定 |
| ------ | ------- | ---- |
| base | 0.486 | BELOW_THRESHOLD |
| 0.8k | 0.488 | BELOW_THRESHOLD |
| 1.6k | 0.628 | BELOW_THRESHOLD |
| 2.5k | 0.567 | BELOW_THRESHOLD |

v0.1.0/v0.2.0と異なり、どのスケールも暫定閾値0.70を超えなかった（Discussion参照：構造体識別は学習されたが、細胞周期状態の識別は不十分）。

### 診断理由検索（異常診断理由検索, n_queries=114, hit-rate@1=hit-rate@5）

| モデル | Hit-rate@1/5 |
| ------ | ------------ |
| base | 0.0% |
| 0.8k | 0.0% |
| 1.6k | **29.8%** |
| 2.5k | 0.0% |

## 重要な実装上の教訓

- **LoRAのみの適応はランダム初期化バックボーンでは機能しない**: 当初、v0.1.0/v0.2.0と同じ「バックボーン凍結+attention projectionsにLoRA」方式を3D Swin Voxel Encoderに適用したところ、15epoch学習後もvoxel embeddingがほぼ入力不変（pairwise cosine類似度0.93〜0.99）に崩壊し、全評価指標がchanceレベルに留まった（学習lossは健全に減少していたにも関わらず）。LoRAは「事前学習済みの良い初期値を小さい差分で適応する」ことを前提とする手法であり、ランダム初期化のバックボーンにはこの前提が成立しない。Voxel EncoderをフルFine-tuningに切り替えることで解消した（詳細: 論文Section 5.3, Decisions参照）。
- **メタデータの列は必ず実データで検証する**: `ChannelNumberStruct`/`ChannelNumber405`等はFOVレベルの値であり、既にクロップ済みの`crop_raw`（常に固定3チャンネル順: 0=DNA, 1=膜, 2=構造タンパク質）には適用できないことを実装時に発見。
- **候補データソースも実データで検証する**: Concept Note候補の「CAAX」構造体はこのquilt3パッケージには存在せず、「TOM20」は正式には`TOMM20`。

## 成果物

- 学習済みモデル: `models/voxel_swin_claip_{0.8k,1.6k,2.5k}/`
- 評価結果: [data/v03_fine_tuning/evaluations/](data/v03_fine_tuning/evaluations)
- 論文原稿（LaTeX）: [paper_cell_3dswin_cl/0_cell_3dswin_cl_v0-3-0/methodology.tex](paper_cell_3dswin_cl/0_cell_3dswin_cl_v0-3-0/methodology.tex)
- 実装計画: [docs/Plan_cell_3dswin_cl_v0-3-0.md](docs/Plan_cell_3dswin_cl_v0-3-0.md)
