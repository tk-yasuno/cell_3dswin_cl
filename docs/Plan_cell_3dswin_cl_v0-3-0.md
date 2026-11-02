Plan: v0.3.0 Cell 3D Voxel × Text Contrastive Learning（3D Swin Transformer + ClinicalBERT、Medical/Material CLAIP基盤）

TL;DR: v0.2.0（`cell_vlm_cl`、本ワークスペースには `0_LogBAK/pre-code_cell_vlm_cl` として保存済み）は、CLIP ViT-L/14-336 + LoRA + InfoNCEで2D顕微鏡細胞画像（Malaria/C-NMC/LDC、6クラス計4,500枚）とテンプレートcaptionを対照学習し、細胞異常スコアリングROC-AUC最大0.985を達成した。本MVP v0.3.0（新規ワークスペース `cell_3dswin_cl`）は、Image Encoderを3D Swin Transformer（MONAI `SwinUNETR`のエンコーダ部）に、Text EncoderをClinicalBERT（`Bio_ClinicalBERT`）に置き換え、Allen Institute hiPSC single-cell 3D蛍光顕微鏡データセット（4構造体×750件、quilt3経由）とテンプレート生成所見テキストを対照学習する「Voxel×Text」2者対照学習（InfoNCE）を実装する。これは、土木構造物の舗装材料における3D顕微鏡異常診断をテキスト推論するdownstream taskへドメイン適応するための方法論を、材料系に直接の実験データが無い現時点で生物系3D画像により実験的に検証するステップである。Gene/Structure Encoderを加えた3者対照学習（Medical CLAIP）、所見テキストの自由生成、3D Swin事前学習重みの導入は、本バージョンではスコープ外とし、v0.4.0以降に先送りする。

## 背景・設計根拠（ユーザー確認済みの分岐点）

- **データセット規模**: Allen Institute hiPSC single-cell image dataset（quilt3パッケージ `aics/hipsc_single_cell_image_dataset`, registry `s3://allencell`, 216,062 single cells / 18,186 FOV, 既に単一細胞ごとにクロップ済みのOME-TIFF `crop_raw`/`crop_seg`を提供）のうち、4構造体 **LMNB1(核膜/Lamin B1)・TOM20(ミトコンドリア)・TUBA1B(微小管)・CAAX(細胞膜)** を選定し、各750件（v0.2.0の6クラス×750枚規模を4クラスで踏襲）、64^3ボクセルにクロップ・リサイズする（ユーザー選定: `small_4x750_64`）。
- **モダリティ範囲**: v0.3.0は **Voxel Encoder(3D Swin) ↔ Finding Encoder(ClinicalBERT)** の2者対照学習のみを実装する。Gene/Structure Encoder（3者目）はv0.4.0に先送りする（ユーザー選定: `two_tower`）。
- **異常プロキシラベル**: Allen Institute付与の `cell_stage` メタデータ（`M0`=interphase, `M1`〜`M7`=mitotic各ステージ）をそのまま使い、`M0`→normal-like、`M1`〜`M7`→abnormal-likeとする（ユーザー選定: `mitotic_stage`）。v0.1.0/v0.2.0の `damage_proxy_labels.yaml`/`cell_anomaly_proxy_labels.yaml`と同じ「プロトタイプ検証用の弱ラベル」という位置づけを踏襲する。
- **所見テキストの生成方法**: Allen Cellデータには自由記述の所見テキストが存在しないため、v0.1.0/v0.2.0と同じテンプレートcaption ensemble方式（構造体表示名×細胞周期状態の組み合わせテンプレート）で生成する。LLMによる言い換え拡張（`llm_expanded`）は行わない（ユーザー選定: `template_ensemble`）。
- **downstream taskの評価範囲**: v0.3.0は「異常診断理由検索」を、学習コーパス中の所見テキストに対する最近傍検索（retrieval）として評価する。自由文生成によるzero-shot診断説明生成はv0.4.0に先送りする（ユーザー選定: `retrieval_only`）。

## 技術的な実現可能性（事前調査で確認済み）

- quilt3アクセス: `quilt3.Package.browse("aics/hipsc_single_cell_image_dataset", registry="s3://allencell")` → `pkg["metadata.csv"]()` でメタデータDataFrameを取得、列に `structure_name`, `cell_stage`, `FOVId`, `this_cell_index`, `crop_raw`, `crop_seg`, `ChannelNumberStruct`, `ChannelNumber405`(DNA), `ChannelNumber638`(membrane) 等を含む（Allen公式チュートリアル `Tutorial 1/2/4` で実例確認済み）。`pkg[subdir][filename].fetch(local_path)` で個別セルのOME-TIFFを取得可能。
- ライセンス: Allen Institute Terms of Use により非商用研究・教育目的限定（CC BY-NC-SA相当、要引用）。v0.2.0のC-NMC/LDCと同じ「非商用研究利用限定」の注記をYAMLに残す。
- 3D Swin Transformer: `monai>=1.3`の`monai.networks.nets.SwinUNETR`から`model.swinViT(x)`でエンコーダのみ利用可能（セグメンテーションデコーダ部は不要、使わない）。
- ClinicalBERT: `emilyalsentzer/Bio_ClinicalBERT`（HuggingFace, MIT license, BERT-base, 実際のMIMIC-III臨床ノートで事前学習済み）。

Steps

Phase 0: 環境構築・実行可能性スモークテスト（独立実行可能）

1. 新規仮想環境 `.venv-cell3dswin` を作成し、`requirements.txt` を新規作成（既存 `cell_vlm_cl`の requirements.txt から共通部分を流用しつつ、以下を追加）
   - `monai>=1.3.0`（3D Swin Transformer/SwinUNETR）
   - `aicsimageio>=4.14.0`（OME-TIFF読み込み。`tifffile`は依存として自動導入される）
   - `quilt3>=5.3.0`（Allen Cell S3データ取得）
   - `transformers>=4.57.0`, `peft>=0.18.0`（Bio_ClinicalBERT + LoRA、CLIP用と共通）
   - `scipy>=1.11.0`（`scipy.ndimage.zoom`によるボクセルリサイズ）
   - `torch==2.6.0`/`torchvision==0.21.0` はv0.1.0と同じCUDA 12.4手順でインストール（CPU版が先に入る問題も同じ注記を踏襲）
   - 既存と同じ: `pandas`, `pyyaml`, `scikit-learn`, `matplotlib`, `seaborn`, `tqdm`, `loguru`
2. `src/models/voxel_swin_claip.py` の最小実装（Step 7の先行部分のみ）を用意した上で、`tests/smoke_test_voxel_swin_claip.py` を新規作成 — ランダムテンソル（`batch=4, channels=2, 64x64x64`）とダミー所見テキストでforward/backwardを1stepだけ実行し、(a) CUDA OOMが起きないこと、(b) `torch.cuda.max_memory_allocated()` が16GB未満であることを確認する。OOM発生時は `feature_size: 24→12` または `voxel_img_size: 64→32` へ縮小してリトライするフォールバックをログ出力する（Phase 3以降の全ハイパーパラメータはこのスモークテストの実測結果で確定させる。v0.1.0の「batch_size 32→8に実測で縮小」と同じ手順） (depends on 7 の最小実装)

Phase 1: データ収集（3D ボクセル + 所見テキスト）（depends on Phase 0, 独立実行可能）

3. `config.yaml` を新規作成し、`v03_dataset` / `v03_contrastive_training` / `v03_evaluation` / `v03_paper` の4セクションを定義
   - `v03_dataset.quilt3_package: "aics/hipsc_single_cell_image_dataset"`, `registry: "s3://allencell"`
   - `v03_dataset.structures: ["LMNB1", "TOM20", "TUBA1B", "CAAX"]`（核膜・ミトコンドリア・微小管・細胞膜）
   - `v03_dataset.structure_display_names`: `{"LMNB1": "nuclear envelope (lamin B1)", "TOM20": "mitochondria (TOM20)", "TUBA1B": "microtubules (alpha-tubulin)", "CAAX": "plasma membrane (CAAX)"}`
   - `v03_dataset.per_class_target: 750`, `test_per_class: 125`（4構造体×750=3,000、うちtest 4×125=500、train+valプール4×625=2,500）
   - `v03_dataset.voxel_shape: [64, 64, 64]`, `channels: ["dna", "structure"]`（`ChannelNumber405`をdna, `ChannelNumberStruct`をstructureとして抽出し2チャンネルで保存。membrane/brightfield/segmentationはv0.3.0では未使用、Further Considerations参照）
   - `v03_dataset.random_seed: 42`, `stratify_by: "structure_name"`
   - `v03_dataset.train_sizes: [800, 1600, 2500]`, `scale_labels: {800: "0.8k", 1600: "1.6k", 2500: "2.5k"}`（v0.1.0/v0.2.0の4段階(1k/2k/3k/4k)から3段階に短縮。理由はDecisions参照）
   - `v03_dataset.finding_templates`: 状態(interphase/mitotic)×構造体表示名のテンプレートensemble（例: interphase→`["a 3D confocal fluorescence volume of an interphase human iPS cell showing an intact {structure}", "a 3D microscopy volume showing normal, non-dividing {structure} morphology", ...]`, mitotic→`["a 3D confocal fluorescence volume of a mitotic human iPS cell showing {structure} reorganization during cell division (stage {stage})", "a 3D microscopy volume showing {structure} undergoing mitotic remodeling", ...]`）
   - `v03_dataset.proxy_labels_file: "cell_anomaly_proxy_labels_voxel.yaml"`
   - `v03_paper.latex_dir`, `figures_dir` はPhase 6で使用
4. `prepare_voxel_cell_dataset.py` を新規作成
   - `quilt3.Package.browse(...)` → `meta_df = pkg["metadata.csv"]()` を取得
   - 構造体ごとに `meta_df.query("structure_name == @s")` し、`random_seed=42`で`per_class_target`件（不足時は全件使用しログ出力、LDCの件数不足時の扱いと同じ方針）をサンプリング。test分は125件を固定ホールドアウトとして先に確保し、残りをtrain+valプールとする
   - 各サンプル行について `row.crop_raw`（`"subdir/filename.ome.tiff"`形式）を `pkg[subdir_name][file_name].fetch(local_tmp_path)` で取得し、`aicsimageio.AICSImage`で読み込み、`row.ChannelNumber405`(dna)・`row.ChannelNumberStruct`(structure)の2チャンネルを`"CZYX"`で抽出
   - `_sample_valid_images`と同様の開封可能性チェック（破損TIFFはスキップし、プールから補充サンプリング）を実装
   - `scipy.ndimage.zoom`で各チャンネルを`voxel_shape=[64,64,64]`にリサイズし、パーセンタイルクリップ(0.5–99.5%)の上でmin-max正規化してfloat16で`data/v03_fine_tuning/voxels/{structure_name}/{CellId}.npy`（shape `[2,64,64,64]`）にキャッシュ保存（学習時に毎回TIFFデコード・S3取得を繰り返さないため）
   - `v03_dataset.finding_templates`から各セルの`cell_stage`に応じてcaptionをensemble選択し、`voxel_path, structure_name, cell_stage, finding_text, split`形式で`data/v03_fine_tuning/{train,val,test}.csv`を出力
   - quilt3が匿名S3アクセスに失敗する場合（要実装時確認）のフォールバックとして、`s3://allencell/{crop_raw path}`への直接`boto3`匿名アクセス(`UNSIGNED`設定)手順をコメントで明記

Phase 2: progressive split（depends on Phase 1）

5. `prepare_progressive_splits.py` を `cell_3dswin_cl` にそのままコピーし、`--version`の`choices`に`"v03"`を追加するのみ（スクリプト本体はCSVの`class_col`/`train_sizes`/`scale_labels`をconfig経由で読む汎用実装のため無変更で流用可能）

Phase 3: モデル実装（depends on Phase 0 の方針確定, 独立実行可能, parallel with Phase 1/2）

6. `src/models/voxel_swin_claip.py` 新規作成（`src/models/clip_lora.py`のMaterialCLIPと同じAPI形状で設計）
   - `VoxelSwinCLAIPConfig`: `voxel_img_size=(64,64,64)`, `in_channels=2`, `feature_size=24`, `lora_r_voxel=8`, `lora_alpha_voxel=16`, `bert_model_id="emilyalsentzer/Bio_ClinicalBERT"`, `lora_r_text=16`, `lora_alpha_text=32`, `shared_dim=512`, `use_checkpoint=True`
   - Voxel tower: `monai.networks.nets.SwinUNETR(img_size=..., in_channels=2, out_channels=2, feature_size=24, use_checkpoint=True)`から`self.swinViT`のみ保持し、decoder部は破棄。`peft.get_peft_model`でLoRAを`target_modules=["qkv", "proj"]`に適用（CLIPの`q_proj/k_proj/v_proj/out_proj`に相当）。`encode_voxel()`は`swinViT(x)[-1]`（最終stage特徴マップ）を`adaptive_avg_pool3d`で1点にプールし、`nn.Linear(feature_size*16, shared_dim)`で射影、L2正規化
   - Text tower: `transformers.AutoModel.from_pretrained(bert_model_id)` + LoRA `target_modules=["query","key","value"]`。`encode_text()`は`[CLS]`トークンの`last_hidden_state`を`nn.Linear(hidden_size, shared_dim)`で射影、L2正規化
   - `logit_scale`: `nn.Parameter(torch.tensor(math.log(1/0.07)))`（CLIP由来の初期値を流用）
   - `forward()`は`MaterialCLIP.forward()`と同じ辞書を返す（`voxel_embeds`/`text_embeds`/`logits_per_voxel`/`logits_per_text`）。`save_adapter()`/`from_pretrained_adapter()`も同様に実装
7. `tests/smoke_test_voxel_swin_claip.py`（Phase 0 Step 2で先行作成したものを本実装に合わせて更新）

Phase 4: 学習パイプライン（depends on Phase 1, 2, 3）

8. `train_contrastive_voxel.py` 新規作成（`train_contrastive_lora.py`の構成を踏襲）
   - `VoxelCaptionDataset`: `__getitem__`で`np.load(voxel_path)`（float16→float32変換）と`finding_text`を返す
   - collate関数で`BertTokenizer`によるテキストトークナイズとvoxelテンソルのスタックを行う
   - 対称InfoNCE損失（`clip_infonce_loss`を関数ごと流用、`logits_per_image`→`logits_per_voxel`に変数名変更のみ）
   - `v03_contrastive_training.training`: `batch_size`はPhase 0のスモークテスト実測値で確定（初期推定4）、`learning_rate: 1e-4`, `num_epochs: 15`（3D 1サンプルあたりの計算コストが2D CLIPより大きいため、v0.1/v0.2の10epochからやや減らした初期値。実測の学習曲線を見て調整可、Decisions参照）、`bf16: true`
   - `--scale {0.8k,1.6k,2.5k,all}`で各スケール独立学習 → `models/voxel_swin_claip_{scale}/`に保存
9. 3スケール分の学習を実行: `python train_contrastive_voxel.py --config config.yaml --scale all`（0.8k/1.6k/2.5kを順次実行） (depends on 8)

Phase 5: 評価（depends on Phase 4）

10. `evaluate_zeroshot_voxel.py` 新規作成 — 4構造体のゼロショット分類精度（`evaluate_zeroshot.py`と同じロジックをvoxel/BERT用に書き換え）。`base`（ランダム初期化voxel encoder + 事前学習済みBio_ClinicalBERT、対照学習前）とのself-baseline比較のみ行う（外部文献のsanity rangeは存在しないため、v0.2.0と同じ扱い）
11. `evaluate_retrieval_voxel.py` 新規作成 — voxel↔text Recall@1,5,10（`evaluate_retrieval.py`を書き換え）
12. `cell_anomaly_proxy_labels_voxel.yaml` 新規作成 — `abnormal_like_rule: "cell_stage != 'M0'"`, `normal_like_rule: "cell_stage == 'M0'"`、ファイル冒頭に「Allen Institutionの細胞周期ステージは正式なground truthだが、舗装材料異常検出という本番タスクへの転用という意味ではプロトタイプ検証専用の代理ラベルである」旨を明記（v0.2.0の`cell_anomaly_proxy_labels.yaml`と同趣旨）
13. `scoring_cell_anomaly_voxel.py` 新規作成 — `scoring_material_durability.py`の`text_prototype()`/`bin_score()`方式を踏襲し、`state_prompts.normal`/`state_prompts.abnormal`（config `v03_evaluation.scoring`）の差分コサイン類似度でスコアリングし、`cell_stage`に基づくproxy labelとのROC-AUCを検証。各モデル・各structureごとのmitotic件数(n)を出力JSONに明記する(depends on 6, 12)
14. `retrieve_diagnosis_reason.py` 新規作成（v0.3.0新規の評価スクリプト、「異常診断理由検索」のdownstream task評価）— テストセット中のabnormal-like(mitotic)ボクセルそれぞれについて、学習コーパス（train+valプールの`finding_text`）に対しvoxel embeddingとtext embeddingのコサイン類似度で最近傍top-kを検索し、正解の`structure_name`と`cell_stage`（mitotic/interphaseの別）が一致する所見が含まれるかで hit-rate@1, hit-rate@5 を算出。base/各スケールモデルで比較表を出力

Phase 6: 結果整理・図表（depends on Phase 5）

15. `create_figures_v03.py` 新規作成（`create_figures_v02.py`を土台に、(a)スケール別ゼロショット精度, (b)Recall@kスケール比較, (c)normal/abnormal-likeの類似度violin plot, (d)学習lossカーブ、(e)新規: diagnosis reason retrieval hit-rate@kの棒グラフ、の5図を出力） → `paper_cell_3dswin_cl/0_cell_3dswin_cl_v0-3-0/figures/`

Phase 7: arXiv原稿執筆（depends on Phase 6、実測結果が出てから着手）

16. `paper_cell_3dswin_cl/0_cell_3dswin_cl_v0-3-0/methodology.tex` 新規作成 — `0_LogBAK/pre-paper_cell_vlm_cl/1_cell_vlm_cl_v0-2-0/methodology.tex`と同一LaTeXテンプレートを踏襲。Related Workにv0.2.0(2D CLIP)→v0.3.0(3D Swin+ClinicalBERT)のアーキテクチャ比較を追加し、Limitationsに(i)Allen Institute非商用ライセンス制約、(ii)mitotic細胞の出現率が低く各構造体のabnormal-like件数が少数である点、(iii)3D Swinがランダム初期化のため外部文献比較基準が存在しない点、(iv)細胞3D画像と道路舗装3Dマイクロ/CT画像の間のドメインギャップの限界、(v)Gene/Structure Encoderと自由文生成を次版に先送りした旨、を明記
17. `pdflatex methodology.tex`でコンパイルし、図表・参照の整合性を確認

Relevant Files

- `I:\ACT2025.5.26-2030\MVP\DAMAGE\cell_3dswin_cl\requirements.txt` — 新規作成（monai/aicsimageio/quilt3/scipy追加）
- `I:\ACT2025.5.26-2030\MVP\DAMAGE\cell_3dswin_cl\config.yaml` — 新規作成（`v03_dataset`/`v03_contrastive_training`/`v03_evaluation`/`v03_paper`）
- `I:\ACT2025.5.26-2030\MVP\DAMAGE\cell_3dswin_cl\tests\smoke_test_voxel_swin_claip.py` — 新規作成（Phase 0の16GB VRAM実現可能性確認）
- `I:\ACT2025.5.26-2030\MVP\DAMAGE\cell_3dswin_cl\prepare_voxel_cell_dataset.py` — 新規作成（quilt3経由のAllen Cell収集・クロップ・caption CSV生成）
- `I:\ACT2025.5.26-2030\MVP\DAMAGE\cell_3dswin_cl\prepare_progressive_splits.py` — `0_LogBAK/pre-code_cell_vlm_cl`からコピー、`--version`choicesに`"v03"`追加のみ
- `I:\ACT2025.5.26-2030\MVP\DAMAGE\cell_3dswin_cl\src\models\voxel_swin_claip.py` — 新規作成（VoxelSwinCLAIP: 3D Swin + ClinicalBERT + LoRA + InfoNCE dual encoder）
- `I:\ACT2025.5.26-2030\MVP\DAMAGE\cell_3dswin_cl\train_contrastive_voxel.py` — 新規作成
- `I:\ACT2025.5.26-2030\MVP\DAMAGE\cell_3dswin_cl\evaluate_zeroshot_voxel.py` — 新規作成
- `I:\ACT2025.5.26-2030\MVP\DAMAGE\cell_3dswin_cl\evaluate_retrieval_voxel.py` — 新規作成
- `I:\ACT2025.5.26-2030\MVP\DAMAGE\cell_3dswin_cl\scoring_cell_anomaly_voxel.py` — 新規作成
- `I:\ACT2025.5.26-2030\MVP\DAMAGE\cell_3dswin_cl\retrieve_diagnosis_reason.py` — 新規作成（v0.3.0新規の「異常診断理由検索」評価）
- `I:\ACT2025.5.26-2030\MVP\DAMAGE\cell_3dswin_cl\cell_anomaly_proxy_labels_voxel.yaml` — 新規作成
- `I:\ACT2025.5.26-2030\MVP\DAMAGE\cell_3dswin_cl\create_figures_v03.py` — 新規作成
- `I:\ACT2025.5.26-2030\MVP\DAMAGE\cell_3dswin_cl\paper_cell_3dswin_cl\0_cell_3dswin_cl_v0-3-0\methodology.tex` — 新規作成
- `I:\ACT2025.5.26-2030\MVP\DAMAGE\cell_3dswin_cl\0_LogBAK\pre-code_cell_vlm_cl\src\models\clip_lora.py` — 参照のみ（VoxelSwinCLAIPのAPI設計の土台）
- `I:\ACT2025.5.26-2030\MVP\DAMAGE\cell_3dswin_cl\0_LogBAK\pre-paper_cell_vlm_cl\1_cell_vlm_cl_v0-2-0\methodology.tex` — 参照のみ（LaTeXテンプレートの土台）

Verification

- `python tests/smoke_test_voxel_swin_claip.py` → CUDA OOMなく1 forward/backward stepが完了し、`torch.cuda.max_memory_allocated() < 16 * 1024**3` を確認。失敗時は`feature_size`/`voxel_img_size`を縮小してパスするまでリトライ
- `python prepare_voxel_cell_dataset.py --config config.yaml` → `data/v03_fine_tuning/{train,val,test}.csv`が生成され、`test.csv`は4構造体×125=500行、train+valプールは4構造体×625=2,500行（構造体ごとの実件数が不足した場合はログに明記された件数で許容）
- `python prepare_progressive_splits.py --config config.yaml --version v03` → `train_0.8k.csv`(800行)/`train_1.6k.csv`(1600行)/`train_2.5k.csv`(2500行)が生成され、各ファイルで4構造体がほぼ均等
- `python train_contrastive_voxel.py --config config.yaml --scale all` → `models/voxel_swin_claip_{0.8k,1.6k,2.5k}/`が保存され、各スケールの学習loss曲線がCUDA OOMなく完走したことを確認
- `python evaluate_zeroshot_voxel.py --config config.yaml --model base` と各`voxel_swin_claip_{scale}` → 4-way分類精度テーブル。domain-adapted > base(chanceレベル25%程度)を確認（外部文献比較なし、v0.2.0と同じ自己ベースライン方式）
- `python evaluate_retrieval_voxel.py --config config.yaml --model ...` → Recall@1,5,10テーブル。domain-adapted > base を確認
- `python scoring_cell_anomaly_voxel.py --config config.yaml --model voxel_swin_claip_{scale}` → `cell_anomaly_voxel_scores_{model}.csv`・`cell_anomaly_voxel_proxy_validation_{model}.json`（`proxy_roc_auc`とmitotic件数nを含む）を出力。目標値: domain-adapted ROC-AUC >= 0.70（v0.1.0の暫定閾値を踏襲、mitotic件数が少数の場合はnを明記した上で参考値として扱う）
- `python retrieve_diagnosis_reason.py --config config.yaml --model ...` → hit-rate@1,@5テーブル。domain-adapted > base を確認
- `create_figures_v03.py` を実行し、`paper_cell_3dswin_cl/0_cell_3dswin_cl_v0-3-0/figures/`に5枚のPNGが生成されることを確認
- `pdflatex methodology.tex` がエラーなくコンパイルされ、PDFの図表参照に欠落がないことを確認

Decisions

- **データセット**: Allen Institute hiPSC single-cell image dataset（quilt3経由, 4構造体 LMNB1/TOMM20/TUBA1B/TJP1 ×各750件, 64^3ボクセル）を採用。構造体は「核膜・ミトコンドリア・微小管・タイトジャンクション(細胞間接着)」という異なる細胞内小器官カテゴリを横断するように選定し、Concept Noteの「細胞小器官レベルで異常形態学を学習」という方向性に対応させた。**実装時の修正**: Concept Note由来の候補名「TOM20」は正式HGNC遺伝子記号`TOMM20`に訂正。候補の細胞膜マーカー「CAAX」は本quilt3パッケージの実際の`metadata.csv`には存在しないことを実装時に確認し、代替として膜関連構造`TJP1`(タイトジャンクション, ZO-1)を採用した。
- **モダリティ範囲**: v0.3.0はVoxel×Textの2者対照学習のみとし、Gene/Structure Encoderによる3者対照学習（Medical CLAIP）はv0.4.0に先送り: 新規の3D Swin + ClinicalBERTアーキテクチャ自体が未検証のため、まず2者で学習・評価パイプラインの健全性を確認してから3者目を追加する方が手戻りが少ないと判断。
- **異常プロキシラベル**: `cell_stage`（M0=interphase vs mitotic）を正式なground truthではなく「プロトタイプ検証用弱ラベル」として扱う。**実装時の修正**: 実データの`cell_stage`列はM1〜M7の個別ステージではなく、結合バケット(`M1M2`/`M3`/`M4M5`/`M6M7_single`/`M6M7_complete`)として提供されていることを確認し、`cell_anomaly_proxy_labels_voxel.yaml`をこの実値に合わせて修正した。v0.1.0/v0.2.0の`damage_proxy_labels.yaml`系列と同じ位置づけ。
- **mitotic細胞のオーバーサンプリング**: mitotic細胞は構造体あたり全体の2〜13%程度(実測)と希少なため、`mitotic_oversample_fraction: 0.2`により各構造体750件のうち最大20%をmitotic細胞に優先配分し、残りをinterphaseで埋める方式を採用（Phase 5のROC-AUC検証がpositiveサンプル不足で破綻しないようにするための実装判断）。
- **所見テキスト**: テンプレートcaption ensemble方式を採用し、LLMによる言い換え拡張は行わない。理由: Bio_ClinicalBERT自体は実臨床ノートで事前学習済みであり、本バージョンの目的は「3D voxel encoderとの対照学習パイプラインが機能するか」の検証であって、テキストの自然さの最適化はスコープ外とした。
- **downstream評価**: 自由文生成ではなく、学習コーパスに対する最近傍検索（`retrieve_diagnosis_reason.py`のhit-rate@k）で「異常診断理由検索」を定量評価する。自由文生成（デコーダ or RAG）はv0.4.0に先送り。
- **progressive scaleを4段階→3段階に短縮**: 3Dボクセル+3D Swinの1サンプルあたりの計算コストは2D CLIPより大きいため、v0.1.0/v0.2.0の1k/2k/3k/4kから0.8k/1.6k/2.5kの3段階に短縮し、GPU時間を抑えつつスケーリング傾向の観察を維持する。
- **3D Swinの初期化**: ランダム初期化を既定とする（CT画像向けMONAI事前学習重みは蛍光顕微鏡とドメインギャップが大きく、転移効果が不明なため）。精度が著しく低い場合のみFurther Considerationsの代替案を検討する。
- **crop_rawの2チャンネル(dna+structure)のみ使用**: membrane染色・brightfield・segmentationマスク(`crop_seg`)はv0.3.0では使わない。理由: まず最小構成で対照学習パイプラインを動かすことを優先し、追加チャンネルによる精度向上はv0.4.0以降の拡張に残す。**実装時の発見**: `crop_raw`は細胞系統によらず常に固定3チャンネル順(0=DNA, 1=膜, 2=構造タンパク質)であり、メタデータの`ChannelNumberStruct`/`ChannelNumber405`列はFOVレベルのチャンネル番号でありcrop_rawには適用できないことが実データ検証で判明（詳細はPlan外だがprepare_voxel_cell_dataset.pyのコメント参照）。チャンネル抽出は固定インデックス(0, 2)に変更した。
- **新規ワークスペース・新規venv**: `cell_vlm_cl`とは別に`cell_3dswin_cl`を新設し、仮想環境も`.venv-cell3dswin`として分離する（monai/aicsimageio/quilt3という重い依存を既存`cell_vlm_cl`の`.venv`に混在させない）。
- **batch_size確定**: Phase 0スモークテスト実測(batch=4→3.26GB, batch=12→8.68GB, batch=16→11.39GB)に基づき、batch_size=12を採用(16GB予算の約54%、安全マージン確保)。
- **Voxel EncoderのLoRA→フルFine-tuningへの切り替え(実装時の重大な修正)**: 当初計画通りVoxel EncoderにもLoRA(r=8, attention projectionsのみ)を適用し全3スケールを15epoch学習したところ、学習lossは健全に減少した(2.5k: 1.83→1.16)にも関わらず、全4評価指標(ゼロショット分類・検索・異常スコアリング・診断理由検索)がchanceレベルに留まった。voxel embeddingを直接検証したところ、異なる構造体間でもpairwise cosine類似度0.93〜0.99と、ほぼ入力不変に崩壊していることを発見。原因は「LoRAは事前学習済みの良い初期値を前提とする手法」であり、ランダム初期化のバックボーン(patch_embed/PatchMergingは凍結・ランダムのまま)にrank-8の小さな差分を加えるだけでは有用な表現を一から学習できないため。Voxel EncoderをフルFine-tuning(全パラメータ学習可能、3.68Mパラメータ、Text Encoder(ClinicalBERT)は事前学習済みのためLoRAのまま維持)に切り替えて全スケールを再学習した結果、embeddingが構造体ごとに明確にクラスタ化し(同一構造体ペア0.98〜0.997 vs 異構造体ペア0.79〜0.93)、ゼロショット精度もchance(25.0%)→48.6%(2.5k)まで改善した。異常スコアリングROC-AUCは0.628(1.6k)止まりで閾値0.70には未到達、診断理由検索もhit-rate@1/5が1.6kで29.8%、他スケールで0.0%と非単調——これらは「構造体識別は学習されたが細胞周期状態の識別は不十分」という素直な(かつ正直な)実験結果として論文のResults/Discussionに記録した。

Further Considerations

- 3者対照学習（Voxel × Text × Gene/Structure-ID）への拡張
  推奨: v0.4.0で、`structure_name`を学習可能な埋め込みテーブル（またはUniProt遺伝子名トークンを入力する小型Transformer）でエンコードする第3タワーを追加し、3ペアのInfoNCE損失の和（voxel-text, voxel-gene, text-gene）として定式化する。
- 所見テキストの自由生成（downstream text generation）
  推奨: v0.4.0でまずRAG方式（`retrieve_diagnosis_reason.py`で取得したtop-k所見文を軽量LLM、例: Swallow-8Bに条件付けして言い換え生成）から着手し、end-to-endデコーダの学習は得られた知見を見てから判断する。
- 3D Swinの事前学習重み導入
  推奨: v0.3.0の結果（ゼロショット精度・ROC-AUC）が著しく低い場合に限り、MONAIのself-supervised CT事前学習チェックポイント（Apache-2.0ライセンス、要ライセンス再確認）を試す。現時点ではランダム初期化を既定とする。
- batch_size/feature_sizeの確定値
  推奨: Phase 0のスモークテスト実測値を正とする。本計画の`batch_size=4`, `feature_size=24`は暫定見積もりであり、実測前の数値として扱う。
- `crop_seg`（形状セグメンテーション）の活用
  推奨: v0.4.0以降で、形状特徴量をチャンネルまたは補助損失として追加し、Allen Institute提供の`features.csv`(形状統計量)と組み合わせた`shape_outlier`異常プロキシの併用を検討する。
- mitotic細胞の出現率の低さ
  推奨: 構造体ごとに750件サンプリングしてもmitotic(abnormal-like)件数が少数（典型的に全体の数%）になるリスクを許容し、`scoring_cell_anomaly_voxel.py`の出力JSONに実際のn数を必ず明記して透明性を確保する。不十分な場合はTutorial 4の手法（`cell_stage!='M0'`のFOVを優先的に多く収集）で底上げを検討する。
- `aicsimageio`のインストール失敗時のフォールバック
  推奨: Windows環境での依存解決に失敗した場合、`tifffile`による直接OME-TIFF読み込み＋手動XMLメタデータパース（LDCデータセットのPascal VOC XML処理と同じ標準ライブラリ`xml.etree.ElementTree`方式）に切り替える。
- quilt3の匿名S3アクセス
  推奨: 環境によってはAWS認証情報の設定を要求される場合があるため、失敗時は`boto3`の`UNSIGNED`設定による直接S3パスフェッチにフォールバックする手順を`prepare_voxel_cell_dataset.py`のコメントに残す。
