# UNI2-h + MLP (CORAL) による病理マルチ概念 5 段階分類

画像 400 枚 → UNI2-h 特徴量 [1536] (center_crop 896x896) → 概念ごとの CORAL ヘッド (8 概念) → 5-Fold CV → Nelder-Mead 閾値最適化

## ファイル構成

| ファイル | フェーズ | 内容 |
|---|---|---|
| `config.py` | 共通 | パス・列名・ハイパーパラメータ（**最初にここを編集**） |
| `00_check_env.py` | 1 | ライブラリ / GPU・VRAM / UNI2-h アクセス権の確認 |
| `01_prepare_data.py` | 1 | `data/labels.csv`（slide_id, image_path, grade）作成・クラス分布集計 |
| `uni_encoder.py` | 2 | UNI2-h の構築（`dynamic_img_size=True`）と 896px 前処理 |
| `02_extract_features.py` | 2 | 全画像を `features/slide_XXX.pt`（[1536]）に保存・検証 |
| `coral_model.py` | 3 | `CoralMLP`, `label_to_levels`, `coral_loss` |
| `04_train_cv.py` | 4 | Stratified 5-Fold CV + AdamW + Early Stopping → OOF 予測 |
| `05_optimize_thresholds.py` | 5 | QWK 最大化の閾値探索・混同行列・誤り分析 |
| `06_predict.py` | 推論 | 新規画像を 5 Fold アンサンブル + 最適閾値で判定 |

## 実行手順

```bash
cd UNI2-h
pip install -r requirements.txt
huggingface-cli login                 # MahmoodLab/UNI2-h の利用承認済みトークン

python 00_check_env.py
# config.py の SOURCE_CSV / IMG_DIR / FILENAME_COL / GRADE_COL を設定
python 01_prepare_data.py
python 02_extract_features.py         # GPU 推奨。中断しても再実行で続きから
python coral_model.py                 # (任意) モデル単体の動作確認
python 04_train_cv.py
python 05_optimize_thresholds.py
python 06_predict.py path/to/new_images/
```

## 入力データ

- **CSV 管理**: `filename,grade` の列を持つ CSV（列名は config で変更可）。グレードが 1〜5 なら自動で 0〜4 に変換。
- **フォルダ管理**: `SOURCE_CSV = None` にし、`IMG_DIR/0/*.png … IMG_DIR/4/*.png` の構成。

## 評価に関する注意

- `05` の **QWK (最適閾値)** は同じ OOF で閾値を決めているため楽観的です。
  併せて出力される **入れ子推定 QWK**（Fold k 以外で閾値最適化 → Fold k に適用）が厳密な値です。
- `INNER_VAL_RATIO = 0.0`（デフォルト）は、ロードマップ通り検証 80 枚で Early Stopping します。
  完全に未知データとして評価したい場合は `0.15` 程度に設定してください。
