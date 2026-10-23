"""
config.py
=========
UNI2-h 特徴量 + 軽量MLP による「10 概念のグレード (1〜5) 推定」プロジェクトの共通設定。
各概念の予測には CORAL Loss（順序回帰）を使います。

全スクリプト (00〜06) はこのファイルの値を参照します。
データの置き場所・列名などは、まずここを自分の環境に合わせて書き換えてください。
"""
from pathlib import Path

# このファイルが置かれている UNI2-h/ ディレクトリと、リポジトリのルート
ROOT_DIR = Path(__file__).resolve().parent
REPO_DIR = ROOT_DIR.parent

# ==============================================================================
# フェーズ 1: データとラベルの紐付け
# ==============================================================================
# 概念CSV: 列 = filename, 概念1, 概念2, ... 概念10（各セルはグレード 1〜5）
CONCEPT_CSV = REPO_DIR / "dataset/ccbm_20251109_annotated/ratings_v1.csv"
# 画像フォルダ（CSV の filename はここからの相対パス）
IMG_DIR = REPO_DIR / "dataset/ccbm_20251109_annotated/data"
FILENAME_COL = "filename"
# 使う概念の列名。None なら CONCEPT_CSV の filename 以外の全列を使う
CONCEPT_COLS = None
# 除外する概念のリスト（学習・評価から除外）
EXCLUDE_CONCEPTS = ["Palate ridge", "Erosion"]

# 層化 K-Fold の分割に使うラベル（任意）。
# クラス CSV (filename, dataset, class) があれば、その class 列で層化する。None なら「全概念の平均グレードの四捨五入」で層化。
CLASS_CSV = REPO_DIR / "dataset/ccbm_20251109_annotated/filename_mapping.csv"
CLASS_COL = "class"

NUM_GRADES = 5                                  # 各概念のグレード数 (1〜5)
GRADE_MIN = 1                                   # 最小グレード（1〜5 表記）
IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
EXPECTED_IMG_SIZE = (900, 900)                  # 想定画像サイズ（違うものは警告のみ）

# フェーズ 1 の出力（以降のフェーズはこの CSV を読み込む）
DATA_DIR = ROOT_DIR / "data"
LABELS_CSV = DATA_DIR / "labels.csv"            # 列: slide_id, image_path, strat_label, grade_<概念名>...
CLASS_DIST_CSV = DATA_DIR / "concept_distribution.csv"
GRADE_PREFIX = "grade_"                         # labels.csv 内の概念列の接頭辞

# ==============================================================================
# フェーズ 2: UNI2-h による特徴量抽出
# ==============================================================================
UNI_MODEL_NAME = "hf-hub:MahmoodLab/UNI2-h"
PATCH_SIZE = 14
INPUT_SIZE = 896                                # 14 の倍数 (896 / 14 = 64 → 64x64 = 4096 トークン)
RESIZE_MODE = "center_crop"                     # "resize": 900→896 に縮小 / "center_crop": 短辺896に縮小して中央896x896をクロップ (アスペクト比保持)
FEATURE_DIM = 1536
FEATURE_DIR = ROOT_DIR / "features"             # slide_001.pt などを保存
USE_AMP = True                                  # CUDA 環境で bf16/fp16 推論を使う
OVERWRITE_FEATURES = False                      # True なら既存の .pt も再計算

# ==============================================================================
# フェーズ 3: 軽量マルチ概念 CORAL モデル
# ==============================================================================
HIDDEN_DIM = 512
DROPOUT = 0.4                                   # 0.3〜0.5 推奨
STANDARDIZE_FEATURES = True                     # 訓練Foldの平均/標準偏差で特徴量を標準化（リークなし）

# ==============================================================================
# フェーズ 4: 5-Fold 交差検証
# ==============================================================================
N_FOLDS = 5
SEED = 42
LR = 1e-4
WEIGHT_DECAY = 0.01
BATCH_SIZE = 32
MAX_EPOCHS = 500
PATIENCE = 30                                   # 検証損失が改善しないエポック数で早期終了
MIN_DELTA = 1e-4
# 早期終了の監視用データの取り方
#   0.0     : 各Foldの検証データで early stopping（OOF評価がわずかに楽観的になる）
#   0.1〜0.2: 訓練データの一部を early stopping 専用に切り出す
INNER_VAL_RATIO = 0.0

OUTPUT_DIR = ROOT_DIR / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"               # fold1.pt 〜 fold5.pt
OOF_CSV = OUTPUT_DIR / "oof_predictions.csv"
OOF_PROBS_NPY = OUTPUT_DIR / "oof_probs.npy"    # [N, 概念数, 4] のハードル確率
FOLD_SUMMARY_CSV = OUTPUT_DIR / "fold_summary.csv"
HISTORY_JSON = OUTPUT_DIR / "train_history.json"

# ==============================================================================
# フェーズ 5: 閾値最適化と最終評価（概念ごとに実施）
# ==============================================================================
INIT_THRESHOLDS = [1.5, 2.5, 3.5, 4.5]          # 連続スコア (1.0〜5.0) を 1〜5 に区切る初期閾値
NM_INITIAL_STEP = 0.25                          # Nelder-Mead 初期シンプレックスの幅
NM_N_RESTARTS = 10                              # 初期値を少しずらした再探索の回数
THRESHOLDS_JSON = OUTPUT_DIR / "thresholds.json"
EVAL_DIR = OUTPUT_DIR / "evaluation"
