"""
config.py
=========
UNI2-h 特徴量 + 軽量MLP (CORAL) による病理5段階分類プロジェクトの共通設定。

全スクリプト (01〜05) はこのファイルの値を参照します。
データの置き場所・列名などは、まずここを自分の環境に合わせて書き換えてください。
"""
from pathlib import Path

# このファイルが置かれている UNI2-h/ ディレクトリ
ROOT_DIR = Path(__file__).resolve().parent

# ==============================================================================
# フェーズ 1: データとラベルの紐付け
# ==============================================================================
# 入力方法は 2 通りから選べます。
#   (A) CSV 管理  : SOURCE_CSV に「画像ファイル名」と「グレード」の列を持つ CSV を指定
#   (B) フォルダ管理: SOURCE_CSV = None とし、IMG_DIR/<grade>/xxx.png の構成にする
#                   （フォルダ名は "0"〜"4" や "grade_1" など、数字を含んでいれば OK）
SOURCE_CSV = ROOT_DIR / "data" / "source_labels.csv"
IMG_DIR = ROOT_DIR / "data" / "images"          # 画像ファイルのルート（CSV の filename はここからの相対パス）
FILENAME_COL = "filename"                       # CSV の画像ファイル名列
GRADE_COL = "grade"                             # CSV の正解グレード列

NUM_CLASSES = 5                                 # グレード数（0〜4）
# 元データのグレードから引く値。None の場合は自動判定（1〜5 なら 1 を引いて 0〜4 に変換）
GRADE_OFFSET = None
IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
EXPECTED_IMG_SIZE = (900, 900)                  # 想定画像サイズ（違うものは警告のみ）

# フェーズ 1 の出力（以降のフェーズはこの CSV を読み込む）
DATA_DIR = ROOT_DIR / "data"
LABELS_CSV = DATA_DIR / "labels.csv"            # 列: slide_id, image_path, grade
CLASS_DIST_CSV = DATA_DIR / "class_distribution.csv"

# ==============================================================================
# フェーズ 2: UNI2-h による特徴量抽出
# ==============================================================================
UNI_MODEL_NAME = "hf-hub:MahmoodLab/UNI2-h"
PATCH_SIZE = 14
INPUT_SIZE = 896                                # 14 の倍数 (896 / 14 = 64 → 64x64 = 4096 トークン)
RESIZE_MODE = "resize"                          # "resize": 900→896 に縮小 / "center_crop": 中央 896 を切り出し
FEATURE_DIM = 1536
FEATURE_DIR = ROOT_DIR / "features"             # slide_001.pt などを保存
USE_AMP = True                                  # CUDA 環境で bf16/fp16 推論を使う（MPS/CPU では無視）
OVERWRITE_FEATURES = False                      # True なら既存の .pt も再計算

# ==============================================================================
# フェーズ 3: 軽量分類モデル (MLP + CORAL)
# ==============================================================================
HIDDEN_DIM = 512
DROPOUT = 0.4                                   # 0.3〜0.5 推奨
STANDARDIZE_FEATURES = True                     # 訓練Foldの平均/標準偏差で特徴量を標準化（Fold内で計算するためリークなし）

# ==============================================================================
# フェーズ 4: 5-Fold 交差検証
# ==============================================================================
N_FOLDS = 5
SEED = 42
LR = 1e-4                                       # 1e-4 または 5e-5
WEIGHT_DECAY = 0.01
BATCH_SIZE = 32
MAX_EPOCHS = 500
PATIENCE = 30                                   # 検証損失が改善しないエポック数で早期終了
MIN_DELTA = 1e-4                                # 改善とみなす最小の損失減少量
# 早期終了の監視用データの取り方
#   0.0     : ロードマップ通り、各Foldの検証80枚で early stopping（OOF評価がわずかに楽観的になる）
#   0.1〜0.2: 訓練320枚の一部を early stopping 専用に切り出し、検証80枚は完全な未知データとして扱う
INNER_VAL_RATIO = 0.0
USE_CLASS_WEIGHTS = False                       # True: クラス不均衡に応じてサンプル重み付き損失を使う

OUTPUT_DIR = ROOT_DIR / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"               # fold1.pt 〜 fold5.pt
OOF_CSV = OUTPUT_DIR / "oof_predictions.csv"
FOLD_SUMMARY_CSV = OUTPUT_DIR / "fold_summary.csv"
HISTORY_JSON = OUTPUT_DIR / "train_history.json"

# ==============================================================================
# フェーズ 5: 閾値最適化と最終評価
# ==============================================================================
INIT_THRESHOLDS = [0.5, 1.5, 2.5, 3.5]
NM_INITIAL_STEP = 0.25                          # Nelder-Mead 初期シンプレックスの幅（QWKは階段関数のため大きめに）
NM_N_RESTARTS = 10                              # 初期値を少しずらした再探索の回数（0 で初期値のみ）
THRESHOLDS_JSON = OUTPUT_DIR / "thresholds.json"
EVAL_DIR = OUTPUT_DIR / "evaluation"
