"""
cem_NEW_pipeline.py
===================
独自データセット（2つのCSVファイル管理）向け CEM / RECEM 学習・評価パイプライン

【入力CSVの形式】
1. クラス分類CSV:
   - 列: filename, dataset, class
   - filename: 画像ファイル名（例: sample001.jpg）
   - dataset : 'train' / 'val' / 'test' の分割指定（固定分割時に使用）
   - class   : あり(1) / なし(0) の2値

2. 概念分類CSV:
   - 列: filename, 概念1, 概念2, ... , 概念10
   - filename: 画像ファイル名
   - 概念[数字]: 各概念の強さを表すグレード（1〜5）

【アプローチB (Ordinal Encoding)】
各概念のグレード1〜5を 4つの累積バイナリフラグ (Grade>=2, >=3, >=4, >=5) に展開
例:
  Grade 1 -> [0, 0, 0, 0]
  Grade 2 -> [1, 0, 0, 0]
  Grade 3 -> [1, 1, 0, 0]
  Grade 4 -> [1, 1, 1, 0]
  Grade 5 -> [1, 1, 1, 1]
元概念が 10 個の場合、展開後の概念総数 n_concepts = 10 * 4 = 40 となります。

【検証モードの切り替え】
main() 内の `USE_KFOLD` フラグにより、以下の2通りを選択可能です:
- USE_KFOLD = False: CSV内の dataset 列 ('train'/'val'/'test') に基づく「固定分割学習」
- USE_KFOLD = True : 全データを対象とした「層化 K-分割交差検証 (Stratified K-Fold CV)」
"""

import os
import time
import datetime
import torch
import torch.multiprocessing

# Mac/Linux環境でのマルチプロセスによる Too many open files エラー対策
try:
    torch.multiprocessing.set_sharing_strategy('file_system')
except RuntimeError:
    pass

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet34
from PIL import Image
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from sklearn.model_selection import StratifiedKFold, train_test_split

from cem.train.training import train_end_to_end_model
from cem.train.evaluate import evaluate_cbm
import cem.train.utils as utils


# ==============================================================================
# 1. カスタム Dataset クラス（アプローチ B: 順序バイナリ展開）
# ==============================================================================
class OrdinalConceptDataset(Dataset):
    """
    DataFrameから画像パス、クラスラベル、およびグレード1〜5の概念ラベルを保持し、
    順序バイナリ（>=2, >=3, >=4, >=5）に展開して (image, y, c) を返す Dataset クラス。
    """
    def __init__(
        self,
        df,
        img_dir="",
        filename_col="filename",
        label_col="class",
        concept_cols=None,
        transform=None,
    ):
        super().__init__()
        self.img_dir = img_dir
        self.transform = transform
        self.filename_col = filename_col
        self.label_col = label_col
        self.concept_cols = concept_cols

        # (1) 画像ファイル名リスト
        self.img_paths = df[filename_col].astype(str).tolist()

        # (2) クラスラベルの数値化 (0: なし, 1: あり)
        raw_y = df[label_col].tolist()
        y_list = []
        for val in raw_y:
            if isinstance(val, (str, bool)):
                y_list.append(1 if str(val).strip().lower() in ["1", "true", "yes", "positive", "あり"] else 0)
            else:
                y_list.append(int(val))
        self.y_data = torch.tensor(y_list, dtype=torch.long)

        # (3) 概念ラベルの順序バイナリ展開 (Ordinal Encoding)
        # grades の shape: [N, M] (N: サンプル数, M: 元概念数)
        grades = df[concept_cols].values.astype(np.float32)
        num_samples, num_orig_concepts = grades.shape

        # 各グレード閾値 (2, 3, 4, 5) に対する 0/1 フラグ行列を作成
        flags = [
            (grades >= 2).astype(np.float32),
            (grades >= 3).astype(np.float32),
            (grades >= 4).astype(np.float32),
            (grades >= 5).astype(np.float32),
        ]
        # 概念ごとに [c0_ge2, c0_ge3, c0_ge4, c0_ge5, c1_ge2, ...] と並ぶように結合
        stacked = np.stack(flags, axis=2)  # shape: [N, M, 4]
        c_all = stacked.reshape(num_samples, num_orig_concepts * 4)  # shape: [N, M * 4]
        self.c_data = torch.tensor(c_all, dtype=torch.float32)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        """
        DataLoader からバッチを生成する際に呼び出されるメソッド。
        戻り値:
            image (torch.FloatTensor): [3, H, W]
            y (torch.LongTensor): スカラー (0 または 1)
            c (torch.FloatTensor): [n_concepts] (各要素は 0.0 または 1.0)
        """
        img_filename = self.img_paths[idx]
        img_path = os.path.join(self.img_dir, img_filename) if self.img_dir else img_filename
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        y = self.y_data[idx]
        c = self.c_data[idx]
        return image, y, c


# ==============================================================================
# 2. 不均衡比率 (imbalance) 計算関数
# ==============================================================================
def compute_concept_imbalance(dataset):
    """
    各展開後概念について、正例数と負例数の比率を計算します。
    imbalance_j = (全サンプル数 - 正例数) / max(正例数, 1)
    これは BCEWithLogitsLoss / BCELoss の正例重み (pos_weight) として利用されます。
    """
    total = len(dataset.c_data)
    ones = dataset.c_data.sum(dim=0).tolist()
    imbalance = [(total - o) / max(o, 1.0) for o in ones]
    return imbalance


# ==============================================================================
# 3. 前処理 (Transform) 取得関数
# ==============================================================================
def get_transforms(img_size=224):
    """ResNetバックボーン用の標準前処理を返します"""
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, eval_transform


# ==============================================================================
# 4. 固定分割用: 2つのCSVを結合して DataLoader を構築する関数
# ==============================================================================
def build_dataloaders_from_csv(
    class_csv,
    concept_csv,
    img_dir="",
    filename_col="filename",
    dataset_col="dataset",
    label_col="class",
    concept_cols=None,
    batch_size=32,
    num_workers=2,
    img_size=224,
):
    """
    クラス分類CSV と 概念分類CSV を filename をキーとして結合し、
    dataset 列 ('train', 'val', 'test') に基づいて分割して DataLoader を生成します。
    """
    # 1. CSVの読み込み
    print(f"Loading class CSV  : {class_csv}")
    class_df = pd.read_csv(class_csv)
    print(f"Loading concept CSV: {concept_csv}")
    concept_df = pd.read_csv(concept_csv)

    # 2. filename をキーに内部結合 (Inner Merge)
    merged_df = pd.merge(class_df, concept_df, on=filename_col, how="inner")
    print(f"結合後のデータ数: {len(merged_df)} 件")

    # 3. 概念列の決定 (未指定なら concept_df の filename 以外の全列を自動取得)
    if concept_cols is None:
        concept_cols = [c for c in concept_df.columns if c != filename_col]
    print(f"検出された元概念 ({len(concept_cols)} 項目): {concept_cols}")

    # 4. dataset 列による train / val / test への分割
    ds_series = merged_df[dataset_col].astype(str).str.strip().str.lower()
    train_df = merged_df[ds_series == "train"].copy()
    val_df = merged_df[ds_series.isin(["val", "valid", "validation"])].copy()
    test_df = merged_df[ds_series == "test"].copy()

    print(f"データ分割内訳 -> Train: {len(train_df)} 件, Val: {len(val_df)} 件, Test: {len(test_df)} 件")
    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError("train / val / test のいずれかのデータ件数が0件です。dataset 列の値を確認してください。")

    # 5. 画像前処理
    train_transform, eval_transform = get_transforms(img_size)

    # 6. Dataset インスタンスの生成
    train_ds = OrdinalConceptDataset(train_df, img_dir, filename_col, label_col, concept_cols, train_transform)
    val_ds = OrdinalConceptDataset(val_df, img_dir, filename_col, label_col, concept_cols, eval_transform)
    test_ds = OrdinalConceptDataset(test_df, img_dir, filename_col, label_col, concept_cols, eval_transform)

    # 7. メタ情報の計算
    n_concepts = train_ds.c_data.shape[1]  # 元概念数 * 4
    n_tasks = 2                           # あり(1) / なし(0) の2クラス分類
    imbalance = compute_concept_imbalance(train_ds)

    print(f"展開後概念数 (n_concepts): {n_concepts} (元概念 {len(concept_cols)} × 4段階)")
    print(f"タスク分類クラス数 (n_tasks): {n_tasks}")

    # 8. DataLoader の生成
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_dl, val_dl, test_dl, imbalance, n_concepts, n_tasks


# ==============================================================================
# 5. K-分割交差検証パイプライン (K-Fold Cross Validation)
# ==============================================================================
def run_kfold_pipeline(
    class_csv,
    concept_csv,
    img_dir="",
    filename_col="filename",
    label_col="class",
    concept_cols=None,
    n_splits=5,
    val_size=0.10,
    batch_size=32,
    num_workers=2,
    img_size=224,
    config=None,
    MODE="RECEM",
    result_dir="./results",
    seed=42,
):
    """
    全データを対象に層化 K-分割交差検証 (Stratified K-Fold CV) を実行します。
    各 Fold でモデルを学習・評価し、最終的に全 Fold の平均精度と標準偏差を出力します。
    """
    # 1. CSV読み込みと結合
    print(f"\n{'='*20} Stratified {n_splits}-Fold Cross Validation {'='*20}")
    print(f"Loading class CSV  : {class_csv}")
    class_df = pd.read_csv(class_csv)
    print(f"Loading concept CSV: {concept_csv}")
    concept_df = pd.read_csv(concept_csv)

    merged_df = pd.merge(class_df, concept_df, on=filename_col, how="inner")
    print(f"全データ件数: {len(merged_df)} 件")

    if concept_cols is None:
        concept_cols = [c for c in concept_df.columns if c != filename_col]
    print(f"検出された元概念 ({len(concept_cols)} 項目): {concept_cols}")

    # 2. 層化K分割のインデックス作成 (class の比率を均等に分割)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_all = merged_df[label_col].values
    splits = list(skf.split(merged_df, y_all))

    train_transform, eval_transform = get_transforms(img_size)

    all_fold_results = []

    # 3. 各 Fold の学習・評価ループ
    for fold, (train_val_idx, test_idx) in enumerate(splits):
        print(f"\n{'#'*25} Fold {fold + 1} / {n_splits} 開始 {'#'*25}")

        test_df = merged_df.iloc[test_idx].copy()
        train_val_df = merged_df.iloc[train_val_idx].copy()

        # train_val からさらに val_size (例: 10%) を検証用に分割
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size,
            stratify=train_val_df[label_col],
            random_state=seed + fold,
        )

        print(f"Fold {fold + 1} 内訳 -> Train: {len(train_df)} 件, Val: {len(val_df)} 件, Test: {len(test_df)} 件")

        # Dataset & DataLoader の生成
        train_ds = OrdinalConceptDataset(train_df, img_dir, filename_col, label_col, concept_cols, train_transform)
        val_ds = OrdinalConceptDataset(val_df, img_dir, filename_col, label_col, concept_cols, eval_transform)
        test_ds = OrdinalConceptDataset(test_df, img_dir, filename_col, label_col, concept_cols, eval_transform)

        n_concepts = train_ds.c_data.shape[1]
        n_tasks = 2
        imbalance = compute_concept_imbalance(train_ds)

        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

        # モデルの学習 (split=fold を渡して各Foldのモデルを分離保存)
        run_name = f"{MODE}_CUSTOM_fold{fold + 1}_{datetime.date.today()}"
        model, _ = train_end_to_end_model(
            n_concepts=n_concepts,
            n_tasks=n_tasks,
            config=config,
            train_dl=train_dl,
            val_dl=val_dl,
            run_name=run_name,
            result_dir=result_dir,
            split=fold,
            imbalance=imbalance,
            rerun=True,
            save_model=True,
            gradient_clip_val=1.0,
            enable_checkpointing=True,
            devices=1,
            accelerator="auto",
        )

        # テストデータでの評価
        eval_trainer = pl.Trainer(accelerator="auto", devices=1, logger=False)
        test_results = evaluate_cbm(
            model=model,
            trainer=eval_trainer,
            config=config,
            run_name=run_name,
            test_dl=test_dl,
            dl_name=f"test_fold_{fold + 1}",
            rerun=True,
        )

        print(f"--- Fold {fold + 1} テスト結果 ---")
        for k, v in test_results.items():
            print(f"  {k}: {v * 100:.2f}%" if isinstance(v, (int, float)) else f"  {k}: {v}")

        all_fold_results.append(test_results)

    # 4. 全 Fold のサマリー集計 (平均値 ± 標準偏差)
    print("\n" + "=" * 25 + f" {n_splits}-Fold 交差検証 最終サマリー " + "=" * 25)
    summary_metrics = ["y_accuracy", "y_auc", "y_f1", "c_accuracy", "c_auc", "c_f1"]
    for metric in summary_metrics:
        vals = [res[metric] for res in all_fold_results if metric in res and isinstance(res[metric], (int, float))]
        if vals:
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals))
            print(f"{metric:<15}: {mean_val * 100:.2f}% ± {std_val * 100:.2f}%")

    return all_fold_results


# ==============================================================================
# 6. メインパイプライン (学習 & 評価)
# ==============================================================================
def main():
    # ----------------------------------------------------
    # 設定セクション: お手元のファイルパス・パラメータを指定してください
    # ----------------------------------------------------
    MODE = "RECEM"  # "RECEM" または "CEM"

    # ★ ここで K-分割交差検証を使うかどうかを選択できます
    # False: CSVの dataset 列 ('train'/'val'/'test') を使った従来の固定分割学習
    # True : 全400枚を対象にした 5-Fold 交差検証 (Stratified K-Fold CV)
    USE_KFOLD = True
    N_SPLITS = 5                        # K-Fold 時の分割数 (例: 5)
    VAL_SIZE_FOR_KFOLD = 0.10           # K-Fold 時、Train+Val から Val に割り当てる比率 (10%)

    # CSVと画像フォルダのパス
    CLASS_CSV = "dataset/ccbm_20251109_annotated/filename_mapping.csv"        # 列: filename, dataset, class
    CONCEPT_CSV = "dataset/ccbm_20251109_annotated/ratings_v1.csv"            # 列: filename, 概念1, 概念2, ...
    IMG_DIR = "dataset/ccbm_20251109_annotated/data"                          # 画像フォルダのルートパス

    # 列名の設定
    FILENAME_COL = "filename"
    DATASET_COL = "dataset"
    LABEL_COL = "class"
    CONCEPT_COLS = None                 # None の場合は concept.csv の filename 以外の全列を自動取得

    # 学習パラメータ
    BATCH_SIZE = 32
    NUM_WORKERS = 2                     # Macでの安定動作のため 2 を推奨
    MAX_EPOCHS = 2                      # まずは2エポックで動作確認
    CHECK_VAL_EVERY_N_EPOCH = 1         # 1エポックごとに検証

    result_dir = f"./results/custom_{MODE.lower()}_{datetime.date.today()}"
    os.makedirs(result_dir, exist_ok=True)

    # ----------------------------------------------------
    # モデルおよび学習設定 (config)
    # ----------------------------------------------------
    is_recem = (MODE == "RECEM")

    config = {
        "architecture": "RECEM" if is_recem else "ConceptEmbeddingModel",
        "emb_size": 8,
        "c_extractor_arch": utils.wrap_pretrained_model(resnet34, pretrain_model=True),

        # 学習ハイパーパラメータ
        "optimizer": "sgd",
        "learning_rate": 0.001,
        "weight_decay": 4e-05,
        "momentum": 0.9,
        "lr_scheduler_factor": 0.1,
        "lr_scheduler_patience": 10,
        "concept_loss_weight": 1.0,
        "task_loss_weight": 1.0,
        "training_intervention_prob": 0.25,
        "max_epochs": MAX_EPOCHS,
        "concept_warmup_epochs": 0,
        "check_val_every_n_epoch": CHECK_VAL_EVERY_N_EPOCH,
        "num_workers": NUM_WORKERS,
        "num_load_workers": NUM_WORKERS,

        # Early Stopping 設定
        "early_stopping_monitor": "val_loss",
        "early_stopping_best_model": True,
        "early_stopping_mode": "min",

        # RECEM 特有のハイパーパラメータ
        "beta_max": 0.5 if is_recem else 0,
        "beta_hsic": 0.1 if is_recem else 0,
        "lambda_dis": 0.1 if is_recem else 0,
        "lambda_rec": 0.1 if is_recem else 0,
        "grl_lambda": 1.0 if is_recem else 0,
        "ema_momentum": 0.9 if is_recem else 0,

        # CEM 共通の必須キー
        "weight_loss": True,
        "shared_prob_gen": True,
        "embedding_activation": "leakyrelu",
        "c2y_layers": [],
        "sigmoidal_prob": True,
        "sigmoidal_extra_capacity": False,
        "extra_dims": 0,
        "bottleneck_nonlinear": None,
    }

    # ====================================================
    # 実行分岐: K-Fold 交差検証 or 通常の固定分割学習
    # ====================================================
    if USE_KFOLD:
        # ------------------------------------------------
        # パターン A: K-分割交差検証
        # ------------------------------------------------
        run_kfold_pipeline(
            class_csv=CLASS_CSV,
            concept_csv=CONCEPT_CSV,
            img_dir=IMG_DIR,
            filename_col=FILENAME_COL,
            label_col=LABEL_COL,
            concept_cols=CONCEPT_COLS,
            n_splits=N_SPLITS,
            val_size=VAL_SIZE_FOR_KFOLD,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            config=config,
            MODE=MODE,
            result_dir=result_dir,
            seed=42,
        )

    else:
        # ------------------------------------------------
        # パターン B: 従来の固定分割 (dataset 列による分割)
        # ------------------------------------------------
        # 1. データローダーとメタ情報の準備
        print("--- データのロードと結合・前処理 (固定分割) ---")
        train_dl, val_dl, test_dl, imbalance, n_concepts, n_tasks = build_dataloaders_from_csv(
            class_csv=CLASS_CSV,
            concept_csv=CONCEPT_CSV,
            img_dir=IMG_DIR,
            filename_col=FILENAME_COL,
            dataset_col=DATASET_COL,
            label_col=LABEL_COL,
            concept_cols=CONCEPT_COLS,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
        )

        resume_ckpt = None

        # 2. 学習パイプラインの実行
        print(f"\n--- {MODE} の学習開始 (固定分割) ---")
        model, train_eval_results = train_end_to_end_model(
            n_concepts=n_concepts,
            n_tasks=n_tasks,
            config=config,
            train_dl=train_dl,
            val_dl=val_dl,
            run_name=f"{MODE}_CUSTOM_{datetime.date.today()}",
            result_dir=result_dir,
            split=0,
            imbalance=imbalance,
            rerun=True,
            save_model=True,
            gradient_clip_val=1.0,
            ckpt_path=resume_ckpt,
            enable_checkpointing=True,
            devices=1,
            accelerator="auto",
        )

        # 3. テストデータでの評価
        print("\n--- テストデータでの評価 (evaluate_cbm) ---")
        eval_trainer = pl.Trainer(accelerator="auto", devices=1, logger=False)
        test_results = evaluate_cbm(
            model=model,
            trainer=eval_trainer,
            config=config,
            run_name=f"{MODE}_CUSTOM_{datetime.date.today()}",
            test_dl=test_dl,
            dl_name="test",
            rerun=True,
        )

        print("\n" + "=" * 10 + " テスト結果 " + "=" * 10)
        for k, v in test_results.items():
            print(f"{k}: {v * 100:.2f}%" if isinstance(v, (int, float)) else f"{k}: {v}")


if __name__ == "__main__":
    main()
