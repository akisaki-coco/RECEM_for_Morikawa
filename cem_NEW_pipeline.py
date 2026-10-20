"""
cem_NEW_pipeline.py
===================
独自データセット（CSV管理）向け CEM / RECEM 学習・評価パイプライン

【データセット仕様】
- クラスラベル: あり/なし の2値分類 (n_tasks = 2: 0=なし, 1=あり)
- コンセプトラベル: グレード1〜5 (順序尺度)
- アプローチB (Ordinal Encoding):
    グレード1〜5を 4つの累積バイナリフラグ (Grade>=2, >=3, >=4, >=5) に展開
    例:
      Grade 1 -> [0, 0, 0, 0]
      Grade 2 -> [1, 0, 0, 0]
      Grade 3 -> [1, 1, 0, 0]
      Grade 4 -> [1, 1, 1, 0]
      Grade 5 -> [1, 1, 1, 1]
    元コンセプト数が M 個の場合、展開後の概念数 n_concepts = M * 4 となります。
"""

import os
import time
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet34
from PIL import Image
import pandas as pd
import numpy as np
import pytorch_lightning as pl

from cem.train.training import train_end_to_end_model
from cem.train.evaluate import evaluate_cbm
import cem.train.utils as utils


# ==============================================================================
# 1. カスタム Dataset クラス（アプローチ B: 順序バイナリ展開）
# ==============================================================================
class OrdinalConceptDataset(Dataset):
    """
    CSVから画像パス、クラスラベル、およびグレード1〜5のコンセプトラベルを読み込み、
    順序バイナリ（>=2, >=3, >=4, >=5）に展開して (image, y, c) を返す Dataset クラス。
    """
    def __init__(
        self,
        df,
        img_dir="",
        img_col="image_path",
        label_col="class_label",
        concept_cols=None,
        transform=None,
    ):
        super().__init__()
        self.img_dir = img_dir
        self.transform = transform
        self.img_col = img_col
        self.label_col = label_col
        self.concept_cols = concept_cols

        # (1) 画像パスのリスト
        self.img_paths = df[img_col].astype(str).tolist()

        # (2) クラスラベルの数値化 (0: なし, 1: あり)
        raw_y = df[label_col].tolist()
        y_list = []
        for val in raw_y:
            if isinstance(val, (str, bool)):
                # "あり", "positive", "1", True などを 1、それ以外を 0 に統一
                y_list.append(1 if str(val).lower() in ["1", "true", "yes", "positive", "あり"] else 0)
            else:
                y_list.append(int(val))
        self.y_data = torch.tensor(y_list, dtype=torch.long)

        # (3) コンセプトラベルの順序バイナリ展開 (Ordinal Encoding)
        # grades の shape: [N, M] (N: サンプル数, M: 元コンセプト数)
        grades = df[concept_cols].values.astype(np.float32)
        num_samples, num_orig_concepts = grades.shape

        # 各グレード閾値 (2, 3, 4, 5) に対する 0/1 フラグ行列を作成
        flags = [
            (grades >= 2).astype(np.float32),
            (grades >= 3).astype(np.float32),
            (grades >= 4).astype(np.float32),
            (grades >= 5).astype(np.float32),
        ]
        # コンセプトごとに [c0_ge2, c0_ge3, c0_ge4, c0_ge5, c1_ge2, ...] と並ぶように結合
        stacked = np.stack(flags, axis=2)  # shape: [N, M, 4]
        c_all = stacked.reshape(num_samples, num_orig_concepts * 4)  # shape: [N, M * 4]
        self.c_data = torch.tensor(c_all, dtype=torch.float32)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        """
        DataLoader から各サンプルを取り出すメソッド。
        戻り値:
            image (torch.FloatTensor): [3, H, W]
            y (torch.LongTensor): スカラー (0 または 1)
            c (torch.FloatTensor): [n_concepts] (各要素は 0.0 または 1.0)
        """
        # 画像の読み込み
        img_path = self.img_paths[idx]
        if self.img_dir:
            img_path = os.path.join(self.img_dir, img_path)
        image = Image.open(img_path).convert("RGB")

        # 前処理 (Resize, Normalization 等)
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
    各展開後コンセプトについて、正例数と負例数の比率を計算します。
    imbalance_j = (全サンプル数 - 正例数) / max(正例数, 1)
    これは BCEWithLogitsLoss / BCELoss の重みとして利用されます。
    """
    total = len(dataset.c_data)
    ones = dataset.c_data.sum(dim=0).tolist()
    imbalance = [(total - o) / max(o, 1.0) for o in ones]
    return imbalance


# ==============================================================================
# 3. データローダー構築関数
# ==============================================================================
def build_dataloaders(
    train_csv,
    val_csv,
    test_csv,
    img_dir="",
    img_col="image_path",
    label_col="class_label",
    concept_cols=None,
    batch_size=32,
    num_workers=4,
    img_size=224,
):
    """
    CSVファイルを読み込み、Train/Val/Testの各 DataLoader と学習に必要なメタ情報を生成します。
    """
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    # コンセプト列が指定されていない場合、画像パス列とクラス列以外を自動検出
    if concept_cols is None:
        concept_cols = [col for col in train_df.columns if col not in [img_col, label_col]]

    print(f"元コンセプト数: {len(concept_cols)} 列 ({concept_cols})")

    # 画像前処理の定義 (Backbone ResNet 用)
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

    # Dataset インスタンスの生成
    train_ds = OrdinalConceptDataset(train_df, img_dir, img_col, label_col, concept_cols, train_transform)
    val_ds = OrdinalConceptDataset(val_df, img_dir, img_col, label_col, concept_cols, eval_transform)
    test_ds = OrdinalConceptDataset(test_df, img_dir, img_col, label_col, concept_cols, eval_transform)

    # メタ情報の取得
    n_concepts = train_ds.c_data.shape[1]  # 展開後のコンセプト総数 (元コンセプト数 * 4)
    n_tasks = 2                          # あり/なし の 2クラス分類
    imbalance = compute_concept_imbalance(train_ds)

    print(f"展開後コンセプト数 (n_concepts): {n_concepts}")
    print(f"タスク分類クラス数 (n_tasks): {n_tasks}")

    # DataLoader の生成
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_dl, val_dl, test_dl, imbalance, n_concepts, n_tasks


# ==============================================================================
# 4. メインパイプライン (学習 & 評価)
# ==============================================================================
def main():
    # ----------------------------------------------------
    # 設定セクション: お手元のデータ環境に合わせてパス等を指定してください
    # ----------------------------------------------------
    MODE = "RECEM"  # "RECEM" または "CEM"

    TRAIN_CSV = "data/train.csv"          # 学習用CSVパス
    VAL_CSV = "data/val.csv"              # 検証用CSVパス
    TEST_CSV = "data/test.csv"            # テスト用CSVパス
    IMG_DIR = "data/images/"              # 画像保存先ルートディレクトリ (CSVのパスが絶対パスなら "" で可)

    IMG_COL = "image_path"                # 画像ファイル名の列名
    LABEL_COL = "class_label"             # クラス分類ラベルの列名 (あり/なし)
    CONCEPT_COLS = None                   # 特定の列のみ使う場合は ["c1", "c2", ...] (Noneで自動検出)

    BATCH_SIZE = 32
    NUM_WORKERS = 4
    MAX_EPOCHS = 100

    result_dir = f"./results/custom_{MODE.lower()}_{int(time.time())}"
    os.makedirs(result_dir, exist_ok=True)

    # 1. データローダーとメタ情報の準備
    print("--- データのロードと前処理 ---")
    train_dl, val_dl, test_dl, imbalance, n_concepts, n_tasks = build_dataloaders(
        train_csv=TRAIN_CSV,
        val_csv=VAL_CSV,
        test_csv=TEST_CSV,
        img_dir=IMG_DIR,
        img_col=IMG_COL,
        label_col=LABEL_COL,
        concept_cols=CONCEPT_COLS,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    # 2. モデルおよび学習設定 (config)
    is_recem = (MODE == "RECEM")

    config = {
        "architecture": "RECEM" if is_recem else "ConceptEmbeddingModel",
        "emb_size": 8,
        "c_extractor_arch": utils.wrap_pretrained_model(resnet34, pretrain_model=True),

        # 学習ハイパーパラメータ
        "optimizer": "sgd",
        "learning_rate": 0.005,
        "weight_decay": 4e-05,
        "momentum": 0.9,
        "lr_scheduler_factor": 0.1,
        "lr_scheduler_patience": 10,
        "concept_loss_weight": 1.0,
        "task_loss_weight": 1.0,
        "training_intervention_prob": 0.25,
        "max_epochs": MAX_EPOCHS,
        "concept_warmup_epochs": 0,
        "check_val_every_n_epoch": 5,
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

    # 再開用チェックポイントのパス (新規学習時は None)
    resume_ckpt = None

    # 3. 学習パイプラインの実行
    print(f"\n--- {MODE} の学習開始 ---")
    model, train_eval_results = train_end_to_end_model(
        n_concepts=n_concepts,
        n_tasks=n_tasks,
        config=config,
        train_dl=train_dl,
        val_dl=val_dl,
        run_name=f"{MODE}_CUSTOM_{int(time.time())}",
        result_dir=result_dir,
        split=0,
        imbalance=imbalance,
        rerun=True,
        save_model=True,
        gradient_clip_val=1.0,
        ckpt_path=resume_ckpt,
        enable_checkpointing=True,
    )

    # 4. テストデータでの評価
    print("\n--- テストデータでの評価 (evaluate_cbm) ---")
    eval_trainer = pl.Trainer(accelerator="auto", devices=1, logger=False)
    test_results = evaluate_cbm(
        model=model,
        trainer=eval_trainer,
        config=config,
        run_name=f"{MODE}_CUSTOM_{int(time.time())}",
        test_dl=test_dl,
        dl_name="test",
        rerun=True,
    )

    print("\n" + "=" * 10 + " テスト結果 " + "=" * 10)
    for k, v in test_results.items():
        print(f"{k}: {v * 100:.2f}%" if isinstance(v, (int, float)) else f"{k}: {v}")


if __name__ == "__main__":
    main()
