import os
import shutil
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision.models import resnet18

from cem.train.training import train_end_to_end_model
from cem.train.evaluate import evaluate_cbm
import cem.train.utils as utils
import pytorch_lightning as pl


def main():
    # テスト結果の一時保存用フォルダ
    result_dir = "./tmp_recem_test_results"
    os.makedirs(result_dir, exist_ok=True)

    # 1. 小規模ダミーデータローダーの準備
    n_samples = 32
    n_concepts = 6
    n_tasks = 2
    x = torch.randn(n_samples, 3, 64, 64)
    y = torch.randint(0, n_tasks, (n_samples,)).long()
    c = torch.randint(0, 2, (n_samples, n_concepts)).float()

    train_dl = DataLoader(TensorDataset(x, y, c), batch_size=16, shuffle=True)
    val_dl = DataLoader(TensorDataset(x, y, c), batch_size=16, shuffle=False)
    test_dl = DataLoader(TensorDataset(x, y, c), batch_size=16, shuffle=False)

    # 2. training.py 用の config 辞書を作成
    config = {
        "architecture": "ConceptEmbeddingModel",
        "emb_size": 8,
        "c_extractor_arch": utils.wrap_pretrained_model(resnet18, pretrain_model=False),
        "optimizer": "adam",
        "learning_rate": 0.001,
        "weight_decay": 4e-05,
        "concept_loss_weight": 1.0,
        "task_loss_weight": 1.0,
        "training_intervention_prob": 0.25,
        "max_epochs": 2,               # ミニテストのため2エポック
        "concept_warmup_epochs": 1,    # ウォームアップ動作も1エポック確認
        "check_val_every_n_epoch": 1,
        "num_workers": 0,
        "num_load_workers": 0,
        # RECEM用ハイパーパラメータ
        "beta_max": 0.5,
        "beta_hsic": 0.1,
        "lambda_dis": 0.1,
        "lambda_rec": 0.1,
        "grl_lambda": 1.0,
        "ema_momentum": 0.9,
        # construct_model が必須で参照するキー
        "weight_loss": False,
        "shared_prob_gen": True,
        "embedding_activation": "leakyrelu",
        "c2y_layers": [],                     # 分類器の中間層（空リストで線形層）
        "sigmoidal_prob": True,
        "sigmoidal_extra_capacity": False,
        "extra_dims": 0,
        "bottleneck_nonlinear": None
    }

    # 3. 既存の学習パイプライン (train_end_to_end_model) を実行
    print("--- train_end_to_end_model の実行テスト ---")
    model, train_eval_results = train_end_to_end_model(
        n_concepts=n_concepts,
        n_tasks=n_tasks,
        config=config,
        train_dl=train_dl,
        val_dl=val_dl,
        run_name="RECEM_MiniTest",
        result_dir=result_dir,
        split=0,
        rerun=True,
        save_model=True,
    )

    # 4. テストデータでの評価 (evaluate_cbm) を実行
    print("\n--- evaluate_cbm の実行テスト ---")
    eval_trainer = pl.Trainer(accelerator="auto", devices=1, logger=False)
    test_results = evaluate_cbm(
        model=model,
        trainer=eval_trainer,
        config=config,
        run_name="RECEM_MiniTest",
        test_dl=test_dl,
        dl_name="test",
        rerun=True,
    )
    print("テストデータの評価結果:", test_results)

    # 確認が終わったら一時フォルダを削除（残したい場合はコメントアウト）
    shutil.rmtree(result_dir)
    print("\n パイプライン統合テストが正常に完了しました！")


if __name__ == "__main__":
    main()