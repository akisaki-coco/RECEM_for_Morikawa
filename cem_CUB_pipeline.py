from torchvision.models import resnet34
import time
import os
import shutil
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision.models import resnet18

from cem.train.training import train_end_to_end_model
from cem.train.evaluate import evaluate_cbm
import cem.train.utils as utils
import pytorch_lightning as pl

from cem.data.CUB200.cub_loader import generate_data


def main():
    # テスト結果の一時保存用フォルダ
    MODE = "RECEM"
    result_dir = f"./results/cub_{MODE.lower()}_{int(time.time())}"
    os.makedirs(result_dir, exist_ok=True)

    # # 1. 小規模ダミーデータローダーの準備
    # n_samples = 32
    # n_concepts = 6
    # n_tasks = 2
    # x = torch.randn(n_samples, 3, 64, 64)
    # y = torch.randint(0, n_tasks, (n_samples,)).long()
    # c = torch.randint(0, 2, (n_samples, n_concepts)).float()

    # train_dl = DataLoader(TensorDataset(x, y, c), batch_size=16, shuffle=True)
    # val_dl = DataLoader(TensorDataset(x, y, c), batch_size=16, shuffle=False)
    # test_dl = DataLoader(TensorDataset(x, y, c), batch_size=16, shuffle=False)

    # 1. CUBデータセットのロード
    data_config = {
        "dataset": "cub",
        "root_dir": "cem/data/CUB200/",
        "batch_size": 64,
        "num_workers": 4,
        "weight_loss": True,
        "sampling_percent": 1,
        "sampling_groups": False
    }

    train_dl, val_dl, test_dl, imbalance, (n_concepts, n_tasks, concept_group_map) = generate_data(
        config = data_config,
        root_dir = "cem/data/CUB200/",
        seed = 42,
        output_dataset_vars = True
    )

    print("n_concepts: ", n_concepts)
    print("n_tasks: ", n_tasks)
    print("concept_group_map: ", concept_group_map)
    


    # 2. training.py 用の config 辞書を作成
    is_recem = (MODE == "RECEM")

    config = {
        "architecture": "RECEM" if is_recem else "ConceptEmbeddingModel",
        "emb_size": 8,
        "c_extractor_arch": utils.wrap_pretrained_model(resnet34, pretrain_model=True),
        # CEMの学習設定
        "optimizer": "sgd",
        "learning_rate": 0.005,
        "weight_decay": 4e-05,
        "momentum": 0.9,
        "lr_scheduler_factor": 0.1,
        "lr_scheduler_patience": 10,
        "concept_loss_weight": 1.0,
        "task_loss_weight": 1.0,
        "training_intervention_prob": 0.25,
        "max_epochs": 150,               # ミニテストのため2エポック
        "concept_warmup_epochs": 0,    # ウォームアップ動作も1エポック確認
        "check_val_every_n_epoch": 5,
        "num_workers": 4,
        "num_load_workers": 4,
        "dataset_config": data_config,

        "early_stopping_monitor": "val_loss",
        "early_stopping_best_model": True,
        "early_stopping_mode": "min",
        
        
        # RECEM用ハイパーパラメータ
        "beta_max": 0.5 if is_recem else 0,
        "beta_hsic": 0.1 if is_recem else 0,
        "lambda_dis": 0.1 if is_recem else 0,
        "lambda_rec": 0.1 if is_recem else 0,
        "grl_lambda": 1.0 if is_recem else 0,
        "ema_momentum": 0.9 if is_recem else 0,
        # construct_model が必須で参照するキー
        "weight_loss": True,
        "shared_prob_gen": True,
        "embedding_activation": "leakyrelu",
        "c2y_layers": [],                     # 分類器の中間層（空リストで線形層）
        "sigmoidal_prob": True,
        "sigmoidal_extra_capacity": False,
        "extra_dims": 0,
        "bottleneck_nonlinear": None
    }

    # ----------------------------------------------------
    # 中断から再開したい場合は last.ckpt のパスを指定（新規開始時は None）
    # ----------------------------------------------------
    resume_ckpt = None
    # 例（中断後に再開したい時はコメントアウトを解除してパスを指定）:
    # resume_ckpt = "./results/cub_recem_1792423323/checkpoints_RECEM_CUB_1792423323_fold_1/last.ckpt"

    # 3. 既存の学習パイプライン (train_end_to_end_model) を実行
    print("--- train_end_to_end_model の実行テスト ---")
    model, train_eval_results = train_end_to_end_model(
        n_concepts=n_concepts,
        n_tasks=n_tasks,
        config=config,
        train_dl=train_dl,
        val_dl=val_dl,
        run_name=f"{MODE}_CUB_{int(time.time())}",
        result_dir=result_dir,
        split=0,
        imbalance=imbalance,
        rerun=True,
        save_model=True,
        gradient_clip_val=1.0,
        
        # ★ 再開用
        ckpt_path=resume_ckpt,
        enable_checkpointing=True,
    )

    # 4. テストデータでの評価 (evaluate_cbm) を実行
    print("\n--- evaluate_cbm の実行テスト ---")
    eval_trainer = pl.Trainer(accelerator="auto", devices=1, logger=False)
    test_results = evaluate_cbm(
        model=model,
        trainer=eval_trainer,
        config=config,
        run_name=f"{MODE}_CUB_{int(time.time())}",
        test_dl=test_dl,
        dl_name="test",
        rerun=True,
    )

    print("="*10 + "テスト結果" + "="*10)
    for k,v in test_results.items():
        print(f"{k}: {v*100:.2f}%" if isinstance(v, (int, float)) else f"{k}: {v}")

if __name__ == "__main__":
    main()