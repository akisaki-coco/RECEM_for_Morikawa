import torch
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from torchvision.models import resnet18

from cem.models.cem import ConceptEmbeddingModel
import cem.train.utils as utils


def main():
    # 再現性のためのシード固定
    pl.seed_everything(42)

    # 1. 小規模テスト用のハイパーパラメータ設定
    n_samples = 32      # サンプル数（バッチサイズ16 × 2ステップ分）
    batch_size = 16     # HSIC計算のため16以上を推奨
    n_concepts = 6      # 概念数 K
    n_tasks = 3         # タスクのクラス数（多クラス分類）
    emb_size = 8        # 概念埋め込み次元 d

    # 2. ダミーデータの作成 ※順番は必ず (x, y, c) にします
    # x: 画像データ [N, 3, 64, 64] (テスト高速化のため64x64に縮小)
    x_dummy = torch.randn(n_samples, 3, 64, 64)
    # y: タスクラベル [N] (0 ~ n_tasks-1 の整数)
    y_dummy = torch.randint(0, n_tasks, (n_samples,)).long()
    # c: 概念ラベル [N, K] (0.0 または 1.0)
    c_dummy = torch.randint(0, 2, (n_samples, n_concepts)).float()

    dataset = TensorDataset(x_dummy, y_dummy, c_dummy)
    train_dl = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 3. RECEM (ConceptEmbeddingModel) のインスタンス化
    # テストを高速化するため、バックボーンに軽量な resnet18 を指定
    model = ConceptEmbeddingModel(
        n_concepts=n_concepts,
        n_tasks=n_tasks,
        emb_size=emb_size,
        c_extractor_arch=utils.wrap_pretrained_model(resnet18, pretrain_model=False),
        optimizer="adam",
        learning_rate=1e-3,
        # --- RECEM固有パラメータ ---
        beta_max=0.5,
        beta_hsic=0.1,
        lambda_dis=0.1,
        lambda_rec=0.1,
        grl_lambda=1.0,
        ema_momentum=0.9,
    )

    # 学習前のパラメータとバッファの状態を記録（更新確認用）
    init_dis_weight = model.disentangle_encoder[0].weight.clone().detach()
    init_rec_weight = model.reconstruct_decoder[0].weight.clone().detach()
    print("学習前の重心初期化状態:", model.semantic_mean_initialized.tolist())

    # 4. PyTorch Lightning Trainer で 2エポックだけ学習を実行
    trainer = pl.Trainer(
        max_epochs=2,
        accelerator="auto",
        devices=1,
        enable_checkpointing=False,
        logger=False,
        num_sanity_val_steps=0,
    )

    print("\n--- ミニ学習テスト開始 (2 Epochs) ---")
    trainer.fit(model, train_dl, val_dl)

    # 5. 学習後の検証チェック
    print("\n--- RECEM 動作検証結果 ---")
    # チェック①: 意味的重心バッファ (semantic_mean_pos) が更新されたか
    is_initialized = model.semantic_mean_initialized.all().item()
    mean_norm = torch.norm(model.semantic_mean_pos).item()
    print(f"[チェック1] 意味的重心バッファの初期化完了: {is_initialized} (L2ノルム: {mean_norm:.4f})")

    # チェック②: 分離エンコーダ (E_dis) の重みが更新されたか
    dis_diff = torch.norm(model.disentangle_encoder[0].weight.detach() - init_dis_weight).item()
    print(f"[チェック2] 分離エンコーダ (E_dis) の重み変化量: {dis_diff:.6f}")

    # チェック③: 再構成デコーダ (D) の重みが更新されたか
    rec_diff = torch.norm(model.reconstruct_decoder[0].weight.detach() - init_rec_weight).item()
    print(f"[チェック3] 再構成デコーダ (D) の重み変化量: {rec_diff:.6f}")

    assert is_initialized and mean_norm > 0, "エラー: 意味的重心バッファが更新されていません！"
    assert dis_diff > 0, "エラー: 分離エンコーダに勾配が流れていません！"
    assert rec_diff > 0, "エラー: 再構成デコーダに勾配が流れていません！"

    # 6. 推論（predict）モードでの動作チェック
    print("\n--- 推論 (Predict) モードのテスト ---")
    preds = trainer.predict(model, val_dl)
    c_sem, bottleneck, y_pred = preds[0][0], preds[0][1], preds[0][2]
    print(f"推論出力の形状 -> c_sem: {c_sem.shape}, bottleneck: {bottleneck.shape}, y_pred: {y_pred.shape}")
    print("\n すべての単体ミニ学習テストに合格しました！")


if __name__ == "__main__":
    main()