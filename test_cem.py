import torch
from cem.models.cem import ConceptEmbeddingModel

# 1. ダミーのハイパーパラメータとモデルの準備
n_concepts = 10
n_tasks = 2
batch_size = 4

# ConceptEmbeddingModelのインスタンス化（RECEM用パラメータが追加されているか確認）
model = ConceptEmbeddingModel(
    n_concepts=n_concepts,
    n_tasks=n_tasks,
    emb_size=16,
    beta_max=0.5,
    beta_hsic=0.1,
    lambda_dis=0.1,
    lambda_rec=0.1,
)

model.train() # 学習モードに設定（GRLや再構成、HSICが有効になるか）

# 2. ダミーデータの作成 (画像入力 x, 概念ラベル c, タスクラベル y)
dummy_x = torch.randn(batch_size, 3, 224, 224)
dummy_c = torch.randint(0, 2, (batch_size, n_concepts)).float()
dummy_y = torch.randint(0, n_tasks, (batch_size,))

# 3. 順伝播と損失計算（_run_step）のテスト
print("--- 順伝播および損失計算のテスト中 ---")
try:
    loss, result = model._run_step((dummy_x, dummy_y, dummy_c), batch_idx=0, train=True)
    print(" 成功！ 順伝播およびRECEMの損失計算が正常に完了しました。")
    print(f"   - 合計Loss: {loss.item():.4f}")
    print(f"   - ログ出力された指標: {list(result.keys())}")
except Exception as e:
    print(" エラーが発生しました。実装を確認してください。")
    raise e