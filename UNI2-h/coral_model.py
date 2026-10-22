"""
coral_model.py
==============
フェーズ 3: 軽量分類モデル（MLP + CORAL Layer）と CORAL Loss。

CORAL (COnsistent RAnk Logits; Cao et al., 2020)
    K=5 段階の順序分類を、K-1=4 個の二値タスク「grade > k ?」(k=0,1,2,3) に分解する。
    4 タスクは重み w を共有し、バイアス b_k だけが独立:
        logit_k = w・h + b_k
    重み共有により、各ハードルの確率が順序を保ちやすく（rank consistency）、
    少数データでもパラメータ数が少なく過学習しにくい。

ネットワーク構成:
    [B, 1536] → Linear(1536→512) → LayerNorm → ReLU → Dropout
              → Linear(512→1, bias=False) + [b1, b2, b3, b4]  →  [B, 4] のロジット
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CoralLayer(nn.Module):
    """共有重み Linear(in→1) と K-1 個の独立バイアスを持つ CORAL 出力層。"""

    def __init__(self, in_features: int, num_classes: int = 5):
        super().__init__()
        self.num_tasks = num_classes - 1
        self.fc = nn.Linear(in_features, 1, bias=False)
        # 初期バイアスは降順 (例: [1.0, 0.75, 0.5, 0.25]) にして、学習開始時点から順序を満たす
        init_bias = torch.arange(self.num_tasks, 0, -1, dtype=torch.float32) / self.num_tasks
        self.bias = nn.Parameter(init_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x) + self.bias  # [B, 1] + [K-1] → [B, K-1]


class CoralMLP(nn.Module):
    """UNI2-h 特徴量 [B, 1536] から 4 つのハードルロジット [B, 4] を出力する MLP。"""

    def __init__(self, in_dim: int = 1536, hidden_dim: int = 512, num_classes: int = 5, dropout: float = 0.4):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.coral = CoralLayer(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.coral(self.backbone(x))


# ------------------------------------------------------------------------------
# ラベル変換・損失・スコア化
# ------------------------------------------------------------------------------
def label_to_levels(labels: torch.Tensor, num_classes: int = 5) -> torch.Tensor:
    """
    グレード (0〜K-1) を K-1 次元の累積バイナリに変換する。
        0 → [0, 0, 0, 0]
        1 → [1, 0, 0, 0]
        2 → [1, 1, 0, 0]
        3 → [1, 1, 1, 0]
        4 → [1, 1, 1, 1]
    """
    labels = labels.long().view(-1, 1)
    ks = torch.arange(num_classes - 1, device=labels.device).view(1, -1)
    return (labels > ks).float()


def coral_loss(logits: torch.Tensor, levels: torch.Tensor, sample_weights: torch.Tensor = None) -> torch.Tensor:
    """
    CORAL Loss: 各ハードルの BCEWithLogits をタスク方向に合計し、バッチ方向に平均する。
        logits : [B, K-1]
        levels : [B, K-1] (label_to_levels の出力)
        sample_weights : [B] (任意。クラス不均衡対策の重み)
    """
    per_task = F.binary_cross_entropy_with_logits(logits, levels, reduction="none")  # [B, K-1]
    per_sample = per_task.sum(dim=1)                                                 # [B]
    if sample_weights is not None:
        return (per_sample * sample_weights).sum() / sample_weights.sum()
    return per_sample.mean()


def logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    """各ハードル P(grade > k) の確率 [B, K-1]。"""
    return torch.sigmoid(logits)


def logits_to_score(logits: torch.Tensor) -> torch.Tensor:
    """4 つのハードル確率の和 = 期待グレードに相当する連続スコア (0.0〜4.0)。"""
    return torch.sigmoid(logits).sum(dim=1)


def logits_to_label(logits: torch.Tensor) -> torch.Tensor:
    """CORAL 標準の予測（確率 > 0.5 のハードル数）。閾値最適化前のベースライン。"""
    return (torch.sigmoid(logits) > 0.5).sum(dim=1)


if __name__ == "__main__":
    # 動作確認: python coral_model.py
    torch.manual_seed(0)
    model = CoralMLP()
    x = torch.randn(8, 1536)
    y = torch.tensor([0, 1, 2, 3, 4, 0, 2, 4])
    logits = model(x)
    levels = label_to_levels(y)
    assert logits.shape == (8, 4), logits.shape
    assert levels.tolist()[3] == [1, 1, 1, 0]
    loss = coral_loss(logits, levels)
    loss.backward()
    score = logits_to_score(logits)
    assert ((score >= 0) & (score <= 4)).all()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"logits {tuple(logits.shape)}  loss={loss.item():.4f}  params={n_params:,}")
    print("score:", [round(s, 3) for s in score.tolist()])
    print("OK")
