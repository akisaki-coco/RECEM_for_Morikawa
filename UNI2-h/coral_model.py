"""
coral_model.py
==============
フェーズ 3: 軽量マルチ概念モデル（共有MLP + 概念ごとの CORAL ヘッド）と CORAL Loss。

CORAL (COnsistent RAnk Logits; Cao et al., 2020)
    各概念のグレード (K=5 段階) を、K-1=4 個の二値タスク「grade >= k+1 ?」に分解する。
    1 つの概念内では重み w を 4 タスクで共有し、バイアス b_k だけが独立:
        logit_{m,k} = w_m・h + b_{m,k}
    概念 m ごとに別の (w_m, b_m) を持つ（概念ごとに独立した CORAL 層）。

ネットワーク構成:
    [B, 1536] → Linear(1536→512) → LayerNorm → ReLU → Dropout   （全概念で共有）
              → 概念ごとの CORAL ヘッド × M                      → [B, M, 4] のロジット
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiConceptCoralLayer(nn.Module):
    """M 個の概念それぞれに、共有重み 1 本 + K-1 個の独立バイアスを持つ CORAL 層を並列に適用する。"""

    def __init__(self, in_features: int, num_concepts: int, num_grades: int = 5):
        super().__init__()
        self.num_concepts = num_concepts
        self.num_tasks = num_grades - 1
        self.weight = nn.Parameter(torch.empty(num_concepts, in_features))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        # 初期バイアスは降順にして、学習開始時点から b1 > b2 > b3 > b4 の順序を満たす
        init_bias = torch.linspace(1.0, -1.0, self.num_tasks).repeat(num_concepts, 1)
        self.bias = nn.Parameter(init_bias)                      # [M, K-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = x @ self.weight.t()                               # [B, M]
        return base.unsqueeze(-1) + self.bias.unsqueeze(0)       # [B, M, K-1]


class MultiConceptCoralMLP(nn.Module):
    """UNI2-h 特徴量 [B, 1536] から 全概念のハードルロジット [B, M, 4] を出力する。"""

    def __init__(self, in_dim: int = 1536, hidden_dim: int = 512, num_concepts: int = 10,
                 num_grades: int = 5, dropout: float = 0.4):
        super().__init__()
        self.num_concepts = num_concepts
        self.num_grades = num_grades
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.coral = MultiConceptCoralLayer(hidden_dim, num_concepts, num_grades)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.coral(self.backbone(x))


# ------------------------------------------------------------------------------
# ラベル変換・損失・スコア化   （内部グレードは 0〜K-1。外部の 1〜K 表記からは -1 して渡す）
# ------------------------------------------------------------------------------
def label_to_levels(labels: torch.Tensor, num_grades: int = 5) -> torch.Tensor:
    """
    グレード (0〜K-1) を累積バイナリに変換する。形状 [...] → [..., K-1]
        0 → [0,0,0,0]   1 → [1,0,0,0]   2 → [1,1,0,0]   3 → [1,1,1,0]   4 → [1,1,1,1]
    （1〜5 表記なら Grade1→[0,0,0,0], Grade3→[1,1,0,0], Grade5→[1,1,1,1]）
    """
    ks = torch.arange(num_grades - 1, device=labels.device)
    return (labels.long().unsqueeze(-1) > ks).float()


def coral_loss(logits: torch.Tensor, levels: torch.Tensor, concept_weights: torch.Tensor = None) -> torch.Tensor:
    """
    マルチ概念 CORAL Loss。
        各概念: 4 ハードルの BCEWithLogits を合計
        全体  : 概念方向に平均 → バッチ方向に平均
    logits, levels : [B, M, K-1]
    concept_weights: [M] (任意。概念ごとの重み)
    """
    per_task = F.binary_cross_entropy_with_logits(logits, levels, reduction="none")   # [B, M, K-1]
    per_concept = per_task.sum(dim=-1)                                                # [B, M]
    if concept_weights is not None:
        per_concept = per_concept * concept_weights.view(1, -1)
    return per_concept.mean()


def logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    """各ハードル P(grade >= k+1) の確率 [..., K-1]。"""
    return torch.sigmoid(logits)


def logits_to_score(logits: torch.Tensor, grade_min: int = 1) -> torch.Tensor:
    """連続スコア = grade_min + 4 ハードル確率の和（1〜5 表記なら 1.0〜5.0）。形状 [..., K-1] → [...]"""
    return grade_min + torch.sigmoid(logits).sum(dim=-1)


def logits_to_grade(logits: torch.Tensor, grade_min: int = 1) -> torch.Tensor:
    """CORAL 標準の予測（確率 > 0.5 のハードル数）。閾値最適化前のベースライン。"""
    return grade_min + (torch.sigmoid(logits) > 0.5).sum(dim=-1)


if __name__ == "__main__":
    # 動作確認: python coral_model.py
    torch.manual_seed(0)
    B, M = 8, 10
    model = MultiConceptCoralMLP(num_concepts=M)
    x = torch.randn(B, 1536)
    y = torch.randint(0, 5, (B, M))                      # 内部表記 0〜4
    logits = model(x)
    levels = label_to_levels(y)
    assert logits.shape == (B, M, 4), logits.shape
    assert levels.shape == (B, M, 4)
    assert label_to_levels(torch.tensor([3])).tolist() == [[1.0, 1.0, 1.0, 0.0]]
    loss = coral_loss(logits, levels)
    loss.backward()
    assert model.coral.weight.grad.shape == (M, 512)
    score = logits_to_score(logits)
    assert score.shape == (B, M) and ((score >= 1) & (score <= 5)).all()
    print(f"logits {tuple(logits.shape)}  loss={loss.item():.4f}  params={sum(p.numel() for p in model.parameters()):,}")
    print("OK")
