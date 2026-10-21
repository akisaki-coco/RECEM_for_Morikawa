"""
coral.py
========
Consistent Rank Logits (CORAL) モジュール

順序回帰（Ordinal Regression）のための構成要素:
1. CoralLayer: 重みベクトルを1つ共有し、バイアスをグレード数分 (K-1個) 持つ全結合層
2. MultiConceptCoralLayer: 複数の概念それぞれに対して CoralLayer を適用する複合層
3. CoralLoss: 各グレード閾値のバイナリ交差エントロピー (BCE) を合算する損失関数
4. ユーティリティ関数: 4次元のバイナリ確率から 1〜5 のグレード数値を復元する関数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1. 単一概念用 CORAL 全結合層 (CoralLayer)
# ==============================================================================
class CoralLayer(nn.Module):
    """
    CORAL (Consistent Rank Logits) 全結合層

    【構造】
    - 入力: in_features 次元 (例: 画像特徴量や概念コンテキストベクトル)
    - 重み (weight): [1, in_features] -> 全グレード閾値で「1つの重みベクトル」を共有
    - バイアス (bias): [num_classes - 1] -> グレードの閾値ごとに独立したバイアス

    出力ロジット:
        logits = x @ weight.T + bias  (shape: [batch_size, num_classes - 1])
    """
    def __init__(self, in_features: int, num_classes: int = 5):
        """
        :param in_features: 入力特徴量の次元数
        :param num_classes: グレードの総数 (デフォルト 5段階: 1〜5)
        """
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.num_tasks = num_classes - 1  # 5段階評価なら 4 つの閾値 (>=2, >=3, >=4, >=5)

        # 1. 全グレードで共有する単一の重みベクトル
        self.weight = nn.Parameter(torch.empty(1, in_features))

        # 2. 各グレード閾値ごとの独立したバイアス
        self.bias = nn.Parameter(torch.zeros(self.num_tasks))

        # パラメータの初期化
        self.reset_parameters()

    def reset_parameters(self):
        # 重みの初期化 (Kaiming Uniform)
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

        # バイアスの初期化:
        # 初期状態で b1 > b2 > b3 > b4 (高いグレードほど閾値が高く出にくくなる) ように設定すると学習が安定します
        initial_biases = torch.linspace(1.0, -1.0, steps=self.num_tasks)
        with torch.no_grad():
            self.bias.copy_(initial_biases)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [batch_size, in_features]
        :return: 未正規化ロジット [batch_size, num_tasks]
        """
        # x @ weight.T -> shape: [batch_size, 1]
        base_logit = F.linear(x, self.weight)

        # ブロードキャスト加算で各バイアスを加算 -> shape: [batch_size, num_tasks]
        logits = base_logit + self.bias
        return logits


# ==============================================================================
# 2. 複数概念用 CORAL 層 (MultiConceptCoralLayer)
# ==============================================================================
class MultiConceptCoralLayer(nn.Module):
    """
    元概念が M 個ある場合に、各概念ごとに独立した CoralLayer を並列実行するラッパー層。
    CEM / RECEM の概念予測層として直接組み込みやすい設計です。
    """
    def __init__(self, num_concepts: int, in_features: int, num_classes: int = 5):
        """
        :param num_concepts: 元概念の数 (例: 10)
        :param in_features: 各概念の入力次元 (例: 2 * emb_size)
        :param num_classes: 各概念のグレード数 (例: 5)
        """
        super().__init__()
        self.num_concepts = num_concepts
        self.num_tasks_per_concept = num_classes - 1

        # 各概念ごとの CoralLayer を ModuleList で管理
        self.coral_layers = nn.ModuleList([
            CoralLayer(in_features=in_features, num_classes=num_classes)
            for _ in range(num_concepts)
        ])

    def forward(self, concept_contexts: list[torch.Tensor]) -> torch.Tensor:
        """
        :param concept_contexts: 各概念のコンテキストテンソルのリスト
                                 [ [batch_size, in_features], ... ] (長さ M)
        :return: 全概念のロジットを連結したもの [batch_size, num_concepts * (num_classes - 1)]
        """
        all_logits = []
        for i, layer in enumerate(self.coral_layers):
            ctx = concept_contexts[i]
            logits = layer(ctx)  # [B, 4]
            all_logits.append(logits)

        # 列方向に連結 -> [B, num_concepts * 4]
        return torch.cat(all_logits, dim=-1)


# ==============================================================================
# 3. CORAL 損失関数 (CoralLoss)
# ==============================================================================
class CoralLoss(nn.Module):
    """
    CORAL 用の損失関数
    バッチごとに各グレード閾値の BCE (Binary Cross Entropy) を計算し、合算します。
    """
    def __init__(self, weight=None, pos_weight=None, reduction: str = "mean", from_logits: bool = False):
        """
        :param weight: 各概念・タスクごとの重み [num_tasks] (不均衡補正用)
        :param pos_weight: 各タスクの正例の重み (from_logits=True 時のみ有効)
        :param reduction: 'mean', 'sum', 'none'
        :param from_logits: True の場合は未正規化ロジット、False の場合は Sigmoid 後の確率を入力とする
        """
        super().__init__()
        self.reduction = reduction
        self.from_logits = from_logits

        # weight として渡されたものを pos_weight として解釈できるようにする
        # (imbalance は正例の少なさに対する倍率比率のため、正例に対する重み pos_weight として機能させる)
        if pos_weight is None and weight is not None:
            pos_weight = weight
            weight = None

        # weightが設定されているとき
        if weight is not None:
            if not isinstance(weight, torch.Tensor):
                weight = torch.tensor(weight, dtype=torch.float32)
            self.register_buffer("weight", weight)
        else:
            self.weight = None

        if pos_weight is not None:
            if not isinstance(pos_weight, torch.Tensor):
                pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        :param inputs: 確率 [batch_size, n_concepts] (from_logits=False) または ロジット (from_logits=True)
        :param targets: 順序バイナリの正解ラベル [batch_size, n_concepts] (0.0 または 1.0)
        :return: スカラー損失値
        """
        # デバイスの不一致を自動防止 (万が一 self.weight / self.pos_weight が別デバイスにあっても追従)
        weight = self.weight
        if weight is not None and weight.device != inputs.device:
            weight = weight.to(inputs.device)

        pos_weight = self.pos_weight
        if pos_weight is not None and pos_weight.device != inputs.device:
            pos_weight = pos_weight.to(inputs.device)

        if self.from_logits:
            bce_loss = F.binary_cross_entropy_with_logits(
                inputs,
                targets,
                weight=weight,
                pos_weight=pos_weight,
                reduction="none"
            )  # shape: [batch_size, n_concepts]
        else:
            # CEM標準: Sigmoid 後の確率 (0.0〜1.0) を受け取る場合
            eps = 1e-7
            p = torch.clamp(inputs, eps, 1.0 - eps)
            # 正例 (y=1) のみに pos_weight をかけて不均衡補正 (全部ゼロ予測を防ぐ)
            if pos_weight is not None:
                bce_loss = - (pos_weight * targets * torch.log(p) + (1.0 - targets) * torch.log(1.0 - p))
            else:
                bce_loss = - (targets * torch.log(p) + (1.0 - targets) * torch.log(1.0 - p))

            if weight is not None:
                bce_loss = bce_loss * weight

        # サンプルごとに全概念・全閾値の損失を平均 (CEM標準の BCELoss とスケールを合わせてタスク損失とのバランスを保つ)
        sample_loss = bce_loss.mean(dim=-1)  # shape: [batch_size]

        if self.reduction == "mean":
            return sample_loss.mean()
        elif self.reduction == "sum":
            return sample_loss.sum()
        else:
            return sample_loss


# ==============================================================================
# 4. ユーティリティ: 順序バイナリからグレード数値への復元
# ==============================================================================

# 結果を人間用にわかりやすくする時に使ってね

def ordinal_probs_to_grades(probs: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    モデルが出力した 4 つの閾値確率 (p_ge2, p_ge3, p_ge4, p_ge5) から、
    元のグレード数値 (1 〜 5) を復元します。

    グレード = 1 + (確率 > threshold の個数)
    例:
      [0.1, 0.0, 0.0, 0.0] -> 1 + 0 = グレード 1
      [0.9, 0.8, 0.1, 0.0] -> 1 + 2 = グレード 3
      [0.9, 0.9, 0.9, 0.9] -> 1 + 4 = グレード 5

    :param probs: [batch_size, 4] または [batch_size, M, 4]
    :param threshold: 二値化しきい値 (通常 0.5)
    :return: グレード [batch_size] または [batch_size, M] (値は 1 〜 5)
    """
    binary_preds = (probs > threshold).long()
    grades = 1 + binary_preds.sum(dim=-1)
    return grades


# ==============================================================================
# 動作確認用テストコード
# ==============================================================================
if __name__ == "__main__":
    print("--- CoralLayer の動作テスト ---")
    batch_size = 4
    in_features = 16  # 例: 2 * emb_size (emb_size=8 の場合)
    num_classes = 5

    # 1. CoralLayer のインスタンス化
    layer = CoralLayer(in_features=in_features, num_classes=num_classes)
    dummy_x = torch.randn(batch_size, in_features)
    logits = layer(dummy_x)
    probs = torch.sigmoid(logits)

    print(f"入力 shape : {dummy_x.shape}")
    print(f"出力 shape : {logits.shape} (期待値: [{batch_size}, {num_classes - 1}])")
    print(f"予測確率値 :\n{probs.detach().numpy().round(3)}")

    # 2. 復元関数のテスト
    grades = ordinal_probs_to_grades(probs)
    print(f"復元グレード: {grades.tolist()}")

    # 3. CoralLoss の動作テスト
    print("\n--- CoralLoss の動作テスト ---")
    dummy_targets = torch.tensor([
        [1.0, 1.0, 0.0, 0.0],  # グレード 3
        [1.0, 1.0, 1.0, 1.0],  # グレード 5
        [0.0, 0.0, 0.0, 0.0],  # グレード 1
        [1.0, 0.0, 0.0, 0.0],  # グレード 2
    ])
    criterion = CoralLoss()
    loss = criterion(logits, dummy_targets)
    print(f"算出された損失値: {loss.item():.4f}")

    # 逆伝播 (backward) のテスト
    loss.backward()
    print("逆伝播 (loss.backward()) 成功: 勾配が正常に計算されました")
