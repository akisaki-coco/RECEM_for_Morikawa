"""
patch_resnet.py
===============
Patch-based ResNet バックボーンモジュール

入力画像を高解像度（例: 448x448）のままグリッド状のパッチ（例: 2x2 = 4枚の 224x224 パッチ）に切り分け、
各パッチから CNN で局所特徴量を抽出した後、それらを統合して 1 つの画像特徴量（512次元）を生成します。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50


class PatchResNet(nn.Module):
    """
    画像を 2x2 = 4 つのパッチに分割し、ResNet で特徴抽出・統合するバックボーン
    """
    def __init__(
        self,
        output_dim=512,
        pretrained=True,
        base_arch=resnet34,
        fusion_type="linear",  # "linear" (全結合層で結合) または "mean" (平均プーリング)
    ):
        super().__init__()
        self.fusion_type = fusion_type
        self.output_dim = output_dim or 512

        # 1. バックボーンの ResNet (全結合層 fc の直前まで)
        base_model = base_arch(pretrained=pretrained)
        in_features = base_model.fc.in_features  # resnet18/34 は 512, resnet50 は 2048
        self.in_features = in_features

        # conv1 〜 avgpool までの層を抽出
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])  # 出力: [B, in_features, 1, 1]

        # 2. 統合層
        if self.fusion_type == "linear":
            # 4パッチ分の特徴量 (4 * in_features) を結合して 1 つのベクトル (output_dim) に圧縮
            self.fc = nn.Linear(in_features * 4, self.output_dim)
        else:
            # 4パッチの特徴量を平均した後に次元調整
            self.fc = nn.Linear(in_features, self.output_dim) if in_features != self.output_dim else nn.Identity()

        # CEM 側で list(self.pre_concept_model.modules())[-1].out_features を参照できるようにする
        self.out_features = self.output_dim

    def extract_patches_2x2(self, x: torch.Tensor) -> torch.Tensor:
        """
        [B, C, H, W] -> [B * 4, C, H//2, W//2] に分割
        """
        B, C, H, W = x.shape
        h_mid, w_mid = H // 2, W // 2

        p1 = x[:, :, :h_mid, :w_mid]   # 左上
        p2 = x[:, :, :h_mid, w_mid:]   # 右上
        p3 = x[:, :, h_mid:, :w_mid]   # 左下
        p4 = x[:, :, h_mid:, w_mid:]   # 右下

        # [B, 4, C, H//2, W//2]
        patches = torch.stack([p1, p2, p3, p4], dim=1)
        # ResNet に並列で一括入力できるように [B * 4, C, H//2, W//2] に変形
        return patches.view(B * 4, C, h_mid, w_mid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [Batch_size, Channels, Height, Width]
        :return: 統合された画像特徴量 [Batch_size, output_dim]
        """
        B = x.shape[0]

        # 1. 画像を 2x2 = 4 パッチに切り出し
        patches = self.extract_patches_2x2(x)  # [B * 4, C, H//2, W//2]

        # 2. ResNet で各パッチの特徴を抽出
        feats = self.feature_extractor(patches)  # [B * 4, in_features, 1, 1]
        feats = torch.flatten(feats, 1)          # [B * 4, in_features]

        # 3. 4つのパッチ特徴量を統合
        if self.fusion_type == "linear":
            # 4パッチを横に並べて結合 -> [B, 4 * in_features]
            feats = feats.view(B, 4 * self.in_features)
            out = self.fc(feats)                # [B, output_dim]
        else:
            # 4パッチの平均をとる -> [B, in_features]
            feats = feats.view(B, 4, self.in_features).mean(dim=1)
            out = self.fc(feats)                # [B, output_dim]

        return out


if __name__ == "__main__":
    print("--- PatchResNet 動作テスト ---")
    batch_size = 2
    # 448x448 の画像を入力
    dummy_img = torch.randn(batch_size, 3, 448, 448)

    model = PatchResNet(output_dim=512, pretrained=False, fusion_type="linear")
    out = model(dummy_img)

    print(f"入力 shape : {dummy_img.shape}")
    print(f"出力 shape : {out.shape} (期待値: [{batch_size}, 512])")
    print("PatchResNet テスト成功！")
