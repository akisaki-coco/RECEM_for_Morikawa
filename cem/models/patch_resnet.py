"""
patch_resnet.py
===============
Patch-based ResNet バックボーンモジュール

入力画像をグリッド状のパッチ（例: 2x2 = 4枚、または 5x5 = 25枚の 224x224 パッチ）に切り分け、
各パッチから CNN で局所特徴量を抽出した後、それらを統合して 1 つの画像特徴量（512次元）を生成します。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50


class PatchResNet(nn.Module):
    """
    画像を grid_size x grid_size（例: 2x2=4枚, 5x5=25枚）に分割し、
    ResNet で特徴抽出・統合するバックボーン
    """
    def __init__(
        self,
        output_dim=512,
        pretrained=True,
        base_arch=resnet34,
        grid_size=2,          # 2 (2x2=4パッチ) または 5 (5x5=25パッチ) など
        fusion_type="attention",  # "attention" (重要度加重平均), "mean" (単純平均), "linear" (全結合層)
    ):
        super().__init__()
        self.grid_size = grid_size
        self.num_patches = grid_size * grid_size
        self.fusion_type = fusion_type
        self.output_dim = output_dim or 512

        # 1. バックボーンの ResNet (全結合層 fc の直前まで)
        base_model = base_arch(pretrained=pretrained)
        in_features = base_model.fc.in_features  # resnet18/34 は 512, resnet50 は 2048
        self.in_features = in_features

        # conv1 〜 avgpool までの層を抽出 (各パッチの特徴量: [B, in_features, 1, 1])
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])

        # 2. 統合層
        if self.fusion_type == "linear":
            # 全パッチの特徴量を結合して圧縮
            self.fc = nn.Linear(in_features * self.num_patches, self.output_dim)
        elif self.fusion_type == "attention":
            # アテンション重み算出ネットワーク: 各パッチの重要度スコア (1次元) を予測
            # 黒帯 (パディング領域) を自動的に重みゼロにして無視できるため、パディング画像に最適
            self.attn = nn.Sequential(
                nn.Linear(in_features, 128),
                nn.Tanh(),
                nn.Linear(128, 1),
            )
            self.fc = nn.Linear(in_features, self.output_dim) if in_features != self.output_dim else nn.Identity()
        else:
            # 単純平均プーリング
            self.fc = nn.Linear(in_features, self.output_dim) if in_features != self.output_dim else nn.Identity()

        # CEM 側で list(self.pre_concept_model.modules())[-1].out_features を参照できるようにする
        self.out_features = self.output_dim

    def extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        """
        [B, C, H, W] を [B * (grid_size^2), C, H//grid_size, W//grid_size] に分割
        """
        B, C, H, W = x.shape
        G = self.grid_size
        patch_h = H // G
        patch_w = W // G

        # テンソルを [B, C, G, patch_h, G, patch_w] に変形
        x_patches = x.view(B, C, G, patch_h, G, patch_w)
        # [B, G, G, C, patch_h, patch_w] に並び替え
        x_patches = x_patches.permute(0, 2, 4, 1, 3, 5).contiguous()
        # [B * (G * G), C, patch_h, patch_w] に平坦化
        return x_patches.view(B * G * G, C, patch_h, patch_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [Batch_size, Channels, Height, Width]
        :return: 統合された画像特徴量 [Batch_size, output_dim]
        """
        B = x.shape[0]

        # 1. 画像を G x G パッチに切り出し
        patches = self.extract_patches(x)  # [B * num_patches, C, patch_h, patch_w]

        # 2. ResNet で各パッチの特徴を並列抽出
        feats = self.feature_extractor(patches)  # [B * num_patches, in_features, 1, 1]
        feats = torch.flatten(feats, 1)          # [B * num_patches, in_features]

        # [B, num_patches, in_features] に戻す
        feats_reshaped = feats.view(B, self.num_patches, self.in_features)

        # 3. パッチ特徴量の統合
        if self.fusion_type == "linear":
            flat_feats = feats_reshaped.view(B, self.num_patches * self.in_features)
            out = self.fc(flat_feats)
        elif self.fusion_type == "attention":
            # アテンション重みの計算: [B, num_patches, 1]
            attn_scores = self.attn(feats_reshaped)
            attn_weights = F.softmax(attn_scores, dim=1)  # 全パッチの和が 1
            # 加重平均: [B, in_features]
            weighted_feats = (feats_reshaped * attn_weights).sum(dim=1)
            out = self.fc(weighted_feats)
        else:
            # 単純平均
            mean_feats = feats_reshaped.mean(dim=1)
            out = self.fc(mean_feats)

        return out


def build_patch_resnet(grid_size=2, fusion_type="attention", pretrained=True):
    """wrap_pretrained_model 用のファクトリ関数"""
    def _constructor(output_dim=512):
        return PatchResNet(
            output_dim=output_dim,
            pretrained=pretrained,
            base_arch=resnet34,
            grid_size=grid_size,
            fusion_type=fusion_type,
        )
    return _constructor


if __name__ == "__main__":
    print("--- PatchResNet 5x5 (25パッチ) 動作テスト ---")
    batch_size = 2
    # 1120x1120 の原寸大パディング画像を入力
    dummy_img = torch.randn(batch_size, 3, 1120, 1120)

    model = PatchResNet(output_dim=512, pretrained=False, grid_size=5, fusion_type="attention")
    out = model(dummy_img)

    print(f"入力 shape : {dummy_img.shape}")
    print(f"出力 shape : {out.shape} (期待値: [{batch_size}, 512])")
    print("5x5 PatchResNet テスト成功！")
