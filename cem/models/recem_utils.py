import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

# 1. 勾配反転層 (Gradient Reversal Layer)
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        # 逆伝播時に勾配を -lambda 倍して反転させる
        return -ctx.lambda_ * grads, None

class GRL(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


# 2. HSIC (Hilbert-Schmidt Independence Criterion) 損失
def rbf_kernel(X, sigma=None):
    # ペアワイズ二乗ユークリッド距離: ||x_i - x_j||^2
    pairwise_dist = torch.cdist(X, X, p=2) ** 2
    if sigma is None:
        # Median heuristic によるバンド幅設定
        median_dist = torch.median(pairwise_dist.detach())
        sigma = torch.sqrt(0.5 * median_dist + 1e-6)
    gamma = 1.0 / (2.0 * (sigma ** 2) + 1e-8)
    return torch.exp(-gamma * pairwise_dist)

def compute_hsic(Z, C_tilde):
    """
    Z: 無関連特徴 [B, K * d]
    C_tilde: 真の概念埋め込み [B, K * d]
    """
    B = Z.size(0)
    if B <= 1:
        return torch.tensor(0.0, device=Z.device)
    
    # 特徴次元が大きい場合はL2正規化を入れるとカーネル計算が安定します
    Z_norm = F.normalize(Z, p=2, dim=-1)
    C_norm = F.normalize(C_tilde, p=2, dim=-1)

    K_z = rbf_kernel(Z_norm)
    K_c = rbf_kernel(C_norm)

    # 中心化行列 H = I - (1/B) * 1 * 1^T
    H = torch.eye(B, device=Z.device) - (1.0 / B) * torch.ones((B, B), device=Z.device)
    K_z_centered = H @ K_z @ H
    K_c_centered = H @ K_c @ H

    # Empirical HSIC: Tr(K_z_c * K_c_c) / (B - 1)^2
    hsic = torch.trace(K_z_centered @ K_c_centered) / ((B - 1) ** 2)
    return hsic
