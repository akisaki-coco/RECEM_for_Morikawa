"""
common.py
=========
各フェーズで共有するユーティリティ（シード固定・デバイス選択・QWK・閾値適用・特徴量読み込み）。
"""
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score


def set_seed(seed: int) -> None:
    """乱数シードを固定して再現性を確保する。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer_mps: bool = True) -> torch.device:
    """利用可能なデバイスを CUDA > MPS > CPU の順で選択する。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_mps and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def qwk(y_true, y_pred, num_classes: int = 5) -> float:
    """Quadratic Weighted Kappa（全クラスを labels に明示して計算）。"""
    return float(
        cohen_kappa_score(
            np.asarray(y_true, dtype=int),
            np.asarray(y_pred, dtype=int),
            weights="quadratic",
            labels=list(range(num_classes)),
        )
    )


def apply_thresholds(scores, thresholds) -> np.ndarray:
    """
    連続スコア (0.0〜4.0) を閾値 [t1, t2, t3, t4] で 5 段階 (0〜4) に変換する。
        score <  t1        -> 0
        t1 <= score < t2   -> 1
        ...
        t4 <= score        -> 4
    """
    t = np.sort(np.asarray(thresholds, dtype=float))
    return np.digitize(np.asarray(scores, dtype=float), t).astype(int)


def feature_path(feature_dir: Path, slide_id: str) -> Path:
    return Path(feature_dir) / f"{slide_id}.pt"


def load_feature(path: Path) -> torch.Tensor:
    """保存済み特徴量 (.pt) を [D] の float32 テンソルとして読み込む。"""
    feat = torch.load(path, map_location="cpu")
    return feat.float().reshape(-1)


def load_feature_matrix(df: pd.DataFrame, feature_dir: Path, feature_dim: int) -> torch.Tensor:
    """labels.csv の順番で全特徴量を読み込み [N, D] のテンソルにする。"""
    feats = []
    for sid in df["slide_id"].astype(str):
        p = feature_path(feature_dir, sid)
        if not p.exists():
            raise FileNotFoundError(f"特徴量が見つかりません: {p}（先に 02_extract_features.py を実行してください）")
        f = load_feature(p)
        if f.numel() != feature_dim:
            raise ValueError(f"{p} の次元が {f.numel()} です（期待値: {feature_dim}）")
        feats.append(f)
    return torch.stack(feats, dim=0)
