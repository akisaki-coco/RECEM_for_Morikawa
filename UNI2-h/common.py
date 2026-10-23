"""
common.py
=========
各フェーズで共有するユーティリティ（シード固定・デバイス選択・QWK・閾値適用・特徴量読み込み）。
グレードは外部表記（1〜5）で扱います。モデル内部のみ 0〜4 に変換します。
"""
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score

import config as C


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer_mps: bool = True) -> torch.device:
    """CUDA > MPS > CPU の順で選択する。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_mps and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def qwk(y_true, y_pred, num_grades: int = None, grade_min: int = None) -> float:
    """Quadratic Weighted Kappa（全グレードを labels に明示。グレードは 1〜5 表記）。"""
    num_grades = num_grades or C.NUM_GRADES
    grade_min = C.GRADE_MIN if grade_min is None else grade_min
    labels = list(range(grade_min, grade_min + num_grades))
    y_true, y_pred = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    if len(np.unique(y_true)) <= 1 and len(np.unique(y_pred)) <= 1:
        return 1.0 if y_true[0] == y_pred[0] else 0.0
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=labels))


def apply_thresholds(scores, thresholds, grade_min: int = None) -> np.ndarray:
    """
    連続スコア (1.0〜5.0) を閾値 [t1, t2, t3, t4] で 1〜5 のグレードに変換する。
        score < t1 → 1,  t1 <= score < t2 → 2, ...,  t4 <= score → 5
    """
    grade_min = C.GRADE_MIN if grade_min is None else grade_min
    t = np.sort(np.asarray(thresholds, dtype=float))
    return (np.digitize(np.asarray(scores, dtype=float), t) + grade_min).astype(int)


def concept_names_from_labels(df: pd.DataFrame) -> list:
    """labels.csv の grade_<概念名> 列から概念名リストを得る。"""
    return [c[len(C.GRADE_PREFIX):] for c in df.columns if c.startswith(C.GRADE_PREFIX)]


def grades_from_labels(df: pd.DataFrame) -> np.ndarray:
    """labels.csv から [N, 概念数] のグレード (1〜5) を取り出す。"""
    cols = [c for c in df.columns if c.startswith(C.GRADE_PREFIX)]
    return df[cols].values.astype(int)


def feature_path(feature_dir: Path, slide_id: str) -> Path:
    return Path(feature_dir) / f"{slide_id}.pt"


def load_feature(path: Path) -> torch.Tensor:
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
