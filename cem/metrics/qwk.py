"""
cem/metrics/qwk.py
==================
順序尺度概念（グレード1〜5）のための QWK (Quadratic Weighted Kappa) 評価モジュール
"""

import numpy as np
import torch
from functools import partial
from scipy.optimize import minimize
from sklearn.metrics import cohen_kappa_score
from cem.models.coral import ordinal_probs_to_grades


class OptimizedRounder:
    """
    Nelder-Mead 法を用いて、連続スコア (期待値: 1.0 〜 5.0) を
    1〜K のグレードに離散化するための最適なしきい値を探索するクラス。
    Kaggleコンペ (APTOS, Diabetic Retinopathy等) の定番手法。
    """
    def __init__(self, num_classes: int = 5):
        self.num_classes = num_classes
        # 初期しきい値: 例 5段階評価なら [1.5, 2.5, 3.5, 4.5]
        self.coefficients = [i + 0.5 for i in range(1, num_classes)]
        self.labels = list(range(1, num_classes + 1))

    def _predict_with_coef(self, X: np.ndarray, coef: list[float]) -> np.ndarray:
        """指定したしきい値で離散化"""
        coef_sorted = np.sort(coef)
        res = np.digitize(X, coef_sorted) + 1
        return np.clip(res, 1, self.num_classes)

    def _loss(self, coef: list[float], X: np.ndarray, y: np.ndarray) -> float:
        """目的関数: -QWK (最小化することで QWK を最大化)"""
        X_p = self._predict_with_coef(X, coef)
        if len(np.unique(X_p)) <= 1 and len(np.unique(y)) <= 1:
            score = 1.0 if X_p[0] == y[0] else 0.0
        else:
            try:
                score = cohen_kappa_score(y, X_p, labels=self.labels, weights="quadratic")
            except Exception:
                score = 0.0
        return -score

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        検証データ (X: 連続スコア, y: 正解グレード) を使って
        Nelder-Mead 法により QWK を最大化するしきい値を学習します。
        """
        loss_partial = partial(self._loss, X=X, y=y)
        initial_coef = self.coefficients
        res = minimize(loss_partial, initial_coef, method="Nelder-Mead")
        self.coefficients = np.sort(res.x).tolist()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """学習済みしきい値を用いて離散グレード (1〜5) を予測"""
        return self._predict_with_coef(X, self.coefficients)


def probs_to_expected_grades(sub_probs: torch.Tensor) -> np.ndarray:
    """
    CORAL の累積確率 [N, 4] から期待値としての連続スコア (1.0 〜 5.0) を算出
    E[Y] = 1 + sum_{k=2}^5 P(Y >= k)
    """
    if isinstance(sub_probs, torch.Tensor):
        sub_probs = sub_probs.cpu().detach().numpy()
    return 1.0 + np.sum(sub_probs, axis=-1)


def optimize_concept_thresholds(
    val_c_true: torch.Tensor,
    val_c_pred: torch.Tensor,
    num_classes_per_concept: int = 5,
) -> list[OptimizedRounder]:
    """
    検証データ (Validation Data) を用いて、各概念ごとに
    Nelder-Mead 法により QWK を最大化する最適なしきい値を探索します。

    :param val_c_true: 検証データの正解バイナリ [N, n_concepts]
    :param val_c_pred: 検証データの予測確率 [N, n_concepts]
    :param num_classes_per_concept: グレード数 (通常 5)
    :return: 概念ごとの学習済み OptimizedRounder のリスト
    """
    if isinstance(val_c_true, torch.Tensor):
        val_c_true = val_c_true.cpu().detach()
    if isinstance(val_c_pred, torch.Tensor):
        val_c_pred = val_c_pred.cpu().detach()

    tasks_per_concept = num_classes_per_concept - 1
    num_orig_concepts = val_c_true.shape[-1] // tasks_per_concept

    rounders = []
    for m in range(num_orig_concepts):
        start_idx = m * tasks_per_concept
        end_idx = (m + 1) * tasks_per_concept

        sub_true = val_c_true[:, start_idx:end_idx]
        sub_pred = val_c_pred[:, start_idx:end_idx]

        # 正解グレード (1〜5)
        y_val = ordinal_probs_to_grades(sub_true, threshold=0.5).numpy()
        # 予測の連続スコア (期待値: 1.0〜5.0)
        X_val = probs_to_expected_grades(sub_pred)

        # Nelder-Mead 最適化
        rounder = OptimizedRounder(num_classes=num_classes_per_concept)
        rounder.fit(X_val, y_val)
        rounders.append(rounder)

    return rounders


def compute_concept_qwk_with_rounders(
    c_true: torch.Tensor,
    c_pred: torch.Tensor,
    rounders: list[OptimizedRounder],
    num_classes_per_concept: int = 5,
) -> tuple[float, list[float]]:
    """
    検証データで最適化した rounders (しきい値) を適用してテストデータの QWK を計算します。

    :param c_true: テストデータの正解バイナリ [N, n_concepts]
    :param c_pred: テストデータの予測確率 [N, n_concepts]
    :param rounders: optimize_concept_thresholds で得られた各概念の rounder
    :return: (平均 QWK, 各概念の QWK リスト)
    """
    if isinstance(c_true, torch.Tensor):
        c_true = c_true.cpu().detach()
    if isinstance(c_pred, torch.Tensor):
        c_pred = c_pred.cpu().detach()

    tasks_per_concept = num_classes_per_concept - 1
    num_orig_concepts = c_true.shape[-1] // tasks_per_concept

    qwk_list = []
    labels = list(range(1, num_classes_per_concept + 1))

    for m in range(num_orig_concepts):
        start_idx = m * tasks_per_concept
        end_idx = (m + 1) * tasks_per_concept

        sub_true = c_true[:, start_idx:end_idx]
        sub_pred = c_pred[:, start_idx:end_idx]

        true_grades = ordinal_probs_to_grades(sub_true, threshold=0.5).numpy()
        pred_scores = probs_to_expected_grades(sub_pred)
        pred_grades = rounders[m].predict(pred_scores)

        if len(np.unique(true_grades)) <= 1 and len(np.unique(pred_grades)) <= 1:
            score = 1.0 if true_grades[0] == pred_grades[0] else 0.0
        else:
            try:
                score = cohen_kappa_score(
                    true_grades,
                    pred_grades,
                    labels=labels,
                    weights="quadratic",
                )
            except Exception:
                score = 0.0

        qwk_list.append(float(score))

    mean_qwk = float(np.mean(qwk_list))
    return mean_qwk, qwk_list


def compute_concept_qwk(
    c_true: torch.Tensor,
    c_pred: torch.Tensor,
    num_classes_per_concept: int = 5,
    threshold: float = 0.5,
) -> tuple[float, list[float]]:
    """
    40 次元の順序バイナリ (c_true, c_pred) から各概念のグレード (1〜5) を復元し、
    各概念の QWK およびその平均値を計算します (固定しきい値 0.5 版)。
    """
    if isinstance(c_true, torch.Tensor):
        c_true = c_true.cpu().detach()
    if isinstance(c_pred, torch.Tensor):
        c_pred = c_pred.cpu().detach()

    # 1 概念あたりの閾値数 (例: 5段階評価なら 4)
    tasks_per_concept = num_classes_per_concept - 1
    num_orig_concepts = c_true.shape[-1] // tasks_per_concept

    qwk_list = []
    labels = list(range(1, num_classes_per_concept + 1))  # [1, 2, 3, 4, 5]

    for m in range(num_orig_concepts):
        # 概念 m に対応する 4 スライスのインデックスを取得
        start_idx = m * tasks_per_concept
        end_idx = (m + 1) * tasks_per_concept

        sub_true = c_true[:, start_idx:end_idx]
        sub_pred = c_pred[:, start_idx:end_idx]

        # 4次元バイナリから 1〜5 のグレード数値に復元
        true_grades = ordinal_probs_to_grades(sub_true, threshold=threshold).numpy()
        pred_grades = ordinal_probs_to_grades(sub_pred, threshold=threshold).numpy()

        # クラスが1種類しかない場合のゼロ除算対策
        if len(np.unique(true_grades)) <= 1 and len(np.unique(pred_grades)) <= 1:
            score = 1.0 if true_grades[0] == pred_grades[0] else 0.0
        else:
            try:
                # 2乗重み付きカッパ係数 (QWK) の計算
                score = cohen_kappa_score(
                    true_grades,
                    pred_grades,
                    labels=labels,
                    weights="quadratic"
                )
            except Exception:
                score = 0.0

        qwk_list.append(float(score))

    mean_qwk = float(np.mean(qwk_list))
    return mean_qwk, qwk_list

