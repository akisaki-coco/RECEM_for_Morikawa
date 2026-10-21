"""
cem/metrics/qwk.py
==================
順序尺度概念（グレード1〜5）のための QWK (Quadratic Weighted Kappa) 評価モジュール
"""

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score
from cem.models.coral import ordinal_probs_to_grades


def compute_concept_qwk(
    c_true: torch.Tensor,
    c_pred: torch.Tensor,
    num_classes_per_concept: int = 5,
    threshold: float = 0.5,
) -> tuple[float, list[float]]:
    """
    40 次元の順序バイナリ (c_true, c_pred) から各概念のグレード (1〜5) を復元し、
    各概念の QWK およびその平均値を計算します。

    :param c_true: 正解バイナリテンソル [N, n_concepts] (例: [N, 40])
    :param c_pred: 予測確率テンソル [N, n_concepts] (例: [N, 40])
    :param num_classes_per_concept: 各概念のグレード数 (デフォルト: 5)
    :param threshold: 確率を二値化するしきい値 (通常 0.5)
    :return: (平均 QWK, 各概念の QWK リスト)
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
