"""
05_tune_boost.py
================
第1閾値 t1（Grade 1と2の境界）の下限ガードを段階的に変化させ、
「Grade 1 の過大評価（1→2誤認）」と「QWK / Accuracy」のトレードオフを検証・比較するスクリプト。

モデルの再学習は不要で、outputs/oof_predictions.csv を使って数秒で完了します。

使い方:
    python 05_tune_boost.py
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

import config as C
from common import apply_thresholds, qwk

GRADES = list(range(C.GRADE_MIN, C.GRADE_MIN + C.NUM_GRADES))


def _neg_qwk(t, scores, y, min_t1=None):
    t_sorted = np.sort(t)
    # ペナルティ: t1 が min_t1 を下回った場合に大きなペナルティ
    penalty = 0.0
    if min_t1 is not None and t_sorted[0] < min_t1:
        penalty = 10.0 * (min_t1 - t_sorted[0]) ** 2
    return -qwk(y, apply_thresholds(scores, t_sorted)) + penalty


def _nelder_mead(x0, scores, y, min_t1=None):
    x0 = np.asarray(x0, dtype=float)
    if min_t1 is not None and x0[0] < min_t1:
        x0[0] = min_t1
    simplex = np.vstack([x0] + [x0 + C.NM_INITIAL_STEP * np.eye(len(x0))[i] for i in range(len(x0))])
    res = minimize(
        _neg_qwk,
        x0,
        args=(scores, y, min_t1),
        method="Nelder-Mead",
        options={"initial_simplex": simplex, "xatol": 1e-4, "fatol": 1e-6, "maxiter": 2000},
    )
    t = np.sort(res.x)
    if min_t1 is not None and t[0] < min_t1:
        t[0] = min_t1
    return t, qwk(y, apply_thresholds(scores, t))


def optimize_thresholds(scores, y, min_t1=None, seed=C.SEED):
    rng = np.random.default_rng(seed)
    init = np.asarray(C.INIT_THRESHOLDS, dtype=float)
    if min_t1 is not None and init[0] < min_t1:
        init[0] = min_t1
    starts = [init] + [np.sort(init + rng.normal(0, 0.2, size=len(init))) for _ in range(C.NM_N_RESTARTS)]
    best_t, best_q = init, qwk(y, apply_thresholds(scores, init))
    for x0 in starts:
        t, q = _nelder_mead(x0, scores, y, min_t1=min_t1)
        if q > best_q + 1e-9:
            best_t, best_q = t, q
    return best_t, best_q


def nested_evaluate(scores, y, folds, min_t1=None):
    pred = np.zeros(len(y), dtype=int)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        t, _ = optimize_thresholds(scores[tr], y[tr], min_t1=min_t1)
        pred[te] = apply_thresholds(scores[te], t)
    return pred


def run_experiment(min_t1_candidates=[None, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]):
    oof = pd.read_csv(C.OOF_CSV)
    concepts = [c[:-len("_true")] for c in oof.columns if c.endswith("_true")]
    folds = oof["fold"].values

    print(f"[INFO] 評価対象概念数: {len(concepts)} 個")
    print(f"[INFO] min_t1（第1閾値の下限ガード）を段階的に変化させて比較します...\n")

    summary_rows = []

    for min_t1 in min_t1_candidates:
        tag = f"{min_t1:.1f}" if min_t1 is not None else "制限なし (現状)"
        qwk_list, acc_list, within1_list, g1_to_g2_list, total_g1_list = [], [], [], [], []

        for name in concepts:
            y = oof[f"{name}_true"].values.astype(int)
            s = oof[f"{name}_score"].values

            # 入れ子推定（未知データでの公平な評価）
            pred = nested_evaluate(s, y, folds, min_t1=min_t1)

            qwk_val = qwk(y, pred)
            acc = float((pred == y).mean())
            within1 = float((np.abs(pred - y) <= 1).mean())

            # Grade 1 -> 2 の誤認数
            g1_mask = y == 1
            g1_to_2 = int(((pred == 2) & g1_mask).sum())
            total_g1 = int(g1_mask.sum())

            qwk_list.append(qwk_val)
            acc_list.append(acc)
            within1_list.append(within1)
            g1_to_g2_list.append(g1_to_2)
            total_g1_list.append(total_g1)

        summary_rows.append({
            "min_t1 (下限ガード)": tag,
            "平均 QWK (nested)": round(float(np.mean(qwk_list)), 4),
            "平均 正解率 (Acc)": f"{np.mean(acc_list)*100:.2f}%",
            "±1 許容正解率": f"{np.mean(within1_list)*100:.2f}%",
            "Grade 1→2 誤認数 (全概念合計)": sum(g1_to_g2_list),
            "Grade 1 誤認率": f"{(sum(g1_to_g2_list) / sum(total_g1_list))*100:.1f}%",
        })

    res_df = pd.DataFrame(summary_rows)
    print("=" * 80)
    print("【min_t1 段階調整の比較結果】")
    print("=" * 80)
    print(res_df.to_string(index=False))
    print("=" * 80)

    out_path = C.OUTPUT_DIR / "tune_boost_summary.csv"
    res_df.to_csv(out_path, index=False)
    print(f"\n保存先: {out_path}")


if __name__ == "__main__":
    run_experiment()
