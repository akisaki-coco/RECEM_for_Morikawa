"""
05_optimize_thresholds.py
=========================
フェーズ 5: Nelder-Mead 法による閾値最適化と最終評価

    1. OOF 連続スコア (0.0〜4.0) と正解ラベルから、QWK を最大化する閾値 [t1, t2, t3, t4] を探索
       （初期値 [0.5, 1.5, 2.5, 3.5]、scipy.optimize.minimize(method="Nelder-Mead")）
    2. 最適閾値で 5 段階予測を決定し、混同行列・QWK・Accuracy・クラス別 Precision/Recall/F1 を算出
    3. どのグレード間の誤りが多いかを分析
    4. 参考: 「閾値も未知 Fold で評価する」入れ子推定（Fold k 以外で閾値最適化 → Fold k に適用）

出力 (outputs/):
    thresholds.json
    evaluation/summary.json, confusion_matrix.csv, confusion_matrix.png,
    evaluation/classification_report.csv, error_pairs.csv, oof_final_predictions.csv, score_distribution.png

使い方:
    python 05_optimize_thresholds.py
"""
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import config as C
from common import apply_thresholds, qwk

K = C.NUM_CLASSES
LABELS = list(range(K))


# ------------------------------------------------------------------------------
# 閾値最適化
# ------------------------------------------------------------------------------
def _neg_qwk(t, scores, y):
    return -qwk(y, apply_thresholds(scores, t), K)


def _nelder_mead(x0, scores, y):
    x0 = np.asarray(x0, dtype=float)
    # QWK は階段状の関数なので、初期シンプレックスを大きめに取らないと動かない
    simplex = np.vstack([x0] + [x0 + C.NM_INITIAL_STEP * np.eye(len(x0))[i] for i in range(len(x0))])
    res = minimize(
        _neg_qwk, x0, args=(scores, y), method="Nelder-Mead",
        options={"initial_simplex": simplex, "xatol": 1e-4, "fatol": 1e-6, "maxiter": 2000},
    )
    return np.sort(res.x), -res.fun


def optimize_thresholds(scores, y, seed=C.SEED):
    """初期値 [0.5,1.5,2.5,3.5] とその摂動から Nelder-Mead を実行し、最良 QWK の閾値を返す。"""
    rng = np.random.default_rng(seed)
    init = np.asarray(C.INIT_THRESHOLDS, dtype=float)
    starts = [init] + [np.sort(init + rng.normal(0, 0.2, size=len(init))) for _ in range(C.NM_N_RESTARTS)]
    best_t, best_q = init, qwk(y, apply_thresholds(scores, init), K)
    for x0 in starts:
        t, q = _nelder_mead(x0, scores, y)
        # 同点の場合は初期値に近い（＝極端でない）閾値を優先
        if q > best_q + 1e-9 or (abs(q - best_q) <= 1e-9 and np.abs(t - init).sum() < np.abs(best_t - init).sum()):
            best_t, best_q = t, q
    return best_t, best_q


def nested_threshold_estimate(oof):
    """Fold k の閾値を Fold k 以外の OOF で最適化して適用する（閾値の過適合を除いた QWK 推定）。"""
    pred = np.zeros(len(oof), dtype=int)
    for f in sorted(oof["fold"].unique()):
        tr, te = oof["fold"] != f, oof["fold"] == f
        t, _ = optimize_thresholds(oof.loc[tr, "score"].values, oof.loc[tr, "grade"].values)
        pred[te.values] = apply_thresholds(oof.loc[te, "score"].values, t)
    return qwk(oof["grade"], pred, K)


# ------------------------------------------------------------------------------
# 可視化（matplotlib が無い環境ではスキップ）
# ------------------------------------------------------------------------------
def plot_confusion(cm, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    cm_norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, mat, title, fmt in [(axes[0], cm, "Confusion Matrix (count)", "d"), (axes[1], cm_norm, "Normalized by true grade", ".2f")]:
        im = ax.imshow(mat, cmap="Blues")
        ax.set_xticks(LABELS); ax.set_yticks(LABELS)
        ax.set_xlabel("Predicted grade"); ax.set_ylabel("True grade"); ax.set_title(title)
        thr = mat.max() / 2
        for i in LABELS:
            for j in LABELS:
                ax.text(j, i, format(mat[i, j], fmt), ha="center", va="center", color="white" if mat[i, j] > thr else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_score_distribution(oof, thresholds, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    rng = np.random.default_rng(0)
    for g in LABELS:
        s = oof.loc[oof["grade"] == g, "score"].values
        ax.scatter(s, g + rng.uniform(-0.2, 0.2, len(s)), s=12, alpha=0.6, label=f"Grade {g} (n={len(s)})")
    for i, t in enumerate(thresholds):
        ax.axvline(t, color="red", ls="--", lw=1)
        ax.text(t, K - 0.4, f"t{i+1}={t:.2f}", color="red", ha="center", fontsize=8)
    for t in C.INIT_THRESHOLDS:
        ax.axvline(t, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("OOF continuous score (sum of 4 hurdle probabilities)")
    ax.set_ylabel("True grade"); ax.set_yticks(LABELS); ax.set_xlim(-0.1, K - 0.9)
    ax.set_title("Score distribution & optimized thresholds (red) vs. initial (gray)")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ------------------------------------------------------------------------------
def main():
    C.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    oof = pd.read_csv(C.OOF_CSV)
    y, scores = oof["grade"].values.astype(int), oof["score"].values

    # ---- 1. 閾値最適化 ----
    init = np.asarray(C.INIT_THRESHOLDS)
    qwk_init = qwk(y, apply_thresholds(scores, init), K)
    qwk_coral = qwk(y, oof["pred_coral"].values, K)
    thresholds, qwk_opt = optimize_thresholds(scores, y)
    qwk_nested = nested_threshold_estimate(oof)

    print("=== 閾値最適化 (Nelder-Mead) ===")
    print(f"  初期閾値   {np.round(init, 3).tolist()}  →  QWK {qwk_init:.4f}")
    print(f"  最適閾値   {np.round(thresholds, 3).tolist()}  →  QWK {qwk_opt:.4f}")
    print(f"  (参考) CORAL 標準予測 p>0.5         QWK {qwk_coral:.4f}")
    print(f"  (参考) 入れ子推定（閾値も未知Foldで評価）QWK {qwk_nested:.4f}  ← 論文等で報告する際はこちらが厳密")

    with open(C.THRESHOLDS_JSON, "w") as f:
        json.dump({"thresholds": thresholds.tolist(), "qwk_oof": qwk_opt}, f, indent=2)

    # ---- 2. 最終評価 ----
    pred = apply_thresholds(scores, thresholds)
    oof["pred_opt"] = pred
    oof.to_csv(C.EVAL_DIR / "oof_final_predictions.csv", index=False)

    cm = confusion_matrix(y, pred, labels=LABELS)
    cm_df = pd.DataFrame(cm, index=[f"true_{g}" for g in LABELS], columns=[f"pred_{g}" for g in LABELS])
    cm_df.to_csv(C.EVAL_DIR / "confusion_matrix.csv")
    plot_confusion(cm, C.EVAL_DIR / "confusion_matrix.png")
    plot_score_distribution(oof, thresholds, C.EVAL_DIR / "score_distribution.png")

    report = classification_report(y, pred, labels=LABELS, target_names=[f"Grade {g}" for g in LABELS], output_dict=True, zero_division=0)
    rep_df = pd.DataFrame(report).T
    # グレードごとの正解率 = そのグレードの画像のうち正しく当てた割合（= Recall）
    per_grade_acc = {f"Grade {g}": cm[g, g] / max(cm[g].sum(), 1) for g in LABELS}
    rep_df["per_grade_accuracy"] = pd.Series(per_grade_acc)
    rep_df.to_csv(C.EVAL_DIR / "classification_report.csv")

    acc = accuracy_score(y, pred)
    mae = float(np.abs(pred - y).mean())
    within1 = float((np.abs(pred - y) <= 1).mean())

    print("\n=== 混同行列 (行: 正解, 列: 予測) ===")
    print(cm_df.to_string())
    print("\n=== クラス別指標 ===")
    print(rep_df.loc[[f"Grade {g}" for g in LABELS], ["precision", "recall", "f1-score", "per_grade_accuracy", "support"]].round(3).to_string())
    print(f"\nQWK {qwk_opt:.4f} | Accuracy {acc:.4f} | Macro-F1 {report['macro avg']['f1-score']:.4f} | MAE {mae:.3f} | ±1 Accuracy {within1:.4f}")

    # ---- 3. 誤り分析 ----
    pairs = []
    for i in LABELS:
        for j in LABELS:
            if i != j and cm[i, j] > 0:
                pairs.append({
                    "true": i, "pred": j, "count": int(cm[i, j]),
                    "ratio_of_true": round(cm[i, j] / cm[i].sum(), 3),
                    "distance": abs(i - j), "direction": "過大評価" if j > i else "過小評価",
                })
    err_df = pd.DataFrame(pairs, columns=["true", "pred", "count", "ratio_of_true", "distance", "direction"])
    err_df = err_df.sort_values(["count", "distance"], ascending=[False, False])
    err_df.to_csv(C.EVAL_DIR / "error_pairs.csv", index=False)

    n_err = int((pred != y).sum())
    print(f"\n=== 誤り分析（誤分類 {n_err} / {len(y)} 枚） ===")
    if n_err:
        adj = int(err_df.loc[err_df["distance"] == 1, "count"].sum())
        far = n_err - adj
        over = int(err_df.loc[err_df["direction"] == "過大評価", "count"].sum())
        print(f"  隣接グレードの誤り (|差|=1): {adj} 枚 ({adj / n_err * 100:.1f}%)")
        print(f"  2段階以上の誤り   (|差|>=2): {far} 枚 ({far / n_err * 100:.1f}%)")
        print(f"  過大評価 {over} 枚 / 過小評価 {n_err - over} 枚")
        print("  誤りの多い組み合わせ Top 5:")
        for r in err_df.head(5).itertuples():
            print(f"    Grade {r.true} → Grade {r.pred} と誤認: {r.count} 枚（Grade {r.true} の {r.ratio_of_true * 100:.1f}%）")

    summary = {
        "thresholds_init": init.tolist(), "thresholds_opt": thresholds.tolist(),
        "qwk_init": qwk_init, "qwk_opt": qwk_opt, "qwk_coral": qwk_coral, "qwk_nested": qwk_nested,
        "accuracy": acc, "macro_f1": report["macro avg"]["f1-score"], "weighted_f1": report["weighted avg"]["f1-score"],
        "mae": mae, "within1_accuracy": within1,
        "per_grade_accuracy": {k: float(v) for k, v in per_grade_acc.items()},
        "per_grade_f1": {f"Grade {g}": report[f"Grade {g}"]["f1-score"] for g in LABELS},
    }
    with open(C.EVAL_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n保存: {C.THRESHOLDS_JSON}\n保存: {C.EVAL_DIR}/")


if __name__ == "__main__":
    main()
