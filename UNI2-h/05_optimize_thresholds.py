"""
05_optimize_thresholds.py
=========================
フェーズ 5: 概念ごとの Nelder-Mead 閾値最適化と最終評価

    各概念 m について:
      1. OOF 連続スコア (1.0〜5.0) と正解グレードから、QWK を最大化する 4 閾値を探索
         （初期値 [1.5, 2.5, 3.5, 4.5]、scipy.optimize.minimize(method="Nelder-Mead")）
      2. 最適閾値で 1〜5 を決定し、QWK / Accuracy / ±1 Accuracy / MAE / 混同行列を算出
      3. 入れ子推定（Fold k 以外で閾値最適化 → Fold k に適用）の QWK も算出（厳密な性能推定）
    全概念の平均と、誤りの多い概念・グレード間の分析も出力する。

出力 (outputs/):
    thresholds.json                     概念ごとの最適閾値
    evaluation/concept_summary.csv      概念別の指標一覧
    evaluation/confusion_<概念>.csv     概念別の混同行列
    evaluation/confusion_matrices.png   全概念の混同行列
    evaluation/error_pairs.csv          概念ごとの誤りパターン
    evaluation/oof_final_predictions.csv

使い方:
    python 05_optimize_thresholds.py
"""
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import confusion_matrix, f1_score

import config as C
from common import apply_thresholds, qwk

GRADES = list(range(C.GRADE_MIN, C.GRADE_MIN + C.NUM_GRADES))


def _neg_qwk(t, scores, y):
    return -qwk(y, apply_thresholds(scores, t))


def _nelder_mead(x0, scores, y):
    x0 = np.asarray(x0, dtype=float)
    # QWK は階段状の関数なので、初期シンプレックスを大きめに取る
    simplex = np.vstack([x0] + [x0 + C.NM_INITIAL_STEP * np.eye(len(x0))[i] for i in range(len(x0))])
    res = minimize(_neg_qwk, x0, args=(scores, y), method="Nelder-Mead",
                   options={"initial_simplex": simplex, "xatol": 1e-4, "fatol": 1e-6, "maxiter": 2000})
    return np.sort(res.x), -res.fun


def optimize_thresholds(scores, y, seed=C.SEED):
    """初期値とその摂動から Nelder-Mead を実行し、最良 QWK の閾値を返す。"""
    rng = np.random.default_rng(seed)
    init = np.asarray(C.INIT_THRESHOLDS, dtype=float)
    starts = [init] + [np.sort(init + rng.normal(0, 0.2, size=len(init))) for _ in range(C.NM_N_RESTARTS)]
    best_t, best_q = init, qwk(y, apply_thresholds(scores, init))
    for x0 in starts:
        t, q = _nelder_mead(x0, scores, y)
        if q > best_q + 1e-9 or (abs(q - best_q) <= 1e-9 and np.abs(t - init).sum() < np.abs(best_t - init).sum()):
            best_t, best_q = t, q
    return best_t, best_q


def nested_qwk(scores, y, folds):
    """Fold k の閾値を Fold k 以外で最適化して適用したときの QWK。"""
    pred = np.zeros(len(y), dtype=int)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        t, _ = optimize_thresholds(scores[tr], y[tr])
        pred[te] = apply_thresholds(scores[te], t)
    return qwk(y, pred)


def plot_confusions(cms, names, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    n = len(names)
    cols = min(5, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, cm, name in zip(axes.ravel(), cms, names):
        ax.axis("on")
        ax.imshow(cm, cmap="Blues")
        ax.set_title(name, fontsize=9)
        ax.set_xticks(range(len(GRADES))); ax.set_xticklabels(GRADES)
        ax.set_yticks(range(len(GRADES))); ax.set_yticklabels(GRADES)
        ax.set_xlabel("pred", fontsize=8); ax.set_ylabel("true", fontsize=8)
        thr = cm.max() / 2 if cm.max() else 1
        for i in range(len(GRADES)):
            for j in range(len(GRADES)):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7, color="white" if cm[i, j] > thr else "black")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    C.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    oof = pd.read_csv(C.OOF_CSV)
    concepts = [c[:-len("_true")] for c in oof.columns if c.endswith("_true")]
    folds = oof["fold"].values
    init = np.asarray(C.INIT_THRESHOLDS)

    thresholds, rows, cms, err_rows = {}, [], [], []
    final = oof[["slide_id", "fold"]].copy()

    for name in concepts:
        y = oof[f"{name}_true"].values.astype(int)
        s = oof[f"{name}_score"].values
        t, q_opt = optimize_thresholds(s, y)
        pred = apply_thresholds(s, t)
        thresholds[name] = t.tolist()
        final[f"{name}_true"], final[f"{name}_pred_opt"] = y, pred

        cm = confusion_matrix(y, pred, labels=GRADES)
        cms.append(cm)
        pd.DataFrame(cm, index=[f"true_{g}" for g in GRADES], columns=[f"pred_{g}" for g in GRADES]).to_csv(C.EVAL_DIR / f"confusion_{name}.csv")

        rows.append({
            "concept": name,
            "qwk_init": qwk(y, apply_thresholds(s, init)),
            "qwk_coral": qwk(y, oof[f"{name}_pred_coral"].values),
            "qwk_opt": q_opt,
            "qwk_nested": nested_qwk(s, y, folds),
            "accuracy": float((pred == y).mean()),
            "within1": float((np.abs(pred - y) <= 1).mean()),
            "mae": float(np.abs(pred - y).mean()),
            "macro_f1": float(f1_score(y, pred, labels=GRADES, average="macro", zero_division=0)),
            "t1": t[0], "t2": t[1], "t3": t[2], "t4": t[3],
        })
        for i, g_true in enumerate(GRADES):
            for j, g_pred in enumerate(GRADES):
                if i != j and cm[i, j] > 0:
                    err_rows.append({"concept": name, "true": g_true, "pred": g_pred, "count": int(cm[i, j]),
                                     "ratio_of_true": round(cm[i, j] / cm[i].sum(), 3), "distance": abs(i - j),
                                     "direction": "過大評価" if j > i else "過小評価"})

    summary = pd.DataFrame(rows)
    mean_row = {"concept": "平均", **{c: summary[c].mean() for c in summary.columns if c != "concept"}}
    summary_out = pd.concat([summary, pd.DataFrame([mean_row])], ignore_index=True)
    summary_out.to_csv(C.EVAL_DIR / "concept_summary.csv", index=False)
    final.to_csv(C.EVAL_DIR / "oof_final_predictions.csv", index=False)
    err_df = pd.DataFrame(err_rows).sort_values(["count", "distance"], ascending=False)
    err_df.to_csv(C.EVAL_DIR / "error_pairs.csv", index=False)
    plot_confusions(cms, concepts, C.EVAL_DIR / "confusion_matrices.png")
    with open(C.THRESHOLDS_JSON, "w") as f:
        json.dump({"thresholds": thresholds, "qwk_oof": {r["concept"]: r["qwk_opt"] for r in rows}}, f, indent=2, ensure_ascii=False)

    show = summary_out[["concept", "qwk_init", "qwk_coral", "qwk_opt", "qwk_nested", "accuracy", "within1", "mae"]]
    print("=== 概念別の評価（OOF, Nelder-Mead 最適化） ===")
    print(show.round(4).to_string(index=False))
    print("\n  qwk_coral : CORAL 標準 (p>0.5)   qwk_init : 初期閾値   qwk_opt : 最適閾値（楽観的）")
    print("  qwk_nested: 閾値も未知 Fold で評価（報告する際はこちらが厳密）")

    worst = summary.sort_values("qwk_nested").head(3)
    print("\n=== QWK（入れ子推定）が低い概念 Top 3 ===")
    for r in worst.itertuples():
        print(f"  {r.concept}: {r.qwk_nested:.4f}")

    # 誤り分析（全概念の合計）
    tot_err = int(err_df["count"].sum()) if len(err_df) else 0
    if tot_err:
        adj = int(err_df.loc[err_df["distance"] == 1, "count"].sum())
        over = int(err_df.loc[err_df["direction"] == "過大評価", "count"].sum())
        print(f"\n=== 誤り分析（全概念合計 {tot_err} 件） ===")
        print(f"  隣接グレードの誤り: {adj} 件 ({adj / tot_err * 100:.1f}%) / 2段階以上: {tot_err - adj} 件")
        print(f"  過大評価 {over} 件 / 過小評価 {tot_err - over} 件")
        print("  誤りの多い (概念, 正解→予測) Top 5:")
        for r in err_df.head(5).itertuples():
            print(f"    {r.concept}: Grade {r.true} → {r.pred} と誤認 {r.count} 枚（Grade {r.true} の {r.ratio_of_true * 100:.1f}%）")

    print(f"\n保存: {C.THRESHOLDS_JSON}\n保存: {C.EVAL_DIR}/")


if __name__ == "__main__":
    main()
