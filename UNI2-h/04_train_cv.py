"""
04_train_cv.py
==============
フェーズ 4: 5 分割層化交差検証による マルチ概念 MLP + CORAL の学習

    - 10 概念すべてを 1 つのモデルで同時に学習（損失 = 各概念の CORAL Loss の平均）
    - AdamW、Early Stopping（最良エポックの重みを復元）
    - 各 Fold の検証データの予測を全サンプル分つなぎ、Out-of-Fold (OOF) 予測として保存

出力:
    outputs/oof_predictions.csv  列: slide_id, fold, <概念>_true, <概念>_score, <概念>_pred_coral ...
    outputs/oof_probs.npy        [N, 概念数, 4] のハードル確率
    outputs/fold_summary.csv     Fold ごとの最良エポック・損失・概念平均 QWK
    outputs/models/fold{k}.pt    各 Fold のモデル（標準化パラメータ込み）

使い方:
    python 04_train_cv.py
"""
import copy
import json

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

import config as C
from common import concept_names_from_labels, grades_from_labels, load_feature_matrix, qwk, set_seed
from coral_model import MultiConceptCoralMLP, coral_loss, label_to_levels, logits_to_grade, logits_to_probs, logits_to_score


@torch.no_grad()
def evaluate(model, X, levels):
    model.eval()
    logits = model(X)
    return coral_loss(logits, levels).item(), logits


def mean_concept_qwk(grades_true, grades_pred):
    """概念ごとの QWK の平均。grades_*: [N, M]（1〜5 表記）"""
    return float(np.mean([qwk(grades_true[:, m], grades_pred[:, m]) for m in range(grades_true.shape[1])]))


def train_one_fold(X_fit, y_fit, X_es, y_es, num_concepts, device, fold):
    """y_* は内部表記 0〜4 の [N, M]。X_es/y_es は Early Stopping 監視用。"""
    set_seed(C.SEED + fold)
    model = MultiConceptCoralMLP(C.FEATURE_DIM, C.HIDDEN_DIM, num_concepts, C.NUM_GRADES, C.DROPOUT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    lv_fit, lv_es = label_to_levels(y_fit, C.NUM_GRADES), label_to_levels(y_es, C.NUM_GRADES)

    gen = torch.Generator().manual_seed(C.SEED + fold)
    best = {"loss": float("inf"), "epoch": 0, "state": None}
    history = {"train_loss": [], "es_loss": [], "es_qwk": []}
    wait = 0
    for epoch in range(1, C.MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(X_fit), generator=gen).to(device)
        total, n = 0.0, 0
        for i in range(0, len(perm), C.BATCH_SIZE):
            idx = perm[i: i + C.BATCH_SIZE]
            loss = coral_loss(model(X_fit[idx]), lv_fit[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
            n += len(idx)

        es_loss, es_logits = evaluate(model, X_es, lv_es)
        es_pred = logits_to_grade(es_logits).cpu().numpy()
        history["train_loss"].append(total / n)
        history["es_loss"].append(es_loss)
        history["es_qwk"].append(mean_concept_qwk(y_es.cpu().numpy() + C.GRADE_MIN, es_pred))

        if es_loss < best["loss"] - C.MIN_DELTA:
            best = {"loss": es_loss, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
            wait = 0
        else:
            wait += 1
            if wait >= C.PATIENCE:
                break
    model.load_state_dict(best["state"])
    print(f"  stop@{epoch:3d}  best_epoch={best['epoch']:3d}  es_loss={best['loss']:.4f}")
    return model, best, history


def main():
    set_seed(C.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 小さな MLP は CPU/CUDA で十分

    df = pd.read_csv(C.LABELS_CSV)
    concepts = concept_names_from_labels(df)
    M = len(concepts)
    X_all = load_feature_matrix(df, C.FEATURE_DIR, C.FEATURE_DIM)       # [N, 1536]
    G_all = grades_from_labels(df)                                       # [N, M] (1〜5)
    Y_all = torch.tensor(G_all - C.GRADE_MIN, dtype=torch.long)          # [N, M] (0〜4)
    strat = df["strat_label"].astype(str).values
    print(f"[INFO] 特徴量 {tuple(X_all.shape)} / 概念数 {M} / device={device}")

    C.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    skf = StratifiedKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=C.SEED)

    oof_logits = torch.zeros(len(df), M, C.NUM_GRADES - 1)
    oof_fold = np.zeros(len(df), dtype=int)
    summaries, histories = [], {}

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(df)), strat), start=1):
        if C.INNER_VAL_RATIO > 0:
            fit_idx, es_idx = train_test_split(tr_idx, test_size=C.INNER_VAL_RATIO, stratify=strat[tr_idx], random_state=C.SEED + fold)
        else:
            fit_idx, es_idx = tr_idx, va_idx

        # 訓練データの統計量だけで標準化
        if C.STANDARDIZE_FEATURES:
            mean, std = X_all[fit_idx].mean(0), X_all[fit_idx].std(0).clamp(min=1e-6)
        else:
            mean, std = torch.zeros(C.FEATURE_DIM), torch.ones(C.FEATURE_DIM)
        norm = lambda idx: ((X_all[idx] - mean) / std).to(device)

        print(f"[Fold {fold}/{C.N_FOLDS}] fit={len(fit_idx)}  es={len(es_idx)}  val={len(va_idx)}")
        model, best, hist = train_one_fold(norm(fit_idx), Y_all[fit_idx].to(device), norm(es_idx), Y_all[es_idx].to(device), M, device, fold)

        va_loss, va_logits = evaluate(model, norm(va_idx), label_to_levels(Y_all[va_idx], C.NUM_GRADES).to(device))
        va_logits = va_logits.cpu()
        oof_logits[va_idx] = va_logits
        oof_fold[va_idx] = fold

        va_score = logits_to_score(va_logits).numpy()
        va_true = G_all[va_idx]
        summaries.append({
            "fold": fold, "n_train": len(fit_idx), "n_val": len(va_idx),
            "best_epoch": best["epoch"], "es_loss": round(best["loss"], 5), "val_loss": round(va_loss, 5),
            "val_qwk_coral": round(mean_concept_qwk(va_true, logits_to_grade(va_logits).numpy()), 4),
            "val_qwk_round": round(mean_concept_qwk(va_true, np.clip(np.rint(va_score), C.GRADE_MIN, C.GRADE_MIN + C.NUM_GRADES - 1)), 4),
            "val_mae": round(float(np.abs(va_score - va_true).mean()), 4),
        })
        histories[f"fold{fold}"] = hist

        torch.save({
            "state_dict": model.state_dict(), "mean": mean, "std": std, "fold": fold, "best_epoch": best["epoch"],
            "concepts": concepts,
            "model_kwargs": {"in_dim": C.FEATURE_DIM, "hidden_dim": C.HIDDEN_DIM, "num_concepts": M,
                             "num_grades": C.NUM_GRADES, "dropout": C.DROPOUT},
        }, C.MODEL_DIR / f"fold{fold}.pt")

    assert (oof_fold > 0).all(), "OOF 予測が埋まっていないサンプルがあります"

    # ---- OOF 保存 ----
    probs = logits_to_probs(oof_logits).numpy()                          # [N, M, 4]
    scores = logits_to_score(oof_logits).numpy()                         # [N, M]
    preds = logits_to_grade(oof_logits).numpy()                          # [N, M]
    np.save(C.OOF_PROBS_NPY, probs)
    oof = pd.DataFrame({"slide_id": df["slide_id"], "fold": oof_fold})
    for m, name in enumerate(concepts):
        oof[f"{name}_true"] = G_all[:, m]
        oof[f"{name}_score"] = scores[:, m]
        oof[f"{name}_pred_coral"] = preds[:, m]
    oof.to_csv(C.OOF_CSV, index=False)

    summary = pd.DataFrame(summaries)
    summary.to_csv(C.FOLD_SUMMARY_CSV, index=False)
    with open(C.HISTORY_JSON, "w") as f:
        json.dump(histories, f)

    print("\n=== Fold Summary ===")
    print(summary.to_string(index=False))
    rounded = np.clip(np.rint(scores), C.GRADE_MIN, C.GRADE_MIN + C.NUM_GRADES - 1)
    print("\n=== OOF QWK（概念別） ===")
    print(f"  {'concept':<20s} {'CORAL(p>0.5)':>12s} {'score四捨五入':>12s}")
    for m, name in enumerate(concepts):
        print(f"  {name:<20s} {qwk(G_all[:, m], preds[:, m]):12.4f} {qwk(G_all[:, m], rounded[:, m]):12.4f}")
    print(f"  {'平均':<20s} {mean_concept_qwk(G_all, preds):12.4f} {mean_concept_qwk(G_all, rounded):12.4f}")
    if C.INNER_VAL_RATIO == 0:
        print("[NOTE] 検証 Fold で Early Stopping しているため、わずかに楽観的な推定です（config.INNER_VAL_RATIO 参照）。")
    print(f"\n保存: {C.OOF_CSV}\n保存: {C.OOF_PROBS_NPY}\n保存: {C.FOLD_SUMMARY_CSV}\n保存: {C.MODEL_DIR}/fold*.pt")


if __name__ == "__main__":
    main()
