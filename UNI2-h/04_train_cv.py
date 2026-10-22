"""
04_train_cv.py
==============
フェーズ 4: 5 分割層化交差検証 (Stratified 5-Fold CV) による MLP + CORAL の学習

    - 400 枚を 5 Fold に層化分割（各 Fold: 訓練 320 / 検証 80）
    - AdamW (lr=1e-4, weight_decay=0.01)
    - エポックごとに検証損失を監視し、Early Stopping（最良エポックの重みを復元）
    - 各 Fold の検証データに対する予測（4 つのハードル確率と連続スコア 0.0〜4.0）を
      400 枚分つなぎ合わせて Out-of-Fold (OOF) 予測として保存

出力:
    outputs/oof_predictions.csv  列: slide_id, grade, fold, prob_gt0..prob_gt3, score, pred_coral
    outputs/fold_summary.csv     Fold ごとの最良エポック・検証損失・QWK など
    outputs/train_history.json   学習曲線
    outputs/models/fold{k}.pt    各 Fold のモデル（標準化パラメータ込み。06_predict.py で使用）

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
from common import load_feature_matrix, qwk, set_seed
from coral_model import CoralMLP, coral_loss, label_to_levels, logits_to_label, logits_to_probs, logits_to_score


def make_sample_weights(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    """クラス頻度の逆数に比例するサンプル重み（平均が 1 になるよう正規化）。"""
    counts = torch.bincount(y, minlength=num_classes).float().clamp(min=1)
    w = (len(y) / (num_classes * counts))[y]
    return w / w.mean()


@torch.no_grad()
def evaluate(model, X, y, levels):
    model.eval()
    logits = model(X)
    loss = coral_loss(logits, levels).item()
    return loss, logits


def train_one_fold(X_fit, y_fit, X_es, y_es, device, fold):
    """1 Fold 分の学習。X_es/y_es は Early Stopping の監視用データ。"""
    set_seed(C.SEED + fold)
    model = CoralMLP(C.FEATURE_DIM, C.HIDDEN_DIM, C.NUM_CLASSES, C.DROPOUT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)

    lv_fit = label_to_levels(y_fit, C.NUM_CLASSES)
    lv_es = label_to_levels(y_es, C.NUM_CLASSES)
    sw_fit = make_sample_weights(y_fit, C.NUM_CLASSES) if C.USE_CLASS_WEIGHTS else None

    gen = torch.Generator().manual_seed(C.SEED + fold)
    best = {"loss": float("inf"), "epoch": 0, "state": None}
    history = {"train_loss": [], "es_loss": [], "es_qwk": []}
    wait = 0

    for epoch in range(1, C.MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(X_fit), generator=gen).to(device)
        total, n = 0.0, 0
        for i in range(0, len(perm), C.BATCH_SIZE):
            idx = perm[i : i + C.BATCH_SIZE]
            logits = model(X_fit[idx])
            loss = coral_loss(logits, lv_fit[idx], sw_fit[idx] if sw_fit is not None else None)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
            n += len(idx)

        es_loss, es_logits = evaluate(model, X_es, y_es, lv_es)
        es_qwk = qwk(y_es.cpu(), logits_to_label(es_logits).cpu(), C.NUM_CLASSES)
        history["train_loss"].append(total / n)
        history["es_loss"].append(es_loss)
        history["es_qwk"].append(es_qwk)

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 小さな MLP のため MPS より CPU の方が速い

    df = pd.read_csv(C.LABELS_CSV)
    X_all = load_feature_matrix(df, C.FEATURE_DIR, C.FEATURE_DIM)          # [N, 1536]
    y_all = torch.tensor(df["grade"].values, dtype=torch.long)              # [N]
    print(f"[INFO] 特徴量 {tuple(X_all.shape)}  device={device}")
    print(f"[INFO] グレード分布: {np.bincount(y_all.numpy(), minlength=C.NUM_CLASSES).tolist()}")

    C.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    skf = StratifiedKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=C.SEED)

    oof_logits = torch.zeros(len(df), C.NUM_CLASSES - 1)
    oof_fold = np.zeros(len(df), dtype=int)
    summaries, histories = [], {}

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y_all)), y_all.numpy()), start=1):
        # Early Stopping 監視用データの決定
        if C.INNER_VAL_RATIO > 0:
            fit_idx, es_idx = train_test_split(
                tr_idx, test_size=C.INNER_VAL_RATIO, stratify=y_all[tr_idx].numpy(), random_state=C.SEED + fold
            )
        else:
            fit_idx, es_idx = tr_idx, va_idx

        # 訓練データの統計量だけで標準化（検証データの情報は使わない）
        if C.STANDARDIZE_FEATURES:
            mean = X_all[fit_idx].mean(0)
            std = X_all[fit_idx].std(0).clamp(min=1e-6)
        else:
            mean = torch.zeros(C.FEATURE_DIM)
            std = torch.ones(C.FEATURE_DIM)
        norm = lambda idx: ((X_all[idx] - mean) / std).to(device)

        print(f"[Fold {fold}/{C.N_FOLDS}] fit={len(fit_idx)}  es={len(es_idx)}  val={len(va_idx)}")
        model, best, hist = train_one_fold(
            norm(fit_idx), y_all[fit_idx].to(device), norm(es_idx), y_all[es_idx].to(device), device, fold
        )

        # 検証 Fold の予測（OOF）
        y_va = y_all[va_idx]
        va_loss, va_logits = evaluate(model, norm(va_idx), y_va.to(device), label_to_levels(y_va, C.NUM_CLASSES).to(device))
        va_logits = va_logits.cpu()
        oof_logits[va_idx] = va_logits
        oof_fold[va_idx] = fold

        va_score = logits_to_score(va_logits).numpy()
        summaries.append({
            "fold": fold,
            "n_train": len(fit_idx),
            "n_val": len(va_idx),
            "best_epoch": best["epoch"],
            "es_loss": round(best["loss"], 5),
            "val_loss": round(va_loss, 5),
            "val_qwk_coral": round(qwk(y_va, logits_to_label(va_logits), C.NUM_CLASSES), 4),
            "val_qwk_round": round(qwk(y_va, np.clip(np.rint(va_score), 0, C.NUM_CLASSES - 1), C.NUM_CLASSES), 4),
            "val_mae": round(float(np.abs(va_score - y_va.numpy()).mean()), 4),
        })
        histories[f"fold{fold}"] = hist

        torch.save({
            "state_dict": model.state_dict(),
            "mean": mean,
            "std": std,
            "fold": fold,
            "best_epoch": best["epoch"],
            "model_kwargs": {"in_dim": C.FEATURE_DIM, "hidden_dim": C.HIDDEN_DIM, "num_classes": C.NUM_CLASSES, "dropout": C.DROPOUT},
        }, C.MODEL_DIR / f"fold{fold}.pt")

    # ---- OOF 予測の保存 ----
    probs = logits_to_probs(oof_logits).numpy()
    oof = pd.DataFrame({"slide_id": df["slide_id"], "grade": y_all.numpy(), "fold": oof_fold})
    for k in range(C.NUM_CLASSES - 1):
        oof[f"prob_gt{k}"] = probs[:, k]
    oof["score"] = probs.sum(1)
    oof["pred_coral"] = logits_to_label(oof_logits).numpy()
    assert (oof["fold"] > 0).all(), "OOF 予測が埋まっていないサンプルがあります"
    oof.to_csv(C.OOF_CSV, index=False)

    summary = pd.DataFrame(summaries)
    summary.to_csv(C.FOLD_SUMMARY_CSV, index=False)
    with open(C.HISTORY_JSON, "w") as f:
        json.dump(histories, f)

    print("\n=== Fold Summary ===")
    print(summary.to_string(index=False))
    print(f"\nOOF QWK (CORAL 標準: p>0.5 の数) : {qwk(oof['grade'], oof['pred_coral'], C.NUM_CLASSES):.4f}")
    print(f"OOF QWK (スコア四捨五入)          : {qwk(oof['grade'], np.clip(np.rint(oof['score']), 0, C.NUM_CLASSES - 1), C.NUM_CLASSES):.4f}")
    print(f"OOF MAE (スコア vs 正解)          : {np.abs(oof['score'] - oof['grade']).mean():.4f}")
    if C.INNER_VAL_RATIO == 0:
        print("[NOTE] 検証 Fold で Early Stopping しているため、上記はわずかに楽観的な推定です（config.INNER_VAL_RATIO 参照）。")
    print(f"\n保存: {C.OOF_CSV}\n保存: {C.FOLD_SUMMARY_CSV}\n保存: {C.MODEL_DIR}/fold*.pt")


if __name__ == "__main__":
    main()
