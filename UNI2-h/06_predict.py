"""
06_predict.py
=============
学習済みモデルによる新規画像の推論（5 Fold アンサンブル + 最適閾値）

    1. 画像を UNI2-h で [1536] 特徴量に変換（02 と同じ前処理）
    2. outputs/models/fold1〜5.pt それぞれで連続スコア (0.0〜4.0) を算出し平均
    3. outputs/thresholds.json の閾値で 5 段階 (0〜4) に変換

使い方:
    python 06_predict.py path/to/img1.png path/to/img2.png
    python 06_predict.py path/to/image_dir --out predictions.csv
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch

import config as C
from common import apply_thresholds, get_device
from coral_model import CoralMLP, logits_to_probs


def collect_images(inputs):
    paths = []
    for p in map(Path, inputs):
        if p.is_dir():
            paths += sorted(f for f in p.rglob("*") if f.suffix.lower() in C.IMG_EXTS)
        elif p.exists():
            paths.append(p)
        else:
            print(f"[WARN] 見つかりません: {p}")
    return paths


def load_fold_models():
    ckpts = sorted(C.MODEL_DIR.glob("fold*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"{C.MODEL_DIR} に学習済みモデルがありません（先に 04_train_cv.py を実行）")
    models = []
    for ck in ckpts:
        d = torch.load(ck, map_location="cpu")
        m = CoralMLP(**d["model_kwargs"])
        m.load_state_dict(d["state_dict"])
        m.eval()
        models.append((m, d["mean"], d["std"]))
    return models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="画像ファイルまたはディレクトリ")
    parser.add_argument("--out", default=str(C.OUTPUT_DIR / "predictions.csv"))
    args = parser.parse_args()

    with open(C.THRESHOLDS_JSON) as f:
        thresholds = json.load(f)["thresholds"]
    models = load_fold_models()
    paths = collect_images(args.inputs)
    print(f"[INFO] 画像 {len(paths)} 枚 / Fold モデル {len(models)} 個 / 閾値 {[round(t, 3) for t in thresholds]}")

    from uni_encoder import build_transform, build_uni2h, extract_feature, load_image

    device = get_device()
    encoder = build_uni2h(C.UNI_MODEL_NAME, dynamic_img_size=True).to(device)
    transform = build_transform(C.INPUT_SIZE, C.RESIZE_MODE)

    rows = []
    with torch.no_grad():
        for p in paths:
            feat = extract_feature(encoder, transform(load_image(p)), device, use_amp=C.USE_AMP)  # [1536]
            probs = torch.stack([logits_to_probs(m(((feat - mean) / std).unsqueeze(0)))[0] for m, mean, std in models])
            mean_probs = probs.mean(0)                     # Fold 平均のハードル確率 [4]
            score = float(probs.sum(1).mean())             # Fold 平均の連続スコア
            grade = int(apply_thresholds([score], thresholds)[0])
            rows.append({"image_path": str(p), "score": round(score, 4), "pred_grade": grade,
                         **{f"prob_gt{k}": round(float(v), 4) for k, v in enumerate(mean_probs)}})
            print(f"  {p.name}: score={score:.3f} → Grade {grade}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"保存: {args.out}")


if __name__ == "__main__":
    main()
