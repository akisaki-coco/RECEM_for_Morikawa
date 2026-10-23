"""
06_predict.py
=============
学習済みモデルによる新規画像の推論（5 Fold アンサンブル + 概念ごとの最適閾値）

    1. 画像を UNI2-h で [1536] 特徴量に変換（02 と同じ前処理）
    2. fold1〜5 それぞれで、各概念の連続スコア (1.0〜5.0) を算出して平均
    3. thresholds.json の概念ごとの閾値で 1〜5 のグレードに変換

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
from coral_model import MultiConceptCoralMLP, logits_to_score


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
    models, concepts = [], None
    for ck in ckpts:
        d = torch.load(ck, map_location="cpu")
        m = MultiConceptCoralMLP(**d["model_kwargs"])
        m.load_state_dict(d["state_dict"])
        m.eval()
        models.append((m, d["mean"], d["std"]))
        concepts = d["concepts"]
    return models, concepts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="画像ファイルまたはディレクトリ")
    parser.add_argument("--out", default=str(C.OUTPUT_DIR / "predictions.csv"))
    args = parser.parse_args()

    with open(C.THRESHOLDS_JSON) as f:
        thresholds = json.load(f)["thresholds"]          # {概念名: [t1..t4]}
    models, concepts = load_fold_models()
    paths = collect_images(args.inputs)
    print(f"[INFO] 画像 {len(paths)} 枚 / Fold モデル {len(models)} 個 / 概念 {len(concepts)} 個")

    from uni_encoder import build_transform, build_uni2h, extract_feature, load_image

    device = get_device()
    encoder = build_uni2h(C.UNI_MODEL_NAME, dynamic_img_size=True).to(device)
    transform = build_transform(C.INPUT_SIZE, C.RESIZE_MODE)

    rows = []
    with torch.no_grad():
        for p in paths:
            feat = extract_feature(encoder, transform(load_image(p)), device, use_amp=C.USE_AMP)      # [1536]
            # 各 Fold のスコア [M] を平均
            scores = torch.stack([
                logits_to_score(m(((feat - mean) / std).unsqueeze(0)))[0] for m, mean, std in models
            ]).mean(0)
            row = {"image_path": str(p)}
            for k, name in enumerate(concepts):
                s = float(scores[k])
                row[f"{name}_score"] = round(s, 4)
                row[f"{name}_grade"] = int(apply_thresholds([s], thresholds[name])[0])
            rows.append(row)
            print(f"  {p.name}: " + " ".join(f"{n}={row[f'{n}_grade']}" for n in concepts))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"保存: {args.out}")


if __name__ == "__main__":
    main()
