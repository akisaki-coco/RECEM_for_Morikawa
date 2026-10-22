"""
01_prepare_data.py
==================
フェーズ 1-b: データと正解ラベルの紐付け・クラス不均衡の集計

入力（config.py で指定）:
    (A) SOURCE_CSV: 画像ファイル名列 (FILENAME_COL) とグレード列 (GRADE_COL) を持つ CSV
    (B) SOURCE_CSV = None: IMG_DIR/<grade>/*.png のフォルダ構成

出力:
    data/labels.csv              列: slide_id, image_path, grade (0〜4), source_name
    data/class_distribution.csv  グレードごとの枚数・割合

使い方:
    python 01_prepare_data.py
"""
import re
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

import config as C


def _resolve_path(p: str) -> Path:
    p = Path(str(p).strip())
    return p if p.is_absolute() else Path(C.IMG_DIR) / p


def load_from_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in (C.FILENAME_COL, C.GRADE_COL):
        if col not in df.columns:
            sys.exit(f"[ERROR] {csv_path} に列 '{col}' がありません。列一覧: {list(df.columns)}")
    df = df[[C.FILENAME_COL, C.GRADE_COL]].rename(columns={C.FILENAME_COL: "source_name", C.GRADE_COL: "grade"})
    n_before = len(df)
    df = df.dropna(subset=["grade"])
    if len(df) < n_before:
        print(f"[WARN] グレードが欠損している {n_before - len(df)} 行を除外しました")
    df["image_path"] = df["source_name"].map(lambda s: str(_resolve_path(s)))
    return df


def load_from_folders(img_dir: Path) -> pd.DataFrame:
    rows = []
    for sub in sorted(p for p in Path(img_dir).iterdir() if p.is_dir()):
        m = re.search(r"\d+", sub.name)
        if m is None:
            print(f"[WARN] フォルダ名に数字がないためスキップ: {sub.name}")
            continue
        for f in sorted(sub.rglob("*")):
            if f.suffix.lower() in C.IMG_EXTS:
                rows.append({"source_name": str(f.relative_to(img_dir)), "grade": int(m.group()), "image_path": str(f)})
    if not rows:
        sys.exit(f"[ERROR] {img_dir} 以下に画像が見つかりません")
    return pd.DataFrame(rows)


def normalize_grades(df: pd.DataFrame) -> pd.DataFrame:
    grades = df["grade"].astype(float)
    if not (grades == grades.round()).all():
        sys.exit("[ERROR] グレードに整数でない値が含まれています")
    grades = grades.astype(int)
    offset = C.GRADE_OFFSET
    if offset is None:
        offset = 1 if (grades.min() >= 1 and grades.max() == C.NUM_CLASSES) else 0
    if offset:
        print(f"[INFO] グレードを {offset} だけシフトします（例: {grades.min()}〜{grades.max()} → {grades.min()-offset}〜{grades.max()-offset}）")
    df["grade"] = grades - offset
    bad = df[(df["grade"] < 0) | (df["grade"] >= C.NUM_CLASSES)]
    if len(bad):
        sys.exit(f"[ERROR] 0〜{C.NUM_CLASSES-1} の範囲外のグレードがあります:\n{bad.head()}")
    return df


def validate_images(df: pd.DataFrame) -> pd.DataFrame:
    missing, size_mismatch = [], []
    for p in df["image_path"]:
        path = Path(p)
        if not path.exists():
            missing.append(p)
            continue
        with Image.open(path) as im:
            if im.size != tuple(C.EXPECTED_IMG_SIZE):
                size_mismatch.append((path.name, im.size))
    if missing:
        print(f"[ERROR] 画像が見つかりません: {len(missing)} 件（先頭5件）")
        for p in missing[:5]:
            print("   ", p)
        sys.exit(1)
    if size_mismatch:
        print(f"[WARN] {C.EXPECTED_IMG_SIZE} 以外のサイズの画像が {len(size_mismatch)} 件あります（前処理で 896 に揃えます）: {size_mismatch[:3]}")
    return df


def main():
    if C.SOURCE_CSV is not None and Path(C.SOURCE_CSV).exists():
        print(f"[INFO] CSV から読み込み: {C.SOURCE_CSV}")
        df = load_from_csv(Path(C.SOURCE_CSV))
    elif C.SOURCE_CSV is None:
        print(f"[INFO] フォルダ構成から読み込み: {C.IMG_DIR}")
        df = load_from_folders(Path(C.IMG_DIR))
    else:
        sys.exit(f"[ERROR] SOURCE_CSV が見つかりません: {C.SOURCE_CSV}（config.py を確認してください）")

    if df["source_name"].duplicated().any():
        dup = df[df["source_name"].duplicated(keep=False)]
        sys.exit(f"[ERROR] 同じ画像が重複しています:\n{dup.head()}")

    df = normalize_grades(df)
    df = validate_images(df)

    # ファイル名でソートして slide_001, slide_002, ... の ID を付与（再実行しても同じ ID になる）
    df = df.sort_values("source_name").reset_index(drop=True)
    width = max(3, len(str(len(df))))
    df.insert(0, "slide_id", [f"slide_{i+1:0{width}d}" for i in range(len(df))])
    df = df[["slide_id", "image_path", "grade", "source_name"]]

    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.LABELS_CSV, index=False)

    # クラス不均衡の集計
    counts = df["grade"].value_counts().reindex(range(C.NUM_CLASSES), fill_value=0)
    dist = pd.DataFrame({"grade": counts.index, "count": counts.values, "ratio": (counts.values / len(df)).round(4)})
    dist.to_csv(C.CLASS_DIST_CSV, index=False)

    print(f"\n総画像数: {len(df)}")
    print("グレード別枚数:")
    for _, r in dist.iterrows():
        bar = "█" * int(round(r["ratio"] * 50))
        print(f"  Grade {int(r['grade'])}: {int(r['count']):4d} 枚 ({r['ratio']*100:5.1f}%) {bar}")
    min_count = int(counts.min())
    if min_count < C.N_FOLDS:
        print(f"[WARN] 最少クラスが {min_count} 枚で、{C.N_FOLDS}-Fold の層化分割ができません。")
    elif counts.max() / max(min_count, 1) > 3:
        print("[INFO] クラス不均衡が大きめです。config.USE_CLASS_WEIGHTS = True も検討してください。")
    print(f"\n保存: {C.LABELS_CSV}\n保存: {C.CLASS_DIST_CSV}")


if __name__ == "__main__":
    main()
