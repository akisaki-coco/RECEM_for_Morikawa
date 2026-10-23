"""
01_prepare_data.py
==================
フェーズ 1-b: 画像と 10 概念のグレード (1〜5) の紐付け、分布の集計

入力 (config.py):
    CONCEPT_CSV : 列 = filename, 概念1, ..., 概念10（グレード 1〜5）
    CLASS_CSV   : (任意) 列 = filename, class。層化 K-Fold の分割用ラベルとして使用

出力:
    data/labels.csv               列: slide_id, image_path, strat_label, grade_<概念名> × 概念数, source_name
    data/concept_distribution.csv 概念 × グレードの枚数表

使い方:
    python 01_prepare_data.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import config as C


def main():
    csv = Path(C.CONCEPT_CSV)
    if not csv.exists():
        sys.exit(f"[ERROR] CONCEPT_CSV が見つかりません: {csv}（config.py を確認してください）")
    df = pd.read_csv(csv)
    # 列名前後の余分な空白を除去 (例: 'Vasodilation ' -> 'Vasodilation')
    df.columns = [c.strip() for c in df.columns]

    if C.FILENAME_COL not in df.columns:
        sys.exit(f"[ERROR] 列 '{C.FILENAME_COL}' がありません。列一覧: {list(df.columns)}")

    concept_cols = C.CONCEPT_COLS or [c for c in df.columns if c != C.FILENAME_COL]
    concept_cols = [c.strip() for c in concept_cols]

    # 除外概念のフィルタリング
    exclude = getattr(C, "EXCLUDE_CONCEPTS", []) or []
    exclude_clean = [c.strip() for c in exclude]
    if exclude_clean:
        concept_cols = [c for c in concept_cols if c not in exclude_clean]
        print(f"[INFO] 除外した概念 ({len(exclude_clean)} 個): {exclude_clean}")

    missing = [c for c in concept_cols if c not in df.columns]
    if missing:
        sys.exit(f"[ERROR] 概念列が見つかりません: {missing}")
    print(f"[INFO] 対象概念数: {len(concept_cols)} → {concept_cols}")

    df = df[[C.FILENAME_COL] + concept_cols].copy()
    if df[C.FILENAME_COL].duplicated().any():
        sys.exit(f"[ERROR] 重複した filename があります:\n{df[df[C.FILENAME_COL].duplicated(keep=False)].head()}")

    # 欠損・範囲チェック
    n0 = len(df)
    df = df.dropna(subset=concept_cols)
    if len(df) < n0:
        print(f"[WARN] 概念グレードに欠損がある {n0 - len(df)} 行を除外しました")
    grades = df[concept_cols].astype(float)
    if not (grades.values == np.round(grades.values)).all():
        sys.exit("[ERROR] 整数でないグレードが含まれています")
    lo, hi = C.GRADE_MIN, C.GRADE_MIN + C.NUM_GRADES - 1
    bad = ((grades < lo) | (grades > hi)).any(axis=1)
    if bad.any():
        sys.exit(f"[ERROR] {lo}〜{hi} の範囲外のグレードがあります:\n{df[bad].head()}")
    df[concept_cols] = grades.astype(int)

    # 画像の存在・サイズ確認
    img_paths, size_warn = [], []
    for fn in df[C.FILENAME_COL].astype(str):
        p = Path(fn) if Path(fn).is_absolute() else Path(C.IMG_DIR) / fn
        if not p.exists():
            sys.exit(f"[ERROR] 画像が見つかりません: {p}")
        with Image.open(p) as im:
            if im.size != tuple(C.EXPECTED_IMG_SIZE):
                size_warn.append((p.name, im.size))
        img_paths.append(str(p))
    if size_warn:
        print(f"[WARN] {C.EXPECTED_IMG_SIZE} 以外のサイズが {len(size_warn)} 件（前処理で 896 に揃えます）: {size_warn[:3]}")
    df["image_path"] = img_paths

    # 層化 K-Fold 用ラベル（全概念を同時に層化することはできないため、代表ラベルを 1 つ決める）
    strat = None
    if C.CLASS_CSV is not None and Path(C.CLASS_CSV).exists():
        cls = pd.read_csv(C.CLASS_CSV)
        cls.columns = [c.strip() for c in cls.columns]
        if C.FILENAME_COL in cls.columns and C.CLASS_COL in cls.columns:
            m = dict(zip(cls[C.FILENAME_COL].astype(str).str.strip(), cls[C.CLASS_COL]))
            s = df[C.FILENAME_COL].astype(str).str.strip().map(m)
            if s.notna().all():
                strat = s.astype(str)
                print(f"[INFO] 層化ラベル: {C.CLASS_CSV} の '{C.CLASS_COL}' 列")
            else:
                print(f"[WARN] CLASS_CSV に無い画像が {int(s.isna().sum())} 枚あるため、層化ラベルは平均グレードで代用します")
    if strat is None:
        strat = np.rint(df[concept_cols].mean(axis=1)).astype(int).astype(str)
        print("[INFO] 層化ラベル: 全概念の平均グレード（四捨五入）")
    df["strat_label"] = strat.values

    # slide_id 付与（ファイル名順で固定）
    df = df.sort_values(C.FILENAME_COL).reset_index(drop=True)
    width = max(3, len(str(len(df))))
    df.insert(0, "slide_id", [f"slide_{i + 1:0{width}d}" for i in range(len(df))])
    out = df[["slide_id", "image_path", "strat_label"]].copy()
    for c in concept_cols:
        out[f"{C.GRADE_PREFIX}{c}"] = df[c].values
    out["source_name"] = df[C.FILENAME_COL].values

    # 最小クラスが Fold 数未満だと層化できない
    vc = out["strat_label"].value_counts()
    if vc.min() < C.N_FOLDS:
        print(f"[WARN] 層化ラベル '{vc.idxmin()}' が {vc.min()} 枚で、{C.N_FOLDS}-Fold の層化分割に足りません。")

    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(C.LABELS_CSV, index=False)

    # 概念 × グレード の分布
    grade_range = list(range(lo, hi + 1))
    dist = pd.DataFrame({c: df[c].value_counts().reindex(grade_range, fill_value=0) for c in concept_cols}).T
    dist.columns = [f"grade{g}" for g in grade_range]
    dist.index.name = "concept"
    dist.to_csv(C.CLASS_DIST_CSV)

    print(f"\n総画像数: {len(out)}")
    print("概念ごとのグレード分布（枚数）:")
    print(dist.to_string())
    sparse = [(c, g) for c in concept_cols for g in grade_range if dist.loc[c, f"grade{g}"] < C.N_FOLDS]
    if sparse:
        print(f"\n[INFO] {C.N_FOLDS} 枚未満のグレードがある概念: {sparse[:10]}{' ...' if len(sparse) > 10 else ''}")
        print("       該当グレードは学習・QWK が不安定になりやすい点に注意してください。")
    print(f"\n保存: {C.LABELS_CSV}\n保存: {C.CLASS_DIST_CSV}")


if __name__ == "__main__":
    main()
