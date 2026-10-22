"""
02_extract_features.py
======================
フェーズ 2: UNI2-h による特徴量抽出とキャッシュ化

    1. 各画像を 896x896 に揃え、UNI2-h 標準正規化を適用
    2. dynamic_img_size=True の UNI2-h に 1 枚ずつ入力（64x64=4096 トークン）
    3. CLS 特徴 [1536] を features/slide_XXX.pt として保存
    4. 全ファイルの存在と形状 [1536] をアサーションで確認

使い方:
    python 02_extract_features.py              # 未抽出の画像のみ処理
    python 02_extract_features.py --verify     # 抽出せず、保存済み .pt の検証のみ
"""
import argparse
import time

import pandas as pd
import torch

import config as C
from common import feature_path, get_device


def verify_features(df: pd.DataFrame) -> None:
    """全 slide_id の .pt が存在し、形状が [1536] か [1, 1536] で、NaN/Inf を含まないことを確認する。"""
    for sid in df["slide_id"]:
        p = feature_path(C.FEATURE_DIR, sid)
        assert p.exists(), f"特徴量ファイルがありません: {p}"
        f = torch.load(p, map_location="cpu")
        assert tuple(f.shape) in [(C.FEATURE_DIM,), (1, C.FEATURE_DIM)], f"{p.name} の形状が不正: {tuple(f.shape)}"
        assert torch.isfinite(f).all(), f"{p.name} に NaN/Inf が含まれます"
    print(f"[OK] {len(df)} 個すべての特徴量が存在し、形状は [{C.FEATURE_DIM}] です。")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="検証のみ行う")
    args = parser.parse_args()

    df = pd.read_csv(C.LABELS_CSV)
    if args.verify:
        verify_features(df)
        return

    from uni_encoder import build_transform, build_uni2h, extract_feature, load_image

    assert C.INPUT_SIZE % C.PATCH_SIZE == 0, "INPUT_SIZE は 14 の倍数にしてください"
    n_tok = (C.INPUT_SIZE // C.PATCH_SIZE) ** 2
    C.FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    todo = df if C.OVERWRITE_FEATURES else df[[not feature_path(C.FEATURE_DIR, s).exists() for s in df["slide_id"]]]
    print(f"[INFO] 対象 {len(todo)} / {len(df)} 枚（入力 {C.INPUT_SIZE}x{C.INPUT_SIZE} → {n_tok} トークン）")

    if len(todo):
        device = get_device()
        print(f"[INFO] UNI2-h を読み込み中...  device={device}")
        model = build_uni2h(C.UNI_MODEL_NAME, dynamic_img_size=True).to(device)
        transform = build_transform(C.INPUT_SIZE, C.RESIZE_MODE)

        t0 = time.time()
        for i, row in enumerate(todo.itertuples(index=False), start=1):
            x = transform(load_image(row.image_path))
            assert x.shape == (3, C.INPUT_SIZE, C.INPUT_SIZE), x.shape
            feat = extract_feature(model, x, device, use_amp=C.USE_AMP)
            torch.save(feat.clone(), feature_path(C.FEATURE_DIR, row.slide_id))
            if i % 20 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"  {i:4d}/{len(todo)}  {el:6.1f}s  (残り約 {el / i * (len(todo) - i):5.0f}s)")
        if device.type == "cuda":
            print(f"[INFO] 最大 VRAM 使用量: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

    verify_features(df)


if __name__ == "__main__":
    main()
