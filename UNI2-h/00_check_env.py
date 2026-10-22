"""
00_check_env.py
===============
フェーズ 1-a: 環境セットアップの確認
    - 必要ライブラリ (torch, timm, huggingface_hub, sklearn, scipy, pandas) のバージョン
    - GPU (CUDA / MPS) と VRAM 容量
    - Hugging Face のログイン状態と MahmoodLab/UNI2-h へのアクセス権

使い方:
    python 00_check_env.py
"""
import importlib
import sys

REQUIRED = ["torch", "torchvision", "timm", "huggingface_hub", "sklearn", "scipy", "pandas", "numpy", "PIL"]


def check_libs() -> bool:
    print("=== ライブラリ ===")
    ok = True
    for name in REQUIRED:
        try:
            mod = importlib.import_module(name)
            print(f"  [OK] {name:16s} {getattr(mod, '__version__', '')}")
        except ImportError:
            print(f"  [NG] {name:16s} 未インストール → pip install -r requirements.txt")
            ok = False
    return ok


def check_gpu() -> None:
    import torch

    print("\n=== GPU ===")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            vram = p.total_memory / 1024**3
            flag = "OK" if vram >= 12 else "注意: 12GB 未満"
            print(f"  [CUDA:{i}] {p.name}  VRAM {vram:.1f} GB  ({flag})")
        print(f"  bf16 対応: {torch.cuda.is_bf16_supported()}")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        print("  [MPS] Apple Silicon GPU を使用（ユニファイドメモリ。特徴量抽出は CUDA より低速）")
    else:
        print("  [CPU] GPU が見つかりません。特徴量抽出に時間がかかります（学習・評価は CPU でも十分高速）")


def check_hf() -> None:
    print("\n=== Hugging Face / UNI2-h ===")
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        print("  huggingface_hub 未インストール")
        return
    try:
        user = HfApi().whoami()
        print(f"  ログイン中: {user.get('name')}")
    except Exception:
        print("  未ログイン → `huggingface-cli login` を実行してください（または環境変数 HF_TOKEN を設定）")
        return
    try:
        hf_hub_download("MahmoodLab/UNI2-h", filename="config.json")
        print("  [OK] MahmoodLab/UNI2-h へのアクセス権あり")
    except Exception as e:
        print(f"  [NG] UNI2-h にアクセスできません（利用申請の承認待ちの可能性）: {type(e).__name__}")


if __name__ == "__main__":
    print(f"Python {sys.version.split()[0]}\n")
    if check_libs():
        check_gpu()
        check_hf()
