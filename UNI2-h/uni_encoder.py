"""
uni_encoder.py
==============
UNI2-h (MahmoodLab/UNI2-h) の読み込みと前処理。
02_extract_features.py（学習用特徴量のキャッシュ）と 06_predict.py（新規画像の推論）で共有します。

UNI2-h は ViT-H/14 相当（patch 14, embed 1536, depth 24, register tokens 8）。
dynamic_img_size=True により位置埋め込みが補間され、224 以外の解像度（ここでは 896）も入力できます。
    896 / 14 = 64  →  64 x 64 = 4096 パッチトークン (+ CLS 1 + register 8)
num_classes=0 のため、出力は CLS トークンの [1536] ベクトル（画像全体の要約）です。
"""
import torch
from PIL import Image
from torchvision import transforms

# UNI2-h の標準正規化（ImageNet の Mean / Std）
UNI_MEAN = (0.485, 0.456, 0.406)
UNI_STD = (0.229, 0.224, 0.225)


def build_uni2h(model_name: str = "hf-hub:MahmoodLab/UNI2-h", dynamic_img_size: bool = True) -> torch.nn.Module:
    """
    UNI2-h を timm で構築し、Hugging Face Hub から重みを読み込む。
    事前に `huggingface-cli login`（または環境変数 HF_TOKEN）でアクセス権付きトークンを設定しておくこと。
    """
    import timm  # 遅延インポート（学習・評価フェーズでは timm 不要）

    timm_kwargs = {
        "img_size": 224,
        "patch_size": 14,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": 1536,
        "mlp_ratio": 2.66667 * 2,
        "num_classes": 0,
        "no_embed_class": True,
        "mlp_layer": timm.layers.SwiGLUPacked,
        "act_layer": torch.nn.SiLU,
        "reg_tokens": 8,
        "dynamic_img_size": dynamic_img_size,
    }
    model = timm.create_model(model_name, pretrained=True, **timm_kwargs)
    model.eval()
    return model


def build_transform(input_size: int = 896, resize_mode: str = "resize") -> transforms.Compose:
    """
    900x900 の画像を 14 の倍数 (896x896) に揃え、UNI2-h の標準正規化を適用する。
        resize      : 900 → 896 に縮小（画像全体の情報を保持。縮小率 0.996 で劣化はほぼ無し）
        center_crop : 中央の 896x896 を切り出し（画素を補間しないが、外周 2px を捨てる）
    """
    if resize_mode == "resize":
        geom = [transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)]
    elif resize_mode == "center_crop":
        geom = [
            # 画像が input_size より小さい場合に備え、短辺を input_size 以上に揃えてから切り出す
            transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(input_size),
        ]
    else:
        raise ValueError(f"未知の RESIZE_MODE: {resize_mode}")
    return transforms.Compose(geom + [transforms.ToTensor(), transforms.Normalize(UNI_MEAN, UNI_STD)])


def load_image(path) -> Image.Image:
    return Image.open(path).convert("RGB")


def autocast_context(device: torch.device, use_amp: bool):
    """CUDA では bf16（非対応GPUは fp16）で推論し、VRAM と時間を節約する。"""
    if use_amp and device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)
    return torch.autocast(device_type="cpu", enabled=False)


@torch.inference_mode()
def extract_feature(model, image_tensor: torch.Tensor, device: torch.device, use_amp: bool = True) -> torch.Tensor:
    """
    前処理済み画像 [3, H, W] を UNI2-h に通し、[1536] の特徴ベクトル (float32, CPU) を返す。
    """
    x = image_tensor.unsqueeze(0).to(device)
    with autocast_context(device, use_amp):
        feat = model(x)  # [1, 1536]
    return feat.float().squeeze(0).cpu()
