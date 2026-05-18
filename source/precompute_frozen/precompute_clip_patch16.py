"""
Precompute CLIP ViT-B/16 visual features from shot frames.

Output structure:
  features/clip_vitb16_l2/<video_id>/<video_id>_<shot_id>.npz

Each npz contains:
  key='features' -> float32 tensor [N_frames, 16, D]

Default setup:
  - Model: openai/clip-vit-base-patch16
  - Layer: -2 (second last hidden state)
  - Spatial pooling: 4x4 grid -> 16 region vectors per frame
  - HF cache: features/weights/hf_cache
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


AIC_ROOT_HARDCODE = Path(__file__).resolve().parent.parent.parent   # D:/aic
TIME_EXTRACT_DIR = AIC_ROOT_HARDCODE / "dataset" / "time_extract"
FEATURES_DIR = AIC_ROOT_HARDCODE / "data" / "features"
HF_CACHE_DIR = AIC_ROOT_HARDCODE / "data" / "features" / "weights" / "hf_cache"


def setup_hf_cache(hf_cache_dir: Path) -> None:
    hf_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_cache_dir)
    os.environ["HF_HUB_CACHE"] = str(hf_cache_dir / "hub")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_cache_dir / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(hf_cache_dir / "transformers")


def load_clip_vision(
    model_id: str,
    hf_cache_dir: Path,
    device: str,
    allow_online: bool,
):
    from transformers import AutoImageProcessor, CLIPVisionModel

    setup_hf_cache(hf_cache_dir)

    local_only = not allow_online
    try:
        processor = AutoImageProcessor.from_pretrained(
            model_id,
            cache_dir=str(hf_cache_dir),
            local_files_only=local_only,
        )
        model = CLIPVisionModel.from_pretrained(
            model_id,
            cache_dir=str(hf_cache_dir),
            local_files_only=local_only,
        )
    except Exception as exc:
        if local_only:
            raise RuntimeError(
                f"Cannot load {model_id} from local hf_cache ({hf_cache_dir}). "
                f"Use --allow-online once to download. Original error: {exc}"
            ) from exc
        raise

    model.to(device).eval()
    return model, processor


def extract_patch_tokens(hidden: torch.Tensor) -> torch.Tensor:
    """
    hidden: [B, N_tokens, D]
    Return patch tokens only: [B, N_patches, D]
    """
    n_tokens = hidden.shape[1]

    if int(n_tokens ** 0.5) ** 2 == n_tokens:
        return hidden

    if n_tokens > 1 and int((n_tokens - 1) ** 0.5) ** 2 == (n_tokens - 1):
        return hidden[:, 1:, :]

    raise RuntimeError(f"Cannot infer patch tokens from N_tokens={n_tokens}")


def spatial_pool(tokens_2d: torch.Tensor, grid: int) -> torch.Tensor:
    """
    tokens_2d: [N_patches, D]
    Return: [grid*grid, D]
    """
    n_patches, dim = tokens_2d.shape
    side = int(n_patches ** 0.5)
    if side * side != n_patches:
        raise RuntimeError(f"Patch count is not square: {n_patches}")

    x = tokens_2d.view(side, side, dim).permute(2, 0, 1).unsqueeze(0)  # [1, D, H, W]
    x = torch.nn.functional.adaptive_avg_pool2d(x, grid)  # [1, D, g, g]
    x = x.squeeze(0).reshape(dim, grid * grid).T  # [g*g, D]
    return x


@torch.no_grad()
def encode_images(
    images: list[Image.Image],
    model,
    processor,
    device: str,
    layer_idx: int,
    grid: int,
) -> torch.Tensor:
    """
    Return: [B, grid*grid, D]
    """
    inputs = processor(images=images, return_tensors="pt").to(device)
    outputs = model(**inputs, output_hidden_states=True, return_dict=True)

    hidden_states = outputs.hidden_states
    if hidden_states is None or len(hidden_states) == 0:
        raise RuntimeError("No hidden_states returned by CLIPVisionModel")

    if abs(layer_idx) > len(hidden_states):
        use_idx = -1
    else:
        use_idx = layer_idx

    hidden = hidden_states[use_idx]  # [B, N_tokens, D]
    patch_hidden = extract_patch_tokens(hidden)

    regions = []
    for i in range(patch_hidden.shape[0]):
        regions.append(spatial_pool(patch_hidden[i], grid=grid))

    return torch.stack(regions, dim=0)


def resolve_image_path(image_field: str) -> Path:
    path = Path(image_field)
    if path.is_absolute():
        return path
    return AIC_ROOT_HARDCODE / path


def process_shot(
    shot_json: Path,
    out_npz: Path,
    model,
    processor,
    device: str,
    layer_idx: int,
    grid: int,
    batch_frames: int,
) -> bool:
    rows = json.loads(shot_json.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        return False

    hidden_size = int(model.config.hidden_size)
    empty_region = torch.zeros((grid * grid, hidden_size), dtype=torch.float32)

    images = []
    valid_flags = []

    for row in rows:
        image_field = row.get("image") if isinstance(row, dict) else None
        if not image_field:
            images.append(None)
            valid_flags.append(False)
            continue

        img_path = resolve_image_path(str(image_field))
        if not img_path.exists():
            images.append(None)
            valid_flags.append(False)
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            images.append(None)
            valid_flags.append(False)
            continue

        images.append(img)
        valid_flags.append(True)

    valid_images = [img for img, ok in zip(images, valid_flags) if ok]
    encoded_valid = {}

    if valid_images:
        offset = 0
        for start in range(0, len(valid_images), batch_frames):
            batch = valid_images[start : start + batch_frames]
            feats = encode_images(
                batch,
                model=model,
                processor=processor,
                device=device,
                layer_idx=layer_idx,
                grid=grid,
            )
            for j, feat in enumerate(feats):
                encoded_valid[offset + j] = feat.detach().cpu()
            offset += len(batch)

    all_feats = []
    valid_idx = 0
    for ok in valid_flags:
        if ok:
            all_feats.append(encoded_valid[valid_idx])
            valid_idx += 1
        else:
            all_feats.append(empty_region)

    feat_tensor = torch.stack(all_feats, dim=0)  # [N_frames, grid*grid, D]
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, features=feat_tensor.numpy().astype(np.float32, copy=False))
    return True


def iter_shots(video_filter: str | None, feature_subdir: str):
    for vid_dir in sorted(TIME_EXTRACT_DIR.iterdir()):
        if not vid_dir.is_dir():
            continue
        if video_filter and vid_dir.name != video_filter:
            continue

        for shot_json in sorted(vid_dir.glob("*.json")):
            out_npz = FEATURES_DIR / feature_subdir / vid_dir.name / f"{shot_json.stem}.npz"
            yield shot_json, out_npz


def main():
    parser = argparse.ArgumentParser(description="Precompute CLIP patch16 visual features")
    parser.add_argument("--model-id", type=str, default="openai/clip-vit-base-patch16")
    parser.add_argument("--layer-idx", type=int, default=-2)
    parser.add_argument("--feature-subdir", type=str, default="clip_vitb16_l2")
    parser.add_argument("--grid", type=int, default=4)
    parser.add_argument("--batch-frames", type=int, default=16)
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--allow-online",
        action="store_true",
        help="Allow downloading model if not available in local hf_cache",
    )
    args = parser.parse_args()

    print(f"Device: {args.device}")
    print(f"Model : {args.model_id}")
    print(f"Layer : {args.layer_idx}")
    print(f"Output: {FEATURES_DIR / args.feature_subdir}")
    print(f"HF cache: {HF_CACHE_DIR}")

    model, processor = load_clip_vision(
        model_id=args.model_id,
        hf_cache_dir=HF_CACHE_DIR,
        device=args.device,
        allow_online=args.allow_online,
    )

    pairs = list(iter_shots(video_filter=args.video, feature_subdir=args.feature_subdir))
    print(f"Total shots: {len(pairs)}")

    if args.resume:
        pairs = [(j, o) for j, o in pairs if not o.exists()]
        print(f"Pending shots: {len(pairs)}")

    ok = 0
    err = 0
    for shot_json, out_npz in tqdm(pairs, desc="CLIP-B16 precompute"):
        try:
            process_shot(
                shot_json=shot_json,
                out_npz=out_npz,
                model=model,
                processor=processor,
                device=args.device,
                layer_idx=args.layer_idx,
                grid=args.grid,
                batch_frames=args.batch_frames,
            )
            ok += 1
        except Exception as exc:
            print(f"\n[ERR] {shot_json}: {exc}", file=sys.stderr)
            err += 1

    print(f"\nDone. OK={ok} | ERR={err}")


if __name__ == "__main__":
    main()
