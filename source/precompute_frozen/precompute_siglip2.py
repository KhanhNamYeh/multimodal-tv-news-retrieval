"""
precompute_siglip2.py
=====================
Extract SigLIP-2 So400m/14-384 features cho toàn bộ frames.

Theo kiến trúc mô hình:
  - Backbone : google/siglip2-so400m-patch14-384
  - Layer    : penultimate (index -2, tức layer 35/37)
  - Pooling  : 4×4 spatial grid → 16 region vectors
    - Output   : Tensor [N_frames, 16, 1152] mỗi shot

Cấu trúc lưu (xem structure.md):
  features/siglip2/
  └── K01_V001/
      ├── K01_V001_000.npz  ← float32, key='features', shape [N_frames, 16, 1152]
      ├── K01_V001_001.npz
      └── ...

Model weights được tải về lần đầu và cache tại:
  features/weights/siglip2/   (dùng lại cho lần sau, không cần internet)

Chạy:
  py precompute_siglip2.py                   # toàn bộ
  py precompute_siglip2.py --video K01_V001  # 1 video
    py precompute_siglip2.py --resume          # bỏ qua shot đã có .npz
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

# ── Đường dẫn gốc (chỉnh nếu chạy từ nơi khác) ─────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent.parent.parent   # D:/aic
TIME_EXTRACT  = BASE_DIR / "dataset" / "time_extract"
FRAME_DIR     = BASE_DIR / "dataset" / "frame_extracted"
FEAT_DIR      = BASE_DIR / "data" / "features" / "siglip2"
WEIGHT_DIR    = BASE_DIR / "data" / "features" / "weights" / "siglip2"
HF_CACHE_DIR  = BASE_DIR / "data" / "features" / "weights" / "hf_cache"

MODEL_ID      = "google/siglip2-so400m-patch14-384"
IMAGE_SIZE    = 384          # native resolution của model
SPATIAL_GRID  = 4            # 4×4 → 16 region vectors
LAYER_IDX     = -2           # penultimate hidden state (layer 35/37)
EMBED_DIM     = 1152         # So400m hidden dim
BATCH_FRAMES  = 8            # số frame encode mỗi lần (tune theo VRAM)


# ─────────────────────────────────────────────────────────────────────────────
def load_model(weight_dir: Path, device: str):
    """
    Lần đầu: tải từ HuggingFace, lưu vào weight_dir.
    Lần sau : load từ weight_dir (offline, không cần internet).
    """
    from transformers import AutoImageProcessor, AutoModel

    weight_dir.mkdir(parents=True, exist_ok=True)
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Ép cache HuggingFace về thư mục local có quyền ghi
    os.environ["HF_HOME"] = str(HF_CACHE_DIR)
    os.environ["HF_HUB_CACHE"] = str(HF_CACHE_DIR / "hub")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE_DIR / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE_DIR / "transformers")
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    cache_kwargs = {"cache_dir": str(HF_CACHE_DIR)}
    config_file = weight_dir / "config.json"

    if config_file.exists():
        print(f"[SigLIP-2] Load weights từ local: {weight_dir}")
        model     = AutoModel.from_pretrained(str(weight_dir), local_files_only=True)
        processor = AutoImageProcessor.from_pretrained(str(weight_dir), local_files_only=True)
    else:
        print(f"[SigLIP-2] Tải model từ HuggingFace: {MODEL_ID}")
        print("  (sẽ mất vài phút lần đầu, sau đó dùng local cache)")
        print(f"  HF cache: {HF_CACHE_DIR}")
        model     = AutoModel.from_pretrained(MODEL_ID, **cache_kwargs)
        processor = AutoImageProcessor.from_pretrained(MODEL_ID, **cache_kwargs)
        # Lưu lại để dùng offline
        model.save_pretrained(str(weight_dir))
        processor.save_pretrained(str(weight_dir))
        print(f"  Đã lưu weights → {weight_dir}")

    model = model.vision_model   # chỉ cần visual encoder
    model.to(device).eval()
    return model, processor


# ─────────────────────────────────────────────────────────────────────────────
def spatial_pool_4x4(patch_tokens: torch.Tensor) -> torch.Tensor:
    """
    patch_tokens: [N_patches, D]  (không kể CLS token)
    SigLIP-2 So400m/14 với ảnh 384×384 → 27×27 = 729 patches.

    Chia 729 patches thành lưới 4×4 = 16 vùng (mỗi vùng ~45 patches),
    mean pool trong từng vùng → [16, D].

    Lý do dùng 4×4: hàng cuối ~ chyron/subtitle, góc ~ logo kênh,
    trung tâm ~ main visual — phân biệt ngữ nghĩa quan trọng cho bản tin.
    """
    N, D = patch_tokens.shape
    side = int(N ** 0.5)
    if side * side != N:
        raise ValueError(f"Số patch tokens không tạo được lưới vuông: N={N}")

    tokens = patch_tokens.view(side, side, D)

    # Chia đều về 4×4 cell (dùng adaptive average pool)
    tokens = tokens.permute(2, 0, 1).unsqueeze(0)  # [1, D, 27, 27]
    pooled = torch.nn.functional.adaptive_avg_pool2d(tokens, SPATIAL_GRID)
    # [1, D, 4, 4] → [16, D]
    pooled = pooled.squeeze(0).view(D, -1).T        # [16, D]
    return pooled


@torch.no_grad()
def encode_frames(images: list[Image.Image],
                  model, processor,
                  device: str) -> torch.Tensor:
    """
    Encode batch ảnh → [B, 16, D].
    Lấy hidden state ở LAYER_IDX (penultimate), tự nhận diện có/không CLS,
    rồi spatial pool 4×4.
    """
    try:
        inputs = processor(
            images=images,
            return_tensors="pt",
            do_resize=True,
            size={"height": IMAGE_SIZE, "width": IMAGE_SIZE},
        ).to(device)
    except TypeError:
        # Fallback cho version processor cũ/khác chữ ký hàm
        inputs = processor(images=images, return_tensors="pt").to(device)
    out    = model(**inputs, output_hidden_states=True)

    # hidden_states: tuple, mỗi phần tử [B, N_tokens, D]
    # Một số version/model có thể trả hidden_states=None dù đã bật output_hidden_states.
    hidden_states = getattr(out, "hidden_states", None)
    if hidden_states is not None and len(hidden_states) > 0:
        idx = LAYER_IDX if len(hidden_states) >= abs(LAYER_IDX) else -1
        hidden = hidden_states[idx]
    else:
        hidden = getattr(out, "last_hidden_state", None)
        if hidden is None:
            raise RuntimeError("Model output không có hidden_states/last_hidden_state")

    # Có model trả về token gồm CLS, có model chỉ patch tokens.
    # Chọn cách tách patch sao cho N_patches là số chính phương (27x27, ...).
    n_tokens = hidden.shape[1]
    if int(n_tokens ** 0.5) ** 2 == n_tokens:
        patches = hidden
    elif n_tokens > 1 and int((n_tokens - 1) ** 0.5) ** 2 == (n_tokens - 1):
        patches = hidden[:, 1:, :]
    else:
        raise RuntimeError(
            f"Không xác định được patch tokens từ N_tokens={n_tokens}"
        )

    # Spatial pool từng ảnh
    results = []
    for b in range(patches.shape[0]):
        region = spatial_pool_4x4(patches[b])   # [16, D]
        results.append(region)

    return torch.stack(results)  # [B, 16, D]


# ─────────────────────────────────────────────────────────────────────────────
def process_shot(shot_json: Path, model, processor,
                 device: str, out_path: Path) -> bool:
    """
    Đọc 1 shot JSON, encode tất cả frames, lưu .npz float32
    với key='features' và shape [N_frames, 16, D].
    Trả về True nếu thành công.
    """
    entries = json.loads(shot_json.read_text("utf-8"))
    if not entries:
        return False

    all_regions = []
    batch_imgs  = []
    batch_valid = []   # True = ảnh load được, False = fallback zero

    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("image"):
            batch_imgs.append(None)
            batch_valid.append(False)
            continue

        img_path = BASE_DIR / entry["image"]
        if img_path.exists():
            try:
                img = Image.open(img_path).convert("RGB")
                batch_imgs.append(img)
                batch_valid.append(True)
            except Exception:
                batch_imgs.append(None)
                batch_valid.append(False)
        else:
            batch_imgs.append(None)
            batch_valid.append(False)

    # Encode theo batch
    valid_imgs = [img for img, v in zip(batch_imgs, batch_valid) if v]
    encoded    = {}   # index trong valid_imgs → tensor [16, D]

    if valid_imgs:
        vi = 0
        for b_start in range(0, len(valid_imgs), BATCH_FRAMES):
            b_imgs = valid_imgs[b_start : b_start + BATCH_FRAMES]
            feats  = encode_frames(b_imgs, model, processor, device)
            for j, feat in enumerate(feats):
                encoded[vi + j] = feat.cpu()
            vi += len(b_imgs)

    # Ghép lại theo thứ tự, zero cho frame không đọc được
    vi = 0
    for v in batch_valid:
        if v:
            all_regions.append(encoded[vi])
            vi += 1
        else:
            all_regions.append(torch.zeros(SPATIAL_GRID * SPATIAL_GRID, EMBED_DIM))

    feat_tensor = torch.stack(all_regions)  # [N_frames, 16, D]
    feat_np = feat_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, features=feat_np)
    return True


# ─────────────────────────────────────────────────────────────────────────────
def get_all_shots(video_filter: str | None = None) -> list[tuple[Path, Path]]:
    """
    Liệt kê tất cả (shot_json, out_npz) cần xử lý.
    shot_json: time_extract/K01_V001/K01_V001_001.json
    out_npz  : features/siglip2/K01_V001/K01_V001_001.npz
    """
    pairs = []
    for vid_dir in sorted(TIME_EXTRACT.iterdir()):
        if not vid_dir.is_dir():
            continue
        if video_filter and vid_dir.name != video_filter:
            continue
        for shot_json in sorted(vid_dir.glob("*.json")):
            out_npz = FEAT_DIR / vid_dir.name / (shot_json.stem + ".npz")
            pairs.append((shot_json, out_npz))
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Precompute SigLIP-2 features")
    parser.add_argument("--video",  type=str, default=None,
                        help="Chỉ xử lý 1 video (vd: K01_V001). Mặc định = tất cả.")
    parser.add_argument("--resume", action="store_true",
                        help="Bỏ qua shot đã có file .npz")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Device: {args.device}")
    model, processor = load_model(WEIGHT_DIR, args.device)

    shots = get_all_shots(video_filter=args.video)
    print(f"Tổng shots: {len(shots)}")

    if args.resume:
        shots = [(j, p) for j, p in shots if not p.exists()]
        print(f"Chưa có .npz: {len(shots)}")

    ok = err = 0
    for shot_json, out_pt in tqdm(shots, desc="SigLIP-2"):
        try:
            process_shot(shot_json, model, processor, args.device, out_pt)
            ok += 1
        except Exception as e:
            print(f"\n[ERR] {shot_json.name}: {e}", file=sys.stderr)
            err += 1

    print(f"\nHoàn thành: {ok} OK  |  {err} lỗi")
    print(f"Features lưu tại: {FEAT_DIR}")


if __name__ == "__main__":
    main()
