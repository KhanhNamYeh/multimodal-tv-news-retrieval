"""Re-extract SigLIP-2 visual features (4×4 region-pooled) for keyframe videos.

Usage:
    python source/preprocess/precompute_siglip2.py                # all K09_*
    python source/preprocess/precompute_siglip2.py --video K09_V001  # single video
    python source/preprocess/precompute_siglip2.py --only-missing    # skip existing non-zero

Output format per shot:
    data/features/siglip2/{video_id}/{video_id}_{shot_id}.npz
        features: float32 [N_frames, 16, 1152]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "source"


def load_model(weights_dir: Path, device: str):
    from transformers import SiglipImageProcessor, SiglipVisionModel

    model = SiglipVisionModel.from_pretrained(
        str(weights_dir), local_files_only=True
    ).eval().to(device)
    processor = SiglipImageProcessor.from_pretrained(
        str(weights_dir), local_files_only=True
    )
    for p in model.parameters():
        p.requires_grad = False
    return model, processor


@torch.no_grad()
def extract_regions(model, processor, image_paths: list[Path], device: str,
                    batch_size: int = 8) -> np.ndarray:
    """Extract 4×4 region-pooled features for a list of frame images.

    Returns: float32 array [N_frames, 16, 1152]
    """
    all_regions = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        imgs = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=imgs, return_tensors="pt").to(device)

        out = model(**inputs)
        h = out.last_hidden_state  # [B, 729, 1152]
        # SigLIP2 patch14-384: 27×27 = 729 patches (no CLS token)
        g = 27
        spatial = h.reshape(-1, g, g, h.shape[-1]).permute(0, 3, 1, 2)
        pooled = F.adaptive_avg_pool2d(spatial, (4, 4))  # [B, D, 4, 4]
        regions = pooled.permute(0, 2, 3, 1).reshape(-1, 16, h.shape[-1])
        all_regions.append(regions.cpu().float().numpy())

    return np.concatenate(all_regions, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Precompute SigLIP-2 4×4 region features")
    parser.add_argument("--video", type=str, default=None,
                        help="Single video ID (e.g. K09_V001). Default: all K09_*")
    parser.add_argument("--prefix", type=str, default="K09",
                        help="Process all videos matching this prefix")
    parser.add_argument("--only-missing", action="store_true",
                        help="Skip shots that already have non-zero features")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    frame_dir = ROOT / "dataset" / "frame_extracted"
    out_dir = ROOT / "data" / "features" / "siglip2"
    weights_dir = ROOT / "data" / "features" / "weights" / "siglip2"

    if not frame_dir.exists():
        raise FileNotFoundError(f"frame_extracted not found: {frame_dir}")
    if not weights_dir.exists():
        raise FileNotFoundError(f"SigLIP2 weights not found: {weights_dir}")

    # Determine which videos to process
    if args.video:
        video_ids = [args.video]
    else:
        video_ids = sorted(
            d.name for d in frame_dir.iterdir()
            if d.is_dir() and d.name.startswith(args.prefix)
        )

    if not video_ids:
        print(f"No videos found with prefix '{args.prefix}' in {frame_dir}")
        return

    print(f"Device: {args.device}")
    print(f"Videos to process: {len(video_ids)}")
    print(f"Output: {out_dir}")

    model, processor = load_model(weights_dir, args.device)
    print("Model loaded.\n")

    total_shots = 0
    total_skipped = 0

    for vid in tqdm(video_ids, desc="Videos"):
        vid_frame_dir = frame_dir / vid
        if not vid_frame_dir.exists():
            continue

        vid_out_dir = out_dir / vid
        vid_out_dir.mkdir(parents=True, exist_ok=True)

        # Each subdirectory is a shot: K09_V001_000, K09_V001_001, ...
        shot_dirs = sorted(d for d in vid_frame_dir.iterdir() if d.is_dir())

        for shot_dir in shot_dirs:
            shot_name = shot_dir.name  # e.g. K09_V001_000
            out_path = vid_out_dir / f"{shot_name}.npz"

            # Skip if already valid
            if args.only_missing and out_path.exists():
                try:
                    existing = np.load(out_path)["features"]
                    if existing.max() > 0:
                        total_skipped += 1
                        continue
                except Exception:
                    pass

            # Collect frame images
            frames = sorted(
                list(shot_dir.glob("*.jpg")) + list(shot_dir.glob("*.png"))
            )
            if not frames:
                continue

            # Extract
            regions = extract_regions(
                model, processor, frames, args.device, args.batch_size
            )  # [N, 16, 1152]

            np.savez_compressed(out_path, features=regions)
            total_shots += 1

    print(f"\nDone. Extracted: {total_shots} shots, skipped: {total_skipped}")


if __name__ == "__main__":
    main()
