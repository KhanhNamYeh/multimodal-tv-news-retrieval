"""FAISS index build/load + dataset-side document encoding."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import (
    BASE_DIR,
    CHECKPOINT,
    DATA_CFG,
    INDEX_BATCH_SIZE,
    INDEX_PATH,
    META_PATH,
    SOURCE_DIR,
)
from .model_loader import DEVICE, load_main_model

if str(SOURCE_DIR / "train") not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR / "train"))

try:
    import faiss  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("faiss is required (pip install faiss-cpu).") from exc

from dataset import VietnameseNewsDataset, collate_fn  # noqa: E402

KEYFRAMES_DIR = BASE_DIR / "dataset" / "keyframes"
FRAME_EXTRACTED_DIR = BASE_DIR / "dataset" / "frame_extracted"


def _keyframe_video_ids() -> set[str]:
    if not KEYFRAMES_DIR.exists():
        return set()
    return {p.name for p in KEYFRAMES_DIR.iterdir() if p.is_dir()}


_valid_video_cache: dict[str, bool] = {}


def _video_has_valid_features(video_id: str) -> bool:
    """Check (cached) if a video has non-zero SigLIP2 features.

    Probes the first NPZ in the video dir — if it's zero, all shots are
    likely zero (feature extraction failed for the whole video).
    """
    if video_id in _valid_video_cache:
        return _valid_video_cache[video_id]
    vis_dir = BASE_DIR / "data" / "features" / "siglip2" / video_id
    if not vis_dir.exists():
        _valid_video_cache[video_id] = False
        return False
    try:
        import numpy as _np
        npz = next(vis_dir.glob("*.npz"), None)
        valid = npz is not None and float(_np.load(npz)["features"].max()) > 0
    except Exception:
        valid = False
    _valid_video_cache[video_id] = valid
    return valid


def build_keyframe_loader(split: str, batch_size: int = INDEX_BATCH_SIZE) -> DataLoader:
    dataset = VietnameseNewsDataset(
        split=split,
        base_dir=BASE_DIR,
        max_frames=DATA_CFG["max_frames"],
        max_segments=DATA_CFG["max_segments"],
        max_seg_tokens=DATA_CFG["max_seg_tokens"],
        visual_feature_subdir=DATA_CFG["visual_feature_subdir"],
    )
    # NOTE: keyframe filter disabled — all K09_* keyframe videos have
    # zero-filled SigLIP2 features. The validity filter below handles this.
    # keyframe_videos = _keyframe_video_ids()
    # if keyframe_videos:
    #     dataset.samples = [s for s in dataset.samples if s.get("video_id") in keyframe_videos]
    # Filter out samples with zero/missing visual features (e.g. K09_* not extracted).
    dataset.samples = [
        s for s in dataset.samples
        if _video_has_valid_features(s["video_id"])
    ]
    if not dataset.samples:
        raise RuntimeError(f"No samples with valid visual features for split={split}.")
    return DataLoader(
        dataset, batch_size=batch_size, num_workers=0, shuffle=False, collate_fn=collate_fn,
    )


def encode_docs(model: torch.nn.Module, loader: DataLoader):
    doc_embs, shot_ids = [], []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Index docs", leave=False, dynamic_ncols=True):
            with torch.amp.autocast(
                device_type="cuda" if DEVICE.type == "cuda" else "cpu",
                enabled=(DEVICE.type == "cuda"), dtype=torch.bfloat16,
            ):
                e_plus, _, _ = model(
                    seg_tokens=batch["token_reprs"].to(DEVICE),
                    seg_pooled=batch["segment_pooled"].to(DEVICE),
                    visual_features=batch["visual_features"].to(DEVICE),
                    seg_timestamps=batch["seg_timestamps"].to(DEVICE),
                    frame_timestamps=batch["frame_timestamps"].to(DEVICE),
                    seg_mask=batch["segment_mask"].to(DEVICE),
                    visual_mask=batch["visual_mask"].to(DEVICE),
                    query_emb=None,
                    token_mask=batch["token_mask"].to(DEVICE),
                )
            doc_embs.append(e_plus.detach().cpu().numpy().astype("float32"))
            shot_ids.extend(batch["shot_id"])
    if not doc_embs:
        raise RuntimeError("No documents indexed.")
    vecs = np.concatenate(doc_embs, axis=0)
    faiss.normalize_L2(vecs)
    return vecs, shot_ids


def build_index(ckpt_path: Path | str | None = None, splits=("test",)):
    model = load_main_model(ckpt_path or CHECKPOINT)
    all_vecs, all_shot_ids = [], []
    for split in splits:
        loader = build_keyframe_loader(split)
        vecs, shot_ids = encode_docs(model, loader)
        all_vecs.append(vecs)
        all_shot_ids.extend(shot_ids)
    mat = np.concatenate(all_vecs, axis=0)
    index = faiss.IndexFlatIP(mat.shape[1])
    index.add(mat)
    faiss.write_index(index, str(INDEX_PATH))
    META_PATH.write_text(json.dumps({"shot_ids": all_shot_ids}, indent=2), encoding="utf-8")
    return index, all_shot_ids


def load_index():
    if not INDEX_PATH.exists() or not META_PATH.exists():
        return None, None
    index = faiss.read_index(str(INDEX_PATH))
    meta = json.loads(META_PATH.read_text("utf-8"))
    return index, meta["shot_ids"]


def get_or_build_index(splits=("test",)):
    index, shot_ids = load_index()
    if index is None:
        index, shot_ids = build_index(splits=splits)
    return index, shot_ids
