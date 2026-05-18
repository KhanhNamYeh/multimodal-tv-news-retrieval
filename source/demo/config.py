"""Demo configuration: edit BASE_DIR + checkpoint path to point at your data."""
from __future__ import annotations

import os
from pathlib import Path


# Demo folder (this file's directory).
DEMO_DIR = Path(__file__).resolve().parent

# Source folder.
SOURCE_DIR = DEMO_DIR.parent

# Project root — auto-detected from file location (source/demo/config.py → <root>).
# Override with AIC_BASE_DIR env var if needed.
_DEFAULT_ROOT = str(SOURCE_DIR.parent)
BASE_DIR = Path(os.environ.get("AIC_BASE_DIR", _DEFAULT_ROOT))

# Default checkpoint = Variant 3 best (gate_init=-1.0, R@1=73.84).
# Place pretrain_v3_gate_neg1.pth at the project root (see README).
CHECKPOINT = Path(os.environ.get(
    "AIC_CHECKPOINT",
    str(BASE_DIR / "pretrain_v3_gate_neg1.pth"),
))

# FAISS index files live alongside the demo so the artefact is self-contained.
INDEX_PATH = DEMO_DIR / "faiss_v3.index"
META_PATH = DEMO_DIR / "faiss_v3_meta.json"


# Variant 3 architecture + dataset window settings.
MODEL_CFG = {
    "dim": 1024,
    "vis_dim": 1152,
    "n_layers": 2,
    "n_heads": 16,
    "n_kv_heads": 4,
    "delta": 100.0,
    "rope": "videorope2d",
    "cross_attn": "local_global",
    "use_filter": False,
    "base_temporal": 500_000.0,
    "base_spatial": 10_000.0,
    "lg_window_seconds": 8.0,
    "lg_n_global_tokens": 16,
}

DATA_CFG = {
    "max_frames": 25,
    "max_segments": 15,
    "max_seg_tokens": 128,
    "visual_feature_subdir": "siglip2",
    "eval_query_max_length": 256,
    "eval_query_batch_size": 16,
}

INDEX_BATCH_SIZE = 1
THUMB_MAX_WIDTH = 240
