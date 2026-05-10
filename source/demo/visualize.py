"""Visualize visual cross-attention of Variant 3.

Two attention streams in each ``FusionBlockLocalGlobal`` look at images:

- ``local_ca``  : segment tokens → all (frame, region) pairs, masked by an 8-second
                  temporal window. Shows *which frame* + *which 4×4 region* each
                  segment attends to under tight temporal locality.
- ``global_ca`` : segment tokens → 16 region-pooled tokens (mean over frames).
                  Shows *spatial* preference (which of the 4×4 regions matters
                  globally for this shot).

We capture the post-softmax weights ``p`` ∈ ``[B, H, Tq, Tk]`` directly off the
attention modules, aggregate over heads + segment query tokens, then render
heatmaps overlaid on the actual frame images.
"""
from __future__ import annotations

import base64
import io
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
SOURCE_DIR = HERE.parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from model.attention import GatedAttentionVideoRoPE  # noqa: E402
from model.blocks import FusionBlockLocalGlobal  # noqa: E402

REGION_GRID = 4   # SigLIP-2 features are pooled to a 4x4 region grid.
REGIONS = REGION_GRID * REGION_GRID  # = 16


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------

def set_capture(model: torch.nn.Module, on: bool) -> None:
    for m in model.modules():
        if isinstance(m, GatedAttentionVideoRoPE):
            m._capture_attn = on
            if not on:
                m._last_attn = None


def collect_visual_attn(model: torch.nn.Module) -> List[Dict[str, torch.Tensor]]:
    """Per-block attention dict with keys ``local`` and ``global``.

    Shapes (post softmax, on CPU):
      - ``local``  : ``[H, M*T, N*R]``
      - ``global`` : ``[H, M*T, R]``
    """
    out: List[Dict[str, torch.Tensor]] = []
    for block in model.blocks:
        if not isinstance(block, FusionBlockLocalGlobal):
            continue
        d: Dict[str, torch.Tensor] = {}
        # Cast to float32 because autocast may leave attention in bf16 — and
        # ``.numpy()`` does not support bf16.
        if block.local_ca._last_attn is not None:
            d["local"] = block.local_ca._last_attn[0].float().cpu()  # drop batch dim
        if block.global_ca._last_attn is not None:
            d["global"] = block.global_ca._last_attn[0].float().cpu()
        if d:
            out.append(d)
    return out


# ---------------------------------------------------------------------------
# Reduction helpers
# ---------------------------------------------------------------------------

def _agg_over_heads_and_query(attn: torch.Tensor) -> torch.Tensor:
    """[H, Tq, Tk] → [Tk]: mean over heads then over query positions."""
    return attn.mean(dim=0).mean(dim=0)


def reduce_local_to_frame_region(local_attn: torch.Tensor, n_frames: int,
                                 n_regions: int = REGIONS) -> np.ndarray:
    """``[H, M*T, N*R]`` → ``[N, R]`` distribution over (frame, region)."""
    flat = _agg_over_heads_and_query(local_attn)            # [N*R]
    flat = flat.view(n_frames, n_regions).numpy()
    s = flat.sum()
    return flat / s if s > 0 else flat


def reduce_global_to_region(global_attn: torch.Tensor) -> np.ndarray:
    """``[H, M*T, R]`` → ``[R]`` distribution over the 16 spatial regions."""
    flat = _agg_over_heads_and_query(global_attn).numpy()   # [R]
    s = flat.sum()
    return flat / s if s > 0 else flat


def reduce_local_aggregate_blocks(blocks_attn: List[Dict[str, torch.Tensor]],
                                  n_frames: int) -> np.ndarray:
    """Average ``local`` over all fusion blocks → ``[N, R]``."""
    mats = [
        reduce_local_to_frame_region(b["local"], n_frames)
        for b in blocks_attn if "local" in b
    ]
    if not mats:
        return np.zeros((n_frames, REGIONS), dtype=np.float32)
    return np.mean(np.stack(mats, axis=0), axis=0)


def reduce_global_aggregate_blocks(blocks_attn: List[Dict[str, torch.Tensor]]) -> np.ndarray:
    mats = [
        reduce_global_to_region(b["global"])
        for b in blocks_attn if "global" in b
    ]
    if not mats:
        return np.zeros((REGIONS,), dtype=np.float32)
    return np.mean(np.stack(mats, axis=0), axis=0)


# ---------------------------------------------------------------------------
# Rendering (PIL, no matplotlib needed)
# ---------------------------------------------------------------------------

def _viridis(value: float) -> Tuple[int, int, int]:
    """Cheap viridis-like colormap (no matplotlib dep)."""
    v = float(np.clip(value, 0.0, 1.0))
    anchors = [
        (68,   1,  84),
        (59,  82, 139),
        (33, 144, 141),
        ( 94, 201,  98),
        (253, 231,  37),
    ]
    seg = v * (len(anchors) - 1)
    i = int(seg)
    f = seg - i
    if i >= len(anchors) - 1:
        return anchors[-1]
    a = anchors[i]
    b = anchors[i + 1]
    return tuple(int(a[k] * (1 - f) + b[k] * f) for k in range(3))


def overlay_region_attn(frame_img: Image.Image, region_weights: np.ndarray,
                        alpha: float = 0.5, max_w: int = 320) -> Image.Image:
    """Overlay a 4×4 attention heatmap on a frame image."""
    img = frame_img.convert("RGB")
    if img.width > max_w:
        h = int(img.height * (max_w / img.width))
        img = img.resize((max_w, max(h, 1)))
    W, H = img.size
    cw, ch = W / REGION_GRID, H / REGION_GRID

    rw = region_weights.reshape(REGION_GRID, REGION_GRID).astype(np.float32)
    if rw.max() > rw.min():
        rw_n = (rw - rw.min()) / (rw.max() - rw.min())
    else:
        rw_n = np.zeros_like(rw)

    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    pixels = overlay.load()
    for r in range(REGION_GRID):
        for c in range(REGION_GRID):
            color = _viridis(float(rw_n[r, c]))
            a = int(255 * alpha * float(rw_n[r, c]))
            x0, y0 = int(c * cw), int(r * ch)
            x1, y1 = int((c + 1) * cw), int((r + 1) * ch)
            for y in range(y0, y1):
                for x in range(x0, x1):
                    pixels[x, y] = (color[0], color[1], color[2], a)

    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def render_global_region_grid(region_weights: np.ndarray, cell: int = 56) -> Image.Image:
    """Render the 16-cell global attention grid as a standalone heatmap."""
    rw = region_weights.reshape(REGION_GRID, REGION_GRID).astype(np.float32)
    if rw.max() > rw.min():
        rw_n = (rw - rw.min()) / (rw.max() - rw.min())
    else:
        rw_n = np.zeros_like(rw)
    W = H = cell * REGION_GRID
    img = Image.new("RGB", (W, H), (10, 10, 20))
    px = img.load()
    for r in range(REGION_GRID):
        for c in range(REGION_GRID):
            color = _viridis(float(rw_n[r, c]))
            for y in range(r * cell, (r + 1) * cell):
                for x in range(c * cell, (c + 1) * cell):
                    px[x, y] = color
    for r in range(REGION_GRID + 1):
        y = min(r * cell, H - 1)
        for x in range(W):
            px[x, y] = (0, 0, 0)
    for c in range(REGION_GRID + 1):
        x = min(c * cell, W - 1)
        for y in range(H):
            px[x, y] = (0, 0, 0)
    return img


def img_to_uri(img: Image.Image, fmt: str = "JPEG", quality: int = 85) -> str:
    buf = io.BytesIO()
    if fmt == "JPEG":
        img.convert("RGB").save(buf, format=fmt, quality=quality, optimize=True)
    else:
        img.save(buf, format=fmt)
    return f"data:image/{fmt.lower()};base64,{base64.b64encode(buf.getvalue()).decode()}"


# ---------------------------------------------------------------------------
# Forward pass with attention capture
# ---------------------------------------------------------------------------

def _slice_one(batch: Dict, idx: int) -> Dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v[idx:idx + 1]
        elif isinstance(v, list):
            out[k] = [v[idx]]
        else:
            out[k] = v
    return out


def find_sample(loader, shot_id: str) -> Dict | None:
    for batch in loader:
        if shot_id in batch["shot_id"]:
            return _slice_one(batch, batch["shot_id"].index(shot_id))
    return None


def forward_with_capture(model: torch.nn.Module, sample: Dict,
                         device: torch.device) -> List[Dict[str, torch.Tensor]]:
    """Run a single-sample forward and return per-block visual attention."""
    set_capture(model, True)
    try:
        with torch.inference_mode():
            with torch.amp.autocast(
                device_type="cuda" if device.type == "cuda" else "cpu",
                enabled=(device.type == "cuda"), dtype=torch.bfloat16,
            ):
                _ = model(
                    seg_tokens=sample["token_reprs"].to(device),
                    seg_pooled=sample["segment_pooled"].to(device),
                    visual_features=sample["visual_features"].to(device),
                    seg_timestamps=sample["seg_timestamps"].to(device),
                    frame_timestamps=sample["frame_timestamps"].to(device),
                    seg_mask=sample["segment_mask"].to(device),
                    visual_mask=sample["visual_mask"].to(device),
                    query_emb=None,
                    token_mask=sample["token_mask"].to(device),
                )
        blocks_attn = collect_visual_attn(model)
    finally:
        set_capture(model, False)
    return blocks_attn
