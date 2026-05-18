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
from PIL import Image, ImageDraw, ImageFont

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
        # Also capture self-attention for inspection.
        if block.self_attn._last_attn is not None:
            d["self"] = block.self_attn._last_attn[0].float().cpu()
        if d:
            out.append(d)
    return out


# ---------------------------------------------------------------------------
# Reduction helpers
# ---------------------------------------------------------------------------

def _agg_over_heads_and_query(attn: torch.Tensor) -> torch.Tensor:
    """[H, Tq, Tk] → [Tk]: mean over heads then over query positions."""
    return attn.mean(dim=0).mean(dim=0)


def _per_head_agg_query(attn: torch.Tensor) -> torch.Tensor:
    """[H, Tq, Tk] → [H, Tk]: mean over query positions only (keep heads)."""
    return attn.mean(dim=1)


def reduce_local_to_frame_region(local_attn: torch.Tensor, n_frames: int,
                                 n_regions: int = REGIONS) -> np.ndarray:
    """``[H, M*T, N*R]`` → ``[N, R]`` distribution over (frame, region)."""
    flat = _agg_over_heads_and_query(local_attn)            # [N*R]
    flat = flat.view(n_frames, n_regions).numpy()
    s = flat.sum()
    return flat / s if s > 0 else flat


def reduce_local_per_head(local_attn: torch.Tensor, n_frames: int,
                          n_regions: int = REGIONS) -> np.ndarray:
    """``[H, M*T, N*R]`` → ``[H, N, R]`` per-head distribution."""
    per_head = _per_head_agg_query(local_attn)  # [H, N*R]
    H = per_head.shape[0]
    per_head = per_head.view(H, n_frames, n_regions).numpy()
    sums = per_head.sum(axis=(1, 2), keepdims=True)
    sums = np.where(sums > 0, sums, 1.0)
    return per_head / sums


def reduce_global_to_region(global_attn: torch.Tensor) -> np.ndarray:
    """``[H, M*T, R]`` → ``[R]`` distribution over the 16 spatial regions."""
    flat = _agg_over_heads_and_query(global_attn).numpy()   # [R]
    s = flat.sum()
    return flat / s if s > 0 else flat


def reduce_global_per_head(global_attn: torch.Tensor) -> np.ndarray:
    """``[H, M*T, R]`` → ``[H, R]`` per-head distribution."""
    per_head = _per_head_agg_query(global_attn).numpy()  # [H, R]
    sums = per_head.sum(axis=1, keepdims=True)
    sums = np.where(sums > 0, sums, 1.0)
    return per_head / sums


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
# Statistics
# ---------------------------------------------------------------------------

def compute_entropy(weights: np.ndarray) -> float:
    """Shannon entropy of a probability distribution (in bits)."""
    w = weights.flatten()
    w = w[w > 0]
    if len(w) == 0:
        return 0.0
    return float(-np.sum(w * np.log2(w)))


def compute_stats(weights: np.ndarray) -> dict:
    """Compute summary statistics for an attention distribution."""
    w = weights.flatten()
    return {
        "min": float(w.min()),
        "max": float(w.max()),
        "mean": float(w.mean()),
        "std": float(w.std()),
        "entropy": compute_entropy(w),
        "max_entropy": float(np.log2(len(w))) if len(w) > 0 else 0.0,
        "sparsity": float((w < w.mean()).sum() / max(len(w), 1)),
    }


# ---------------------------------------------------------------------------
# Rendering (PIL, no matplotlib needed)
# ---------------------------------------------------------------------------

def _inferno(value: float) -> Tuple[int, int, int]:
    """Inferno-like colormap — much higher contrast than viridis for attention."""
    v = float(np.clip(value, 0.0, 1.0))
    anchors = [
        (0,     0,   4),   # near-black
        (40,   11,  84),   # deep purple
        (101,  21, 110),   # purple
        (159,  42,  99),   # magenta
        (212,  72,  66),   # red-orange
        (245, 125,  21),   # orange
        (250, 193,  39),   # yellow
        (252, 255, 164),   # bright white-yellow
    ]
    seg = v * (len(anchors) - 1)
    i = int(seg)
    f = seg - i
    if i >= len(anchors) - 1:
        return anchors[-1]
    a = anchors[i]
    b = anchors[i + 1]
    return tuple(int(a[k] * (1 - f) + b[k] * f) for k in range(3))


# Keep old name as alias
_viridis = _inferno


def _get_font(size: int = 14):
    """Try to load a TTF font, fall back to PIL default."""
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except (IOError, OSError):
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", size)
        except (IOError, OSError):
            return ImageFont.load_default()


def overlay_region_attn(frame_img: Image.Image, region_weights: np.ndarray,
                        alpha: float = 0.55, max_w: int = 320) -> Image.Image:
    """Overlay a 4×4 attention heatmap on a frame image.

    Uses fixed-alpha overlay so attention regions are ALWAYS visible,
    with color intensity encoding the attention strength.
    """
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
        rw_n = np.full_like(rw, 0.5)

    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    pixels = overlay.load()
    for r in range(REGION_GRID):
        for c in range(REGION_GRID):
            color = _inferno(float(rw_n[r, c]))
            # Fixed alpha so ALL cells are visible, brighter = more attention
            a = int(255 * alpha * (0.3 + 0.7 * float(rw_n[r, c])))
            x0, y0 = int(c * cw), int(r * ch)
            x1, y1 = int((c + 1) * cw), int((r + 1) * ch)
            for y in range(y0, y1):
                for x in range(x0, x1):
                    pixels[x, y] = (color[0], color[1], color[2], a)

    result = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    # Draw grid lines for clarity
    draw = ImageDraw.Draw(result)
    for r in range(REGION_GRID + 1):
        y = min(int(r * ch), H - 1)
        draw.line([(0, y), (W - 1, y)], fill=(255, 255, 255), width=1)
    for c in range(REGION_GRID + 1):
        x = min(int(c * cw), W - 1)
        draw.line([(x, 0), (x, H - 1)], fill=(255, 255, 255), width=1)

    # Draw percentage labels
    font = _get_font(max(10, int(min(cw, ch) * 0.28)))
    total = rw.sum()
    for r in range(REGION_GRID):
        for c in range(REGION_GRID):
            pct = (rw[r, c] / total * 100) if total > 0 else 0
            label = f"{pct:.1f}%"
            cx_pos = int((c + 0.5) * cw)
            cy_pos = int((r + 0.5) * ch)
            # Text with shadow for readability
            draw.text((cx_pos + 1, cy_pos + 1), label, fill=(0, 0, 0), font=font, anchor="mm")
            draw.text((cx_pos, cy_pos), label, fill=(255, 255, 255), font=font, anchor="mm")

    return result


def render_global_region_grid(region_weights: np.ndarray, cell: int = 72) -> Image.Image:
    """Render the 16-cell global attention grid as a standalone heatmap with labels."""
    rw = region_weights.reshape(REGION_GRID, REGION_GRID).astype(np.float32)
    if rw.max() > rw.min():
        rw_n = (rw - rw.min()) / (rw.max() - rw.min())
    else:
        rw_n = np.full_like(rw, 0.5)
    W = H = cell * REGION_GRID
    img = Image.new("RGB", (W, H), (10, 10, 20))
    px = img.load()
    for r in range(REGION_GRID):
        for c in range(REGION_GRID):
            color = _inferno(float(rw_n[r, c]))
            for y in range(r * cell, (r + 1) * cell):
                for x in range(c * cell, (c + 1) * cell):
                    px[x, y] = color
    # Grid lines
    draw = ImageDraw.Draw(img)
    for r in range(REGION_GRID + 1):
        y = min(r * cell, H - 1)
        draw.line([(0, y), (W - 1, y)], fill=(255, 255, 255), width=2)
    for c in range(REGION_GRID + 1):
        x = min(c * cell, W - 1)
        draw.line([(x, 0), (x, H - 1)], fill=(255, 255, 255), width=2)

    # Percentage labels
    font = _get_font(max(12, cell // 5))
    total = rw.sum()
    for r in range(REGION_GRID):
        for c in range(REGION_GRID):
            pct = (rw[r, c] / total * 100) if total > 0 else 0
            label = f"{pct:.1f}%"
            cx_pos = int((c + 0.5) * cell)
            cy_pos = int((r + 0.5) * cell)
            draw.text((cx_pos + 1, cy_pos + 1), label, fill=(0, 0, 0), font=font, anchor="mm")
            draw.text((cx_pos, cy_pos), label, fill=(255, 255, 255), font=font, anchor="mm")

    return img


def render_per_head_grids(per_head_weights: np.ndarray, cell: int = 40) -> Image.Image:
    """Render per-head attention grids side by side.

    ``per_head_weights`` has shape ``[H, R]`` (global) or ``[H]`` that we reshape.
    Returns a single wide image with all heads.
    """
    if per_head_weights.ndim == 1:
        per_head_weights = per_head_weights.reshape(1, -1)
    H_heads = per_head_weights.shape[0]
    n_cols = min(H_heads, 8)
    n_rows = (H_heads + n_cols - 1) // n_cols

    grid_w = cell * REGION_GRID
    gap = 4
    label_h = 18
    total_w = n_cols * grid_w + (n_cols - 1) * gap
    total_h = n_rows * (grid_w + label_h) + (n_rows - 1) * gap

    canvas = Image.new("RGB", (total_w, total_h), (30, 30, 40))
    draw = ImageDraw.Draw(canvas)
    font = _get_font(11)

    for hi in range(H_heads):
        row_i = hi // n_cols
        col_i = hi % n_cols
        ox = col_i * (grid_w + gap)
        oy = row_i * (grid_w + label_h + gap)

        rw = per_head_weights[hi].reshape(REGION_GRID, REGION_GRID).astype(np.float32)
        if rw.max() > rw.min():
            rw_n = (rw - rw.min()) / (rw.max() - rw.min())
        else:
            rw_n = np.full_like(rw, 0.5)

        for r in range(REGION_GRID):
            for c in range(REGION_GRID):
                color = _inferno(float(rw_n[r, c]))
                for y in range(r * cell, (r + 1) * cell):
                    for x in range(c * cell, (c + 1) * cell):
                        canvas.putpixel((ox + x, oy + y), color)

        # Grid lines
        for r in range(REGION_GRID + 1):
            y = min(r * cell, grid_w - 1)
            draw.line([(ox, oy + y), (ox + grid_w - 1, oy + y)], fill=(180, 180, 180), width=1)
        for c in range(REGION_GRID + 1):
            x = min(c * cell, grid_w - 1)
            draw.line([(ox + x, oy), (ox + x, oy + grid_w - 1)], fill=(180, 180, 180), width=1)

        # Head label
        ent = compute_entropy(per_head_weights[hi])
        draw.text((ox + grid_w // 2, oy + grid_w + 2), f"H{hi} ent={ent:.2f}",
                  fill=(220, 220, 220), font=font, anchor="mt")

    return canvas


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


def forward_with_full_capture(model: torch.nn.Module, sample: Dict,
                              device: torch.device) -> Dict:
    """Run forward and capture every signal worth inspecting.

    Returns a dict with:
      ``H``           : {name: cpu float32 tensor} of intermediate activations
      ``blocks_attn`` : list per fusion block of {self_attn|local_ca|global_ca: weights}
      ``e_plus``, ``e_narr``, ``temperature`` : model outputs (CPU)
    """
    H: Dict[str, torch.Tensor] = {}
    hooks = []

    def make_hook(name):
        def fn(_, _inp, out):
            t = out if torch.is_tensor(out) else (out[0] if isinstance(out, tuple) else None)
            if t is not None:
                H[name] = t.detach().float().cpu()
        return fn

    # Turn on attention capture.
    set_capture(model, True)

    for bi, blk in enumerate(model.blocks):
        if not isinstance(blk, FusionBlockLocalGlobal):
            continue
        B = f"B{bi}"
        for aname in ("self_attn", "local_ca", "global_ca"):
            a = getattr(blk, aname, None)
            if a is None:
                continue
            hooks.append(a.register_forward_hook(make_hook(f"{B}.{aname}_out")))
        for mod_name in ("ffn1", "ffn2", "norm1", "norm2", "norm3", "norm4"):
            mod = getattr(blk, mod_name, None)
            if mod is not None:
                hooks.append(mod.register_forward_hook(make_hook(f"{B}.{mod_name}")))

    for global_name in ("visual_proj", "final_norm"):
        m = getattr(model, global_name, None)
        if m is not None:
            hooks.append(m.register_forward_hook(make_hook(global_name)))

    e_plus = e_narr = temperature = None
    try:
        with torch.inference_mode():
            with torch.amp.autocast(
                device_type="cuda" if device.type == "cuda" else "cpu",
                enabled=(device.type == "cuda"), dtype=torch.bfloat16,
            ):
                e_plus, e_narr, temperature = model(
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
        for h in hooks:
            h.remove()
        set_capture(model, False)

    return {
        "H": H,
        "blocks_attn": blocks_attn,
        "e_plus": e_plus.detach().float().cpu() if e_plus is not None else None,
        "e_narr": e_narr.detach().float().cpu() if e_narr is not None else None,
        "temperature": temperature.detach().float().cpu() if temperature is not None else None,
    }


def render_joint_frame_region_heatmap(joint: np.ndarray, n_frames: int,
                                      cell_w: int = 36, cell_h: int = 28) -> Image.Image:
    """Render a ``[N_frames, REGIONS]`` heatmap with row=frame, col=region.

    This is the **VideoRoPE 2D** joint surface — exactly the (temporal,
    spatial) space that local cross-attention operates over.
    """
    arr = joint[:n_frames].astype(np.float32)
    if arr.size == 0:
        return Image.new("RGB", (10, 10), (24, 24, 30))
    if arr.max() > arr.min():
        norm = (arr - arr.min()) / (arr.max() - arr.min())
    else:
        norm = np.full_like(arr, 0.5)

    n_cols = REGIONS  # 16
    label_left = 64
    label_top = 30
    img_w = label_left + cell_w * n_cols + 8
    img_h = label_top + cell_h * n_frames + 8
    img = Image.new("RGB", (img_w, img_h), (24, 24, 30))
    draw = ImageDraw.Draw(img)
    font = _get_font(11)

    for c in range(n_cols):
        x = label_left + c * cell_w + cell_w // 2
        rr, cc = c // REGION_GRID, c % REGION_GRID
        draw.text((x, label_top - 6), f"({rr},{cc})",
                  fill=(220, 220, 220), font=font, anchor="mb")

    total = arr.sum()
    for r in range(n_frames):
        draw.text((label_left - 6, label_top + r * cell_h + cell_h // 2),
                  f"F{r:02d}", fill=(220, 220, 220), font=font, anchor="rm")
        for c in range(n_cols):
            color = _inferno(float(norm[r, c]))
            x0 = label_left + c * cell_w
            y0 = label_top + r * cell_h
            draw.rectangle([x0, y0, x0 + cell_w - 1, y0 + cell_h - 1], fill=color)
            if cell_w >= 30 and cell_h >= 22:
                pct = (arr[r, c] / total * 100) if total > 0 else 0
                cx, cy = x0 + cell_w // 2, y0 + cell_h // 2
                draw.text((cx + 1, cy + 1), f"{pct:.1f}",
                          fill=(0, 0, 0), font=font, anchor="mm")
                draw.text((cx, cy), f"{pct:.1f}",
                          fill=(255, 255, 255), font=font, anchor="mm")

    # Outer grid lines
    for c in range(n_cols + 1):
        x = label_left + c * cell_w
        draw.line([(x, label_top), (x, label_top + n_frames * cell_h)],
                  fill=(80, 80, 90), width=1)
    for r in range(n_frames + 1):
        y = label_top + r * cell_h
        draw.line([(label_left, y), (label_left + n_cols * cell_w, y)],
                  fill=(80, 80, 90), width=1)
    return img


def render_segment_frame_matrix(sim: np.ndarray, top_k: int = 3,
                                cell_w: int = 42, cell_h: int = 28) -> Image.Image:
    """Render a ``[M, N]`` segment × frame heatmap with top-K cells outlined.

    ``sim`` is typically a cosine-similarity matrix. The image colours each
    cell by min-max-normalised value (inferno), prints the raw value, and
    draws a yellow ranked border around the top-K cells so the eye lands
    there first.
    """
    if sim.size == 0:
        return Image.new("RGB", (10, 10), (24, 24, 30))
    M, N = sim.shape
    arr = sim.astype(np.float32)
    rng = arr.max() - arr.min()
    norm = (arr - arr.min()) / rng if rng > 0 else np.full_like(arr, 0.5)

    label_left = 70
    label_top = 30
    img_w = label_left + cell_w * N + 8
    img_h = label_top + cell_h * M + 8
    img = Image.new("RGB", (img_w, img_h), (24, 24, 30))
    draw = ImageDraw.Draw(img)
    font = _get_font(11)
    font_small = _get_font(10)

    for c in range(N):
        x = label_left + c * cell_w + cell_w // 2
        draw.text((x, label_top - 6), f"F{c:02d}",
                  fill=(220, 220, 220), font=font, anchor="mb")
    for r in range(M):
        draw.text((label_left - 6, label_top + r * cell_h + cell_h // 2),
                  f"S{r:02d}", fill=(220, 220, 220), font=font, anchor="rm")
        for c in range(N):
            color = _inferno(float(norm[r, c]))
            x0 = label_left + c * cell_w
            y0 = label_top + r * cell_h
            draw.rectangle([x0, y0, x0 + cell_w - 1, y0 + cell_h - 1], fill=color)
            if cell_w >= 32 and cell_h >= 22:
                cx = x0 + cell_w // 2
                cy = y0 + cell_h // 2
                txt = f"{arr[r, c]:.2f}"
                draw.text((cx + 1, cy + 1), txt, fill=(0, 0, 0),
                          font=font, anchor="mm")
                draw.text((cx, cy), txt, fill=(255, 255, 255),
                          font=font, anchor="mm")

    # Outer grid lines
    for c in range(N + 1):
        x = label_left + c * cell_w
        draw.line([(x, label_top), (x, label_top + M * cell_h)],
                  fill=(80, 80, 90), width=1)
    for r in range(M + 1):
        y = label_top + r * cell_h
        draw.line([(label_left, y), (label_left + N * cell_w, y)],
                  fill=(80, 80, 90), width=1)

    # Highlight top-K cells with a yellow ranked border.
    flat = arr.flatten()
    top_idx = np.argsort(-flat)[:top_k]
    for rank, idx in enumerate(top_idx, 1):
        r = int(idx // N)
        c = int(idx % N)
        x0 = label_left + c * cell_w
        y0 = label_top + r * cell_h
        for w in range(3):
            draw.rectangle(
                [x0 - w, y0 - w, x0 + cell_w - 1 + w, y0 + cell_h - 1 + w],
                outline=(255, 220, 0), width=1,
            )
        draw.text((x0 + 3, y0 + 2), f"#{rank}",
                  fill=(255, 220, 0), font=font_small, anchor="lt")
    return img


def render_per_frame_region_grid(per_frame: np.ndarray, n_frames: int,
                                 cell: int = 28) -> Image.Image:
    """Render one mini 4x4 grid per frame, laid out left→right.

    ``per_frame`` shape: ``[N_frames, 16]``. Normalisation is per-row so
    each frame's pattern is visible regardless of mass.
    """
    arr = per_frame[:n_frames].astype(np.float32)
    n = arr.shape[0]
    grid_w = cell * REGION_GRID
    gap = 6
    label_h = 16
    total_w = n * grid_w + (n - 1) * gap
    total_h = grid_w + label_h
    img = Image.new("RGB", (max(total_w, 1), total_h), (24, 24, 30))
    draw = ImageDraw.Draw(img)
    font = _get_font(10)
    for fi in range(n):
        rw = arr[fi].reshape(REGION_GRID, REGION_GRID)
        rng = rw.max() - rw.min()
        rw_n = (rw - rw.min()) / rng if rng > 0 else np.full_like(rw, 0.5)
        ox = fi * (grid_w + gap)
        for r in range(REGION_GRID):
            for c in range(REGION_GRID):
                color = _inferno(float(rw_n[r, c]))
                x0 = ox + c * cell
                y0 = r * cell
                draw.rectangle([x0, y0, x0 + cell - 1, y0 + cell - 1], fill=color)
        for k in range(REGION_GRID + 1):
            y = min(k * cell, grid_w - 1)
            draw.line([(ox, y), (ox + grid_w - 1, y)], fill=(140, 140, 160), width=1)
            x = ox + min(k * cell, grid_w - 1)
            draw.line([(x, 0), (x, grid_w - 1)], fill=(140, 140, 160), width=1)
        draw.text((ox + grid_w // 2, grid_w + 2), f"F{fi:02d}",
                  fill=(220, 220, 220), font=font, anchor="mt")
    return img


# ---------------------------------------------------------------------------
# Higher-level analysis (per-head entropy, residual contribution, verdict)
# ---------------------------------------------------------------------------

def per_head_row_entropy(attn: torch.Tensor, n_rows: int = 50) -> np.ndarray:
    """``[H, Tq, Tk]`` → mean row entropy per head (bits)."""
    if attn is None:
        return np.zeros((0,), dtype=np.float32)
    nH, Tq, Tk = attn.shape
    out = np.zeros(nH, dtype=np.float32)
    for hi in range(nH):
        rows = attn[hi, :min(Tq, n_rows)].numpy()
        es = []
        for r in rows:
            s = r.sum()
            if s > 0:
                p = r / s
                p = p[p > 0]
                es.append(float(-np.sum(p * np.log2(p))))
        out[hi] = float(np.mean(es)) if es else 0.0
    return out


def head_status_marker(avg_ent: float, max_ent: float) -> Tuple[str, str]:
    if max_ent <= 0:
        return ("⬜", "uniform")
    ratio = avg_ent / max_ent
    if ratio < 0.85:
        return ("🔥", "FOCUSED")
    if ratio < 0.95:
        return ("⚡", "partial")
    return ("⬜", "uniform")
