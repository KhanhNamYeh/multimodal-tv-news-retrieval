"""Gradio UI for Variant 3 retrieval demo.

Run with::

    python -m source.demo.app
    # or:
    cd source/demo && python app.py
    # Force CPU-only inference:
    python app.py cpu        # (or: python app.py --cpu)
"""
from __future__ import annotations

import base64
import html
import io
import json
import os
import random
import sys
from functools import lru_cache
from pathlib import Path

# ---------------------------------------------------------------------------
# CPU flag — must be parsed BEFORE any torch import so CUDA_VISIBLE_DEVICES
# takes effect and ``torch.cuda.is_available()`` returns False downstream.
# ---------------------------------------------------------------------------
FORCE_CPU = any(a.lower() in ("cpu", "--cpu", "-cpu") for a in sys.argv[1:])
if FORCE_CPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # Strip the flag so other libs (gradio, argparse...) don't choke on it.
    sys.argv = [sys.argv[0]] + [a for a in sys.argv[1:]
                                 if a.lower() not in ("cpu", "--cpu", "-cpu")]
    # Hard-disable CUDA — env var alone is unreliable on some torch builds
    # (NVML may still report a device, leaving torch.cuda.is_available() True).
    import torch as _torch_early
    _torch_early.cuda.is_available = lambda: False
    if hasattr(_torch_early.backends, "cuda"):
        try:
            _torch_early.backends.cuda.matmul.allow_tf32 = False
        except Exception:
            pass

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Ensure local imports work whether run as ``python -m source.demo.app`` or
# ``python app.py`` from inside this folder.
HERE = Path(__file__).resolve().parent
if __package__ in (None, ""):
    sys.path.insert(0, str(HERE.parent.parent))
    from source.demo.config import (  # noqa: E402
        BASE_DIR, CHECKPOINT, DATA_CFG, MODEL_CFG, THUMB_MAX_WIDTH,
    )
    from source.demo import model_loader as _model_loader  # noqa: E402
    if FORCE_CPU:
        _model_loader.DEVICE = torch.device("cpu")
    from source.demo.model_loader import encode_queries, load_main_model  # noqa: E402
    DEVICE = _model_loader.DEVICE
    from source.demo.retrieval import build_index, build_keyframe_loader, get_or_build_index  # noqa: E402
    from source.demo import visualize as viz  # noqa: E402
else:
    from .config import BASE_DIR, CHECKPOINT, DATA_CFG, MODEL_CFG, THUMB_MAX_WIDTH
    from . import model_loader as _model_loader
    if FORCE_CPU:
        _model_loader.DEVICE = torch.device("cpu")
    from .model_loader import encode_queries, load_main_model
    DEVICE = _model_loader.DEVICE
    from .retrieval import build_index, build_keyframe_loader, get_or_build_index
    from . import visualize as viz

import faiss  # noqa: E402

FRAME_EXTRACTED_DIR = BASE_DIR / "dataset" / "frame_extracted"


@lru_cache(maxsize=1)
def _index_cache():
    return get_or_build_index(splits=("test",))


def _load_test_queries():
    q_items = []
    for d in (BASE_DIR / "data" / "test", BASE_DIR / "data" / "test2"):
        if not (d.exists() and any(d.glob("*.json"))):
            continue
        for jf in sorted(d.glob("*.json")):
            video_id = jf.stem
            for row in json.loads(jf.read_text("utf-8")):
                sid = str(row.get("id", "")).zfill(3)
                q = str(row.get("positive", "")).strip()
                if sid and q:
                    q_items.append({"shot_id": f"{video_id}_{sid}", "query": q})
        break
    seen, dedup = set(), []
    for x in q_items:
        if x["shot_id"] not in seen:
            seen.add(x["shot_id"])
            dedup.append(x)
    return dedup


def _load_transcript_text(video_id: str, shot_id: str) -> str:
    shot_id = str(shot_id).zfill(3)
    tpath = BASE_DIR / "dataset" / "transcripts" / video_id / f"{shot_id}.json"
    if not tpath.exists():
        return ""
    entries = json.loads(tpath.read_text("utf-8"))
    return " ".join([str(e.get("text", "")).strip() for e in entries if isinstance(e, dict)]).strip()


def _load_segments(video_id: str, shot_id: str) -> list:
    """Return the raw transcript entries (start/end/text) for a shot."""
    shot_id = str(shot_id).zfill(3)
    tpath = BASE_DIR / "dataset" / "transcripts" / video_id / f"{shot_id}.json"
    if not tpath.exists():
        return []
    entries = json.loads(tpath.read_text("utf-8"))
    return [e for e in entries if isinstance(e, dict)]


def _image_to_data_uri(img: Image.Image, max_w: int = THUMB_MAX_WIDTH) -> str:
    if img.width > max_w:
        h = int(img.height * (max_w / img.width))
        img = img.resize((max_w, max(h, 1)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80, optimize=True)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


def _load_frame_data_uris(video_id: str, shot_id: str):
    shot_id = str(shot_id).zfill(3)
    vdir = FRAME_EXTRACTED_DIR / video_id / f"{video_id}_{shot_id}"
    if not vdir.exists():
        return []
    images = sorted(list(vdir.glob("*.jpg")) + list(vdir.glob("*.png")))
    return [_image_to_data_uri(Image.open(p).convert("RGB")) for p in images]


def _search(query: str, top_k: int = 10) -> str:
    index, shot_ids = _index_cache()

    if not query:
        q_items = _load_test_queries()
        if not q_items:
            return "<div class='layer'>No query provided and no test queries found.</div>"
        query = random.choice(q_items)["query"]

    q_emb = encode_queries(
        [query],
        max_length=DATA_CFG["eval_query_max_length"],
        batch_size=DATA_CFG["eval_query_batch_size"],
    )
    q_np = q_emb.detach().cpu().numpy().astype("float32")
    faiss.normalize_L2(q_np)
    scores, idxs = index.search(q_np, int(top_k))

    layers = [
        "<div class='layer layer-header'>"
        f"<div class='layer-head'>Query</div>"
        f"<div class='layer-text'>{html.escape(query)}</div>"
        "</div>"
    ]
    for rank, (score, idx) in enumerate(zip(scores[0], idxs[0]), start=1):
        shot_id = shot_ids[idx]
        video_id, sid = shot_id.rsplit("_", 1)
        transcript = _load_transcript_text(video_id, sid)
        transcript_html = html.escape(transcript) if transcript else "<i>No transcript.</i>"
        frame_uris = _load_frame_data_uris(video_id, sid)
        if frame_uris:
            frames_html = "".join(
                f"<img src='{u}' loading='lazy' alt='{video_id}_{sid} frame' />"
                for u in frame_uris
            )
        else:
            frames_html = "<div class='layer-text'><i>No frames found.</i></div>"
        layers.append(
            "<div class='layer'>"
            f"<div class='layer-head'>Layer {rank:02d} | {video_id} | {shot_id} | score={score:.4f}</div>"
            f"<div class='layer-text'>{transcript_html}</div>"
            f"<div class='frames'>{frames_html}</div>"
            "</div>"
        )
    return "".join(layers)


def _rebuild_index() -> str:
    build_index(CHECKPOINT, splits=("test",))
    _index_cache.cache_clear()
    return "<div class='layer'>Index rebuilt.</div>"


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
body, .gradio-container {
  font-family: 'Space Grotesk', ui-sans-serif, system-ui, sans-serif;
  background: radial-gradient(1200px 600px at 10% -10%, #f5bfff 0%, #f5f0ff 50%, #fff5e6 100%);
  color: #000 !important;
}
.gradio-container, .gradio-container * { color: #000; }
.gradio-container { max-width: 1200px !important; }
#title {
  background: linear-gradient(90deg, #0b0b0b, #7a1313, #0b0b0b);
  color: #fff !important; padding: 16px 18px; border-radius: 12px;
  text-transform: uppercase; letter-spacing: 2px;
}
#title * { color: #fff !important; }
.card { border: 1px solid #2b2b2b; border-radius: 14px; padding: 16px; background: #fff;
        box-shadow: 0 10px 25px rgba(0,0,0,0.08); color: #000; }
.layer { border: 1px solid #111; border-radius: 12px; padding: 14px; background: #fff;
         color: #000; margin-bottom: 14px; }
.layer-header { background: #fff8f0; }
.layer-head { font-weight: 700; margin-bottom: 8px; letter-spacing: 0.5px; color: #000; }
.layer-text { white-space: pre-wrap; line-height: 1.5; color: #000; }
.frames { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
.frames img { width: 220px; height: auto; border-radius: 8px; border: 1px solid #ddd; }
.attn-row { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 10px; align-items: flex-start; }
.attn-cell { display: flex; flex-direction: column; align-items: center;
             border: 1px solid #222; border-radius: 10px; padding: 8px; background: #fafafa; color: #000; }
.attn-cell img { display: block; max-width: 360px; border-radius: 6px; }
.attn-cell .cap { font-size: 12px; color: #000; margin-top: 6px; font-weight: 600; }
.attn-grid { display: flex; gap: 16px; flex-wrap: wrap; }
.attn-block { border: 1px solid #111; border-radius: 12px; padding: 14px; background: #fff;
              flex: 1 1 100%; margin-bottom: 14px; color: #000; }
.attn-block h3 { margin: 0 0 8px 0; color: #000; }
.stats-table { border-collapse: collapse; margin: 8px 0; font-size: 13px; color: #000; }
.stats-table td, .stats-table th { border: 1px solid #ccc; padding: 4px 10px; text-align: center; color: #000; }
.stats-table th { background: #f0f0f0; font-weight: 600; color: #000; }
.stats-table td.hi { background: #fff3cd; font-weight: 700; color: #000; }
.bar-row { display: flex; align-items: center; gap: 6px; margin: 2px 0; font-size: 12px; color: #000; }
.bar-label { width: 60px; text-align: right; font-weight: 600; color: #000; }
.bar-bg { flex: 1; height: 18px; background: #eee; border-radius: 4px; overflow: hidden; position: relative; }
.bar-fill { height: 100%; border-radius: 4px; transition: width 0.3s; }
.bar-val { position: absolute; right: 6px; top: 1px; font-size: 11px; color: #000; font-weight: 600; }
.per-head-img { border: 1px solid #333; border-radius: 8px; margin-top: 8px; }
.section-label { font-weight: 700; font-size: 14px; margin: 12px 0 6px; color: #000;
                 border-bottom: 2px solid #e44; padding-bottom: 4px; display: inline-block; }
.residual-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                 gap: 14px; margin-top: 8px; }
.residual-block { border: 1px solid #ccc; border-radius: 8px; padding: 10px; background: #fafafa; color: #000; }
.stats-table.compact { font-size: 11px; }
.stats-table.compact td, .stats-table.compact th { padding: 2px 6px; }
.preview-frames { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }
.preview-frame { display: flex; flex-direction: column; align-items: center;
                 border: 1px solid #ddd; border-radius: 6px; padding: 4px; background: #fff; }
.preview-frame img { width: 140px; height: auto; border-radius: 4px; }
.preview-frame .cap { font-size: 11px; color: #000; margin-top: 4px; font-weight: 600; }
details summary { color: #000 !important; }
p, div, span, label, b, i, h1, h2, h3, h4, td, th { color: #000 !important; }
"""


# ---------------------------------------------------------------------------
# Inspect tab
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _inspect_model():
    return load_main_model()


@lru_cache(maxsize=1)
def _inspect_loader():
    return build_keyframe_loader("test")


def _list_frame_paths(video_id: str, sid: str):
    sid = str(sid).zfill(3)
    vdir = FRAME_EXTRACTED_DIR / video_id / f"{video_id}_{sid}"
    if not vdir.exists():
        return []
    return sorted(list(vdir.glob("*.jpg")) + list(vdir.glob("*.png")))


def _retrieve_topk(query: str, k: int = 10):
    """Return (resolved_query, [(shot_id, score), ...]).

    If ``query`` is empty, a random test query is chosen so the demo
    still works without manual input.
    """
    if not query:
        items = _load_test_queries()
        if not items:
            return None, []
        pick = random.choice(items)
        query = pick["query"]
    index, shot_ids = _index_cache()
    q_emb = encode_queries([query],
                           max_length=DATA_CFG["eval_query_max_length"],
                           batch_size=DATA_CFG["eval_query_batch_size"])
    q_np = q_emb.detach().cpu().numpy().astype("float32")
    faiss.normalize_L2(q_np)
    scores, idxs = index.search(q_np, int(max(1, k)))
    return query, [(shot_ids[int(i)], float(s)) for s, i in zip(scores[0], idxs[0])]


def _candidate_label(rank: int, shot_id: str, score: float) -> str:
    return f"#{rank:02d}  {shot_id}  ·  score={score:.4f}"


def _render_attn_cell(uri: str, caption: str) -> str:
    return (f"<div class='attn-cell'>"
            f"<img src='{uri}' loading='lazy' />"
            f"<div class='cap'>{html.escape(caption)}</div></div>")


# ---------------------------------------------------------------------------
# Analysis section renderers (factored out of _inspect)
# ---------------------------------------------------------------------------

def _render_stats_html(label: str, weights: np.ndarray) -> str:
    s = viz.compute_stats(weights)
    return (
        f"<table class='stats-table'><tr>"
        f"<th>{html.escape(label)}</th>"
        f"<th>Min</th><th>Max</th><th>Mean</th><th>Std</th>"
        f"<th>Entropy</th><th>Max Ent</th><th>Sparsity</th></tr>"
        f"<tr><td></td>"
        f"<td>{s['min']:.4f}</td>"
        f"<td class='hi'>{s['max']:.4f}</td>"
        f"<td>{s['mean']:.4f}</td>"
        f"<td>{s['std']:.4f}</td>"
        f"<td class='hi'>{s['entropy']:.2f} bits</td>"
        f"<td>{s['max_entropy']:.2f}</td>"
        f"<td>{s['sparsity']:.1%}</td>"
        f"</tr></table>"
    )


def _render_frame_bars(frame_mass: np.ndarray, n_show: int) -> str:
    """Horizontal bar chart showing attention mass per frame."""
    mx = frame_mass[:n_show].max() if n_show > 0 else 1.0
    mx = max(mx, 1e-8)
    bars = []
    for fi in range(n_show):
        pct = frame_mass[fi] / mx * 100
        color = f"hsl({120 - int(pct * 1.2)}, 80%, 50%)"
        bars.append(
            f"<div class='bar-row'>"
            f"<div class='bar-label'>F{fi:02d}</div>"
            f"<div class='bar-bg'>"
            f"<div class='bar-fill' style='width:{pct:.1f}%; background:{color};'></div>"
            f"<div class='bar-val'>{frame_mass[fi]:.4f}</div>"
            f"</div></div>"
        )
    return f"<div style='max-width:520px;'>{''.join(bars)}</div>"


def _render_region_bars(region_mass: np.ndarray) -> str:
    mx = max(float(region_mass.max()), 1e-8)
    bars = []
    for ri in range(len(region_mass)):
        pct = float(region_mass[ri]) / mx * 100
        rr, cc = ri // 4, ri % 4
        color = f"hsl({240 - int(pct * 1.4)}, 80%, 55%)"
        bars.append(
            f"<div class='bar-row'>"
            f"<div class='bar-label'>R{ri:02d} ({rr},{cc})</div>"
            f"<div class='bar-bg'>"
            f"<div class='bar-fill' style='width:{pct:.1f}%; background:{color};'></div>"
            f"<div class='bar-val'>{region_mass[ri]:.4f}</div>"
            f"</div></div>"
        )
    return f"<div style='max-width:520px;'>{''.join(bars)}</div>"


def _render_gate_table(model: torch.nn.Module) -> str:
    """Section A — per-block sigmoid gate values per attention type."""
    rows = ["<tr><th>Block</th><th>self_attn</th><th>local_ca</th><th>global_ca</th></tr>"]
    from model.blocks import FusionBlockLocalGlobal  # local import to avoid early load
    for bi, blk in enumerate(model.blocks):
        if not isinstance(blk, FusionBlockLocalGlobal):
            continue
        cells = [f"<td>B{bi}</td>"]
        for aname in ("self_attn", "local_ca", "global_ca"):
            a = getattr(blk, aname, None)
            if a is None or not hasattr(a, "gate"):
                cells.append("<td>—</td>")
                continue
            g = torch.sigmoid(a.gate).detach().cpu()
            cells.append(
                f"<td>min={g.min():.3f}<br>max={g.max():.3f}<br>"
                f"<b>mean={g.mean():.3f}</b></td>"
            )
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return ("<table class='stats-table'>" + "".join(rows) + "</table>")


def _render_residual_table(H: dict, n_blocks: int) -> str:
    """Section D — per-block residual contribution as bars."""
    out = ["<div class='residual-grid'>"]
    for bi in range(n_blocks):
        parts = []
        for label in ("self_attn_out", "local_ca_out", "global_ca_out", "ffn1", "ffn2"):
            k = f"B{bi}.{label}"
            if k in H:
                parts.append((label, float(H[k].norm().item())))
        total = sum(v for _, v in parts) or 1.0
        bars = [f"<div class='residual-block'><b>Block B{bi}</b>"]
        for label, norm in parts:
            pct = norm / total * 100
            short = label.replace("_out", "")
            color = {
                "self_attn": "#2563eb",
                "local_ca": "#ea580c",
                "global_ca": "#0891b2",
                "ffn1": "#7c3aed",
                "ffn2": "#9333ea",
            }.get(short, "#666")
            bars.append(
                f"<div class='bar-row'>"
                f"<div class='bar-label'>{short}</div>"
                f"<div class='bar-bg'>"
                f"<div class='bar-fill' style='width:{pct:.1f}%; background:{color};'></div>"
                f"<div class='bar-val'>{norm:.1f}  ({pct:.1f}%)</div>"
                f"</div></div>"
            )
        bars.append("</div>")
        out.append("".join(bars))
    out.append("</div>")
    return "".join(out)


def _render_per_head_entropy(attn: torch.Tensor, label: str) -> str:
    """Section C — per-head row entropy with FOCUSED / partial / uniform marker."""
    if attn is None or attn.numel() == 0:
        return ""
    nH, Tq, Tk = attn.shape
    max_e = float(np.log2(max(Tk, 2)))
    ents = viz.per_head_row_entropy(attn)
    rows = [
        f"<tr><th>{html.escape(label)}</th>"
        f"<th>head</th><th>avg row entropy</th><th>max entropy</th>"
        f"<th>ratio</th><th>status</th></tr>"
    ]
    for hi, e in enumerate(ents):
        emoji, status = viz.head_status_marker(float(e), max_e)
        ratio = (e / max_e) if max_e > 0 else 0.0
        css = "hi" if status == "FOCUSED" else ""
        rows.append(
            f"<tr><td></td><td>H{hi:02d}</td>"
            f"<td class='{css}'>{e:.2f} bits</td>"
            f"<td>{max_e:.2f}</td>"
            f"<td>{ratio:.0%}</td>"
            f"<td>{emoji} {status}</td></tr>"
        )
    return "<table class='stats-table'>" + "".join(rows) + "</table>"


def _render_norms_table(H: dict) -> str:
    """Section B — all intermediate tensor norms."""
    rows = [
        "<tr><th>name</th><th>shape</th>"
        "<th>norm</th><th>min</th><th>max</th><th>std</th></tr>"
    ]
    for k in sorted(H.keys()):
        t = H[k]
        rows.append(
            f"<tr><td>{html.escape(k)}</td>"
            f"<td>{list(t.shape)}</td>"
            f"<td>{t.norm().item():.2f}</td>"
            f"<td>{t.min().item():.4f}</td>"
            f"<td>{t.max().item():.4f}</td>"
            f"<td>{t.std().item():.6f}</td></tr>"
        )
    return "<table class='stats-table compact'>" + "".join(rows) + "</table>"


def _render_cross_modal(H: dict, sample: dict, n_frames: int) -> str:
    """Section E — visual_proj ↔ text token cosine similarity."""
    vis = H.get("visual_proj")
    if vis is None or vis.norm().item() == 0:
        return ("<div class='attn-block'><h3>E. Cross-modal similarity</h3>"
                "<i>⚠ visual_proj is zero — modality alignment dead.</i></div>")
    vf = vis[0, :n_frames].reshape(-1, vis.shape[-1])
    tf = sample["token_reprs"][0].reshape(-1, sample["token_reprs"].shape[-1])
    tm = sample["token_mask"][0].reshape(-1) > 0
    tf = tf[tm]
    if vf.shape[0] == 0 or tf.shape[0] == 0:
        return ("<div class='attn-block'><h3>E. Cross-modal similarity</h3>"
                "<i>No valid tokens or frames.</i></div>")
    vn = F.normalize(vf.float(), dim=-1)
    tn = F.normalize(tf.float(), dim=-1)
    sim = tn @ vn.T

    header = (
        f"<table class='stats-table'><tr>"
        f"<th>range</th><th>min</th><th>max</th><th>mean</th><th>std</th></tr>"
        f"<tr><td>cosine(text × visual)</td>"
        f"<td>{sim.min():.4f}</td>"
        f"<td class='hi'>{sim.max():.4f}</td>"
        f"<td>{sim.mean():.4f}</td>"
        f"<td>{sim.std():.4f}</td></tr></table>"
    )
    # Per-frame mean/max/std
    rows = [
        "<tr><th>frame</th><th>mean</th><th>max</th><th>std</th></tr>"
    ]
    for fi in range(n_frames):
        fs = sim[:, fi * viz.REGIONS:(fi + 1) * viz.REGIONS]
        if fs.numel() == 0:
            continue
        rows.append(
            f"<tr><td>F{fi:02d}</td>"
            f"<td>{fs.mean():.4f}</td>"
            f"<td class='hi'>{fs.max():.4f}</td>"
            f"<td>{fs.std():.4f}</td></tr>"
        )
    table = "<table class='stats-table'>" + "".join(rows) + "</table>"
    return (
        "<div class='attn-block'><h3>E. Cross-modal cosine similarity "
        "(projected visual ↔ text tokens)</h3>"
        + header + table + "</div>"
    )


def _render_verdict(H: dict, blocks_attn: list, e_plus, e_narr, temperature) -> str:
    """Section F — which signals are SHOW / weak / dead."""
    rows = ["<tr><th>signal</th><th>status</th><th>metric</th></tr>"]
    for bi, ba in enumerate(blocks_attn):
        for aname in ("local_ca", "global_ca", "self_attn"):
            key = aname if aname in ba else None
            if key is None:
                continue
            attn = ba[key]
            nH, Tq, Tk = attn.shape
            max_e = float(np.log2(max(Tk, 2)))
            ents = viz.per_head_row_entropy(attn)
            avg_ent = float(np.mean(ents)) if len(ents) else max_e
            ent_ratio = avg_ent / max_e if max_e > 0 else 1.0
            out_k = f"B{bi}.{aname}_out"
            out_norm = float(H[out_k].norm().item()) if out_k in H else 0.0
            alive = out_norm > 1.0
            focused = ent_ratio < 0.85
            if alive and focused:
                status, css = "✅ SHOW", "hi"
            elif alive:
                status, css = "⚠️ weak", ""
            else:
                status, css = "❌ dead", ""
            rows.append(
                f"<tr><td>B{bi}.{aname}</td>"
                f"<td class='{css}'>{status}</td>"
                f"<td>ent={avg_ent:.2f}/{max_e:.2f} "
                f"({ent_ratio:.0%}) · out_norm={out_norm:.1f}</td></tr>"
            )
    for bi in range(len(blocks_attn)):
        for fn in ("ffn1", "ffn2"):
            k = f"B{bi}.{fn}"
            if k in H:
                n = float(H[k].norm().item())
                rows.append(
                    f"<tr><td>B{bi}.{fn}</td>"
                    f"<td class='{'hi' if n > 10 else ''}'>"
                    f"{'✅ SHOW' if n > 10 else '❌ dead'}</td>"
                    f"<td>norm={n:.1f}</td></tr>"
                )
    # Final outputs
    if temperature is not None:
        tau = float(torch.exp(temperature).item())
        rows.append(
            f"<tr><td>temperature</td><td>—</td>"
            f"<td>exp(log_tau) = {tau:.4f}</td></tr>"
        )
    if e_plus is not None:
        rows.append(
            f"<tr><td>e_plus</td><td>—</td>"
            f"<td>norm = {float(e_plus.norm().item()):.4f}</td></tr>"
        )
    if e_narr is not None:
        rows.append(
            f"<tr><td>e_narr</td><td>—</td>"
            f"<td>norm = {float(e_narr.norm().item()):.4f}</td></tr>"
        )
    return "<table class='stats-table'>" + "".join(rows) + "</table>"


def _render_videorope_spotlight(local_joint: np.ndarray, n_frames_show: int,
                                frame_paths: list, top_k: int) -> str:
    """VideoRoPE 2D focus: temporal × spatial joint surface + marginals."""
    if local_joint.size == 0:
        return ""
    joint = local_joint[:n_frames_show]
    frame_mass = joint.sum(axis=1)
    region_mass = joint.sum(axis=0)

    # 1) Full frame × region joint heatmap.
    joint_img = viz.render_joint_frame_region_heatmap(joint, n_frames_show)
    joint_uri = viz.img_to_uri(joint_img, fmt="PNG")

    # 2) Per-frame mini grids for top-K frames.
    top_idx = np.argsort(-frame_mass)[:top_k]
    top_idx_sorted = sorted(top_idx.tolist())
    if top_idx_sorted:
        per_frame_arr = joint[top_idx_sorted]
        pf_img = viz.render_per_frame_region_grid(per_frame_arr, len(top_idx_sorted))
        pf_uri = viz.img_to_uri(pf_img, fmt="PNG")
        pf_caption = "top frames · 4×4 mini-grids (per-frame normalised)"
    else:
        pf_uri, pf_caption = None, None

    # 3) Spatial marginal as a 4×4 grid (region_mass).
    region_grid_img = viz.render_global_region_grid(region_mass)
    region_grid_uri = viz.img_to_uri(region_grid_img, fmt="PNG")

    # 4) Entropies along each axis.
    fm_norm = frame_mass / max(frame_mass.sum(), 1e-12)
    rm_norm = region_mass / max(region_mass.sum(), 1e-12)
    H_t = float(viz.compute_entropy(fm_norm))
    H_s = float(viz.compute_entropy(rm_norm))
    max_Ht = float(np.log2(max(len(frame_mass), 2)))
    max_Hs = float(np.log2(max(len(region_mass), 2)))

    rope_cfg = (
        f"<table class='stats-table'><tr>"
        f"<th>VideoRoPE 2D config</th>"
        f"<th>base_temporal</th><th>base_spatial</th>"
        f"<th>window (s)</th><th>H_temporal</th><th>H_spatial</th></tr>"
        f"<tr><td></td>"
        f"<td>{MODEL_CFG['base_temporal']:.0f}</td>"
        f"<td>{MODEL_CFG['base_spatial']:.0f}</td>"
        f"<td>{MODEL_CFG['lg_window_seconds']:.1f}</td>"
        f"<td class='hi'>{H_t:.2f}/{max_Ht:.2f} bits</td>"
        f"<td class='hi'>{H_s:.2f}/{max_Hs:.2f} bits</td>"
        f"</tr></table>"
    )

    cells = [_render_attn_cell(joint_uri,
                                "Joint frame × region surface (VideoRoPE 2D space)")]
    if pf_uri is not None:
        cells.append(_render_attn_cell(pf_uri, pf_caption))
    cells.append(_render_attn_cell(region_grid_uri,
                                    "Spatial marginal (sum over frames)"))

    return (
        "<div class='attn-block'>"
        "<h3>🎯 VideoRoPE 2D spotlight — temporal × spatial attention surface "
        "(aggregated local_ca over all blocks)</h3>"
        "<p style='font-size:13px; color:#444'>"
        "VideoRoPE 2D embeds <b>(t, row, col)</b> positions into the rotation. "
        "Temporal axis uses a long base (500K) for time encoding, the two spatial "
        "axes (row, col of the 4×4 grid) use a smaller base (10K). The 8-second "
        "window restricts local_ca to nearby frames only. Entropy below tells "
        "you how concentrated attention is along each axis.</p>"
        + rope_cfg
        + "<div class='section-label'>Temporal marginal — attention mass per frame</div>"
        + _render_frame_bars(frame_mass, n_frames_show)
        + "<div class='section-label'>Spatial marginal — attention mass per region</div>"
        + _render_region_bars(region_mass)
        + f"<div class='attn-row'>{''.join(cells)}</div>"
        "</div>"
    )


def _render_block(block_attn: dict, label: str,
                  n_frames_padded: int, n_frames_show: int,
                  frame_paths: list, top_frames_per_block: int,
                  show_self: bool = True) -> str:
    parts = [f"<div class='attn-block'><h3>{html.escape(label)}</h3>"]

    # Compute local first so we can reuse its top frames for the global overlay.
    l_t = block_attn.get("local_ca", block_attn.get("local"))
    if isinstance(l_t, torch.Tensor):
        local = viz.reduce_local_to_frame_region(l_t, n_frames_padded)
    elif isinstance(l_t, np.ndarray):
        local = l_t
    else:
        local = np.zeros((n_frames_padded, viz.REGIONS))
    local = local[:n_frames_show]
    frame_mass = local.sum(axis=1)
    top_idx = np.argsort(-frame_mass)[:top_frames_per_block]

    # --- Global attention ---
    g_t = block_attn.get("global_ca", block_attn.get("global"))
    if isinstance(g_t, torch.Tensor):
        g = viz.reduce_global_to_region(g_t)
    elif isinstance(g_t, np.ndarray):
        g = g_t
    else:
        g = np.zeros((viz.REGIONS,))
    g_img = viz.render_global_region_grid(g)
    g_uri = viz.img_to_uri(g_img, fmt="PNG")

    parts.append("<div class='section-label'>Global CA — spatial region preference "
                 "(frame-agnostic, overlaid on top local frames for context)</div>")
    parts.append(_render_stats_html("global", g))
    g_cells = [_render_attn_cell(g_uri, "global · 4×4 region grid (% per cell)")]
    for fi in top_idx[:3]:
        if fi >= len(frame_paths):
            continue
        try:
            img = Image.open(frame_paths[fi])
            overlay = viz.overlay_region_attn(img, g)
            g_cells.append(_render_attn_cell(
                viz.img_to_uri(overlay),
                f"global on F{fi:02d} · same spatial pattern, different frame",
            ))
        except Exception:
            continue
    parts.append(f"<div class='attn-row'>{''.join(g_cells)}</div>")

    if isinstance(g_t, torch.Tensor):
        g_ph = viz.reduce_global_per_head(g_t)
        ph_img = viz.render_per_head_grids(g_ph)
        ph_uri = viz.img_to_uri(ph_img, fmt="PNG")
        parts.append(
            f"<details><summary style='cursor:pointer;font-weight:600;margin:6px 0;'>"
            f"▶ Per-head global attention ({g_ph.shape[0]} heads)</summary>"
            f"<img src='{ph_uri}' class='per-head-img' loading='lazy' /></details>"
        )
        parts.append(
            f"<details><summary style='cursor:pointer;font-weight:600;margin:6px 0;'>"
            f"▶ Per-head entropy (global_ca)</summary>"
            f"{_render_per_head_entropy(g_t, 'global_ca')}</details>"
        )

    # --- Local attention ---
    parts.append("<div class='section-label'>Local CA — frame × region attention "
                 "(VideoRoPE 2D, 8s window)</div>")
    parts.append(_render_stats_html("local", local))

    parts.append(
        f"<details open><summary style='cursor:pointer;font-weight:600;margin:6px 0;'>"
        f"📊 Frame attention distribution ({n_frames_show} frames)</summary>"
        f"{_render_frame_bars(frame_mass, n_frames_show)}</details>"
    )

    # Joint frame × region mini heatmap for this block.
    joint_img = viz.render_joint_frame_region_heatmap(local, n_frames_show)
    joint_uri = viz.img_to_uri(joint_img, fmt="PNG")
    parts.append(
        f"<details open><summary style='cursor:pointer;font-weight:600;margin:6px 0;'>"
        f"🧩 Joint (frame × region) heatmap — VideoRoPE 2D surface</summary>"
        f"<div class='attn-row'>"
        f"{_render_attn_cell(joint_uri, 'rows = frames · cols = 16 regions')}"
        f"</div></details>"
    )

    cells = []
    for fi in top_idx:
        if fi >= len(frame_paths):
            continue
        img = Image.open(frame_paths[fi])
        overlay = viz.overlay_region_attn(img, local[fi])
        cells.append(_render_attn_cell(
            viz.img_to_uri(overlay),
            f"frame {fi:02d} · mass={frame_mass[fi]:.4f} · "
            f"peak={local[fi].max():.4f} ({local[fi].argmax()//4},{local[fi].argmax()%4})"
        ))
    if cells:
        parts.append(f"<div class='attn-row'>{''.join(cells)}</div>")

    if isinstance(l_t, torch.Tensor):
        parts.append(
            f"<details><summary style='cursor:pointer;font-weight:600;margin:6px 0;'>"
            f"▶ Per-head entropy (local_ca)</summary>"
            f"{_render_per_head_entropy(l_t, 'local_ca')}</details>"
        )

    # --- Self-attention (was captured but never rendered before) ---
    s_t = block_attn.get("self_attn", block_attn.get("self"))
    if show_self and isinstance(s_t, torch.Tensor):
        # Reduce across heads + query positions → key distribution
        s_flat = s_t.mean(dim=0).mean(dim=0).numpy()
        parts.append("<div class='section-label'>Self-attention — segment ↔ segment</div>")
        parts.append(_render_stats_html("self_attn", s_flat))
        parts.append(
            f"<details><summary style='cursor:pointer;font-weight:600;margin:6px 0;'>"
            f"▶ Per-head entropy (self_attn)</summary>"
            f"{_render_per_head_entropy(s_t, 'self_attn')}</details>"
        )

    parts.append("</div>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Preview pane (all frames + all segment texts for the chosen shot)
# ---------------------------------------------------------------------------

def _preview_shot(chosen_shot: str | None) -> str:
    if not chosen_shot:
        return ""
    video_id, sid = chosen_shot.rsplit("_", 1)
    frames_uris = _load_frame_data_uris(video_id, sid)
    segments = _load_segments(video_id, sid)

    frames_html = "".join(
        f"<div class='preview-frame'>"
        f"<img src='{u}' loading='lazy' />"
        f"<div class='cap'>F{i:02d}</div></div>"
        for i, u in enumerate(frames_uris)
    ) or "<i>No frames on disk.</i>"

    seg_rows = []
    for i, s in enumerate(segments):
        ts = float(s.get("start", 0))
        end = s.get("end")
        end_str = f" → {float(end):.1f}s" if end is not None else ""
        text = html.escape(str(s.get("text", "")).strip())
        seg_rows.append(
            f"<tr><td>S{i:02d}</td>"
            f"<td>{ts:.1f}s{end_str}</td>"
            f"<td style='text-align:left'>{text}</td></tr>"
        )
    if seg_rows:
        seg_table = (
            "<table class='stats-table' style='width:100%'><tr>"
            "<th>seg</th><th>time</th><th>text</th></tr>"
            + "".join(seg_rows) + "</table>"
        )
    else:
        seg_table = "<i>No transcript on disk.</i>"

    return (
        "<div class='card'>"
        f"<h3>Preview · {html.escape(chosen_shot)} — "
        f"N = {len(frames_uris)} frames · M = {len(segments)} segments</h3>"
        f"<div class='section-label'>All frames (N = {len(frames_uris)})</div>"
        f"<div class='preview-frames'>{frames_html}</div>"
        f"<div class='section-label'>All segments (M = {len(segments)})</div>"
        + seg_table + "</div>"
    )


# ---------------------------------------------------------------------------
# Top-3 segment × last fusion block VideoRoPE routing
# ---------------------------------------------------------------------------

def _query_segment_similarity(query: str, sample: dict) -> np.ndarray:
    """Cosine similarity between BGE-M3 query embedding and each segment_pooled.

    Returns ``[M]`` array; invalid segments are -inf so they sort last.
    """
    if not query:
        return np.zeros(sample["segment_pooled"].shape[1], dtype=np.float32)
    q_emb = encode_queries(
        [query],
        max_length=DATA_CFG["eval_query_max_length"],
        batch_size=DATA_CFG["eval_query_batch_size"],
    ).detach().cpu().float()
    qn = F.normalize(q_emb, dim=-1)                          # [1, D]
    sp = sample["segment_pooled"][0].float()                  # [M, D]
    sn = F.normalize(sp, dim=-1)
    sims = (sn @ qn.T).squeeze(-1).numpy().astype(np.float32)
    mask = sample["segment_mask"][0].numpy() > 0
    sims_masked = np.where(mask, sims, -np.inf).astype(np.float32)
    return sims_masked


def _slice_segment_local(local_attn: torch.Tensor, seg_idx: int,
                         T: int, token_mask_m: torch.Tensor,
                         n_frames_padded: int) -> np.ndarray:
    """``[H, M*T, N*R]`` → ``[N, R]`` for one segment, weighted by its token mask."""
    rows = local_attn[:, seg_idx * T:(seg_idx + 1) * T, :]    # [H, T, N*R]
    tm = token_mask_m.float()
    denom = float(tm.sum().item())
    if denom > 0:
        weighted = (rows * tm.view(1, T, 1)).sum(dim=1) / denom  # [H, N*R]
    else:
        weighted = rows.mean(dim=1)
    avg = weighted.mean(dim=0).numpy()                         # [N*R]
    arr = avg.reshape(n_frames_padded, viz.REGIONS)
    s = arr.sum()
    return arr / s if s > 0 else arr


def _slice_segment_global(global_attn: torch.Tensor, seg_idx: int,
                          T: int, token_mask_m: torch.Tensor) -> np.ndarray:
    """``[H, M*T, R]`` → ``[R]`` for one segment, weighted by its token mask."""
    rows = global_attn[:, seg_idx * T:(seg_idx + 1) * T, :]   # [H, T, R]
    tm = token_mask_m.float()
    denom = float(tm.sum().item())
    if denom > 0:
        weighted = (rows * tm.view(1, T, 1)).sum(dim=1) / denom  # [H, R]
    else:
        weighted = rows.mean(dim=1)
    avg = weighted.mean(dim=0).numpy()                         # [R]
    s = avg.sum()
    return avg / s if s > 0 else avg


def _render_topk_segments_last_block(
    blocks_attn: list, sample: dict, query: str,
    frame_paths: list, n_frames_show: int, n_frames_padded: int,
    top_n_seg: int = 3,
) -> str:
    """Top-N segments by query cosine sim → show each one's last-block routing."""
    if not blocks_attn:
        return ""
    last = blocks_attn[-1]
    local_attn = last.get("local_ca", last.get("local"))
    global_attn = last.get("global_ca", last.get("global"))
    if local_attn is None and global_attn is None:
        return ""
    if not isinstance(local_attn, torch.Tensor):
        local_attn = None
    if not isinstance(global_attn, torch.Tensor):
        global_attn = None

    sims = _query_segment_similarity(query, sample)
    valid = np.isfinite(sims)
    order = np.argsort(-sims)
    top_idx = [int(i) for i in order if valid[i]][:top_n_seg]
    if not top_idx:
        return ""

    shot_id = sample["shot_id"][0] if isinstance(sample["shot_id"], list) else sample["shot_id"]
    video_id, sid = shot_id.rsplit("_", 1)
    segments = _load_segments(video_id, sid)

    M_total, T = sample["token_mask"].shape[1], sample["token_mask"].shape[2]
    token_mask_b = sample["token_mask"][0]      # [M, T]

    parts = [
        "<div class='attn-block'>"
        "<h3>🎯 Top-3 segments × last fusion block — how each segment routes "
        "through VideoRoPE 2D</h3>"
        "<p style='font-size:13px; color:#444'>"
        "Top segments are ranked by <b>cosine(query, BGE-M3 segment_pooled)</b>. "
        "For each one we slice the <b>last fusion block</b>'s "
        f"local_ca (rows [s·T : (s+1)·T] of the {M_total}×{T} segment-token grid) "
        "and average them into one (frame × region) distribution — exactly "
        "the VideoRoPE 2D surface that segment sees. "
        "Different segments often light up different frames; that's the "
        "temporal routing local_ca's 8s window + temporal RoPE enables.</p>"
    ]

    # Tiny summary of the top-3 sims.
    sim_rows = ["<tr><th>rank</th><th>segment</th><th>cosine</th><th>text</th></tr>"]
    for r, m in enumerate(top_idx, 1):
        seg_t = ""
        if m < len(segments):
            ts = float(segments[m].get("start", 0))
            seg_t = f"<b>t={ts:.1f}s</b> · " + html.escape(
                str(segments[m].get("text", "")).strip()
            )
        sim_rows.append(
            f"<tr><td>#{r}</td><td>S{m:02d}</td>"
            f"<td class='hi'>{sims[m]:.4f}</td>"
            f"<td style='text-align:left'>{seg_t}</td></tr>"
        )
    parts.append("<table class='stats-table' style='width:100%'>"
                 + "".join(sim_rows) + "</table>")

    for r, m in enumerate(top_idx, 1):
        seg_t = ""
        if m < len(segments):
            ts = float(segments[m].get("start", 0))
            seg_t = f"<b>t = {ts:.1f}s</b> · " + html.escape(
                str(segments[m].get("text", "")).strip()
            )
        parts.append(
            f"<div style='border:1px solid #ccc;border-radius:10px;padding:12px;"
            f"margin-top:14px;background:#fff8f0;'>"
            f"<div style='font-size:14px;font-weight:700'>"
            f"#{r}  Segment S{m:02d}  ·  query similarity = {sims[m]:.4f}</div>"
            f"<div style='font-size:13px;margin-top:4px'>{seg_t}</div>"
        )

        tm_m = token_mask_b[m]
        n_valid_tokens = int(tm_m.sum().item())

        # Default top_f for global overlay if local is missing → evenly spaced.
        top_f = np.linspace(0, max(n_frames_show - 1, 0), 3, dtype=int)

        if local_attn is not None:
            local_seg = _slice_segment_local(
                local_attn, m, T, tm_m, n_frames_padded,
            )[:n_frames_show]
            joint_img = viz.render_joint_frame_region_heatmap(local_seg, n_frames_show)
            joint_uri = viz.img_to_uri(joint_img, fmt="PNG")

            frame_mass = local_seg.sum(axis=1)
            top_f = np.argsort(-frame_mass)[:3]
            cells = []
            for fi in top_f:
                if fi >= len(frame_paths):
                    continue
                img = Image.open(frame_paths[fi])
                overlay = viz.overlay_region_attn(img, local_seg[fi])
                cells.append(_render_attn_cell(
                    viz.img_to_uri(overlay),
                    f"F{fi:02d} · mass={frame_mass[fi]:.4f}",
                ))

            parts.append(
                "<div class='section-label'>Local CA · last block · this segment</div>"
            )
            parts.append(
                f"<div style='font-size:12px;color:#444'>"
                f"weighted over {n_valid_tokens} valid query tokens</div>"
            )
            parts.append(_render_stats_html(f"local_ca · S{m:02d}", local_seg))
            parts.append(
                f"<div class='attn-row'>"
                f"{_render_attn_cell(joint_uri, 'joint (frame × region) — VideoRoPE 2D surface')}"
                f"{''.join(cells)}</div>"
            )

        if global_attn is not None:
            global_seg = _slice_segment_global(global_attn, m, T, tm_m)
            g_img = viz.render_global_region_grid(global_seg)
            g_uri = viz.img_to_uri(g_img, fmt="PNG")
            parts.append(
                "<div class='section-label'>Global CA · last block · this segment "
                "(frame-agnostic; overlaid on this segment's top local frames)</div>"
            )
            parts.append(_render_stats_html(f"global_ca · S{m:02d}", global_seg))
            g_cells = [_render_attn_cell(g_uri, "4×4 region grid (% per cell)")]
            for fi in top_f[:3]:
                if fi >= len(frame_paths):
                    continue
                try:
                    img = Image.open(frame_paths[fi])
                    overlay = viz.overlay_region_attn(img, global_seg)
                    g_cells.append(_render_attn_cell(
                        viz.img_to_uri(overlay),
                        f"global on F{fi:02d}",
                    ))
                except Exception:
                    continue
            parts.append(f"<div class='attn-row'>{''.join(g_cells)}</div>")

        parts.append("</div>")

    parts.append("</div>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Segment × Frame interaction matrix
# ---------------------------------------------------------------------------

def _last_block_idx(H: dict) -> int | None:
    """Find the highest block index that has a norm4 hook."""
    last = None
    for k in H.keys():
        if k.startswith("B") and k.endswith(".norm4"):
            try:
                bi = int(k.split(".")[0][1:])
            except ValueError:
                continue
            if last is None or bi > last:
                last = bi
    return last


def _segment_frame_interaction(capture: dict, sample: dict):
    """Cosine matrix ``[M_valid, N_valid]`` between post-fusion segment vectors
    (last block's ``norm4`` output, which is exactly ``norm(y + local_ca + global_ca)``
    — i.e. the tensor that feeds FFN2) and per-frame visual vectors
    (``visual_proj`` averaged over the 16 regions).

    Returns ``(sim_matrix, M_valid, N_valid)`` or ``(None, 0, 0)`` if hooks
    are missing.
    """
    H = capture["H"]
    last_bi = _last_block_idx(H)
    if last_bi is None or "visual_proj" not in H:
        return None, 0, 0
    n4 = H.get(f"B{last_bi}.norm4")
    if n4 is None:
        return None, 0, 0

    token_mask = sample["token_mask"][0]              # [M, T]
    M, T = token_mask.shape
    segment_mask = sample["segment_mask"][0].numpy() > 0
    visual_mask = sample["visual_mask"][0].numpy() > 0
    N_padded = sample["visual_features"].shape[1]

    # norm4: [1, M*T, D] → [M, T, D]
    if n4.dim() == 3 and n4.shape[1] == M * T:
        seg_grid = n4[0].view(M, T, -1)
    elif n4.dim() == 4:
        seg_grid = n4[0]
    else:
        seg_grid = n4.reshape(M, T, -1)

    tm = token_mask.float().unsqueeze(-1)              # [M, T, 1]
    denom = tm.sum(dim=1).clamp_min(1e-6)              # [M, 1]
    seg_pooled = (seg_grid * tm).sum(dim=1) / denom    # [M, D]

    # visual_proj: [1, N, 16, D] (Linear preserves leading dims). Fallback to flat.
    vp = H["visual_proj"]
    if vp.dim() == 4:
        frame_vec = vp[0].mean(dim=1)                  # [N, D]
    elif vp.dim() == 3 and vp.shape[1] == N_padded * 16:
        frame_vec = vp[0].view(N_padded, 16, -1).mean(dim=1)
    else:
        return None, 0, 0

    sn = F.normalize(seg_pooled.float(), dim=-1)
    fn = F.normalize(frame_vec.float(), dim=-1)
    sim = (sn @ fn.T).numpy()                          # [M, N]

    M_valid = int(segment_mask.sum())
    N_valid = int(visual_mask.sum())
    return sim[:M_valid, :N_valid], M_valid, N_valid


def _render_segment_frame_interaction(
    capture: dict, sample: dict,
    frame_paths: list, n_frames_show: int,
    top_k_cells: int = 3,
) -> str:
    sim, M_valid, N_valid = _segment_frame_interaction(capture, sample)
    if sim is None or sim.size == 0:
        return ""
    # Clip frames to what we can actually display.
    sim = sim[:, :max(min(N_valid, n_frames_show), 1)]
    M, N = sim.shape
    if M == 0 or N == 0:
        return ""

    shot_id = sample["shot_id"][0] if isinstance(sample["shot_id"], list) else sample["shot_id"]
    video_id, sid = shot_id.rsplit("_", 1)
    segments = _load_segments(video_id, sid)

    mat_img = viz.render_segment_frame_matrix(sim, top_k=top_k_cells)
    mat_uri = viz.img_to_uri(mat_img, fmt="PNG")

    flat = sim.flatten()
    top_idx = np.argsort(-flat)[:top_k_cells]
    top_cells = [(int(i // N), int(i % N), float(flat[i])) for i in top_idx]

    rows = [
        "<tr><th>rank</th><th>seg</th><th>frame</th>"
        "<th>cosine</th><th>segment text</th></tr>"
    ]
    visual_cells = []
    for rank, (m, f, sc) in enumerate(top_cells, 1):
        seg_text = ""
        if m < len(segments):
            ts = float(segments[m].get("start", 0))
            seg_text = (f"<b>t = {ts:.1f}s</b> · "
                        + html.escape(str(segments[m].get("text", "")).strip()))
        rows.append(
            f"<tr><td>#{rank}</td><td>S{m:02d}</td><td>F{f:02d}</td>"
            f"<td class='hi'>{sc:.4f}</td>"
            f"<td style='text-align:left'>{seg_text}</td></tr>"
        )
        if 0 <= f < len(frame_paths):
            try:
                img = Image.open(frame_paths[f]).convert("RGB")
                if img.width > 300:
                    h = int(img.height * (300 / img.width))
                    img = img.resize((300, max(h, 1)))
                visual_cells.append(_render_attn_cell(
                    viz.img_to_uri(img, fmt="JPEG"),
                    f"#{rank}  S{m:02d} ↔ F{f:02d}  ·  cos = {sc:.4f}",
                ))
            except Exception:
                continue

    table = ("<table class='stats-table' style='width:100%'>"
             + "".join(rows) + "</table>")

    return (
        "<div class='attn-block'>"
        "<h3>🔗 Segment × Frame interaction — post (local + global) → FFN, last block</h3>"
        "<p style='font-size:13px; color:#444'>"
        "Cosine similarity between <b>post-fusion segment vectors</b> "
        "(<code>norm4(y + local_ca + global_ca)</code> of the last block, "
        "i.e. exactly the tensor that feeds FFN2, pooled per segment with "
        "<code>token_mask</code>) and <b>per-frame visual vectors</b> "
        "(<code>visual_proj</code> averaged over the 16 region tokens). "
        "Rows = segments, cols = frames. The 3 strongest cells are outlined "
        "in yellow with rank labels.</p>"
        + f"<div class='attn-row'>"
          f"{_render_attn_cell(mat_uri, f'{M} segments × {N} frames · cosine')}"
          f"</div>"
        + "<div class='section-label'>Top-3 (segment, frame) interactions</div>"
        + table
        + (f"<div class='attn-row'>{''.join(visual_cells)}</div>" if visual_cells else "")
        + "</div>"
    )


def _inspect(query: str, chosen_shot: str | None,
             top_frames_per_block: int = 6,
             show_full_norms: bool = False) -> str:
    """Render the full inspection report for the user-picked shot."""
    top_frames_per_block = int(top_frames_per_block)
    if not chosen_shot:
        # Fall back to top-1 so the page is never empty.
        used_query, items = _retrieve_topk(query, k=1)
        if not items:
            return "<div class='attn-block'>No query and no test queries on disk.</div>"
        query = used_query
        chosen_shot = items[0][0]

    shot_id = chosen_shot
    video_id, sid = shot_id.rsplit("_", 1)

    model = _inspect_model()
    sample = viz.find_sample(_inspect_loader(), shot_id)
    if sample is None:
        return f"<div class='attn-block'>Shot {shot_id} not in test loader.</div>"

    capture = viz.forward_with_full_capture(model, sample, DEVICE)
    H = capture["H"]
    blocks_attn = capture["blocks_attn"]

    n_frames_eff = int(sample["visual_mask"].sum().item())
    n_frames_padded = sample["visual_features"].shape[1]
    frame_paths = _list_frame_paths(video_id, sid)
    if not frame_paths:
        return f"<div class='attn-block'>No frame images for {shot_id}.</div>"
    if len(frame_paths) > n_frames_padded:
        idx = np.linspace(0, len(frame_paths) - 1, n_frames_padded, dtype=int)
        frame_paths = [frame_paths[i] for i in idx]
    n_frames_show = min(len(frame_paths), n_frames_eff)

    n_blocks = len(blocks_attn)
    first_block = blocks_attn[0] if blocks_attn else {}
    if "local_ca" in first_block:
        n_heads = first_block["local_ca"].shape[0]
    elif "local" in first_block:
        n_heads = first_block["local"].shape[0]
    else:
        n_heads = "?"

    header = (
        "<div class='attn-block layer-header'>"
        f"<h3>🔍 Inspect Variant 3 — VideoRoPE 2D + Local-Global CA</h3>"
        f"<div><b>Query:</b> {html.escape(query or '(random)')}</div>"
        f"<div><b>Chosen shot:</b> {html.escape(shot_id)} "
        f"(<b>{n_frames_eff}</b> frames, 4×4 spatial regions = 16 per frame, "
        f"<b>{n_blocks}</b> fusion blocks, <b>{n_heads}</b> attention heads, "
        f"<b>{MODEL_CFG['lg_window_seconds']:.0f}s</b> local window, "
        f"base_t={MODEL_CFG['base_temporal']:.0f}, "
        f"base_s={MODEL_CFG['base_spatial']:.0f})</div>"
        "<p style='margin-top:8px; color:#444; font-size:13px'>"
        "🟠 <b>local_ca</b>: segment tokens → (frame × region) under the 8s temporal "
        "window. The <i>frame</i> axis is rotated by base_temporal; the two "
        "<i>spatial</i> axes (row, col of the 4×4 grid) by base_spatial.<br>"
        "🔵 <b>global_ca</b>: segment tokens → 16 region-pooled tokens (mean over "
        "frames) — pure spatial preference, no temporal structure.<br>"
        "🟣 <b>self_attn</b>: segment ↔ segment fusion within the block.<br>"
        "📊 <b>Stats</b>: Entropy ↑ = uniform attention, Entropy ↓ = focused. "
        "Sparsity = fraction of cells below mean.</p>"
        "</div>"
    )

    sections = [header]

    # === VideoRoPE 2D spotlight (aggregated local_ca) ===
    if blocks_attn:
        agg_local = viz.reduce_local_aggregate_blocks(blocks_attn, n_frames_padded)
        sections.append(_render_videorope_spotlight(
            agg_local, n_frames_show, frame_paths, top_frames_per_block,
        ))

    # === Top-3 segments × last fusion block ===
    sections.append(_render_topk_segments_last_block(
        blocks_attn, sample, query, frame_paths,
        n_frames_show, n_frames_padded, top_n_seg=3,
    ))

    # === Segment × Frame interaction matrix (post local+global → FFN) ===
    sections.append(_render_segment_frame_interaction(
        capture, sample, frame_paths, n_frames_show, top_k_cells=3,
    ))

    # === A. Gate values ===
    sections.append(
        "<div class='attn-block'>"
        "<h3>A. Sigmoid gate values — what fraction of each path is open</h3>"
        "<p style='font-size:13px; color:#444'>Closer to 0 means the gate is "
        "suppressing that path; closer to 1 means it's wide open.</p>"
        + _render_gate_table(model) + "</div>"
    )

    # === D. Residual stream contribution ===
    sections.append(
        "<div class='attn-block'>"
        "<h3>D. Residual stream contribution — L2 norms by source</h3>"
        "<p style='font-size:13px; color:#444'>Per block, which sub-module "
        "actually <i>moves</i> the residual stream (after the gate scaling).</p>"
        + _render_residual_table(H, n_blocks) + "</div>"
    )

    # === Per-block detailed rendering ===
    for i, ba in enumerate(blocks_attn):
        sections.append(_render_block(
            ba, f"Block {i + 1} of {n_blocks}",
            n_frames_padded, n_frames_show, frame_paths,
            top_frames_per_block, show_self=True,
        ))

    # === Aggregated block ===
    if blocks_attn:
        agg = {
            "local_ca": viz.reduce_local_aggregate_blocks(blocks_attn, n_frames_padded),
            "global_ca": viz.reduce_global_aggregate_blocks(blocks_attn),
        }
        sections.append(_render_block(
            agg, "⬛ Aggregated (mean over all blocks)",
            n_frames_padded, n_frames_show, frame_paths,
            top_frames_per_block, show_self=False,
        ))

    # === E. Cross-modal similarity ===
    sections.append(_render_cross_modal(H, sample, n_frames_eff))

    # === F. Verdict ===
    sections.append(
        "<div class='attn-block'>"
        "<h3>F. Verdict — which signals are worth showing</h3>"
        + _render_verdict(H, blocks_attn,
                          capture["e_plus"], capture["e_narr"], capture["temperature"])
        + "</div>"
    )

    # === B. Intermediate norms (collapsible — long table) ===
    if show_full_norms:
        sections.append(
            "<div class='attn-block'>"
            "<h3>B. All intermediate tensor norms</h3>"
            f"<details open><summary>show {len(H)} tensors</summary>"
            + _render_norms_table(H) + "</details></div>"
        )
    else:
        sections.append(
            "<div class='attn-block'>"
            "<h3>B. All intermediate tensor norms</h3>"
            f"<details><summary>▶ show {len(H)} tensors</summary>"
            + _render_norms_table(H) + "</details></div>"
        )

    return "".join(sections)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_ui():
    with gr.Blocks(css=CSS) as demo:
        gr.Markdown("# News Retrival System",
                    elem_id="title")
        with gr.Tabs():
            with gr.Tab("Retrieve"):
                with gr.Row():
                    with gr.Column(scale=2):
                        query = gr.Textbox(label="Query",
                                           placeholder="Type query or leave empty for random")
                    with gr.Column(scale=1):
                        top_k = gr.Slider(1, 20, value=10, step=1, label="Top K")
                with gr.Row():
                    search_btn = gr.Button("Search", variant="primary")
                    rebuild_btn = gr.Button("Rebuild index")
                with gr.Row():
                    output_html = gr.HTML(elem_classes=["card"])
                search_btn.click(_search, inputs=[query, top_k], outputs=[output_html])
                rebuild_btn.click(_rebuild_index, outputs=[output_html])

            with gr.Tab("Inspect attention"):
                gr.Markdown(
                    "Inspect what the model attends to in a retrieved shot. "
                    "**Step 1:** retrieve top-K candidates for a query. "
                    "**Step 2:** pick one from the dropdown and run the full "
                    "VideoRoPE 2D / local-global CA inspection on it."
                )
                with gr.Row():
                    with gr.Column(scale=3):
                        iquery = gr.Textbox(
                            label="Query",
                            placeholder="Type query or leave empty for random",
                        )
                    with gr.Column(scale=1):
                        itopk = gr.Slider(1, 20, value=10, step=1,
                                          label="Top-K candidates")
                with gr.Row():
                    retrieve_btn = gr.Button("① Retrieve top-K",
                                             variant="primary")
                with gr.Row():
                    candidates_dd = gr.Dropdown(
                        label="② Candidate shot to inspect (rank · shot · score)",
                        choices=[], value=None, interactive=True,
                    )
                with gr.Row():
                    with gr.Column(scale=1):
                        n_frames = gr.Slider(
                            2, 12, value=6, step=1,
                            label="Top frames to show per block",
                        )
                    with gr.Column(scale=1):
                        show_norms = gr.Checkbox(
                            label="Expand intermediate norm table (Section B)",
                            value=False,
                        )
                with gr.Row():
                    inspect_btn = gr.Button("③ Run inspection",
                                            variant="primary")
                retrieve_status = gr.Markdown()
                preview_html = gr.HTML(elem_classes=["card"])
                inspect_html = gr.HTML(elem_classes=["card"])

                def _on_retrieve(q, k):
                    used_q, items = _retrieve_topk(q, int(k))
                    if not items:
                        return (
                            gr.update(choices=[], value=None),
                            "No candidates found.",
                            "",
                        )
                    choices = [
                        (_candidate_label(i + 1, sid, sc), sid)
                        for i, (sid, sc) in enumerate(items)
                    ]
                    note = (f"Top-{len(items)} for **{html.escape(used_q)}** — "
                            f"pick one to inspect.")
                    first = choices[0][1]
                    return (
                        gr.update(choices=choices, value=first),
                        note,
                        _preview_shot(first),
                    )

                retrieve_btn.click(
                    _on_retrieve,
                    inputs=[iquery, itopk],
                    outputs=[candidates_dd, retrieve_status, preview_html],
                )
                candidates_dd.change(
                    _preview_shot,
                    inputs=[candidates_dd],
                    outputs=[preview_html],
                )
                inspect_btn.click(
                    _inspect,
                    inputs=[iquery, candidates_dd, n_frames, show_norms],
                    outputs=[inspect_html],
                )
    return demo


def main():
    print(f"[demo] BASE_DIR    = {BASE_DIR}")
    print(f"[demo] CHECKPOINT  = {CHECKPOINT}")
    print(f"[demo] DEVICE      = {DEVICE}"
          + ("  (forced via --cpu)" if FORCE_CPU else ""))
    if FORCE_CPU:
        try:
            torch.set_num_threads(max(1, (os.cpu_count() or 4)))
        except Exception:
            pass
    _index_cache()
    demo = build_ui()
    # Bind to all interfaces but print a clickable localhost URL — some
    # browsers (Edge) refuse to navigate to 0.0.0.0.
    print("[demo] Open in browser: http://127.0.0.1:7860")
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
