"""Gradio UI for Variant 3 retrieval demo.

Run with::

    python -m source.demo.app
    # or:
    cd source/demo && python app.py
"""
from __future__ import annotations

import base64
import html
import io
import json
import random
import sys
from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image

# Ensure local imports work whether run as ``python -m source.demo.app`` or
# ``python app.py`` from inside this folder.
HERE = Path(__file__).resolve().parent
if __package__ in (None, ""):
    sys.path.insert(0, str(HERE.parent.parent))
    from source.demo.config import (  # noqa: E402
        BASE_DIR, CHECKPOINT, DATA_CFG, THUMB_MAX_WIDTH,
    )
    from source.demo.model_loader import DEVICE, encode_queries, load_main_model  # noqa: E402
    from source.demo.retrieval import build_index, build_keyframe_loader, get_or_build_index  # noqa: E402
    from source.demo import visualize as viz  # noqa: E402
else:
    from .config import BASE_DIR, CHECKPOINT, DATA_CFG, THUMB_MAX_WIDTH
    from .model_loader import DEVICE, encode_queries, load_main_model
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
}
.gradio-container { max-width: 1100px !important; }
#title {
  background: linear-gradient(90deg, #0b0b0b, #7a1313, #0b0b0b);
  color: #fff; padding: 16px 18px; border-radius: 12px;
  text-transform: uppercase; letter-spacing: 2px;
}
.card { border: 1px solid #2b2b2b; border-radius: 14px; padding: 16px; background: #fff;
        box-shadow: 0 10px 25px rgba(0,0,0,0.08); }
.layer { border: 1px solid #111; border-radius: 12px; padding: 14px; background: #fff;
         color: #000; margin-bottom: 14px; }
.layer-header { background: #fff8f0; }
.layer-head { font-weight: 700; margin-bottom: 8px; letter-spacing: 0.5px; color: #000; }
.layer-text { white-space: pre-wrap; line-height: 1.5; color: #000; }
.frames { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
.frames img { width: 220px; height: auto; border-radius: 8px; border: 1px solid #ddd; }
.attn-row { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 10px; align-items: flex-start; }
.attn-cell { display: flex; flex-direction: column; align-items: center;
             border: 1px solid #222; border-radius: 10px; padding: 8px; background: #fafafa; }
.attn-cell img { display: block; max-width: 320px; border-radius: 6px; }
.attn-cell .cap { font-size: 12px; color: #333; margin-top: 6px; font-weight: 600; }
.attn-grid { display: flex; gap: 16px; flex-wrap: wrap; }
.attn-block { border: 1px solid #111; border-radius: 12px; padding: 14px; background: #fff;
              flex: 1 1 100%; }
.attn-block h3 { margin: 0 0 8px 0; }
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


def _retrieve_top1(query: str):
    if not query:
        items = _load_test_queries()
        if not items:
            return None, None
        pick = random.choice(items)
        query = pick["query"]
    index, shot_ids = _index_cache()
    q_emb = encode_queries([query],
                           max_length=DATA_CFG["eval_query_max_length"],
                           batch_size=DATA_CFG["eval_query_batch_size"])
    q_np = q_emb.detach().cpu().numpy().astype("float32")
    faiss.normalize_L2(q_np)
    _, idxs = index.search(q_np, 1)
    return query, shot_ids[int(idxs[0][0])]


def _render_attn_cell(uri: str, caption: str) -> str:
    return (f"<div class='attn-cell'>"
            f"<img src='{uri}' loading='lazy' />"
            f"<div class='cap'>{html.escape(caption)}</div></div>")


def _inspect(query: str, top_frames_per_block: int = 6) -> str:
    top_frames_per_block = int(top_frames_per_block)  # Gradio slider → float
    query, shot_id = _retrieve_top1(query)
    if shot_id is None:
        return "<div class='attn-block'>No query and no test queries on disk.</div>"
    video_id, sid = shot_id.rsplit("_", 1)

    model = _inspect_model()
    sample = viz.find_sample(_inspect_loader(), shot_id)
    if sample is None:
        return f"<div class='attn-block'>Shot {shot_id} not in test loader.</div>"
    blocks_attn = viz.forward_with_capture(model, sample, DEVICE)

    n_frames_eff = int(sample["visual_mask"].sum().item())
    n_frames_padded = sample["visual_features"].shape[1]
    frame_paths = _list_frame_paths(video_id, sid)
    if not frame_paths:
        return f"<div class='attn-block'>No frame images for {shot_id}.</div>"

    # Align frame_paths to feature indices: if the dataset resampled frames
    # via np.linspace (n_actual > max_frames), do the same here so frame_paths[i]
    # corresponds to the i-th feature row.
    if len(frame_paths) > n_frames_padded:
        idx = np.linspace(0, len(frame_paths) - 1, n_frames_padded, dtype=int)
        frame_paths = [frame_paths[i] for i in idx]
    n_frames_show = min(len(frame_paths), n_frames_eff)

    header = (
        "<div class='attn-block layer-header'>"
        f"<h3>Inspect Variant 3 attention</h3>"
        f"<div><b>Query:</b> {html.escape(query)}</div>"
        f"<div><b>Top-1 shot:</b> {html.escape(shot_id)} "
        f"(<b>{n_frames_eff}</b> frames, 4×4 spatial regions = 16 per frame)</div>"
        "<p style='margin-top:8px; color:#444; font-size:13px'>"
        "<b>local_ca</b>: segment tokens → (frame × region) under an 8 s temporal window. "
        "Bright cells = the segment's narration looks at that region of that frame.<br>"
        "<b>global_ca</b>: segment tokens → 16 region-pooled tokens (mean over frames). "
        "Bright cells = which 4×4 spatial slot dominates the shot embedding overall.</p>"
        "</div>"
    )

    sections = [header]

    def _render_block(block_attn: dict, label: str) -> str:
        if "global" in block_attn:
            g = (viz.reduce_global_to_region(block_attn["global"])
                 if isinstance(block_attn["global"], torch.Tensor)
                 else block_attn["global"])
        else:
            g = np.zeros((viz.REGIONS,))
        g_img = viz.render_global_region_grid(g)
        g_uri = viz.img_to_uri(g_img, fmt="PNG")

        if "local" in block_attn:
            local = (viz.reduce_local_to_frame_region(block_attn["local"], n_frames_padded)
                     if isinstance(block_attn["local"], torch.Tensor)
                     else block_attn["local"])
        else:
            local = np.zeros((n_frames_padded, viz.REGIONS))
        local = local[:n_frames_show]

        frame_mass = local.sum(axis=1)
        top_idx = np.argsort(-frame_mass)[:top_frames_per_block]

        cells = [_render_attn_cell(g_uri, "global · 4×4 region grid")]
        for fi in top_idx:
            if fi >= len(frame_paths):
                continue
            img = Image.open(frame_paths[fi])
            overlay = viz.overlay_region_attn(img, local[fi])
            cells.append(_render_attn_cell(
                viz.img_to_uri(overlay),
                f"local · frame {fi:02d}  (mass={frame_mass[fi]:.3f})"
            ))

        return (f"<div class='attn-block'>"
                f"<h3>{html.escape(label)}</h3>"
                f"<div class='attn-row'>{''.join(cells)}</div></div>")

    for i, ba in enumerate(blocks_attn):
        sections.append(_render_block(ba, f"Block {i + 1} of {len(blocks_attn)}"))

    if blocks_attn:
        agg = {
            "local": viz.reduce_local_aggregate_blocks(blocks_attn, n_frames_padded),
            "global": viz.reduce_global_aggregate_blocks(blocks_attn),
        }
        sections.append(_render_block(agg, "Aggregated (mean over all blocks)"))

    return "".join(sections)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_ui():
    with gr.Blocks(css=CSS) as demo:
        gr.Markdown("# Variant 3 Demo  ·  NoFilter + VideoRoPE 2D + Local-Global CA",
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
                    "Visualize what the model attends to in the **top-1 retrieved shot**. "
                    "Local CA = which (frame, region) per segment, under an 8 s window. "
                    "Global CA = which of the 16 spatial regions matters overall."
                )
                with gr.Row():
                    with gr.Column(scale=2):
                        iquery = gr.Textbox(label="Query",
                                            placeholder="Type query or leave empty for random")
                    with gr.Column(scale=1):
                        n_frames = gr.Slider(2, 12, value=6, step=1,
                                             label="Top frames to show per block")
                inspect_btn = gr.Button("Inspect", variant="primary")
                inspect_html = gr.HTML(elem_classes=["card"])
                inspect_btn.click(_inspect, inputs=[iquery, n_frames], outputs=[inspect_html])
    return demo


def main():
    print(f"[demo] BASE_DIR    = {BASE_DIR}")
    print(f"[demo] CHECKPOINT  = {CHECKPOINT}")
    print(f"[demo] DEVICE      = {DEVICE}")
    _index_cache()
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
