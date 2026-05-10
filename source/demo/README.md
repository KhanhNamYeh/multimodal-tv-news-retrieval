# Variant 3 Demo

Gradio web app for the project's main retrieval model: **NoFilter + VideoRoPE 2D + Local-Global Cross-Attention** (Variant 3 of `ablation_study_1.ipynb`).

## Layout

```
demo/
├── app.py                 # Gradio UI (Retrieve + Inspect attention tabs)
├── config.py              # Paths + model/data configs (edit BASE_DIR if needed)
├── model_loader.py        # Build Variant 3 + load BGE-M3 query encoder
├── retrieval.py           # FAISS index build/load + document encoding
├── visualize.py           # Capture & render visual attention overlays
├── faiss_v3.index         # Pre-built FAISS index (created on first run)
├── faiss_v3_meta.json     # shot_id ↔ vector mapping
├── requirements.txt
└── README.md
```

The model code lives in [`source/model/`](../model/) — this demo only **imports** it, so any change to the model package is picked up automatically.

## Required artefacts

| Path | Purpose |
|------|---------|
| `D:/aic/source/ablation_output/ablation1_nofilter_videorope_lg_L2.pth` | Variant 3 checkpoint |
| `D:/aic/data/features/weights/bgem3/` | BGE-M3 weights (offline copy) |
| `D:/aic/data/features/siglip2/` | Precomputed visual features |
| `D:/aic/data/features/bgem3/` | Precomputed text features |
| `D:/aic/data/test/*.json` | Query split (for fallback random query) |
| `D:/aic/dataset/transcripts/`, `dataset/frame_extracted/` | For result rendering |

Override the project root with::

    $env:AIC_BASE_DIR="D:/aic"               # PowerShell
    export AIC_BASE_DIR=/path/to/aic         # Linux/Mac

Override the checkpoint with `AIC_CHECKPOINT`.

## Run

```powershell
pip install -r requirements.txt
python app.py
```

or, from the project root::

    python -m source.demo.app

The first run builds `faiss_v3.index` (a few minutes on GPU). Subsequent runs reuse it.

## Tabs

- **Retrieve** — Type a query, pick top-K, get ranked shots with frames + transcript.
- **Inspect attention** — For the top-1 retrieved shot:
  - **`global_ca`** heatmap (4×4 spatial regions) — which spatial slot dominates the shot embedding.
  - **`local_ca`** overlay on top frames — which (frame, region) each segment attends to under the 8 s window.
  - Per-block + aggregated views.

## Switching ablation variants

Edit `config.py::MODEL_CFG` (and point `CHECKPOINT` at the matching `.pth`):

```python
# Variant 1/2 (NoFilter + VideoRoPE 2D, full CA)
MODEL_CFG["cross_attn"] = "full"

# Original baseline (1D RoPE, full CA, with filter)
MODEL_CFG.update(rope="rope1d", cross_attn="full", use_filter=True)
```

The factory raises a clear error for unsupported toggle combinations.
