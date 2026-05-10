# Cấu trúc dự án D:/aic

## Quy ước đặt tên

```
K{channel}_{video}_{shot}
│           │      └── shot (đoạn trong video): 000, 001, 002, ...
│           └── video: V001, V002, ...
└── kênh: K01..K20, L21, L22  (18 kênh truyền hình)
```

Ví dụ: `K02_V007_003` = kênh 2, video 7, shot 3.

---

## Thư mục gốc

```
D:/aic/
├── video/                  ← video gốc .mp4 (chưa xử lý)            [gitignored]
├── frame_extracted/        ← ảnh keyframe đã extract                 [gitignored]
├── time_extract/           ← metadata frames (timestamp+ảnh+caption) [gitignored]
├── transcripts/            ← ASR transcript theo shot                [gitignored]
├── concat/                 ← transcript+caption gộp (pipeline cũ)    [gitignored]
├── dataset/                ← thư mục phụ cũ (keyframes thô)          [gitignored]
├── narvid/                 ← clone NarVid (baseline)                 [gitignored]
├── benchmark/              ← run logs benchmark                      [gitignored]
├── data/                   ← query dataset: train/val/test splits
│   ├── manifest.json
│   ├── train/  val/  test/  test_dedicated/
│   └── features/           ← precomputed .npz/.pt features           [gitignored]
├── source/                 ← toàn bộ code dự án (xem chi tiết bên dưới)
├── best_model.pth          ← checkpoint cũ (không dùng cho demo)     [gitignored]
├── .env                    ← HF_TOKEN (KHÔNG commit)                 [gitignored]
├── .env.example            ← template env vars (commit lên)
├── .gitignore
└── structure.md            ← file này
```

---

## `source/` — code dự án

```
source/
├── model/                  ← Modular model package
│   ├── __init__.py         ← re-export public API
│   ├── components.py       ← RMSNorm, SwiGLUFFN, RotaryPositionalEmbedding (1D), VideoRoPE2D
│   ├── attention.py        ← GatedAttention (1D), GatedAttentionVideoRoPE (2D, có capture hook)
│   ├── blocks.py           ← FusionBlock | FusionBlockVideoRoPE | FusionBlockLocalGlobal
│   ├── fusion_encoder.py   ← FusionEncoder | …NoFilterVideoRoPE | …NoFilterVideoRoPELG (Variant 3 = MAIN)
│   │                          + FusionConfig dataclass + build_fusion_encoder() factory
│   └── model.py            ← shim re-export (tương thích notebook cũ)
│
├── demo/                   ← Web demo độc lập (CHỈ phần này được upload)
│   ├── __init__.py
│   ├── app.py              ← Gradio UI (Retrieve + Inspect attention tabs)
│   ├── config.py           ← BASE_DIR + MODEL_CFG (đổi cấu hình ablation tại đây)
│   ├── model_loader.py     ← load Variant 3 + BGE-M3
│   ├── retrieval.py        ← build/load FAISS index, encode docs
│   ├── visualize.py        ← capture + render attention overlays
│   ├── faiss_v3.index      ← FAISS index (sinh khi chạy lần đầu)
│   ├── faiss_v3_meta.json  ← shot_id ↔ vector mapping
│   ├── requirements.txt
│   └── README.md
│
├── train/                  ← Training pipeline & dataset
│   ├── dataset.py          ← VietnameseNewsDataset, collate_fn, get_dataloader
│   ├── train.py            ← Training loop cũ
│   └── dataloader_vietnamese_narvid.py
│
├── eval/                   ← Evaluation scripts
│   ├── benchmark.py
│   ├── test.py / test1.py / test_llama.py
│
├── preprocess/             ← Tiền xử lý dữ liệu
│   ├── keyframe_extractor.py
│   ├── precompute_siglip2.py
│   ├── precompute_bgem3.py
│   ├── precompute_clip_patch16.py
│   ├── concat_merge.py
│   ├── generate_train.py
│   ├── predata.py
│   ├── split_video_test.py
│   ├── spllit_train.py
│   └── check_missing_shots.py
│
├── api/                    ← API wrappers (Gemma/InternVL/Qwen3 caption + retrieval data gen)
│   ├── api.py              ← HF_TOKEN từ env (.env, KHÔNG hardcode)
│   ├── api_frame_gemma.py
│   └── api_frame_internvl.py
│
├── notebooks/              ← Jupyter notebooks                       [gitignored]
│   ├── ablation_study_1.ipynb     ← MAIN: Variant 3 (= demo's model)
│   ├── ablation_study.ipynb
│   ├── baseline.ipynb
│   ├── benchmark.ipynb
│   ├── continue_train_videorope.ipynb
│   ├── narvid_benchmark.ipynb
│   ├── train_eval_main.ipynb
│   ├── train_narvid_vn.ipynb
│   └── test.ipynb
│
└── ablation_output/        ← Checkpoint .pth (gitignored, chỉ giữ Variant 3)
    ├── .gitkeep                                                       [tracked]
    ├── ablation1_nofilter_videorope_lg_L2.pth                         [tracked: MAIN ckpt]
    └── (các .pth khác)                                                [gitignored]
```

---

## Model architecture (Variant 3 = main)

```
FusionEncoderNoFilterVideoRoPELG    (source/model/fusion_encoder.py)
├── visual_proj: Linear(1152 → 1024)
├── blocks: ModuleList of FusionBlockLocalGlobal × n_layers (=2)
│   └── Mỗi block:
│       ├── self_attn   : GatedAttentionVideoRoPE (GQA + sigmoid gate)
│       ├── ffn1        : SwiGLU
│       ├── local_ca    : GatedAttentionVideoRoPE  (window mask, |Δt| ≤ 8s)
│       ├── global_ca   : GatedAttentionVideoRoPE  (region-pooled tokens)
│       └── ffn2        : SwiGLU
├── final_norm: RMSNorm
└── temperature: learnable (clamp 0.01..1.0)
```

**Toggle ablation qua `FusionConfig`:**

| Tham số       | Variant 3 (main)  | Khác                               |
|---------------|-------------------|------------------------------------|
| `rope`        | `"videorope2d"`   | `"rope1d"` (legacy)                |
| `cross_attn`  | `"local_global"`  | `"full"`                           |
| `use_filter`  | `False`           | `True` (chỉ với rope1d + full)     |
| `n_layers`    | `2`               | `4` (Variant 2)                    |

```python
from model import build_fusion_encoder, FusionConfig
m = build_fusion_encoder(FusionConfig())                    # Variant 3 (mặc định)
m = build_fusion_encoder(FusionConfig(cross_attn="full"))   # Variant 1/2
m = build_fusion_encoder(FusionConfig(rope="rope1d", cross_attn="full", use_filter=True))  # baseline
```

---

## Forward signature (Variant 3)

```python
e_plus, e_narr, temperature = model(
    seg_tokens,        # [B, M, T, 1024]    BGE-M3 token reprs
    seg_pooled,        # [B, M, 1024]       BGE-M3 mean-pooled per segment
    visual_features,   # [B, N, 16, 1152]   SigLIP-2 features (4×4 regions)
    seg_timestamps,    # [B, M]
    frame_timestamps,  # [B, N]
    seg_mask,          # [B, M]
    visual_mask,       # [B, N]
    query_emb=None,    # not used in Variant 3 (NoFilter)
    token_mask=None,   # [B, M, T]
)
```

- `e_plus`  : shot embedding (L2-normalised, [B, 1024])
- `e_narr`  : narration snapshot trước fusion ([B, 1024])
- `temperature` : log-scale temperature

---

## Demo

```
source/demo/app.py
   ↓ uses
source/demo/{model_loader, retrieval, config, visualize}.py
   ↓ uses
source/model/  ← FusionConfig + build_fusion_encoder
   ↓ + checkpoint
source/ablation_output/ablation1_nofilter_videorope_lg_L2.pth
   ↓ + index
source/demo/faiss_v3.index  (auto-built lần đầu)
```

Chạy::

    cd source/demo
    pip install -r requirements.txt
    python app.py            # mở http://localhost:7860

**Tabs:**
- **Retrieve**: query → top-K shots với frame thumbnails + transcript
- **Inspect attention**: hiển thị `local_ca` overlay 4×4 trên top-K frames + `global_ca` heatmap 4×4

Chuyển sang ablation khác: sửa `MODEL_CFG` trong `source/demo/config.py` và đặt `AIC_CHECKPOINT` trỏ tới `.pth` tương ứng.

---

## Dữ liệu (cho training, không upload)

### `data/`

```
data/
├── manifest.json           ← regular vs dedicated_test split
├── train/  K02_V007.json   ← 8 query/shot, 4 hard negatives
├── val/    K02_V007.json   ← 1 query/shot
├── test/   K02_V007.json   ← 1 query/shot
└── features/               [gitignored]
    ├── weights/{siglip2, bgem3}/
    ├── siglip2/{video_id}/{video_id}_{shot_id}.npz   ← [N_frames, 16, 1152]
    └── bgem3/{video_id}/
        ├── {shot_id}_tokens.npz   ← list [Ti, 1024]
        └── {shot_id}_pooled.npz   ← [M_segs, 1024]
```

### `dataset/` *(gitignored)*

```
dataset/
├── frame_extracted/{video_id}/{video_id}_{shot_id}/*.jpg
├── time_extract/{video_id}/{video_id}_{shot_id}.json   ← frame timestamps + caption
├── transcripts/{video_id}/{shot_id}.json                ← ASR text + timestamps
└── keyframes/                                            ← legacy
```

---

## Quan hệ pipeline

```
video/ ──[keyframe_extractor]──→ frame_extracted/
                                    │
                          [api_frame_gemma]
                                    │
                              time_extract/        transcripts/
                                    │                  │
                       [precompute_siglip2]   [precompute_bgem3]
                                    │                  │
                          data/features/siglip2/   data/features/bgem3/
                                    └────────┬─────────┘
                                             │
                                source/train/dataset.py
                                             │
                                    data/train|val|test/
                                             │
                                source/model/  (Variant 3)
                                             │
                                source/train/train.py  hoặc  notebooks/ablation_study_1.ipynb
                                             │
                            source/ablation_output/ablation1_nofilter_videorope_lg_L2.pth
                                             │
                                    source/demo/app.py
```

---

## Thống kê tổng quan

| Mục                  | Số lượng                |
|----------------------|-------------------------|
| Kênh truyền hình     | 18 (K01–K20, L21–L22)   |
| Videos               | ~480                    |
| Shots                | ~9,479                  |
| Shots/split (train/val/test) | 292 / 292 / 292 |
| Queries (train)      | 2,336 (8/shot)          |
| Queries (val/test)   | 292 (1/shot)            |
| Avg frames/shot      | ~18                     |
| Avg transcript segs/shot | ~6                  |
| Variant 3 params     | ~50.5M                  |

---

## Phần được commit lên git (theo `.gitignore`)

✅ Tracked:
- `source/model/`, `source/demo/`, `source/train/`, `source/eval/`, `source/preprocess/`, `source/api/`
- `data/{train,val,test}/*.json`, `data/manifest.json`
- `source/ablation_output/ablation1_nofilter_videorope_lg_L2.pth` (chỉ checkpoint Variant 3)
- `structure.md`, `.gitignore`, `.env.example`

❌ Ignored:
- Toàn bộ raw media (`video/`, `frame_extracted/`, `time_extract/`, `transcripts/`, `concat/`, `dataset/`, `narvid/`, `benchmark/`)
- Precomputed features (`data/features/`)
- Notebooks training (`source/notebooks/*.ipynb`)
- Ablation outputs khác Variant 3
- `best_model.pth` ở root, `*.webm`, run logs
- **`.env`** (chứa secrets — HF_TOKEN, v.v.)
