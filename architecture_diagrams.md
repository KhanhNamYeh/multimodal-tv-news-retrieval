# SYSTEM ARCHITECTURE DIAGRAMS

**Project:** Vietnamese Broadcast News Retrieval via Temporal-Aware Multimodal Representation Learning

---

## 4.1 Overall System Architecture

### 4.1.1 Offline Pipeline (Indexing)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       OFFLINE PIPELINE (Indexing)                        │
└─────────────────────────────────────────────────────────────────────────┘

           ┌──────────────────────────────────────┐
           │   Original program video             │
           │   (15–45 minutes, VTV / HTV / THVL)  │
           └─────────────────┬────────────────────┘
                             │
                             ▼
           ┌──────────────────────────────────────┐
           │   Sub-news segmentation              │
           │   (transition + anchor shot)         │
           │   → 60–120 second news clips         │
           └─────────────────┬────────────────────┘
                             │
                             ▼
           ┌──────────────────────────────────────┐
           │      Sub-news clip (.mp4)            │
           │      Processing unit of the system   │
           └─────────────────┬────────────────────┘
                             │
   ┌─────────────────────────┴─────────────────────────┐
   │                                                   │
   ▼                                                   ▼
[ffmpeg]                                    [Sample all frames]
   │                                                   │
   ▼                                                   ▼
Audio .wav (16 kHz mono)                      [Keyframe Extractor:
   │                                           HSV+DCT+pixel similarity]
   ▼                                                   │
[Silero VAD + PhoWhisper]                              ▼
   │                                          [Cluster + select medoid]
   │                                                   │
   ▼                                                   ▼
Transcript segments                            Representative keyframes
[{start, end, text}]                          [{start, end, image}]
   │                                                   │
   ▼                                                   ▼
[Vietnamese_Embedding (frozen)]              [SigLIP-2 So400m/14-384
                                                       (frozen)]
   │                                          [Penultimate layer (35/37)
   │                                           + Spatial Pool 4×4]
   │                                                   │
   ├─► Token features (Layer 21)                       ▼
   │   [B, M, T, 1024]                          Visual features
   │                                            [B, N, 16, 1152]
   └─► Pooled features (Layer 24)                      │
       [B, M, 1024]                                    │
   │                                                   │
   └───────────────────────┬───────────────────────────┘
                           │
                           ▼
   ┌────────────────────────────────────────────────────┐
   │  Training data generation (Gemma API):             │
   │                                                    │
   │  • 8 train + 1 val + 1 test queries per shot       │
   │  • 4 entity-swapped hard negatives per shot        │
   └─────────────────────────┬──────────────────────────┘
                             │
                             ▼
   ┌────────────────────────────────────────────────────┐
   │  Two-level data partitioning:                      │
   │                                                    │
   │  Level 1: 10% videos held out as out-of-domain     │
   │  Level 2: 80/10/10 train/val/test in-domain split  │
   └─────────────────────────┬──────────────────────────┘
                             │
                             ▼
                ┌──────────────────────────┐
                │  Dataset + DataLoader    │
                └────────────┬─────────────┘
                             │
                             ▼
                ┌──────────────────────────┐
                │     FUSION ENCODER       │
                │       (TRAINABLE)        │
                │                          │
                │  • Visual Projection     │
                │    1152 → 1024           │
                │  • VideoRoPE 2D          │
                │  • Self-Attention        │
                │    (GQA + Gate)          │
                │  • Local-Global Cross-   │
                │    Attention 2-stream    │
                │  • SwiGLU FFN            │
                │  • Uniform Weighted Pool │
                │                          │
                │  ~50M parameters         │
                └────────────┬─────────────┘
                             │
                             ▼
                ┌──────────────────────────┐
                │  Loss function:          │
                │  L = L_retrieval         │
                │    + λ·L_narr            │
                │    + β·L_hard            │
                │    + γ·L_MRL             │
                └────────────┬─────────────┘
                             │
                             ▼
                ┌──────────────────────────┐
                │     Trained model        │
                └────────────┬─────────────┘
                             │
                             ▼
                ┌──────────────────────────┐
                │  Encode all shots in     │
                │  the storage corpus      │
                └────────────┬─────────────┘
                             │
                             ▼
                ┌──────────────────────────┐
                │       FAISS INDEX        │
                │  vector 1024-d (L2 norm) │
                │  Inner Product search    │
                └──────────────────────────┘
```

### 4.1.2 Online Pipeline (Query)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        ONLINE PIPELINE (Query)                           │
└─────────────────────────────────────────────────────────────────────────┘

           ┌──────────────────────────────────────┐
           │   User query                         │
           │   (natural Vietnamese text)          │
           └─────────────────┬────────────────────┘
                             │
                             ▼
           ┌──────────────────────────────────────┐
           │   Vietnamese_Embedding (Layer 24)    │
           │   Mean Pool over attention mask      │
           │   + L2 normalize                     │
           └─────────────────┬────────────────────┘
                             │
                             ▼
                  ┌────────────────────┐
                  │  Query vector q̂    │
                  │  ∈ ℝ^1024          │
                  │  (L2 normalized)   │
                  └─────────┬──────────┘
                            │
                            ▼
           ┌──────────────────────────────────────┐
           │   FAISS ANN search                   │
           │   (Inner Product = cosine)           │
           └─────────────────┬────────────────────┘
                             │
                             ▼
                  ┌────────────────────┐
                  │  Top-K shot_id     │
                  │  + similarity      │
                  └─────────┬──────────┘
                            │
                            ▼
           ┌──────────────────────────────────────┐
           │   Dual Softmax post-processing       │
           │   (reduces hubness, τ = 0.01)        │
           └─────────────────┬────────────────────┘
                             │
                             ▼
                  ┌────────────────────┐
                  │  Re-ranked Top-K   │
                  └─────────┬──────────┘
                            │
                            ▼
           ┌──────────────────────────────────────┐
           │   Display results via Gradio UI:     │
           │   • Representative keyframe thumb    │
           │   • Timestamp (start, end)           │
           │   • Related transcript segment       │
           │   • Score ranking                    │
           └──────────────────────────────────────┘
```

---

## 4.2 Detailed Fusion Encoder Architecture

```
                           FUSION ENCODER
                    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  INPUT:                                                  INPUT:
  Visual features                                         Text segments
  [B, N, 16, 1152]                                        [B, M, T, 1024]
  (from SigLIP-2 + Pool 4×4)                    (from Vietnamese_Embedding L21)
        │                                                          │
        │                                                          │      Pooled features
        │                                                          │      [B, M, 1024]
        │                                                          │      (from Layer 24)
        │                                                          │              │
        ▼                                                          │              ▼
   ┌──────────────────┐                                            │   ┌─────────────────┐
   │ Linear Proj      │                                            │   │ Snapshot:       │
   │ 1152 → 1024      │                                            │   │ ê_narr =        │
   └────────┬─────────┘                                            │   │ MeanPool over   │
            │                                                      │   │ seg_mask        │
            ▼                                                      │   │ + L2 normalize  │
   ┌──────────────────┐                                            │   └────────┬────────┘
   │ Build VideoRoPE  │                                            │            │
   │ 2D positions:    │                                            │            │
   │                  │                                            │            │
   │ • Temporal       │                                            │            │
   │   p_t = t × δ    │                                            │            │
   │   (δ = 100)      │                                            │            │
   │                  │                                            │            │
   │ • Spatial        │                                            │            │
   │   p_r = row      │                                            │            │
   │   p_c = col      │                                            │            │
   │   (4×4 grid)     │                                            │            │
   └────────┬─────────┘                                            │            │
            │                                                      │            │
            └──────────────────────────────┬───────────────────────┘            │
                                           ▼                                    │
              ┌──────────────────────────────────────────────────┐              │
              │           FUSION BLOCK × L = 2                   │              │
              │                                                  │              │
              │  ┌────────────────────────────────────────────┐  │              │
              │  │  Self-Attention (text → text)              │  │              │
              │  │  • GQA (16 heads, 4 KV heads)              │  │              │
              │  │  • VideoRoPE 2D                            │  │              │
              │  │  • Sigmoid Per-Head Gate                   │  │              │
              │  └─────────────────┬──────────────────────────┘  │              │
              │                    ▼                              │              │
              │  ┌────────────────────────────────────────────┐  │              │
              │  │  RMSNorm + Residual                        │  │              │
              │  └─────────────────┬──────────────────────────┘  │              │
              │                    ▼                              │              │
              │  ┌────────────────────────────────────────────┐  │              │
              │  │  FFN-1 (SwiGLU)                            │  │              │
              │  │  hidden = 8/3 × dim                        │  │              │
              │  └─────────────────┬──────────────────────────┘  │              │
              │                    ▼                              │              │
              │  ┌────────────────────────────────────────────┐  │              │
              │  │  LOCAL-GLOBAL CROSS-ATTENTION 2-STREAM     │  │              │
              │  │                                            │  │              │
              │  │  ┌──────────────┐    ┌──────────────────┐  │  │              │
              │  │  │ LOCAL        │    │ GLOBAL           │  │  │              │
              │  │  │ stream       │    │ stream           │  │  │              │
              │  │  │              │    │                  │  │  │              │
              │  │  │ Mask:        │    │ Compress N       │  │  │              │
              │  │  │ |t_q − t_v|  │    │ keyframes into   │  │  │              │
              │  │  │  ≤ 8 seconds │    │ 16 global region │  │  │              │
              │  │  │              │    │ tokens (mean     │  │  │              │
              │  │  │ Captures     │    │ pool over time,  │  │  │              │
              │  │  │ chyron,      │    │ keep 4×4 layout) │  │  │              │
              │  │  │ banner at    │    │                  │  │  │              │
              │  │  │ correct      │    │ Captures scene   │  │  │              │
              │  │  │ narration    │    │ type, character  │  │  │              │
              │  │  │ time         │    │ identity         │  │  │              │
              │  │  │ + own gate   │    │ + own gate       │  │  │              │
              │  │  └──────┬───────┘    └────────┬─────────┘  │  │              │
              │  │         │                     │            │  │              │
              │  │         └───────── + ─────────┘            │  │              │
              │  │            (sum two streams)               │  │              │
              │  └─────────────────┬──────────────────────────┘  │              │
              │                    ▼                              │              │
              │  ┌────────────────────────────────────────────┐  │              │
              │  │  RMSNorm + Residual                        │  │              │
              │  └─────────────────┬──────────────────────────┘  │              │
              │                    ▼                              │              │
              │  ┌────────────────────────────────────────────┐  │              │
              │  │  FFN-2 (SwiGLU)                            │  │              │
              │  └─────────────────┬──────────────────────────┘  │              │
              │                    │                              │              │
              └────────────────────┼──────────────────────────────┘              │
                                   │                                             │
                                   ▼                                             │
                       ┌────────────────────────┐                                │
                       │  RMSNorm (final)       │                                │
                       └───────────┬────────────┘                                │
                                   │                                             │
                                   ▼                                             │
                       ┌────────────────────────┐                                │
                       │  Mean Pool / segment   │                                │
                       │  → seg_repr [B, M, D]  │                                │
                       └───────────┬────────────┘                                │
                                   │                                             │
                                   ▼                                             │
                       ┌────────────────────────┐                                │
                       │  Uniform Weighted Pool │                                │
                       │  over seg_mask         │                                │
                       │  (NO AdaptFilter)      │                                │
                       └───────────┬────────────┘                                │
                                   │                                             │
                                   ▼                                             │
                       ┌────────────────────────┐                                │
                       │  L2 normalize          │                                │
                       │  → ê⁺ ∈ ℝ^1024         │                                │
                       └───────────┬────────────┘                                │
                                   │                                             │
                                   │                                             │
                            (multimodal vector)                          (text-only vector)
                                   │                                             │
                                   ▼                                             ▼
                                  ê⁺                                          ê_narr
                                   │                                             │
                                   └─────────────────────┬───────────────────────┘
                                                         │
                                                         ▼
                                             ┌─────────────────────┐
                                             │       LOSS          │
                                             │                     │
                                             │ L = L_retrieval     │
                                             │   + λ × L_narr      │
                                             │   + β × L_hard      │
                                             │   + γ × L_MRL       │
                                             │                     │
                                             │ λ=0.5, β=0.3, γ=0.1 │
                                             └─────────────────────┘
```

---

## 4.3 Local-Global Cross-Attention Detail

```
                LOCAL-GLOBAL CROSS-ATTENTION 2-STREAM
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   ┌────────────────────────────────────────┐
   │  Segment tokens (Query)                │
   │  [B, M·T, 1024]                        │
   │  Position: (t_seg, 0, 0)               │
   └─────────────────┬──────────────────────┘
                     │
                     ▼
        ┌─────────────────────────┐
        │  Split into 2 parallel  │
        │       streams           │
        └────┬───────────────┬────┘
             │               │
             ▼               ▼
   ┌──────────────────┐  ┌────────────────────────┐
   │  LOCAL STREAM    │  │  GLOBAL STREAM         │
   │                  │  │                        │
   │  K, V:           │  │  K, V:                 │
   │  Visual regions  │  │  Mean pool across N    │
   │  [B, N·16, 1024] │  │  → [B, 16, 1024]       │
   │  Position:       │  │  (keep 4×4 layout,     │
   │  (t_v, row, col) │  │   drop time dim)       │
   │                  │  │  Position:             │
   │                  │  │  (0, row, col)         │
   │                  │  │                        │
   │  Build mask:     │  │                        │
   │  |t_q−t_v| ≤ 8s  │  │  (no temporal mask     │
   │  + visual_mask   │  │   needed)              │
   │                  │  │                        │
   │  ┌────────────┐  │  │  ┌──────────────────┐  │
   │  │ Cross-Attn │  │  │  │   Cross-Attn     │  │
   │  │ with time  │  │  │  │   global         │  │
   │  │ mask       │  │  │  │                  │  │
   │  │ + RoPE 2D  │  │  │  │   + RoPE 2D      │  │
   │  └─────┬──────┘  │  │  └────────┬─────────┘  │
   │        │         │  │           │            │
   │        ▼         │  │           ▼            │
   │  ┌────────────┐  │  │  ┌──────────────────┐  │
   │  │ Sigmoid    │  │  │  │   Sigmoid Gate   │  │
   │  │ Gate       │  │  │  │   g_global       │  │
   │  │ g_local    │  │  │  │   ∈ ℝ^(H × D_h)  │  │
   │  │ ∈ ℝ^(H×Dh) │  │  │  │   init −5.0      │  │
   │  │ init −5.0  │  │  │  │                  │  │
   │  └─────┬──────┘  │  │  └────────┬─────────┘  │
   └────────┼─────────┘  └───────────┼────────────┘
            │                        │
            └────────────┬───────────┘
                         ▼
              ┌─────────────────────────────────┐
              │  Combine streams                │
              │  output = local_out + global_out│
              │                                 │
              │  (each stream's W_o (1024→1024) │
              │   already applied inside its    │
              │   own attention module — no     │
              │   shared projection here)       │
              └─────────────────────────────────┘
```

---

## 4.4 VideoRoPE 2D Detail

```
                       VideoRoPE 2D ENCODING
              ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   Head dimension d_h = 64 (per attention head)
   ┌──────────────────────────────────────────────────────┐
   │                       d_h = 64                        │
   └──────────────────────────────────────────────────────┘
                              │
                              ▼
       ┌──────────────────────┴───────────────────────────┐
       │                                                  │
       ▼                                                  ▼
  ┌─────────────┐                                  ┌──────────────┐
  │  n_t = 32   │                                  │  n_r + n_c   │
  │  TEMPORAL   │                                  │  = 16 + 16   │
  │             │                                  │  SPATIAL     │
  │  base =     │                                  │  base =      │
  │  500,000    │                                  │  10,000      │
  │             │                                  │              │
  │  (low freq, │                                  │  (high freq, │
  │   long time │                                  │   only 4 rows│
  │   range)    │                                  │   /4 cols)   │
  └──────┬──────┘                                  └──────┬───────┘
         │                                                │
         ▼                                                ▼
  ┌────────────────┐                              ┌─────────────────┐
  │ Position:      │                              │ Position:       │
  │                │                              │                 │
  │ Segment:       │                              │ Segment:        │
  │ p_t = t_i × δ  │                              │ p_r = 0         │
  │      + k       │                              │ p_c = 0         │
  │                │                              │                 │
  │ Frame:         │                              │ Frame:          │
  │ p_t = t_j × δ  │                              │ p_r = row       │
  │                │                              │ p_c = col       │
  │ (δ = 100)      │                              │ (∈ {0,1,2,3})   │
  └────────┬───────┘                              └────────┬────────┘
           │                                               │
           ▼                                               ▼
  ┌────────────────┐                              ┌─────────────────┐
  │ Rotation       │                              │ Rotation        │
  │ on 16 pairs    │                              │ on 8+8 pairs    │
  │ (x_2i, x_2i+1) │                              │ for row and col │
  │                │                              │                 │
  └────────┬───────┘                              └────────┬────────┘
           │                                               │
           └───────────────────────┬───────────────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │  Concatenate:          │
                       │  [t_part, r_part,      │
                       │   c_part]              │
                       │  → Output [d_h]        │
                       └────────────────────────┘


   RESULTING PROPERTY — TEMPORAL PROXIMITY PRIOR:
   ─────────────────────────────────────────────────

   • Segment at t = 45s and Frame at t = 45s
        → Q·K^T inner product is highest (same rotation angle)

   • Segment at t = 45s and Frame at t = 80s
        → low inner product due to large angular difference

   → Model automatically prioritizes temporally close
     pairs without explicit supervision
```

---

## 4.5 Spatial Pooling 4×4

```
                 SPATIAL POOLING 4×4 — VISUAL ENCODING
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

           ┌──────────────────────────────────┐
           │  Representative keyframe         │
           │  Image 384 × 384 × 3             │
           └─────────────────┬────────────────┘
                             │
                             ▼
           ┌──────────────────────────────────┐
           │  SigLIP-2 So400m/14-384          │
           │  Vision Transformer (frozen)     │
           │                                  │
           │  Take hidden state at            │
           │  penultimate layer (35/37)       │
           └─────────────────┬────────────────┘
                             │
                             ▼
           ┌──────────────────────────────────┐
           │  Patch tokens                    │
           │  27 × 27 = 729 patches           │
           │  Each patch: 1152 dimensions     │
           └─────────────────┬────────────────┘
                             │
                             ▼
           ┌──────────────────────────────────┐
           │  adaptive_avg_pool2d             │
           │  from (27, 27) to (4, 4)         │
           │  Each region ≈ 45 patches        │
           └─────────────────┬────────────────┘
                             │
                             ▼
           ┌──────────────────────────────────┐
           │  16 region vectors               │
           │  Shape: [16, 1152]               │
           │  (representation of one frame)   │
           └──────────────────────────────────┘


   SEMANTIC ZONING IN THE 4×4 GRID:
   ──────────────────────────────────────────

   ┌──────────┬──────────┬──────────┬──────────┐
   │  R0,C0   │  R0,C1   │  R0,C2   │  R0,C3   │   ← Row 0:
   │  (logo)  │  (breaking news banner)        │     Banner / Channel logo
   ├──────────┼──────────┼──────────┼──────────┤
   │  R1,C0   │  R1,C1   │  R1,C2   │  R1,C3   │
   │  (back-  │  (MAIN VISUAL)      │  (back-  │   ← Rows 1, 2:
   │  drop)   │  Anchor / report    │  drop)   │     Center region
   ├──────────┼──────────┼──────────┼──────────┤     (main scene)
   │  R2,C0   │  R2,C1   │  R2,C2   │  R2,C3   │
   │  (graph- │  (MAIN VISUAL)      │  (graph- │
   │  ics)    │  field reporting    │  ics)    │
   ├──────────┼──────────┼──────────┼──────────┤
   │  R3,C0   │  R3,C1   │  R3,C2   │  R3,C3   │   ← Row 3:
   │  (chyron — names, locations,              │     Chyron / Subtitle
   │   ticker news)                            │
   └──────────┴──────────┴──────────┴──────────┘
```

---

## Summary of Architecture Components

| Component | Type | Size | Role |
|---|---|---|---|
| SigLIP-2 So400m/14-384 | Vision Encoder (frozen) | ~400M params | Encode keyframes into 16 region vectors per frame |
| Vietnamese_Embedding | Text Encoder (frozen) | ~568M params | Encode transcript segments via dual-layer extraction |
| PhoWhisper-large | ASR (frozen) | ~1.5B params | Vietnamese speech-to-text with VAD preprocessing |
| Fusion Encoder | Trainable | ~50M params | Cross-modal fusion with temporal awareness |
| FAISS Index | Vector Database | — | Approximate Nearest Neighbor search at inference |

## Key Design Decisions

1. **Bi-encoder asymmetric architecture** — heavy document encoding offline, lightweight query encoding online.
2. **Spatial Pooling 4×4** instead of global mean pool — preserves spatial information of broadcast graphics.
3. **VideoRoPE 2D** — encodes both temporal axis (real timestamp) and spatial axis (4×4 grid) simultaneously.
4. **Local-Global Cross-Attention 2-stream** — captures both fine temporal alignment and global video context.
5. **Uniform Weighted Pooling** instead of AdaptFilter — empirically shown to perform better.
6. **Multi-component loss** — InfoNCE retrieval + Narration loss + Hard negative loss + Matryoshka loss.
7. **Two-level evaluation protocol** — 10% out-of-domain holdout + 80/10/10 in-domain split.

---

## Loss Function Components

| Component | Symbol | Weight | Purpose |
|---|---|---|---|
| Symmetric InfoNCE (text↔video) | L_retrieval | 1.0 | Main retrieval objective |
| Narration loss | L_narr | λ = 0.5 | Text grounding regularizer |
| Hard negative triplet | L_hard | β = 0.3 | Entity-level discrimination |
| Matryoshka loss | L_MRL | γ = 0.1 | Truncate-friendly embedding |

**Total loss:**

> L = L_retrieval + λ · L_narr + β · L_hard + γ · L_MRL
