"""
precompute_bgem3.py
===================
Extract BGE-M3 (AITeamVN/Vietnamese_Embedding) features cho transcript segments.

Theo kiến trúc mô hình, mỗi segment lưu 2 loại tensor:
  1. tokens_layer21 : hidden states tại layer 21/24 → [T, 1024]
                      dùng làm Query trong cross-attention của Fusion Encoder
  2. pooled_layer24 : mean pool của layer 24 → [1024]
                      dùng làm narration snapshot ê_narr

Cấu trúc lưu (xem structure.md):
  features/bgem3/
  └── K01_V001/
      ├── 001_tokens.npz    ← arr_0..arr_N: ndarray[T_i, 1024], 1 array/segment
      └── 001_pooled.npz    ← pooled: ndarray[M_segs, 1024]

  (tên file theo transcripts/: chỉ có shot_id, không có prefix video)

Model weights được tải về lần đầu và cache tại:
  features/weights/bgem3/   (dùng lại offline cho lần sau)

Chạy:
  py precompute_bgem3.py                   # toàn bộ
  py precompute_bgem3.py --video K01_V001  # 1 video
  py precompute_bgem3.py --resume          # bỏ qua shot đã có
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# ── Đường dẫn gốc ────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent.parent.parent   # D:/aic
TRANSCRIPTS_DIR = BASE_DIR / "dataset" / "transcripts"
FEAT_DIR        = BASE_DIR / "data" / "features" / "bgem3"
WEIGHT_DIR      = BASE_DIR / "data" / "features" / "weights" / "bgem3"
HF_CACHE_DIR    = BASE_DIR / "data" / "features" / "weights" / "hf_cache"

MODEL_ID        = "AITeamVN/Vietnamese_Embedding"
TOKEN_LAYER     = 21          # layer lấy token representations (Q cho cross-attn)
POOL_LAYER      = 24          # layer lấy mean pool (narration snapshot)
MAX_LENGTH      = 512         # BGE-M3 hỗ trợ đến 8192, giới hạn theo GPU mem
BATCH_SEGS      = 16          # số segments encode mỗi lần


# ─────────────────────────────────────────────────────────────────────────────
def load_model(weight_dir: Path, device: str):
    """
    Lần đầu: tải từ HuggingFace, lưu vào weight_dir.
    Lần sau : load từ weight_dir (offline, không cần internet).
    """
    # Ép cache Hugging Face vào thư mục writable trong workspace
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(HF_CACHE_DIR)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE_DIR / "hub")
    os.environ["HF_HUB_CACHE"] = str(HF_CACHE_DIR / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE_DIR / "transformers")

    from transformers import AutoTokenizer, AutoModel

    weight_dir.mkdir(parents=True, exist_ok=True)
    config_file = weight_dir / "config.json"

    if config_file.exists():
        print(f"[BGE-M3] Load weights từ local: {weight_dir}")
        tokenizer = AutoTokenizer.from_pretrained(
            str(weight_dir),
            local_files_only=True,
        )
        model = AutoModel.from_pretrained(
            str(weight_dir),
            local_files_only=True,
        )
    else:
        print(f"[BGE-M3] Tải model từ HuggingFace: {MODEL_ID}")
        print("  (sẽ mất vài phút lần đầu, sau đó dùng local cache)")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            cache_dir=str(HF_CACHE_DIR),
        )
        model = AutoModel.from_pretrained(
            MODEL_ID,
            cache_dir=str(HF_CACHE_DIR),
        )
        # Lưu lại để dùng offline
        tokenizer.save_pretrained(str(weight_dir))
        model.save_pretrained(str(weight_dir))
        print(f"  Đã lưu weights → {weight_dir}")

    model.to(device).eval()
    return tokenizer, model


# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def encode_segments(texts: list[str],
                    tokenizer, model,
                    device: str) -> tuple[list[torch.Tensor], torch.Tensor]:
    """
    Encode batch segments.

    Trả về:
      token_reprs : List[Tensor[T_i, 1024]] — hidden states layer TOKEN_LAYER
                    mỗi tensor có T_i khác nhau (sau khi bỏ padding)
      pooled      : Tensor[B, 1024]          — mean pool layer POOL_LAYER
    """
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).to(device)

    out = model(
        **encoded,
        output_hidden_states=True,
        return_dict=True,
    )

    # ── Token representations tại layer TOKEN_LAYER ──────────────────────────
    # hidden_states: tuple len = n_layers + 1 (bao gồm embedding layer)
    # BGE-M3 (XLM-RoBERTa base) có 24 transformer layers → index 0..24
    # layer TOKEN_LAYER = 21 → index 21
    hidden_token = out.hidden_states[TOKEN_LAYER]   # [B, seq_len, D]

    token_reprs = []
    attn_mask   = encoded["attention_mask"]          # [B, seq_len]
    for b in range(hidden_token.shape[0]):
        # Chỉ giữ token thực (bỏ [PAD])
        valid_len = attn_mask[b].sum().item()
        token_reprs.append(hidden_token[b, :valid_len, :].cpu())  # [T_i, D]

    # ── Mean pool tại layer POOL_LAYER ───────────────────────────────────────
    # Giống cách BGE-M3 encode cho retrieval: weighted mean pool theo attn mask
    hidden_pool = out.hidden_states[POOL_LAYER]  # [B, seq_len, D]
    mask_expanded = attn_mask.unsqueeze(-1).float()
    pooled = (hidden_pool * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
    pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)   # [B, D]
    pooled = pooled.cpu()

    return token_reprs, pooled


# ─────────────────────────────────────────────────────────────────────────────
def process_shot(trans_json: Path, tokenizer, model,
                 device: str,
                 out_tokens: Path, out_pooled: Path) -> bool:
    """
    Đọc 1 shot transcript JSON, encode tất cả segments,
    lưu tokens_layer21 và pooled_layer24.
    """
    segments = json.loads(trans_json.read_text("utf-8"))
    if not segments:
        return False

    texts = [s["text"] for s in segments]

    all_tokens = []   # List[Tensor[T_i, D]]
    all_pooled = []   # List[Tensor[D]]

    for b_start in range(0, len(texts), BATCH_SEGS):
        batch = texts[b_start : b_start + BATCH_SEGS]
        toks, pool = encode_segments(batch, tokenizer, model, device)
        all_tokens.extend(toks)
        all_pooled.append(pool)

    # all_tokens: List[Tensor[T_i, D]] — T_i có thể khác nhau mỗi segment
    # all_pooled: Tensor[M_segs, D]
    pooled_tensor = torch.cat(all_pooled, dim=0)  # [M_segs, D]

    out_tokens.parent.mkdir(parents=True, exist_ok=True)
    # Lưu tokens: arr_0, arr_1, ... arr_N (T_i khác nhau → không stack được)
    np.savez(out_tokens, *[t.numpy() for t in all_tokens])
    # Lưu pooled: key "pooled" → ndarray[M_segs, D]
    np.savez(out_pooled, pooled=pooled_tensor.numpy())
    return True


# ─────────────────────────────────────────────────────────────────────────────
def get_all_shots(video_filter: str | None = None) -> list[tuple[Path, Path, Path]]:
    """
    Liệt kê tất cả (trans_json, out_tokens, out_pooled) cần xử lý.

    transcripts/K01_V001/001.json
    → features/bgem3/K01_V001/001_tokens.npz
    → features/bgem3/K01_V001/001_pooled.npz

    Lưu ý: tên file transcript chỉ có shot_id (không có prefix video),
    khác với time_extract. Xem structure.md.
    """
    trips = []
    for vid_dir in sorted(TRANSCRIPTS_DIR.iterdir()):
        if not vid_dir.is_dir():
            continue
        if video_filter and vid_dir.name != video_filter:
            continue
        for trans_json in sorted(vid_dir.glob("*.json")):
            shot_id    = trans_json.stem              # "001"
            feat_dir   = FEAT_DIR / vid_dir.name      # features/bgem3/K01_V001/
            out_tokens = feat_dir / f"{shot_id}_tokens.npz"
            out_pooled = feat_dir / f"{shot_id}_pooled.npz"
            trips.append((trans_json, out_tokens, out_pooled))
    return trips


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Precompute BGE-M3 features")
    parser.add_argument("--video",  type=str, default=None,
                        help="Chỉ xử lý 1 video (vd: K01_V001). Mặc định = tất cả.")
    parser.add_argument("--resume", action="store_true",
                        help="Bỏ qua shot đã có file .npz")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Device: {args.device}")
    tokenizer, model = load_model(WEIGHT_DIR, args.device)

    shots = get_all_shots(video_filter=args.video)
    print(f"Tổng shots: {len(shots)}")

    if args.resume:
        shots = [(j, t, p) for j, t, p in shots
                 if not t.exists() or not p.exists()]
        print(f"Chưa có .npz: {len(shots)}")

    ok = err = 0
    for trans_json, out_tokens, out_pooled in tqdm(shots, desc="BGE-M3"):
        try:
            process_shot(trans_json, tokenizer, model,
                         args.device, out_tokens, out_pooled)
            ok += 1
        except Exception as e:
            print(f"\n[ERR] {trans_json}: {e}", file=sys.stderr)
            err += 1

    print(f"\nHoàn thành: {ok} OK  |  {err} lỗi")
    print(f"Features lưu tại: {FEAT_DIR}")


if __name__ == "__main__":
    main()