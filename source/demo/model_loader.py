"""Helpers to load the Variant 3 model + the BGE-M3 query encoder."""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import BASE_DIR, CHECKPOINT, MODEL_CFG, SOURCE_DIR

# Make ``source/`` importable so we can ``from model import ...``.
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from model import build_fusion_encoder, FusionConfig  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_main_model(ckpt_path: Path | str | None = None,
                    device: torch.device | None = None) -> torch.nn.Module:
    """Build the main (Variant 3) model and load weights from disk."""
    device = device or DEVICE
    ckpt_path = Path(ckpt_path) if ckpt_path is not None else CHECKPOINT
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = build_fusion_encoder(FusionConfig(**MODEL_CFG)).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


@lru_cache(maxsize=1)
def get_text_encoder():
    """Load BGE-M3 once. Returns ``(tokenizer, model)`` on the active device."""
    from transformers import AutoModel, AutoTokenizer
    bgem3_path = BASE_DIR / "data" / "features" / "weights" / "bgem3"
    tokenizer = AutoTokenizer.from_pretrained(str(bgem3_path), local_files_only=True)
    enc = AutoModel.from_pretrained(str(bgem3_path), local_files_only=True)
    enc.eval().to(DEVICE)
    for p in enc.parameters():
        p.requires_grad = False
    return tokenizer, enc


def encode_queries(texts, max_length: int = 256, batch_size: int | None = None,
                   device: torch.device | None = None) -> torch.Tensor:
    """Mean-pooled BGE-M3 sentence embeddings (L2-normalised)."""
    device = device or DEVICE
    tokenizer, enc = get_text_encoder()
    if not texts:
        return torch.empty((0, enc.config.hidden_size), device=device)
    batch_size = batch_size or len(texts)
    out = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            t = tokenizer(chunk, padding=True, truncation=True,
                          max_length=max_length, return_tensors="pt").to(device)
            if device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    h = enc(**t, return_dict=True).last_hidden_state
            else:
                h = enc(**t, return_dict=True).last_hidden_state
            m = t["attention_mask"].unsqueeze(-1).to(h.dtype)
            pooled = (h * m).sum(1) / m.sum(1).clamp_min(1e-6)
            out.append(F.normalize(pooled.float(), p=2, dim=-1))
    return torch.cat(out, dim=0)
