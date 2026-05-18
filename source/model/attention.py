"""Gated GQA attention modules: 1D RoPE and VideoRoPE 2D variants."""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from .components import RotaryPositionalEmbedding, VideoRoPE2D


def _apply_mask_and_softmax(scores: torch.Tensor, mask: Optional[torch.Tensor]):
    """Mask scores then softmax. Robust against rows where mask is fully zero."""
    all_masked = None
    if mask is not None:
        if mask.dim() == 2:
            key_mask = mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(key_mask == 0, float("-inf"))
            all_masked = (mask.sum(dim=-1) == 0)
            if all_masked.any():
                scores = scores.masked_fill(all_masked.view(-1, 1, 1, 1), 0.0)
        elif mask.dim() == 3:
            full_mask = mask.unsqueeze(1)
            scores = scores.masked_fill(full_mask == 0, float("-inf"))
            all_masked = (mask.sum(dim=-1) == 0)
            if all_masked.any():
                scores = scores.masked_fill(all_masked.unsqueeze(1).unsqueeze(-1), 0.0)
        else:
            raise ValueError("Unsupported mask shape")
    p = torch.softmax(scores, dim=-1)
    if all_masked is not None and all_masked.any():
        if all_masked.dim() == 1:
            p = p.masked_fill(all_masked.view(-1, 1, 1, 1), 0.0)
        else:
            p = p.masked_fill(all_masked.unsqueeze(1).unsqueeze(-1), 0.0)
    return torch.nan_to_num(p, nan=0.0)


class GatedAttention(nn.Module):
    """GQA + sigmoid gate, 1D RoPE."""

    def __init__(self, dim: int = 1024, n_heads: int = 16, n_kv_heads: int = 4):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")
        if n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.kv_repeat = n_heads // n_kv_heads
        self.w_q = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.w_k = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.w_v = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.w_o = nn.Linear(n_heads * self.head_dim, dim, bias=False)
        self.gate = nn.Parameter(torch.full((n_heads, self.head_dim), -1.0))
        self.rope = RotaryPositionalEmbedding(self.head_dim)

    def _shape_q(self, q):
        b, t, _ = q.shape
        return q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

    def _shape_kv(self, x):
        b, t, _ = x.shape
        return x.view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)

    def forward(self, x_q, x_kv, pos_q, pos_k, mask=None):
        q = self._shape_q(self.w_q(x_q))
        k = self._shape_kv(self.w_k(x_kv))
        v = self._shape_kv(self.w_v(x_kv))
        k = k.repeat_interleave(self.kv_repeat, dim=1)
        v = v.repeat_interleave(self.kv_repeat, dim=1)

        q = self.rope.apply(q, pos_q)
        k = self.rope.apply(k, pos_k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            if mask.dim() == 2:
                scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(1) == 0, float("-inf"))
            elif mask.dim() == 3:
                scores = scores.masked_fill(mask.unsqueeze(1) == 0, float("-inf"))
            else:
                raise ValueError("Unsupported mask shape")
        p = torch.softmax(scores, dim=-1)
        o = torch.matmul(p, v)
        g = torch.sigmoid(self.gate).unsqueeze(0).unsqueeze(2)
        o = o * g
        o = o.transpose(1, 2).contiguous().view(x_q.shape[0], x_q.shape[1], self.dim)
        return self.w_o(o)


class GatedAttentionVideoRoPE(nn.Module):
    """GQA + sigmoid gate, VideoRoPE 2D."""

    def __init__(self, dim: int = 1024, n_heads: int = 16, n_kv_heads: int = 4,
                 base_temporal: float = 500_000.0, base_spatial: float = 10_000.0):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.kv_repeat = n_heads // n_kv_heads
        self.w_q = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.w_k = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.w_v = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.w_o = nn.Linear(n_heads * self.head_dim, dim, bias=False)
        self.gate = nn.Parameter(torch.full((n_heads, self.head_dim), -1.0))
        self.rope = VideoRoPE2D(self.head_dim, base_temporal, base_spatial)

        # Visualization hooks (no overhead unless toggled).
        self._capture_attn: bool = False
        self._last_attn: torch.Tensor | None = None  # softmax weights [B, H, Tq, Tk]

    def _shape_q(self, q):
        b, t, _ = q.shape
        return q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

    def _shape_kv(self, x):
        b, t, _ = x.shape
        return x.view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)

    def forward(self, x_q, x_kv, t_q, r_q, c_q, t_k, r_k, c_k, mask=None):
        q = self._shape_q(self.w_q(x_q))
        k = self._shape_kv(self.w_k(x_kv))
        v = self._shape_kv(self.w_v(x_kv))
        k = k.repeat_interleave(self.kv_repeat, dim=1)
        v = v.repeat_interleave(self.kv_repeat, dim=1)

        q = self.rope.apply(q, t_q, r_q, c_q)
        k = self.rope.apply(k, t_k, r_k, c_k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        p = _apply_mask_and_softmax(scores, mask)
        if self._capture_attn:
            self._last_attn = p.detach()
        o = torch.matmul(p, v)
        g = torch.sigmoid(self.gate).unsqueeze(0).unsqueeze(2)
        o = o * g
        o = o.transpose(1, 2).contiguous().view(x_q.shape[0], x_q.shape[1], self.dim)
        return self.w_o(o)
