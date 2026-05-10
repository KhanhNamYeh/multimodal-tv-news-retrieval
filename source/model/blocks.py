"""Fusion blocks: 1D RoPE (full CA), VideoRoPE 2D (full CA), VideoRoPE 2D + Local-Global CA."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .attention import GatedAttention, GatedAttentionVideoRoPE
from .components import RMSNorm, SwiGLUFFN


class FusionBlock(nn.Module):
    """1D RoPE, full cross-attention. Original baseline block."""

    def __init__(self, dim: int = 1024, n_heads: int = 16, n_kv_heads: int = 4,
                 ffn_mult: float = 8.0 / 3.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.self_attn = GatedAttention(dim, n_heads, n_kv_heads)
        self.norm2 = RMSNorm(dim)
        self.ffn1 = SwiGLUFFN(dim, ffn_mult)
        self.norm3 = RMSNorm(dim)
        self.cross_attn = GatedAttention(dim, n_heads, n_kv_heads)
        self.norm4 = RMSNorm(dim)
        self.ffn2 = SwiGLUFFN(dim, ffn_mult)

    def forward(self, seg_tokens, visual_regions, seg_pos, frame_pos,
                seg_mask: Optional[torch.Tensor], visual_mask: Optional[torch.Tensor]):
        y = seg_tokens + self.self_attn(
            self.norm1(seg_tokens), self.norm1(seg_tokens),
            seg_pos, seg_pos, mask=seg_mask,
        )
        y = y + self.ffn1(self.norm2(y))
        z = y + self.cross_attn(
            self.norm3(y), visual_regions, seg_pos, frame_pos, mask=visual_mask,
        )
        z = z + self.ffn2(self.norm4(z))
        return z


class FusionBlockVideoRoPE(nn.Module):
    """VideoRoPE 2D, full cross-attention."""

    def __init__(self, dim: int = 1024, n_heads: int = 16, n_kv_heads: int = 4,
                 ffn_mult: float = 8.0 / 3.0,
                 base_temporal: float = 500_000.0, base_spatial: float = 10_000.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.self_attn = GatedAttentionVideoRoPE(dim, n_heads, n_kv_heads, base_temporal, base_spatial)
        self.norm2 = RMSNorm(dim)
        self.ffn1 = SwiGLUFFN(dim, ffn_mult)
        self.norm3 = RMSNorm(dim)
        self.cross_attn = GatedAttentionVideoRoPE(dim, n_heads, n_kv_heads, base_temporal, base_spatial)
        self.norm4 = RMSNorm(dim)
        self.ffn2 = SwiGLUFFN(dim, ffn_mult)

    def forward(self, seg_tokens, visual_regions,
                seg_t, seg_r, seg_c, vis_t, vis_r, vis_c,
                seg_mask, visual_mask):
        y = seg_tokens + self.self_attn(
            self.norm1(seg_tokens), self.norm1(seg_tokens),
            seg_t, seg_r, seg_c, seg_t, seg_r, seg_c, mask=seg_mask,
        )
        y = y + self.ffn1(self.norm2(y))
        z = y + self.cross_attn(
            self.norm3(y), visual_regions,
            seg_t, seg_r, seg_c, vis_t, vis_r, vis_c, mask=visual_mask,
        )
        z = z + self.ffn2(self.norm4(z))
        return z


class FusionBlockLocalGlobal(nn.Module):
    """VideoRoPE 2D, two-stream cross-attention: Local (window mask) + Global (region-pooled)."""

    def __init__(self, dim: int = 1024, n_heads: int = 16, n_kv_heads: int = 4,
                 ffn_mult: float = 8.0 / 3.0,
                 base_temporal: float = 500_000.0, base_spatial: float = 10_000.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.self_attn = GatedAttentionVideoRoPE(dim, n_heads, n_kv_heads, base_temporal, base_spatial)
        self.norm2 = RMSNorm(dim)
        self.ffn1 = SwiGLUFFN(dim, ffn_mult)
        self.norm3 = RMSNorm(dim)
        self.local_ca = GatedAttentionVideoRoPE(dim, n_heads, n_kv_heads, base_temporal, base_spatial)
        self.global_ca = GatedAttentionVideoRoPE(dim, n_heads, n_kv_heads, base_temporal, base_spatial)
        self.norm4 = RMSNorm(dim)
        self.ffn2 = SwiGLUFFN(dim, ffn_mult)

    def forward(self, seg_tokens, visual_regions, global_tokens,
                seg_t, seg_r, seg_c, vis_t, vis_r, vis_c,
                glob_t, glob_r, glob_c,
                seg_mask, local_mask, global_mask):
        y = seg_tokens + self.self_attn(
            self.norm1(seg_tokens), self.norm1(seg_tokens),
            seg_t, seg_r, seg_c, seg_t, seg_r, seg_c, mask=seg_mask,
        )
        y = y + self.ffn1(self.norm2(y))

        y_norm = self.norm3(y)
        local_out = self.local_ca(
            y_norm, visual_regions,
            seg_t, seg_r, seg_c, vis_t, vis_r, vis_c, mask=local_mask,
        )
        zero_t_q = torch.zeros_like(seg_t)
        global_out = self.global_ca(
            y_norm, global_tokens,
            zero_t_q, seg_r, seg_c, glob_t, glob_r, glob_c, mask=global_mask,
        )
        z = y + local_out + global_out
        z = z + self.ffn2(self.norm4(z))
        return z
