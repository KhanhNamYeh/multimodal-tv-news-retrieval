"""Building blocks: RMSNorm, SwiGLU FFN, 1D RoPE, VideoRoPE 2D."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, ffn_mult: float = 8.0 / 3.0):
        super().__init__()
        ffn_dim = int(ffn_mult * dim)
        self.w_gate = nn.Linear(dim, ffn_dim, bias=False)
        self.w_up = nn.Linear(dim, ffn_dim, bias=False)
        self.w_down = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class RotaryPositionalEmbedding(nn.Module):
    """1D RoPE."""

    def __init__(self, dim: int, base: float = 500_000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE head dim must be even")
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def get_cos_sin(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        freqs = positions.unsqueeze(-1) * self.inv_freq.unsqueeze(0).unsqueeze(0)
        cos = freqs.cos().unsqueeze(1)
        sin = freqs.sin().unsqueeze(1)
        return cos, sin

    def apply(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        cos, sin = self.get_cos_sin(positions)
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        x_even_rot = x_even * cos - x_odd * sin
        x_odd_rot = x_even * sin + x_odd * cos
        out = torch.zeros_like(x)
        out[..., ::2] = x_even_rot
        out[..., 1::2] = x_odd_rot
        return out


class VideoRoPE2D(nn.Module):
    """VideoRoPE: temporal LTA + spatial axial 2D (row, col)."""

    def __init__(self, head_dim: int, base_temporal: float = 500_000.0,
                 base_spatial: float = 10_000.0):
        super().__init__()
        if head_dim % 4 != 0:
            raise ValueError("head_dim must be divisible by 4 for VideoRoPE2D")
        n_t = head_dim // 2
        n_r = head_dim // 4
        n_c = head_dim - n_t - n_r
        self.n_t, self.n_r, self.n_c = n_t, n_r, n_c

        inv_t = 1.0 / (base_temporal ** (torch.arange(0, n_t, 2).float() / n_t))
        inv_r = 1.0 / (base_spatial ** (torch.arange(0, n_r, 2).float() / n_r))
        inv_c = 1.0 / (base_spatial ** (torch.arange(0, n_c, 2).float() / n_c))
        self.register_buffer("inv_t", inv_t, persistent=False)
        self.register_buffer("inv_r", inv_r, persistent=False)
        self.register_buffer("inv_c", inv_c, persistent=False)

    @staticmethod
    def _rotate(x: torch.Tensor, pos: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
        freqs = pos.unsqueeze(-1).float() * inv_freq.unsqueeze(0).unsqueeze(0)
        cos = freqs.cos().unsqueeze(1).to(x.dtype)
        sin = freqs.sin().unsqueeze(1).to(x.dtype)
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        x_e = x_even * cos - x_odd * sin
        x_o = x_even * sin + x_odd * cos
        out = torch.zeros_like(x)
        out[..., ::2] = x_e
        out[..., 1::2] = x_o
        return out

    def apply(self, x: torch.Tensor, t_pos: torch.Tensor, r_pos: torch.Tensor,
              c_pos: torch.Tensor) -> torch.Tensor:
        nt, nr, nc = self.n_t, self.n_r, self.n_c
        x_t = x[..., :nt]
        x_r = x[..., nt:nt + nr]
        x_c = x[..., nt + nr:nt + nr + nc]
        x_t = self._rotate(x_t, t_pos, self.inv_t)
        x_r = self._rotate(x_r, r_pos, self.inv_r)
        x_c = self._rotate(x_c, c_pos, self.inv_c)
        return torch.cat([x_t, x_r, x_c], dim=-1)
