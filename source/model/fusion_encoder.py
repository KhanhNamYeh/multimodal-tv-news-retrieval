"""Fusion encoder variants and a unified factory.

The main model is **Variant 3** (NoFilter + VideoRoPE 2D + Local-Global CA, L=2),
exposed as ``FusionEncoderNoFilterVideoRoPELG``.

Use ``build_fusion_encoder(cfg)`` to construct a model from a config dict;
toggling individual components (filter, RoPE family, cross-attention type)
makes ablation studies easy without touching this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    FusionBlock,
    FusionBlockLocalGlobal,
    FusionBlockVideoRoPE,
    FusionBlockVideoRoPEWideCA,
)
from .components import RMSNorm


# ---------------------------------------------------------------------------
# 1) Original baseline: 1D RoPE, full CA, adaptive filter.
# ---------------------------------------------------------------------------

class FusionEncoder(nn.Module):
    """Original encoder: 1D RoPE + full cross-attention + adaptive filter pooling."""

    def __init__(self, dim: int = 1024, vis_dim: int = 1152, n_layers: int = 2,
                 n_heads: int = 16, n_kv_heads: int = 4, delta: float = 100.0):
        super().__init__()
        self.dim = dim
        self.delta = delta

        self.visual_proj = nn.Linear(vis_dim, dim, bias=False)
        self.blocks = nn.ModuleList([
            FusionBlock(dim=dim, n_heads=n_heads, n_kv_heads=n_kv_heads)
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(dim)
        self.filter_threshold = nn.Parameter(torch.tensor(0.0))
        self.temperature = nn.Parameter(torch.log(torch.tensor(1.0 / 0.07)))

    @staticmethod
    def _masked_mean(x, mask, dim):
        m = mask.unsqueeze(-1).to(x.dtype)
        num = (x * m).sum(dim=dim)
        den = m.sum(dim=dim).clamp_min(1e-6)
        return num / den

    def _build_positions(self, seg_timestamps, frame_timestamps, t_tokens, n_regions):
        intra = torch.arange(t_tokens, device=seg_timestamps.device, dtype=seg_timestamps.dtype)
        seg_pos = seg_timestamps.unsqueeze(-1) * self.delta + intra.unsqueeze(0).unsqueeze(0)
        spatial = torch.arange(n_regions, device=frame_timestamps.device, dtype=frame_timestamps.dtype)
        frame_pos = frame_timestamps.unsqueeze(-1) * self.delta + spatial.unsqueeze(0).unsqueeze(0)
        return seg_pos, frame_pos

    @staticmethod
    def _ensure_keep_mask(keep, valid, scores):
        out = keep.clone()
        no_keep = out.sum(dim=1) == 0
        if no_keep.any():
            idx = torch.argmax(scores.masked_fill(~valid, -1e9), dim=1)
            out[no_keep, :] = False
            out[no_keep, idx[no_keep]] = True
        return out

    def forward(self, seg_tokens, seg_pooled, visual_features,
                seg_timestamps, frame_timestamps,
                seg_mask, visual_mask,
                query_emb: Optional[torch.Tensor] = None,
                token_mask: Optional[torch.Tensor] = None):
        b, m, t, d = seg_tokens.shape
        _, n, r, _ = visual_features.shape

        e_narr = self._masked_mean(seg_pooled, seg_mask, dim=1)
        e_narr = F.normalize(e_narr, p=2, dim=-1)

        vis = self.visual_proj(visual_features)
        seg_pos, frame_pos = self._build_positions(seg_timestamps, frame_timestamps, t, r)

        seg_flat = seg_tokens.reshape(b, m * t, d)
        vis_flat = vis.reshape(b, n * r, d)
        seg_pos_flat = seg_pos.reshape(b, m * t)
        frame_pos_flat = frame_pos.reshape(b, n * r)

        if token_mask is None:
            seg_token_mask = seg_mask.unsqueeze(-1).expand(b, m, t).reshape(b, m * t)
        else:
            seg_token_mask = token_mask.reshape(b, m * t)
        visual_region_mask = visual_mask.unsqueeze(-1).expand(b, n, r).reshape(b, n * r)

        for block in self.blocks:
            seg_flat = block(
                seg_tokens=seg_flat, visual_regions=vis_flat,
                seg_pos=seg_pos_flat, frame_pos=frame_pos_flat,
                seg_mask=seg_token_mask, visual_mask=visual_region_mask,
            )

        seg_flat = self.final_norm(seg_flat)
        seg_tokens_out = seg_flat.reshape(b, m, t, d)
        seg_token_mask_4d = seg_token_mask.reshape(b, m, t)
        seg_repr = self._masked_mean(seg_tokens_out, seg_token_mask_4d, dim=2)

        valid_seg = seg_mask > 0
        if query_emb is not None:
            q = F.normalize(query_emb, p=2, dim=-1)
            s = F.normalize(seg_repr, p=2, dim=-1)
            scores = torch.sum(s * q.unsqueeze(1), dim=-1)
            thr = torch.sigmoid(self.filter_threshold)
            keep = valid_seg & (scores > thr)
            keep = self._ensure_keep_mask(keep, valid_seg, scores)
            scores_masked = scores.masked_fill(~keep, float("-inf"))
            weights = torch.softmax(scores_masked, dim=1)
        else:
            weights = valid_seg.float()
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)

        e_plus = torch.sum(seg_repr * weights.unsqueeze(-1), dim=1)
        e_plus = F.normalize(e_plus, p=2, dim=-1)
        return e_plus, e_narr, self.temperature


# ---------------------------------------------------------------------------
# 2) NoFilter + 1D RoPE, full CA (Ablation: isolate rope impact).
# ---------------------------------------------------------------------------

class FusionEncoderNoFilter1DRoPE(nn.Module):
    """NoFilter + 1D RoPE + full cross-attention (for ablation: rope-only comparison)."""

    def __init__(self, dim: int = 1024, vis_dim: int = 1152, n_layers: int = 2,
                 n_heads: int = 16, n_kv_heads: int = 4, delta: float = 100.0):
        super().__init__()
        self.dim = dim
        self.delta = delta
        self.visual_proj = nn.Linear(vis_dim, dim, bias=False)
        self.blocks = nn.ModuleList([
            FusionBlock(dim, n_heads, n_kv_heads, 8.0 / 3.0)
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(dim)
        self.temperature = nn.Parameter(torch.log(torch.tensor(1.0 / 0.07)))

    @staticmethod
    def _masked_mean(x, mask, dim):
        m = mask.unsqueeze(-1).to(x.dtype)
        num = (x * m).sum(dim=dim)
        den = m.sum(dim=dim).clamp_min(1e-6)
        return num / den

    def _build_positions(self, seg_ts, frame_ts, t_tokens, n_regions):
        intra = torch.arange(t_tokens, device=seg_ts.device, dtype=seg_ts.dtype)
        seg_pos = seg_ts.unsqueeze(-1) * self.delta + intra.unsqueeze(0).unsqueeze(0)
        spatial = torch.arange(n_regions, device=frame_ts.device, dtype=frame_ts.dtype)
        frame_pos = frame_ts.unsqueeze(-1) * self.delta + spatial.unsqueeze(0).unsqueeze(0)
        return seg_pos, frame_pos

    def forward(self, seg_tokens, seg_pooled, visual_features,
                seg_timestamps, frame_timestamps,
                seg_mask, visual_mask,
                query_emb: Optional[torch.Tensor] = None,
                token_mask: Optional[torch.Tensor] = None):
        b, m, t, d = seg_tokens.shape
        _, n, r, _ = visual_features.shape

        e_narr = self._masked_mean(seg_pooled, seg_mask, dim=1)
        e_narr = F.normalize(e_narr, p=2, dim=-1)

        vis = self.visual_proj(visual_features)
        seg_pos, frame_pos = self._build_positions(seg_timestamps, frame_timestamps, t, r)

        seg_flat = seg_tokens.reshape(b, m * t, d)
        vis_flat = vis.reshape(b, n * r, d)
        seg_pos_flat = seg_pos.reshape(b, m * t)
        frame_pos_flat = frame_pos.reshape(b, n * r)

        if token_mask is None:
            seg_token_mask = seg_mask.unsqueeze(-1).expand(b, m, t).reshape(b, m * t)
        else:
            seg_token_mask = token_mask.reshape(b, m * t)
        visual_region_mask = visual_mask.unsqueeze(-1).expand(b, n, r).reshape(b, n * r)

        for block in self.blocks:
            seg_flat = block(seg_flat, vis_flat, seg_pos_flat, frame_pos_flat,
                           seg_token_mask, visual_region_mask)

        seg_flat = self.final_norm(seg_flat)
        seg_tokens_out = seg_flat.reshape(b, m, t, d)
        seg_token_mask_4d = seg_token_mask.reshape(b, m, t)
        seg_repr = self._masked_mean(seg_tokens_out, seg_token_mask_4d, dim=2)

        valid = seg_mask > 0
        weights = valid.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        e_plus = torch.sum(seg_repr * weights.unsqueeze(-1), dim=1)
        e_plus = F.normalize(e_plus, p=2, dim=-1)
        return e_plus, e_narr, self.temperature


# ---------------------------------------------------------------------------
# 3) NoFilter + VideoRoPE 2D, full CA (Variant 1/2 of ablation_study_1).
# ---------------------------------------------------------------------------

class FusionEncoderNoFilterVideoRoPE(nn.Module):
    """NoFilter + VideoRoPE 2D + full cross-attention."""

    def __init__(self, dim: int = 1024, vis_dim: int = 1152, n_layers: int = 2,
                 n_heads: int = 16, n_kv_heads: int = 4, delta: float = 100.0,
                 base_temporal: float = 500_000.0, base_spatial: float = 10_000.0,
                 use_gate: bool = True):
        super().__init__()
        self.dim = dim
        self.delta = delta
        self.visual_proj = nn.Linear(vis_dim, dim, bias=False)
        self.blocks = nn.ModuleList([
            FusionBlockVideoRoPE(dim, n_heads, n_kv_heads, 8.0 / 3.0,
                                 base_temporal, base_spatial, use_gate=use_gate)
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(dim)
        self.temperature = nn.Parameter(torch.log(torch.tensor(1.0 / 0.07)))

    @staticmethod
    def _masked_mean(x, mask, dim):
        m = mask.unsqueeze(-1).to(x.dtype)
        num = (x * m).sum(dim=dim)
        den = m.sum(dim=dim).clamp_min(1e-6)
        return num / den

    def _build_positions(self, seg_ts, frame_ts, t_tokens, n_regions):
        b = seg_ts.shape[0]
        intra = torch.arange(t_tokens, device=seg_ts.device, dtype=seg_ts.dtype)
        seg_t = seg_ts.unsqueeze(-1) * self.delta + intra.unsqueeze(0).unsqueeze(0)
        seg_r = torch.zeros_like(seg_t)
        seg_c = torch.zeros_like(seg_t)
        region_idx = torch.arange(n_regions, device=frame_ts.device, dtype=frame_ts.dtype)
        row = region_idx // 4
        col = region_idx % 4
        frame_t = frame_ts.unsqueeze(-1) * self.delta + torch.zeros_like(region_idx).unsqueeze(0).unsqueeze(0)
        frame_r = row.unsqueeze(0).unsqueeze(0).expand(b, frame_ts.shape[1], n_regions).to(frame_ts.dtype)
        frame_c = col.unsqueeze(0).unsqueeze(0).expand(b, frame_ts.shape[1], n_regions).to(frame_ts.dtype)
        return seg_t, seg_r, seg_c, frame_t, frame_r, frame_c

    def forward(self, seg_tokens, seg_pooled, visual_features,
                seg_timestamps, frame_timestamps,
                seg_mask, visual_mask,
                query_emb: Optional[torch.Tensor] = None,
                token_mask: Optional[torch.Tensor] = None):
        b, m, t, d = seg_tokens.shape
        _, n, r, _ = visual_features.shape

        e_narr = self._masked_mean(seg_pooled, seg_mask, dim=1)
        e_narr = F.normalize(e_narr, p=2, dim=-1)

        vis = self.visual_proj(visual_features)
        seg_t, seg_r, seg_c, vis_t, vis_r, vis_c = self._build_positions(
            seg_timestamps, frame_timestamps, t_tokens=t, n_regions=r,
        )

        seg_flat = seg_tokens.reshape(b, m * t, d)
        vis_flat = vis.reshape(b, n * r, d)
        seg_t_flat = seg_t.reshape(b, m * t)
        seg_r_flat = seg_r.reshape(b, m * t)
        seg_c_flat = seg_c.reshape(b, m * t)
        vis_t_flat = vis_t.reshape(b, n * r)
        vis_r_flat = vis_r.reshape(b, n * r)
        vis_c_flat = vis_c.reshape(b, n * r)

        if token_mask is None:
            seg_token_mask = seg_mask.unsqueeze(-1).expand(b, m, t).reshape(b, m * t)
        else:
            seg_token_mask = token_mask.reshape(b, m * t)
        visual_region_mask = visual_mask.unsqueeze(-1).expand(b, n, r).reshape(b, n * r)

        for block in self.blocks:
            seg_flat = block(
                seg_tokens=seg_flat, visual_regions=vis_flat,
                seg_t=seg_t_flat, seg_r=seg_r_flat, seg_c=seg_c_flat,
                vis_t=vis_t_flat, vis_r=vis_r_flat, vis_c=vis_c_flat,
                seg_mask=seg_token_mask, visual_mask=visual_region_mask,
            )

        seg_flat = self.final_norm(seg_flat)
        seg_tokens_out = seg_flat.reshape(b, m, t, d)
        seg_token_mask_4d = seg_token_mask.reshape(b, m, t)
        seg_repr = self._masked_mean(seg_tokens_out, seg_token_mask_4d, dim=2)

        valid = seg_mask > 0
        weights = valid.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        e_plus = torch.sum(seg_repr * weights.unsqueeze(-1), dim=1)
        e_plus = F.normalize(e_plus, p=2, dim=-1)
        return e_plus, e_narr, self.temperature


# ---------------------------------------------------------------------------
# 2b) No-LG ablation: single plain CA but widened to match LG's param count.
# ---------------------------------------------------------------------------

class FusionEncoderNoFilterVideoRoPEWideCA(FusionEncoderNoFilterVideoRoPE):
    """NoFilter + VideoRoPE 2D + 1 plain full CA with ``ca_width_mult x`` width.

    Param-matched No-LG baseline: keeps L=2 and the plain single-CA structure
    (no window mask, no zero-temporal trick), but widens the cross-attention's
    internal dim so the attention parameter count approximates the Full LG.
    """

    def __init__(self, dim: int = 1024, vis_dim: int = 1152, n_layers: int = 2,
                 n_heads: int = 16, n_kv_heads: int = 4, delta: float = 100.0,
                 base_temporal: float = 500_000.0, base_spatial: float = 10_000.0,
                 use_gate: bool = True, ca_width_mult: int = 2):
        nn.Module.__init__(self)
        self.dim = dim
        self.delta = delta
        self.visual_proj = nn.Linear(vis_dim, dim, bias=False)
        self.blocks = nn.ModuleList([
            FusionBlockVideoRoPEWideCA(dim, n_heads, n_kv_heads, 8.0 / 3.0,
                                       base_temporal, base_spatial,
                                       use_gate=use_gate, ca_width_mult=ca_width_mult)
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(dim)
        self.temperature = nn.Parameter(torch.log(torch.tensor(1.0 / 0.07)))


# ---------------------------------------------------------------------------
# 3) Variant 3 (MAIN): NoFilter + VideoRoPE 2D + Local-Global CA.
# ---------------------------------------------------------------------------

class FusionEncoderNoFilterVideoRoPELG(FusionEncoderNoFilterVideoRoPE):
    """NoFilter + VideoRoPE 2D + Local-Global cross-attention. Project's main model."""

    def __init__(self, dim: int = 1024, vis_dim: int = 1152, n_layers: int = 2,
                 n_heads: int = 16, n_kv_heads: int = 4, delta: float = 100.0,
                 base_temporal: float = 500_000.0, base_spatial: float = 10_000.0,
                 window_seconds: float = 8.0, n_global_tokens: int = 16,
                 use_gate: bool = True):
        nn.Module.__init__(self)
        self.dim = dim
        self.delta = delta
        self.window_seconds = window_seconds
        self.n_global_tokens = n_global_tokens
        self.visual_proj = nn.Linear(vis_dim, dim, bias=False)
        self.blocks = nn.ModuleList([
            FusionBlockLocalGlobal(dim, n_heads, n_kv_heads, 8.0 / 3.0,
                                   base_temporal, base_spatial, use_gate=use_gate)
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(dim)
        self.temperature = nn.Parameter(torch.log(torch.tensor(1.0 / 0.07)))

    @staticmethod
    def _build_global_tokens(vis, visual_mask):
        m = visual_mask.unsqueeze(-1).unsqueeze(-1).to(vis.dtype)
        num = (vis * m).sum(dim=1)
        den = m.sum(dim=1).clamp_min(1e-6)
        return num / den

    def _build_local_mask(self, seg_ts, frame_ts, seg_token_mask, visual_mask, m, t, n, r):
        b = seg_ts.shape[0]
        delta_t = seg_ts.unsqueeze(2) - frame_ts.unsqueeze(1)
        within = (delta_t.abs() <= self.window_seconds)
        if visual_mask is not None:
            within = within & visual_mask.unsqueeze(1).bool()
        mask = within.repeat_interleave(t, dim=1)
        if seg_token_mask is not None:
            mask = mask & seg_token_mask.view(b, m * t, 1).bool()
        mask = mask.repeat_interleave(r, dim=2)
        return mask.to(seg_ts.dtype)

    def forward(self, seg_tokens, seg_pooled, visual_features,
                seg_timestamps, frame_timestamps,
                seg_mask, visual_mask,
                query_emb: Optional[torch.Tensor] = None,
                token_mask: Optional[torch.Tensor] = None):
        b, m, t, d = seg_tokens.shape
        _, n, r, _ = visual_features.shape

        e_narr = self._masked_mean(seg_pooled, seg_mask, dim=1)
        e_narr = F.normalize(e_narr, p=2, dim=-1)

        vis = self.visual_proj(visual_features)
        seg_t, seg_r, seg_c, vis_t, vis_r, vis_c = self._build_positions(
            seg_timestamps, frame_timestamps, t_tokens=t, n_regions=r,
        )

        global_tokens = self._build_global_tokens(vis, visual_mask)
        region_idx = torch.arange(r, device=vis.device, dtype=seg_timestamps.dtype)
        glob_t = torch.zeros(b, r, device=vis.device, dtype=seg_timestamps.dtype)
        glob_r = (region_idx // 4).unsqueeze(0).expand(b, r).to(seg_timestamps.dtype)
        glob_c = (region_idx % 4).unsqueeze(0).expand(b, r).to(seg_timestamps.dtype)

        seg_flat = seg_tokens.reshape(b, m * t, d)
        vis_flat = vis.reshape(b, n * r, d)
        seg_t_flat = seg_t.reshape(b, m * t)
        seg_r_flat = seg_r.reshape(b, m * t)
        seg_c_flat = seg_c.reshape(b, m * t)
        vis_t_flat = vis_t.reshape(b, n * r)
        vis_r_flat = vis_r.reshape(b, n * r)
        vis_c_flat = vis_c.reshape(b, n * r)

        if token_mask is None:
            seg_token_mask = seg_mask.unsqueeze(-1).expand(b, m, t).reshape(b, m * t)
        else:
            seg_token_mask = token_mask.reshape(b, m * t)

        local_mask = self._build_local_mask(
            seg_timestamps, frame_timestamps, seg_token_mask, visual_mask, m, t, n, r,
        )
        global_mask = torch.ones(b, r, device=vis.device, dtype=seg_timestamps.dtype)

        for block in self.blocks:
            seg_flat = block(
                seg_tokens=seg_flat, visual_regions=vis_flat, global_tokens=global_tokens,
                seg_t=seg_t_flat, seg_r=seg_r_flat, seg_c=seg_c_flat,
                vis_t=vis_t_flat, vis_r=vis_r_flat, vis_c=vis_c_flat,
                glob_t=glob_t, glob_r=glob_r, glob_c=glob_c,
                seg_mask=seg_token_mask, local_mask=local_mask, global_mask=global_mask,
            )

        seg_flat = self.final_norm(seg_flat)
        seg_tokens_out = seg_flat.reshape(b, m, t, d)
        seg_token_mask_4d = seg_token_mask.reshape(b, m, t)
        seg_repr = self._masked_mean(seg_tokens_out, seg_token_mask_4d, dim=2)

        valid = seg_mask > 0
        weights = valid.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        e_plus = torch.sum(seg_repr * weights.unsqueeze(-1), dim=1)
        e_plus = F.normalize(e_plus, p=2, dim=-1)
        return e_plus, e_narr, self.temperature


# ---------------------------------------------------------------------------
# Unified config + factory for ablation
# ---------------------------------------------------------------------------

@dataclass
class FusionConfig:
    """Config for ``build_fusion_encoder``.

    Toggles map to architectural choices in the ablation study:

    - ``rope``        : ``"videorope2d"`` (Variant 3, default) or ``"rope1d"`` (legacy).
    - ``cross_attn``  : ``"local_global"`` (Variant 3, default), ``"dual_full"``
                        (param-matched No-LG ablation) or ``"full"`` (single CA).
    - ``use_filter``  : ``False`` (NoFilter, Variant 3) or ``True`` (legacy adaptive filter).
                       Only meaningful with ``rope=rope1d`` + ``cross_attn=full``.
    - ``use_gate``    : ``True`` (sigmoid Per-Head Gate, default) or ``False``
                       to ablate the gate while keeping params identical.
    """
    dim: int = 1024
    vis_dim: int = 1152
    n_layers: int = 2
    n_heads: int = 16
    n_kv_heads: int = 4
    delta: float = 100.0
    rope: str = "videorope2d"            # "videorope2d" | "rope1d"
    cross_attn: str = "local_global"     # "local_global" | "dual_full" | "full"
    use_filter: bool = False             # only meaningful when rope=rope1d & cross_attn=full
    use_gate: bool = True
    base_temporal: float = 500_000.0
    base_spatial: float = 10_000.0
    lg_window_seconds: float = 8.0
    lg_n_global_tokens: int = 16

    def to_dict(self):
        return asdict(self)


def build_fusion_encoder(cfg: "FusionConfig | dict") -> nn.Module:
    """Factory: pick the appropriate encoder class from a config."""
    if isinstance(cfg, dict):
        cfg = FusionConfig(**{k: v for k, v in cfg.items() if k in FusionConfig.__dataclass_fields__})

    common = dict(
        dim=cfg.dim, vis_dim=cfg.vis_dim, n_layers=cfg.n_layers,
        n_heads=cfg.n_heads, n_kv_heads=cfg.n_kv_heads, delta=cfg.delta,
    )

    if cfg.rope == "videorope2d" and cfg.cross_attn == "local_global":
        return FusionEncoderNoFilterVideoRoPELG(
            **common,
            base_temporal=cfg.base_temporal, base_spatial=cfg.base_spatial,
            window_seconds=cfg.lg_window_seconds,
            n_global_tokens=cfg.lg_n_global_tokens,
            use_gate=cfg.use_gate,
        )
    if cfg.rope == "videorope2d" and cfg.cross_attn == "full":
        return FusionEncoderNoFilterVideoRoPE(
            **common,
            base_temporal=cfg.base_temporal, base_spatial=cfg.base_spatial,
            use_gate=cfg.use_gate,
        )
    if cfg.rope == "rope1d" and cfg.cross_attn == "full":
        return FusionEncoder(**common)

    raise ValueError(
        f"Unsupported combination rope={cfg.rope!r} cross_attn={cfg.cross_attn!r}. "
        "Supported: (videorope2d, local_global) | (videorope2d, full) | (rope1d, full)."
    )


VARIANT3_DEFAULT = FusionConfig(
    rope="videorope2d", cross_attn="local_global",
    use_filter=False, n_layers=2,
)
