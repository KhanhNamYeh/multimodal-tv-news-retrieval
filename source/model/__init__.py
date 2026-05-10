"""Modular model package.

The main model is **Variant 3** = ``FusionEncoderNoFilterVideoRoPELG``
(NoFilter + VideoRoPE 2D + Local-Global cross-attention, L=2).

For ablation studies use ``build_fusion_encoder(FusionConfig(...))``.
"""
from .attention import GatedAttention, GatedAttentionVideoRoPE
from .blocks import FusionBlock, FusionBlockLocalGlobal, FusionBlockVideoRoPE
from .components import (
    RMSNorm,
    RotaryPositionalEmbedding,
    SwiGLUFFN,
    VideoRoPE2D,
)
from .fusion_encoder import (
    VARIANT3_DEFAULT,
    FusionConfig,
    FusionEncoder,
    FusionEncoderNoFilterVideoRoPE,
    FusionEncoderNoFilterVideoRoPELG,
    build_fusion_encoder,
)

__all__ = [
    "RMSNorm", "SwiGLUFFN", "RotaryPositionalEmbedding", "VideoRoPE2D",
    "GatedAttention", "GatedAttentionVideoRoPE",
    "FusionBlock", "FusionBlockVideoRoPE", "FusionBlockLocalGlobal",
    "FusionEncoder", "FusionEncoderNoFilterVideoRoPE", "FusionEncoderNoFilterVideoRoPELG",
    "FusionConfig", "build_fusion_encoder", "VARIANT3_DEFAULT",
]
