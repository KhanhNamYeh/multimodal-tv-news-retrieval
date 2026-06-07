"""Backward-compatible shim.

Historically this file held every class. The implementation has been split into
``components.py`` / ``attention.py`` / ``blocks.py`` / ``fusion_encoder.py``.
Notebooks doing ``import model`` or ``from model import X`` keep working via
this re-export.
"""
from .attention import GatedAttention, GatedAttentionVideoRoPE
from .blocks import (
    FusionBlock,
    FusionBlockLocalGlobal,
    FusionBlockVideoRoPE,
    FusionBlockVideoRoPEWideCA,
)
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
    FusionEncoderNoFilterVideoRoPEWideCA,
    FusionEncoderNoFilterVideoRoPELG,
    build_fusion_encoder,
)

__all__ = [
    "RMSNorm", "SwiGLUFFN", "RotaryPositionalEmbedding", "VideoRoPE2D",
    "GatedAttention", "GatedAttentionVideoRoPE",
    "FusionBlock", "FusionBlockVideoRoPE", "FusionBlockVideoRoPEWideCA",
    "FusionBlockLocalGlobal",
    "FusionEncoder", "FusionEncoderNoFilterVideoRoPE",
    "FusionEncoderNoFilterVideoRoPEWideCA",
    "FusionEncoderNoFilterVideoRoPELG",
    "FusionConfig", "build_fusion_encoder", "VARIANT3_DEFAULT",
]
