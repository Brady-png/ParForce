"""H3-only model exports for the controlled condition-injection ablation.

The upstream VideoX-Fun package eagerly imports every supported model family.
This isolated experiment ships only the H3 dependencies, so unrelated model
imports must not prevent an H3 run from starting.
"""

from .minimax_h3_transformer3d import MiniMaxH3Transformer3DModel
from .minimax_h3_transformer3d_control import MiniMaxH3ControlTransformer3DModel
from .minimax_h3_vae import AutoencoderKLMiniMaxH3
from .minimax_h3_vae_audio import AutoencoderKLMiniMaxH3Audio

__all__ = [
    "AutoencoderKLMiniMaxH3",
    "AutoencoderKLMiniMaxH3Audio",
    "MiniMaxH3ControlTransformer3DModel",
    "MiniMaxH3Transformer3DModel",
]
