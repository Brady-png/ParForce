"""Compatibility placeholders for non-H3 sparse-attention models.

VideoX-Fun's generic ``attention_utils`` imports these symbols even when the
MiniMax-H3 path exclusively uses PyTorch SDPA.  The original snapshot omitted
the CUDA-only sparse kernel; fail explicitly if an unrelated model selects it.
"""


def _unsupported(*_args, **_kwargs):
    raise RuntimeError("CUDA sparse attention is unavailable in the H3 NPU-only ablation")


class _sparse_linear_attention:
    apply = staticmethod(_unsupported)


get_block_map = _unsupported
mean_pool = _unsupported
sparse_linear_attention_varlen = _unsupported
