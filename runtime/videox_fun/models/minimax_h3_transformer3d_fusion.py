"""Native channel- and token-fusion H3 condition adapters.

This module intentionally derives from the released H3 transformer rather
than the ControlNet implementation.  Consequently, the channel/token
ablations neither allocate nor execute a residual control branch.  Both
adapters start as an exact base-model pass through because their sole
condition projection is zero initialized.
"""

from contextlib import nullcontext
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import register_to_config
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version

from .minimax_h3_transformer3d import (
    MINIMAX_H3_MODALITY_NUM,
    MiniMaxH3Transformer3DModel,
    MiniMaxH3TransformerOutput,
)


class MiniMaxH3ConditionFusionTransformer3DModel(MiniMaxH3Transformer3DModel):
    """H3 with an explicit channel or appended-token PSS+SDS adapter.

    ``channel`` adds a learned zero-initialised projection of the joint
    particle/depth patch to each corresponding video embedding.  ``token``
    appends one projected PSS+SDS token per video token to the packed document;
    target video outputs are still selected only from the original video rows.
    """

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 56,
        attention_head_dim: int = 128,
        hidden_size: int = 5376,
        num_layers: int = 50,
        num_refiner_layers: int = 2,
        ffn_dim: int = 14336,
        in_channels: int = 24,
        audio_in_channels: int = 32,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        text_dim: int = 5120,
        freq_dim: int = 256,
        time_embed_hidden_dim: int = 5376,
        time_embed_dim: int = 2688,
        rope_freq_dim: int = 16,
        rope_theta: float = 10000.0,
        norm_eps: float = 1e-5,
        qk_norm_eps: float = 1e-5,
        final_norm_eps: float = 1e-5,
        condition_in_dim: int = 48,
        injection_mode: str = "channel",
    ) -> None:
        if injection_mode not in {"channel", "token"}:
            raise ValueError(f"Fusion model requires channel or token, got {injection_mode!r}")
        super().__init__(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_refiner_layers=num_refiner_layers,
            ffn_dim=ffn_dim,
            in_channels=in_channels,
            audio_in_channels=audio_in_channels,
            patch_size=patch_size,
            text_dim=text_dim,
            freq_dim=freq_dim,
            time_embed_hidden_dim=time_embed_hidden_dim,
            time_embed_dim=time_embed_dim,
            rope_freq_dim=rope_freq_dim,
            rope_theta=rope_theta,
            norm_eps=norm_eps,
            qk_norm_eps=qk_norm_eps,
            final_norm_eps=final_norm_eps,
        )
        self.injection_mode = injection_mode
        self.condition_in_dim = condition_in_dim
        patch_volume = patch_size[0] * patch_size[1] * patch_size[2]
        self.condition_proj_in = nn.Linear(condition_in_dim * patch_volume, hidden_size)
        nn.init.zeros_(self.condition_proj_in.weight)
        nn.init.zeros_(self.condition_proj_in.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        timestep_indices: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: torch.Tensor,
        video_indices: torch.Tensor,
        audio_indices: torch.Tensor,
        text_indices: torch.Tensor,
        control_rows: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[MiniMaxH3TransformerOutput, Tuple[torch.Tensor, torch.Tensor]]:
        if control_rows is None:
            raise ValueError("channel/token fusion requires one joint PSS+SDS row per video row")
        if control_rows.shape[1] != video_indices.shape[0]:
            raise ValueError(
                f"control rows {control_rows.shape[1]} do not match video rows {video_indices.shape[0]}"
            )
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0
        if USE_PEFT_BACKEND:
            from diffusers.utils import scale_lora_layers
            scale_lora_layers(self, lora_scale)

        if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
            raise ValueError("position_ids must be shaped (sequence, 3)")
        original_length = position_ids.shape[0]
        if token_tags.shape != (original_length,) or timestep_indices.shape != (original_length,):
            raise ValueError("token tags and timestep indices must match the original packed sequence")

        video_embeds = self.proj_in(hidden_states.to(self.proj_in.weight.dtype))
        condition_embeds = self.condition_proj_in(control_rows.to(self.condition_proj_in.weight.dtype))
        if self.injection_mode == "channel":
            video_embeds = video_embeds + condition_embeds.to(video_embeds.dtype)
        audio_embeds = self.audio_proj_in(audio_hidden_states.to(self.audio_proj_in.weight.dtype))
        text_embeds = self.context_embedder(encoder_hidden_states.to(self.context_embedder.weight.dtype))
        text_embeds = self.token_refiner(text_embeds)

        packed = text_embeds.new_zeros((text_embeds.shape[0], original_length, text_embeds.shape[-1]))
        packed = packed.index_copy(1, text_indices, text_embeds)
        packed = packed.index_copy(1, video_indices, video_embeds.to(text_embeds.dtype))
        packed = packed.index_copy(1, audio_indices, audio_embeds.to(text_embeds.dtype))

        if self.injection_mode == "token":
            # Condition tokens inherit their paired video row's spatial and
            # temporal coordinates.  Their learned embeddings distinguish the
            # condition representation while RoPE preserves the pairing.
            packed = torch.cat([packed, condition_embeds.to(text_embeds.dtype)], dim=1)
            token_tags = torch.cat([token_tags, token_tags.index_select(0, video_indices)])
            timestep_indices = torch.cat([timestep_indices, timestep_indices.index_select(0, video_indices)])
            position_ids = torch.cat([position_ids, position_ids.index_select(0, video_indices)], dim=0)

        sequence_length = position_ids.shape[0]
        temb = self.time_proj(timestep)
        temb = self.time_embedder(temb.to(self.time_embedder.linear_1.weight.dtype)).to(packed.dtype)
        adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags.clamp(min=0)

        attention_mask = None
        is_pad = token_tags < 0
        if bool(is_pad.any()):
            attention_mask = is_pad[None, :] == is_pad[:, None]

        padded_length, valid_length = sequence_length, None
        if self.sp_world_size > 1:
            if attention_mask is not None:
                raise ValueError("sequence parallelism requires padless packed input")
            padded_length = sequence_length + (-sequence_length) % self.sp_world_size
            if padded_length != sequence_length:
                pad = padded_length - sequence_length
                packed = F.pad(packed, (0, 0, 0, pad))
                position_ids = F.pad(position_ids, (0, 0, 0, pad))
                adaln_indices = F.pad(adaln_indices, (0, pad))
            valid_length = sequence_length

        rotary_emb = self.rope(position_ids)
        if self.sp_world_size > 1:
            chunk = padded_length // self.sp_world_size
            rows = slice(self.sp_world_rank * chunk, (self.sp_world_rank + 1) * chunk)
            packed = packed[:, rows]
            adaln_indices = adaln_indices[rows]
            rotary_emb = (rotary_emb[0][rows], rotary_emb[1][rows])

        for block in self.transformer_blocks:
            with torch.autograd.graph.save_on_cpu() if self.gradient_checkpointing_save_on_cpu else nullcontext():
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    kwargs = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    packed = torch.utils.checkpoint.checkpoint(
                        block, packed, temb, adaln_indices, rotary_emb, attention_mask, valid_length, **kwargs
                    )
                else:
                    packed = block(packed, temb, adaln_indices, rotary_emb, attention_mask, valid_length)

        if self.sp_world_size > 1:
            packed = self.all_gather(packed.contiguous(), dim=1)[:, :sequence_length]
        packed = self.norm_out(packed, temb, timestep_indices).to(self.proj_out.weight.dtype)
        video_output = self.proj_out(packed).index_select(1, video_indices)
        audio_output = self.audio_proj_out(packed).index_select(1, audio_indices)
        if USE_PEFT_BACKEND:
            from diffusers.utils import unscale_lora_layers
            unscale_lora_layers(self, lora_scale)
        if not return_dict:
            return video_output, audio_output
        return MiniMaxH3TransformerOutput(sample=video_output, audio_sample=audio_output)
