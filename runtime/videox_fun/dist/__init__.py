from .cogvideox_xfuser import CogVideoXMultiGPUsAttnProcessor2_0
from .ernie_image_xfuser import ErnieImageMultiGPUsAttnProcessor
from .flashhead_xfuser import usp_attn_flashhead_forward
# Flux2 is not required by the H3-only ablation.  This lightweight source
# snapshot intentionally does not vendor the Flux2 helper, so keep its import
# optional instead of preventing the unrelated MiniMax-H3 modules from loading.
try:
    from .flux2_xfuser import Flux2MultiGPUsAttnProcessor2_0
except ModuleNotFoundError:
    Flux2MultiGPUsAttnProcessor2_0 = None
from .flux_xfuser import FluxMultiGPUsAttnProcessor2_0
# The H3 trainer owns its FSDP wrapping.  Keep UI-oriented helpers optional:
# this source snapshot does not include ``dist.fsdp``.
try:
    from .fsdp import offload_components_cpu, shard_model
except ModuleNotFoundError:
    offload_components_cpu = None
    shard_model = None
from .fuser import (get_sequence_parallel_rank,
                    get_sequence_parallel_world_size, get_sp_group,
                    get_world_group, init_distributed_environment,
                    initialize_model_parallel, model_parallel_is_initialized,
                    sequence_parallel_all_gather, sequence_parallel_chunk,
                    set_multi_gpus_devices, ulysses_all_to_all,
                    xFuserLongContextAttention)
from .hunyuanvideo_xfuser import HunyuanVideoMultiGPUsAttnProcessor2_0
from .infinitalk_xfuser import usp_attn_infinitetalk_forward
from .lens_xfuser import usp_lens_joint_attention_forward
from .lingbot_video_xfuser import usp_attn_lingbot_video_forward
from .longcatvideo_xfuser import (usp_attn_longcatvideo_avatar_forward,
                                  usp_attn_longcatvideo_forward,
                                  usp_cross_attn_longcatvideo_forward,
                                  usp_rope_longcatvideo_forward)
from .ltx2_xfuser import (LTX2MultiGPUsAttnProcessor,
                          LTX2PerturbedMultiGPUsAttnProcessor)
from .minimax_h3_xfuser import MiniMaxH3MultiGPUsAttnProcessor
from .qwen_xfuser import QwenImageMultiGPUsAttnProcessor2_0
from .qwenimage21_xfuser import QwenImage21MultiGPUsAttnProcessor
from .wan_xfuser import (usp_attn_forward, usp_attn_s2v_forward,
                         usp_attn_self_forcing_forward,
                         usp_attn_sparkwan_forward)
from .z_image_xfuser import ZMultiGPUsSingleStreamAttnProcessor
