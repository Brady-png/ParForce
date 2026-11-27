"""按单张量读取官方权重并写入 FSDP2 本地分片，避免完整 CPU 模型。"""
import math
import torch
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import DTensor, Shard
from peft import LoraConfig
from force_adapter import install_force,route_name
from safetensors import safe_open
from videox_fun.models.minimax_h3_transformer3d_control import MiniMaxH3ControlTransformer3DModel
from videox_fun.models.minimax_h3_transformer3d_fusion import MiniMaxH3ConditionFusionTransformer3DModel
from videox_fun.models.minimax_h3_conversion import MINIMAX_H3_TRANSFORMER_CONFIG, iter_original_transformer_tensors


def _local_tensor(parameter):
    return parameter.to_local() if isinstance(parameter, DTensor) else parameter


def _copy(parameter, source):
    """从全局 CPU 张量切出当前 rank 需要的分片。"""
    if isinstance(parameter, DTensor):
        coordinate = parameter.device_mesh.get_coordinate()
        for mesh_axis, placement in enumerate(parameter.placements):
            if isinstance(placement, Shard):
                extent = source.size(placement.dim)
                chunk = math.ceil(extent / parameter.device_mesh.size(mesh_axis))
                offset = coordinate[mesh_axis] * chunk
                source = source.narrow(placement.dim, min(offset, extent), max(0, min(chunk, extent-offset)))
    target = _local_tensor(parameter)
    if tuple(source.shape) != tuple(target.shape):
        raise ValueError(f'shard shape mismatch {source.shape} vs {target.shape}')
    target.copy_(source.to(device=target.device, dtype=target.dtype))


def load_sharded_h3(base_path, control_path, device, rank, lora_rank=128, force_rank=0,
                    injection_mode="controlnet"):
    """Load one explicit condition-injection variant.

    ``injection_mode`` is deliberately an explicit argument.  The training
    entrypoint rejects unknown modes rather than quietly routing all labels
    through the ControlNet path.  The mode-specific modules are installed in
    the following patch; keeping the API here makes queue manifests auditable.
    """
    if injection_mode not in {"channel", "token", "controlnet"}:
        raise ValueError(f"Unknown injection_mode={injection_mode!r}")
    config = dict(MINIMAX_H3_TRANSFORMER_CONFIG)
    with torch.device('meta'):
        if injection_mode == "controlnet":
            config.update(control_blocks_places=list(range(0, 50, 5)), control_in_dim=48,
                          control_apply_audio=False, inpaint_masked_pixel_mode='post_norm')
            model = MiniMaxH3ControlTransformer3DModel(**config)
            target_modules = r'control_blocks\.\d+\.(?:attn\.(?:to_q|to_k|to_v|to_out\.0)|ff\.net\.(?:0\.proj|2))'
        else:
            config.update(condition_in_dim=48, injection_mode=injection_mode)
            model = MiniMaxH3ConditionFusionTransformer3DModel(**config)
            target_modules = r'transformer_blocks\.\d+\.(?:attn\.(?:to_q|to_k|to_v|to_out\.0)|ff\.net\.(?:0\.proj|2))'
        model.add_adapter(LoraConfig(r=lora_rank, lora_alpha=lora_rank,
            init_lora_weights=True, target_modules=target_modules))
    if force_rank:
        with torch.device('meta'):
            install_force(model, force_rank)
    model.requires_grad_(False)
    for name, param in model.named_parameters():
        if injection_mode == "controlnet":
            trainable = (name.startswith('control_blocks.') and ('lora_' in name or 'before_proj' in name or 'after_proj' in name)) or name.startswith('control_proj_in.')
        else:
            trainable = ('lora_' in name and name.startswith('transformer_blocks.')) or name.startswith('condition_proj_in.')
        param.requires_grad_(trainable)
    if injection_mode == "controlnet":
        assert not any(p.requires_grad for n,p in model.named_parameters() if not n.startswith(('control_blocks.','control_proj_in.')))
    else:
        assert not any(p.requires_grad for n,p in model.named_parameters() if not (n.startswith(('transformer_blocks.','condition_proj_in.'))))
    assert any('lora_' in n and p.requires_grad for n,p in model.named_parameters())
    # FSDP2 requires a uniform original dtype within each sharded block.
    # Keep adapters in bf16 as well; mixed precision still reduces in fp32.
    model.to(dtype=torch.bfloat16)
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    blocks = list(model.transformer_blocks) + list(model.token_refiner.refiner_blocks)
    if injection_mode == "controlnet":
        blocks += list(model.control_blocks)
    for block in blocks:
        fully_shard(block, mp_policy=policy)
    fully_shard(model, mp_policy=policy)
    model.to_empty(device=device)
    params = dict(model.named_parameters())

    def resolve(name):
        if force_rank: name=route_name(name)
        if name in params:
            return name
        stem, leaf = name.rsplit('.', 1)
        candidate = stem + '.base_layer.' + leaf
        if candidate in params:
            return candidate
        raise KeyError(name)

    loaded = set()
    with torch.no_grad():
        # LoRA A 采用每个全局行独立、与分片无关的确定性生成；B 初始化为零。
        for name, param in params.items():
            local = _local_tensor(param)
            local.zero_()
            if 'lora_A.' in name:
                gen = torch.Generator(device='cpu').manual_seed(0)
                source = torch.empty(tuple(param.shape), dtype=torch.float32)
                torch.nn.init.kaiming_uniform_(source, a=math.sqrt(5), generator=gen)
                _copy(param, source)
        control_for_base = (
            {place: idx for idx, place in enumerate(model.control_blocks_places)}
            if injection_mode == "controlnet" else {}
        )
        for key, tensor in iter_original_transformer_tensors(base_path, torch_dtype=torch.bfloat16):
            target_key = resolve(key)
            _copy(params[target_key], tensor)
            loaded.add(target_key)
            if injection_mode == "controlnet" and key.startswith('transformer_blocks.'):
                parts = key.split('.', 2)
                index = int(parts[1])
                if index in control_for_base:
                    control_key = resolve(f'control_blocks.{control_for_base[index]}.{parts[2]}')
                    _copy(params[control_key], tensor)
                    loaded.add(control_key)
            elif injection_mode == "controlnet" and key == 'proj_in.weight' and not control_path:
                source = torch.zeros(tuple(params['control_proj_in.weight'].shape), dtype=tensor.dtype)
                source[:, 96:192] = tensor
                _copy(params['control_proj_in.weight'], source)
            elif injection_mode == "controlnet" and key == 'proj_in.bias' and not control_path:
                _copy(params['control_proj_in.bias'], tensor)
        if control_path:
            if injection_mode != "controlnet":
                raise ValueError("A ControlNet checkpoint cannot be loaded into channel/token fusion")
            with safe_open(control_path, framework='pt', device='cpu') as checkpoint:
                for key in checkpoint.keys():
                    tensor = checkpoint.get_tensor(key)
                    if key == 'control_proj_in.weight':
                        adapted = torch.zeros((tensor.shape[0], 192), dtype=tensor.dtype)
                        adapted[:, 96:] = tensor[:, :96]
                        tensor = adapted
                    target_key = resolve(key)
                    _copy(params[target_key], tensor)
                    loaded.add(target_key)
        model.rope.inv_freq = model.rope._compute_inv_freq(device=device)
    adapter_roots = ('control_proj_in',) if injection_mode == "controlnet" else ('condition_proj_in',)
    expected = {name for name in params if 'lora_' not in name and 'before_proj' not in name and 'after_proj' not in name and not name.startswith(adapter_roots)}
    missing = expected - loaded
    if missing:
        raise RuntimeError(f'Uninitialized checkpoint parameters: {sorted(missing)[:10]}')
    return model


def load_sharded_text(base_path, device, rank, mesh=None):
    """直接载入截断 Qwen3-VL 的本地权重分片，无完整 CPU/GPU 编码器。"""
    import glob
    import os
    from transformers import AutoConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
    config = AutoConfig.from_pretrained(os.path.join(base_path, 'text_encoder'))
    config.text_config.num_hidden_layers = 51
    config._attn_implementation = 'sdpa'
    config.text_config._attn_implementation = 'sdpa'
    config.vision_config._attn_implementation = 'sdpa'
    with torch.device('meta'):
        encoder = Qwen3VLModel(config)
    encoder.requires_grad_(False)
    encoder.to(dtype=torch.bfloat16)
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16)
    for layer in encoder.language_model.layers:
        fully_shard(layer, mesh=mesh, mp_policy=policy)
    # 视觉分支不用在纯文本路径，但也分片存储以保持模型语义。
    fully_shard(encoder, mesh=mesh, mp_policy=policy)
    encoder.to_empty(device=device)
    params = dict(encoder.named_parameters())
    assigned = set()
    with torch.no_grad():
        for shard in sorted(glob.glob(os.path.join(base_path, 'text_encoder', '*.safetensors'))):
            with safe_open(shard, framework='pt', device='cpu') as checkpoint:
                for source_key in checkpoint.keys():
                    if not source_key.startswith('model.'):
                        continue
                    key = source_key[len('model.'):]
                    if key not in params:
                        continue  # 截断掉的 51..63 层。
                    _copy(params[key], checkpoint.get_tensor(source_key))
                    assigned.add(key)
        missing = set(params) - assigned
        if missing:
            raise RuntimeError(f'Text encoder checkpoint parameters missing: {sorted(missing)[:10]}')
        for module in encoder.modules():
            if hasattr(module, 'rope_init_fn'):
                module.inv_freq, module.attention_scaling = module.rope_init_fn(module.config, device=device)
                module.original_inv_freq = module.inv_freq
            elif module.__class__.__name__ == 'Qwen3VLVisionRotaryEmbedding':
                dim = module.inv_freq.numel() * 2
                module.inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    return encoder.eval()
