"""Replay a geometry-only ablation of three archived H3 Teacher 5000 cases."""
import argparse
import datetime
import gc
import hashlib
import inspect
import json
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from safetensors import safe_open

from firstframe_core import assemble_firstframe
from fsdp_loader import _copy, load_sharded_h3, load_sharded_text
from npu_runtime import DIST_BACKEND, local_device, manual_seed, set_device
from physical_data import read_video
from training_core import control_rows, patchify
from videox_fun.models.minimax_h3_vae import AutoencoderKLMiniMaxH3
from videox_fun.pipeline.pipeline_minimax_h3 import (
    audio_latent_num_frames,
    build_packed_sequence,
    build_row_timesteps,
    unpatchify_video_tokens,
)
from videox_fun.utils import MiniMaxH3Scheduler

HEIGHT, WIDTH, FRAMES, FPS = 384, 576, 90, 24
PATCH = (1, 2, 2)
LATENT_THW = (27, 24, 36)
OUTPUT_SHAPE = (FRAMES, HEIGHT, WIDTH, 3)


def geometry(frame, mode):
    if frame.shape[1] * 9 != frame.shape[0] * 16:
        raise ValueError(f'Expected the archived 16:9 input, found {frame.shape}')
    if mode == 'stretch':
        return cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_LANCZOS4)
    canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    canvas[30:354] = cv2.resize(frame, (WIDTH, 324), interpolation=cv2.INTER_LANCZOS4)
    return canvas


def channel_stats(vae, device):
    values = ([.485, .456, .406], [.229, .224, .225], vae.config.latents_mean, vae.config.latents_std)
    return [torch.as_tensor(x, device=device).view(1, -1, 1, 1, 1) for x in values]


def fingerprint(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def load_data(args, row):
    paths = {k: args.root / v for k, v in row['files'].items()}
    for key, path in paths.items():
        with path.open('rb') as f:
            if hashlib.file_digest(f, 'sha256').hexdigest() != row['input_hashes'][key]:
                raise ValueError(f'Source hash mismatch: {path}')
    return paths


def make_video_cache(args, row, mode, device, cache):
    inputs = load_data(args, row)
    origin = Image.open(inputs['first_frame']).convert('RGB')
    image = (np.array(origin.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS))
             if mode == 'stretch' else geometry(np.array(origin), mode))
    vae = AutoencoderKLMiniMaxH3.from_pretrained_original(args.base, device=device).requires_grad_(False)
    pm, ps, lm, ls = channel_stats(vae, device)

    def encode(frames, single=False):
        x = torch.from_numpy(np.stack(frames).transpose(3, 0, 1, 2).copy()).to(device)[None].float() / 255
        x = (x - pm) / ps
        with torch.autocast('npu', dtype=torch.float16):
            params = vae._encode_clip(x) if single else vae._encode(x)
        distribution = DiagonalGaussianDistribution(params)
        sampled = distribution.sample(generator=torch.Generator(device='cpu').manual_seed(42))
        return ((sampled.to(torch.float16).float() - lm) / ls).cpu()

    sample_time = np.arange(FRAMES, dtype=np.float64) / FPS
    def condition(key):
        raw = read_video(inputs[key], sample_time)
        return encode([geometry(frame, mode) for frame in raw])

    controls = control_rows(condition('particle_path'), condition('depth_path'), True, PATCH)
    keyframe_rows = patchify(encode([image], single=True), PATCH)
    if controls.shape != (5832, 192) or keyframe_rows.shape != (216, 96):
        raise ValueError((row['case'], mode, controls.shape, keyframe_rows.shape))
    torch.save({'controls': controls, 'keyframe_rows': keyframe_rows}, cache)
    Image.fromarray(image).save(cache.with_suffix('.png'))


def make_text_cache(args, row, mode, device, rank, cache):
    from transformers import Qwen2TokenizerFast, Qwen3VLProcessor

    mesh = init_device_mesh('npu', (dist.get_world_size(),))
    encoder = load_sharded_text(args.base, device, rank, mesh=mesh)
    processor = Qwen3VLProcessor.from_pretrained(args.base / 'processor')
    tokenizer = Qwen2TokenizerFast.from_pretrained(args.base / 'tokenizer')
    # The archived stretch path fed the original first frame to Qwen-VL.
    image_path = args.root / row['files']['first_frame'] if mode == 'stretch' else cache.with_suffix('.png')
    with Image.open(image_path) as image:
        vision = processor.image_processor(images=[image.convert('RGB')], return_tensors='pt')
    grid = vision['image_grid_thw']
    count = int(grid[0].prod()) // processor.image_processor.merge_size ** 2
    image_pad = tokenizer.convert_tokens_to_ids('<|image_pad|>')
    vision_start = tokenizer.convert_tokens_to_ids('<|vision_start|>')
    vision_end = tokenizer.convert_tokens_to_ids('<|vision_end|>')
    label = tokenizer('<Picture 1>: ', add_special_tokens=False)['input_ids']
    prompt = tokenizer(row['prompt'], add_special_tokens=False)['input_ids']
    ids = torch.tensor([label + [vision_start] + [image_pad] * count + [vision_end] + prompt], device=device)
    tags = torch.tensor([1] * len(label) + [0] * (count + 2) + [1] * len(prompt))
    kwargs = {'pixel_values': vision['pixel_values'].to(device, torch.bfloat16), 'image_grid_thw': grid.to(device)}
    if 'mm_token_type_ids' in inspect.signature(encoder.forward).parameters:
        kwargs['mm_token_type_ids'] = (ids == image_pad).long()
    captured = {}
    def capture(_module, _args, output):
        captured['text'] = output[0] if isinstance(output, (tuple, list)) else output

    hook = encoder.language_model.layers[49].register_forward_hook(capture)
    try:
        encoder(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, **kwargs)
    finally:
        hook.remove()
    text = captured['text'].detach().cpu().to(torch.bfloat16)
    if text.shape[1] != len(tags) or not torch.isfinite(text).all():
        raise ValueError(f'Invalid text cache for {row["case"]}')
    sample = torch.load(cache, map_location='cpu', weights_only=True)
    sample.update(text=text, tags=tags)
    return sample


def denoise(args, row, device, rank, cache):
    model = load_sharded_h3(args.base, args.control, device, rank, lora_rank=128).eval()
    trainable = {name: param for name, param in model.named_parameters() if param.requires_grad}
    with safe_open(args.root / 'teacher5000.safetensors', framework='pt', device='cpu') as adapter:
        if set(adapter.keys()) != set(trainable) or adapter.metadata()['step'] != '5000':
            raise ValueError('The adapter does not match the Teacher 5000 configuration')
        for name, param in trainable.items():
            _copy(param, adapter.get_tensor(name))
    data = torch.load(cache, map_location='cpu', weights_only=True)
    key = data['keyframe_rows'].to(device)
    controls = data['controls'].to(device)
    text = data['text'].to(device)
    audio_frames = audio_latent_num_frames(FRAMES) if args.joint_audio else 0
    layout = build_packed_sequence(data['tags'], *LATENT_THW, audio_frames, PATCH, ('first',))
    torch.manual_seed(row['seed'])
    manual_seed(row['seed'])
    prefix_noise = torch.randn_like(key)
    noise = torch.randn(1, len(controls), 96, device=device)
    proof = {'noise_sha256': fingerprint(noise), 'prefix_noise_sha256': fingerprint(prefix_noise),
             'controls_sha256': fingerprint(data['controls']), 'keyframe_sha256': fingerprint(data['keyframe_rows']),
             'text_sha256': fingerprint(data['text'].view(torch.int16))}
    scheduler = MiniMaxH3Scheduler(shift=12)
    scheduler.set_timesteps(51, device=device)
    audio_schedule = MiniMaxH3Scheduler(shift=3)
    audio_schedule.set_timesteps(51, device=device)
    if len(scheduler.timesteps) != 50:
        raise ValueError('Expected 50 denoiser evaluations')
    indices = {k: getattr(layout, k).to(device) for k in (
        'token_tags', 'position_ids', 'video_indices', 'audio_indices', 'text_indices')}
    audio = torch.randn(1, 2 * audio_frames, model.config.audio_in_channels, device=device)
    proof['audio_noise_sha256'] = fingerprint(audio)
    for i, (t, audio_t) in enumerate(zip(scheduler.timesteps, audio_schedule.timesteps)):
        packed, packed_controls, prefix = assemble_firstframe(noise[0], controls, key, prefix_noise)
        time = float(t)
        times, time_indices = build_row_timesteps(
            layout, time, float(audio_t) if args.joint_audio else time, max(time, .999), 1.)
        fade = min(max((time - .12) / .38, 0), 1)
        strength = (1 - fade) + fade * args.final_control_scale
        prediction, audio_velocity = model(
            hidden_states=packed[None].to(torch.bfloat16), audio_hidden_states=audio.to(torch.bfloat16), encoder_hidden_states=text,
            timestep=times.to(device), timestep_indices=time_indices.to(device),
            control_rows=packed_controls[None].to(torch.bfloat16), control_context_scale=strength,
            return_dict=False, **indices)
        velocity = prediction[:, prefix:].float()
        if not torch.isfinite(velocity).all():
            raise ValueError(f'Nonfinite flow at step {i + 1}')
        noise = scheduler.step(velocity, t, noise, return_dict=False)[0]
        if args.joint_audio:
            audio = audio_schedule.step(audio_velocity.float(), audio_t, audio, return_dict=False)[0]
        if i % 10 == 9 and rank == 0:
            print(f'DENOISE {i + 1}/50', flush=True)
    return noise.cpu(), proof


def decode(args, input_path, device, output_path):
    vae = AutoencoderKLMiniMaxH3.from_pretrained_original(args.base, device=device).requires_grad_(False)
    noise = torch.load(input_path, map_location='cpu', weights_only=True)
    z = unpatchify_video_tokens(noise[0].to(device), *LATENT_THW, 24, PATCH)
    pm, ps, lm, ls = channel_stats(vae, device)
    with torch.autocast('npu', dtype=torch.float16):
        pixels = vae.decode(z * ls + lm, return_dict=False)[0]
    rgb = ((pixels.float() * ps + pm).clamp(0, 1) * 255).byte()[0].permute(1, 2, 3, 0).cpu().numpy()
    if rgb.shape != OUTPUT_SHAPE or not torch.isfinite(pixels).all():
        raise ValueError(f'Invalid decode: {rgb.shape}')
    tmp = output_path.with_suffix('.tmp.mp4')
    subprocess.run([args.ffmpeg, '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', '576x384', '-r', '24', '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', 'fast',
        '-crf', '18', '-pix_fmt', 'yuv420p', '-threads', '2', '-movflags', '+faststart', str(tmp)],
        input=rgb.tobytes(), check=True)
    tmp.replace(output_path)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('vae', 'text', 'denoise', 'decode'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--ffmpeg', required=True)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--joint-audio', action='store_true')
    parser.add_argument('--final-control-scale', type=float, default=1.)
    args = parser.parse_args()
    rows = json.loads((args.root / 'CASES.json').read_text())
    slots = [(row, mode) for row in rows for mode in ('stretch', 'letterbox')]
    device = local_device()
    set_device(device)
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    rank = int(os.environ['RANK'])
    row, mode = slots[min(rank, len(slots) - 1)]
    stem = row['case'] + '_' + mode
    args.root.joinpath('cache').mkdir(exist_ok=True)
    output_dir = args.output_dir or args.root.joinpath('outputs')
    output_dir.mkdir(parents=True, exist_ok=True)
    working_cache = args.root / 'cache' / f'{stem}.pt'
    output = output_dir / stem
    distributed = args.phase in ('text', 'denoise')
    if distributed:
        dist.init_process_group(DIST_BACKEND, timeout=datetime.timedelta(minutes=30), device_id=device)
    if args.phase == 'vae' and rank < len(slots):
        make_video_cache(args, row, mode, device, working_cache)
    elif args.phase == 'text':
        result = make_text_cache(args, row, mode, device, rank, working_cache)
        if rank < len(slots):
            torch.save(result, working_cache)
    elif args.phase == 'denoise':
        latent, proof = denoise(args, row, device, rank, working_cache)
        if rank < len(slots):
            torch.save(latent, output.with_suffix('.pt'))
            proof.update(case=row['case'], geometry=mode, model_step=5000, seed=row['seed'],
                         prompt=row['prompt'], input_sha256=row['input_hashes'], fps=FPS, frames=FRAMES,
                         denoiser_evaluations=50, cfg=1, shift=12,
                         audio_tokens=2 * audio_latent_num_frames(FRAMES) if args.joint_audio else 0,
                         audio_shift=3 if args.joint_audio else None, final_control_scale=args.final_control_scale)
            output.with_suffix('.json').write_text(json.dumps(proof, indent=2) + '\n')
    elif args.phase == 'decode' and rank < len(slots):
        decode(args, output.with_suffix('.pt'), device, output.with_suffix('.mp4'))
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    if rank < len(slots):
        print(f'{args.phase} DONE {stem}', flush=True)


if __name__ == '__main__':
    main()
