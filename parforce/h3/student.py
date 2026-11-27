"""Matched native four-step H3 inference with the released no-DMD and paper G200 adapters."""
import argparse
from datetime import timedelta
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from peft import LoraConfig
from safetensors import safe_open
import torch
import torch.distributed as dist
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from fsdp_loader import _copy
from npu_runtime import DIST_BACKEND, local_device, manual_seed, set_device
from paired_teacher_probe import channel_stats, fingerprint, make_text_cache
from training_core import patchify
from videox_fun.models.minimax_h3_conversion import MINIMAX_H3_TRANSFORMER_CONFIG, iter_original_transformer_tensors
from videox_fun.models.minimax_h3_vae import AutoencoderKLMiniMaxH3
from videox_fun.pipeline.pipeline_minimax_h3 import (
    build_packed_sequence, build_row_timesteps, unpatchify_video_tokens, video_latent_num_frames,
)

WIDTH, HEIGHT = 640, 384
PATCH = (1, 2, 2)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def emit(path, value):
    draft = path.with_suffix(".tmp")
    draft.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    draft.replace(path)


def load_base(base, device):
    name = "videox_fun.models.minimax_h3_transformer3d_student_npu"
    original = importlib.util.find_spec("videox_fun.models.minimax_h3_transformer3d")
    source = Path(original.origin)
    native = importlib.util.module_from_spec(importlib.util.spec_from_file_location(name, source))
    sys.modules[name] = native
    line = "        hidden_states = hidden_states.index_copy(1, audio_indices, audio_embeds.to(text_embeds.dtype))"
    text = source.read_text()
    if text.count(line) != 1:
        raise RuntimeError("Expected exactly one native audio write-back")
    guarded = "        if audio_indices.numel():\n    " + line
    exec(compile(text.replace(line, guarded, 1), str(source), "exec"), vars(native))
    with torch.device("meta"):
        model = native.MiniMaxH3Transformer3DModel(**dict(MINIMAX_H3_TRANSFORMER_CONFIG))
        model.add_adapter(LoraConfig(r=128, lora_alpha=128, init_lora_weights=True,
                                     target_modules=["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"]))
    model.requires_grad_(False).to(dtype=torch.bfloat16)
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for block in list(model.transformer_blocks) + list(model.token_refiner.refiner_blocks):
        fully_shard(block, mp_policy=policy)
    # Keep the input, time and output projections at their released float32 precision.
    # Shard them separately to keep each Ascend FSDP group uniform in dtype.
    for name in ("proj_in", "audio_proj_in", "time_embedder", "proj_out", "audio_proj_out"):
        fully_shard(getattr(model, name).float(), mp_policy=MixedPrecisionPolicy(cast_forward_inputs=False))
    fully_shard(model, mp_policy=MixedPrecisionPolicy(cast_forward_inputs=False))
    model.to_empty(device=device)
    params = dict(model.named_parameters())
    loaded = set()
    for key, tensor in iter_original_transformer_tensors(base, torch_dtype=torch.bfloat16):
        if key not in params:
            root, leaf = key.rsplit(".", 1)
            key = f"{root}.base_layer.{leaf}"
        _copy(params[key], tensor)
        loaded.add(key)
    expected = {key for key in params if "lora_" not in key}
    if loaded != expected:
        raise ValueError(f"Base checkpoint missing tensors: {sorted(expected - loaded)[:8]}")
    model.rope.inv_freq = model.rope._compute_inv_freq(device=device)
    return model.eval()


def use_adapter(model, path):
    model.reshard()
    params = {key: p for key, p in model.named_parameters() if "lora_" in key}
    with safe_open(path, framework="pt", device="cpu") as source:
        assert source.metadata()["alpha"] == "128" and set(source.keys()) == params.keys()
        for name, param in params.items():
            tensor = source.get_tensor(name)
            assert tensor.shape == param.shape, name
            _copy(param, tensor)


def encode_image(vae, path, device):
    with Image.open(path) as picture:
        image = np.asarray(picture.convert("RGB"))
    assert image.shape == (HEIGHT, WIDTH, 3)
    pixel_mean, pixel_std, latent_mean, latent_std = channel_stats(vae, device)
    tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).to(device).float()[None, :, None] / 255
    with torch.autocast("npu", dtype=torch.float16):
        params = vae._encode_clip((tensor - pixel_mean) / pixel_std)
    posterior = DiagonalGaussianDistribution(params)
    sampled = posterior.sample(generator=torch.Generator(device="cpu").manual_seed(42))
    latent = (sampled.to(torch.float16).float() - latent_mean) / latent_std
    return patchify(latent, PATCH).cpu()


def rollout(model, row, cache, device):
    key = cache["keyframe_rows"].to(device)
    text = cache["text"].to(device)
    frames = video_latent_num_frames(row["native_frames"])
    h, w = HEIGHT // 16, WIDTH // 16
    layout = build_packed_sequence(cache["tags"], frames, h, w, 0, PATCH, ("first",))
    assert layout.num_condition_video_rows == len(key)
    torch.manual_seed(row["seed"])
    manual_seed(row["seed"])
    disturbances = torch.randn_like(key)
    prefix = .999 * key + .001 * disturbances
    noise = torch.randn(1, frames * (h // 2) * (w // 2), 96, device=device)
    proof = {"noise_sha256": fingerprint(noise), "prefix_noise_sha256": fingerprint(disturbances),
             "keyframe_sha256": fingerprint(key), "text_sha256": fingerprint(text.view(torch.int16))}
    sigma = torch.linspace(1., 0., 5, device=device, dtype=torch.float32)
    sigma = (6 * sigma) / (1 + 5 * sigma)
    indices = {k: getattr(layout, k).to(device) for k in (
        "token_tags", "position_ids", "video_indices", "audio_indices", "text_indices")}
    empty_audio = torch.empty(1, 0, model.config.audio_in_channels, device=device, dtype=torch.bfloat16)
    for step in range(4):
        t = 1.0 - float(sigma[step])
        times, time_indices = build_row_timesteps(layout, t, t, max(t, .999), 1.)
        prediction, _ = model(
            hidden_states=torch.cat((prefix, noise[0]), dim=0)[None],
            audio_hidden_states=empty_audio, encoder_hidden_states=text, timestep=times.to(device),
            timestep_indices=time_indices.to(device), return_dict=False, **indices,
        )
        velocity = prediction[:, len(prefix):].float()
        assert velocity.shape == noise.shape and torch.isfinite(velocity).all()
        noise = noise + (sigma[step] - sigma[step + 1]) * velocity
    return noise.cpu(), proof


def encode_video(vae, latent, row, ffmpeg, dest, device):
    latent_frames = video_latent_num_frames(row["native_frames"])
    z = unpatchify_video_tokens(latent[0].to(device), latent_frames, HEIGHT // 16, WIDTH // 16, 24, PATCH)
    pm, ps, lm, ls = channel_stats(vae, device)
    with torch.autocast("npu", dtype=torch.float16):
        pixels = vae.decode(z * ls + lm, return_dict=False)[0]
    rgb = ((pixels.float() * ps + pm).clamp(0, 1) * 255).byte()[0].permute(1, 2, 3, 0).cpu().numpy()
    assert rgb.shape == (row["native_frames"], HEIGHT, WIDTH, 3) and torch.isfinite(pixels).all()
    draft = dest.with_suffix(".tmp.mp4")
    x, y, width, height = row.get("letterbox", [0, 0, WIDTH, HEIGHT])
    output_width, output_height = row.get("original_size", [WIDTH, HEIGHT])
    command = [ffmpeg, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{output_width}x{output_height}", "-r", str(row["fps"]), "-i", "pipe:0", "-an",
               "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
               "-threads", "2", "-movflags", "+faststart", str(draft)]
    frames = rgb[:row["frames"]]
    if "letterbox" in row:
        with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
            for frame in frames:
                viewport = Image.fromarray(frame[y:y + height, x:x + width])
                restored = viewport.resize((output_width, output_height), Image.Resampling.LANCZOS)
                encoder.stdin.write(restored.tobytes())
            encoder.stdin.close()
            if encoder.wait():
                raise subprocess.CalledProcessError(encoder.returncode, command)
    else:
        subprocess.run(command, input=frames.tobytes(), check=True)
    draft.replace(dest)


def waves(rows, rank, world):
    for offset in range(0, len(rows), world):
        batch = rows[offset:offset + world]
        yield batch[min(rank, len(batch) - 1)], rank < len(batch)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("vae", "text", "denoise", "decode"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--arm", choices=("student", "turbo"), default="student")
    args = parser.parse_args()
    root = args.root
    rank, world = (int(os.environ[key]) for key in ("RANK", "WORLD_SIZE"))
    rows = json.loads((root / "CASES.json").read_text())
    proof = json.loads((root / "weights/PROVENANCE.json").read_text())
    root.joinpath("cache").mkdir(exist_ok=True)
    input_latents = root / "input_latents"
    input_latents.mkdir(exist_ok=True)
    device = local_device()
    set_device(device)
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    distributed = args.phase in ("text", "denoise")
    if distributed:
        dist.init_process_group(DIST_BACKEND, timeout=timedelta(minutes=30), device_id=device)
    if args.phase == "vae":
        vae = AutoencoderKLMiniMaxH3.from_pretrained_original(args.base, device=device).eval().requires_grad_(False)
        unique = {row["first_frame_sha256"]: row for row in rows}
        for row, writer in waves(list(unique.values()), rank, world):
            if writer:
                first = root / row["first_frame"]
                image_hash = row["first_frame_sha256"]
                assert sha(first) == image_hash
                torch.save({"keyframe_rows": encode_image(vae, first, device)}, input_latents / f"{image_hash}.pt")
                print("VAE", row["case"], flush=True)
    elif args.phase == "text":
        for row, writer in waves(rows, rank, world):
            cache = input_latents / f'{row["first_frame_sha256"]}.pt'
            result = make_text_cache(args, {**row, "files": {"first_frame": row["first_frame"]}},
                                     "stretch", device, rank, cache)
            if writer:
                result["prompt_sha256"] = hashlib.sha256(row["prompt"].encode()).hexdigest()
                torch.save(result, root / "cache" / f'{row["case"]}.pt')
                print("TEXT", row["case"], flush=True)
            del result
            dist.barrier()
    elif args.phase == "denoise":
        model = load_base(args.base, device)
        for arm in (args.arm,):
            use_adapter(model, root / proof[arm]["path"])
            dist.barrier()
            destination = root / "outputs" / arm
            destination.mkdir(parents=True, exist_ok=True)
            for row, writer in waves(rows, rank, world):
                began = time.monotonic()
                cache = torch.load(root / "cache" / f'{row["case"]}.pt', map_location="cpu", weights_only=True)
                assert cache["prompt_sha256"] == hashlib.sha256(row["prompt"].encode()).hexdigest()
                latent, details = rollout(model, row, cache, device)
                if writer:
                    name = destination / row["case"]
                    torch.save(latent, name.with_suffix(".pt"))
                    emit(name.with_suffix(".partial.json"), {
                        **row, **details, "arm": arm, "adapter_sha256": proof[arm]["sha256"],
                        "model_G_step": proof[arm]["G_step"], "denoiser_calls": 4, "cfg": 1, "shift": 6.,
                        "canvas": [WIDTH, HEIGHT], "audio_tokens": 0, "inputs": ["first_frame", "prompt"],
                        "ascend_empty_audio_scatter_guard": True,
                        "denoise_seconds": time.monotonic() - began,
                    })
                    print(arm, "DENOISE", row["case"], flush=True)
                del cache, latent
                dist.barrier()
    else:
        vae = AutoencoderKLMiniMaxH3.from_pretrained_original(args.base, device=device).eval().requires_grad_(False)
        for row, writer in waves(rows, rank, world):
            if not writer:
                continue
            for arm in (args.arm,):
                name = root / "outputs" / arm / row["case"]
                meta = json.loads(name.with_suffix(".partial.json").read_text())
                movie = name.with_suffix(".mp4")
                latent = torch.load(name.with_suffix(".pt"), map_location="cpu", weights_only=True)
                encode_video(vae, latent, row, args.ffmpeg, movie, device)
                emit(name.with_suffix(".json"), {**meta, "video_sha256": sha(movie)})
                print(arm, "DECODE", row["case"], flush=True)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
