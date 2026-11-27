"""Run one released real-first-frame case per rank with the verified Teacher 5000 runtime."""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import torch.distributed as dist

import paired_teacher_probe as runner


def shared_geometry(width, height):
    scale = min(runner.WIDTH / width, runner.HEIGHT / height)
    new_width = 2 * round(width * scale / 2)
    new_height = 2 * round(height * scale / 2)
    x, y = (runner.WIDTH - new_width) // 2, (runner.HEIGHT - new_height) // 2

    def convert(frame, mode):
        # Frozen condition exports may encode the full camera view at an anisotropic raster.
        # Map normalized full-image coordinates into the first-frame viewport; do not
        # letterbox each condition independently or crop its contents.
        source_ratio = frame.shape[1] / frame.shape[0]
        if mode != "letterbox":
            raise ValueError(f"Media aspect differs from the frozen first frame: {frame.shape} vs {(height, width)}")
        canvas = np.zeros((runner.HEIGHT, runner.WIDTH, 3), dtype=np.uint8)
        canvas[y:y + new_height, x:x + new_width] = cv2.resize(
            frame, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4
        )
        return canvas

    return convert, (x, y, new_width, new_height)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("vae", "text", "denoise", "decode"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()
    args.joint_audio = False
    args.final_control_scale = 1.0

    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    world = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    manifest = args.root / ('PENDING_CASES.json' if (args.root / 'PENDING_CASES.json').exists() else 'CASES.json')
    rows = json.loads(manifest.read_text())[args.offset:args.offset + world]
    if not rows:
        raise ValueError(f"Empty inference wave at offset {args.offset}")
    row = rows[min(rank, len(rows) - 1)]
    writer = rank < len(rows)
    device = runner.local_device()
    runner.set_device(device)
    with Image.open(args.root / row["files"]["first_frame"]) as image:
        runner.geometry, viewport = shared_geometry(*image.size)

    cache = args.root / "cache" / f'{row["case"]}.pt'
    output = args.root / "outputs" / f'{row["case"]}'
    mode = "letterbox"
    distributed = args.phase in ("text", "denoise")
    if distributed:
        dist.init_process_group(runner.DIST_BACKEND, timeout=timedelta(minutes=30), device_id=device)
    if args.phase == "vae" and writer:
        runner.make_video_cache(args, row, mode, device, cache)
    elif args.phase == "text":
        result = runner.make_text_cache(args, row, mode, device, rank, cache)
        if writer:
            torch.save(result, cache)
    elif args.phase == "denoise":
        latent, proof = runner.denoise(args, row, device, rank, cache)
        if writer:
            torch.save(latent, output.with_suffix(".pt"))
            proof.update(
                case=row["case"], id=row["id"], geometry=mode, viewport=viewport,
                model_step=5000, seed=row["seed"], prompt=row["prompt"], input_sha256=row["input_hashes"],
                common_geometry_sha256=row["geometry_sha256"], release_gate_sha256=row["gate_sha256"],
                fps=runner.FPS, frames=runner.FRAMES, denoiser_evaluations=50, cfg=1, shift=12,
                audio_shift=None, joint_audio=False, final_control_scale=args.final_control_scale, future_rgb_used=False,
                protocol="historical_video_only_cfg1_shift12_eval50_576x384_f90_fps24",
            )
            output.with_suffix(".partial.json").write_text(json.dumps(proof, indent=2) + "\n")
    elif args.phase == "decode" and writer:
        movie = output.with_suffix(".mp4")
        runner.decode(args, output.with_suffix(".pt"), device, movie)
        proof_file = output.with_suffix(".partial.json")
        proof = json.loads(proof_file.read_text())
        with movie.open("rb") as stream:
            proof["video_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        proof_file.write_text(json.dumps(proof, indent=2) + "\n")
        proof_file.replace(output.with_suffix(".json"))
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    if writer:
        print(f'{args.phase} DONE {row["case"]}', flush=True)


if __name__ == "__main__":
    main()
