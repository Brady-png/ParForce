"""Launch the paper's ParForce H3 inference on an Ascend FSDP runtime."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from PIL import Image

TEACHER_SHA = "e6b267c12454f60b8de7f14bcea8cafbbd9962314f02411056ea6458f8c5a09d"
STUDENT_SHA = "33bd7a437276be6b3184147fc429ac767c7d334456fc1decd67a90f2c84185d2"
CONTROL_SHA = "ecaf457aacc18ca6ed9889a276a36722e88c09962549bcd9e153b97e35ea921e"
ASCEND_FIX_SHA = "004e3993476364d6f826f6fabd8531554b08e69c82fca27b2cc44b58d4d4af21"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def expect(path, expected):
    actual = sha(path)
    if actual != expected:
        raise ValueError(f"{path.name}: SHA-256 {actual} differs from {expected}")


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def args_from_cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", choices=("teacher", "student"))
    p.add_argument("--base", type=Path, required=True, help="Original MiniMax-H3 FL2VA directory")
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--control", type=Path, help="Union 2.0 checkpoint, required by the teacher")
    p.add_argument("--runtime", type=Path, required=True, help="Parent of the verified videox_fun package")
    p.add_argument("--dependencies", type=Path, help="Optional pinned Python package overlay")
    p.add_argument("--first-frame", type=Path, required=True)
    p.add_argument("--prompt-file", type=Path, required=True)
    p.add_argument("--particle", type=Path)
    p.add_argument("--depth", type=Path)
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--devices", type=int, default=8)
    p.add_argument("--ffmpeg", default="ffmpeg")
    p.add_argument("--ffprobe", default="ffprobe")
    p.add_argument("--registered-full-view", action="store_true",
                   help="Input cameras share the complete view even if the encoded raster ratios differ")
    return p.parse_args()


def video_shape(args, path):
    data = json.loads(subprocess.check_output([args.ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,sample_aspect_ratio,duration", "-of", "json", str(path)]))
    frames, = data["streams"]
    if frames.get("sample_aspect_ratio") not in (None, "1:1", "N/A"):
        raise ValueError(f"Expected square sample pixels in {path.name}")
    if float(frames.get("duration", 0)) < 3.75 - 1 / 12:
        raise ValueError(f"Too little registered motion in {path.name} for 90 frames at 24 fps")
    return frames["width"], frames["height"]


def main():
    args = args_from_cli()
    if args.devices < 1:
        raise ValueError("--devices must be positive")
    ffmpeg = shutil.which(args.ffmpeg)
    if ffmpeg is None:
        raise FileNotFoundError(args.ffmpeg)
    root = args.work_dir.resolve()
    source = Path(__file__).resolve().parent / "parforce/h3"
    base = args.base.resolve(strict=True)
    runtime = args.runtime.resolve(strict=True)
    prompt = args.prompt_file.read_text().strip()
    if not prompt:
        raise ValueError("The prompt file is empty")
    expect(runtime / "videox_fun/models/minimax_h3_transformer3d_control.py", ASCEND_FIX_SHA)
    expect(args.adapter, TEACHER_SHA if args.model == "teacher" else STUDENT_SHA)
    root.mkdir(parents=True, exist_ok=False)
    for subfolder in ("cache", "outputs", "inputs", "weights"):
        root.joinpath(subfolder).mkdir()
    first = root / "inputs/first_frame.png"
    with Image.open(args.first_frame) as image:
        width, height = image.size
        image = image.convert("RGB")
        if args.model == "student":
            image = image.resize((640,384), Image.Resampling.LANCZOS)
        image.save(first)
    if args.model == "teacher":
        if not all((args.particle, args.depth, args.control)):
            raise ValueError("Teacher requires --particle, --depth, and --control")
        expect(args.control, CONTROL_SHA)
        args.control = args.control.resolve()
        root.joinpath("teacher5000.safetensors").symlink_to(args.adapter.resolve())
        paths = {"first_frame": first, "particle_path": args.particle.resolve(), "depth_path": args.depth.resolve()}
        for key in ("particle_path", "depth_path"):
            w, h = video_shape(args, paths[key])
            if abs((w/h) / (width/height) - 1) > .006 and not args.registered_full_view:
                raise ValueError(f"{key} aspect differs from the first frame. Verify the full camera view.")
        digest = {key: sha(path) for key, path in paths.items()}
        scale = min(576/width, 384/height)
        rw, rh = 2 * round(width*scale/2), 2 * round(height*scale/2)
        geometry = [(576-rw)//2,(384-rh)//2,rw,rh]
        geometry_sha = hashlib.sha256(json.dumps(geometry).encode()).hexdigest()
        gate_sha = hashlib.sha256(json.dumps(digest,sort_keys=True).encode()).hexdigest()
        dump(root / "CASES.json", [{
            "case": "result", "id": "result", "seed": args.seed, "prompt": prompt,
            "files": {key: str(path) for key,path in paths.items()}, "input_hashes": digest,
            "geometry_sha256": geometry_sha, "gate_sha256": gate_sha,
        }])
        entry = source / "run_teacher5000.py"
        extra = ["--control", str(args.control)]
    else:
        root.joinpath("weights/student.safetensors").symlink_to(args.adapter.resolve())
        dump(root / "weights/PROVENANCE.json", {"student":{
            "path": "weights/student.safetensors","sha256": STUDENT_SHA,"G_step": 200}})
        dump(root / "CASES.json", [{
            "case": "result", "seed": args.seed, "prompt": prompt, "first_frame": "inputs/first_frame.png",
            "first_frame_sha256": sha(first), "native_frames": 90, "frames": 90, "fps": 24,
        }])
        entry = source / "student.py"
        extra = []
    paths = [str(runtime)] + ([str(args.dependencies.resolve())] if args.dependencies else []) + [str(source)]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(paths + ([os.environ["PYTHONPATH"]] if "PYTHONPATH" in os.environ else [])))
    for phase in ("vae", "text", "denoise", "decode"):
        cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone",
               f"--nproc_per_node={args.devices}", str(entry), phase, "--root", str(root),
               "--base", str(base), "--ffmpeg", ffmpeg] + extra
        subprocess.run(cmd, env=env, check=True)
    print(root / ("outputs/result.mp4" if args.model == "teacher" else "outputs/student/result.mp4"))


if __name__ == "__main__":
    main()
