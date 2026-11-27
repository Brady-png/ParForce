<h1 align="center">Particle Forcing</h1>

<p align="center"><b>Guiding Spatio-Temporal Layouts, Learning Physics-Aware Generation</b></p>

<p align="center">
  <a href="https://huggingface.co/BRADY99/ParForce">Model weights</a> ·
  <a href="examples/showcase/manifest.json">Showcase inputs</a> ·
  <a href="https://huggingface.co/BRADY99/ParForce/tree/main/examples/sim2real">Sim-to-Real</a>
</p>

<p align="center">
  <img src="assets/teaser.webp" width="960" alt="ParForce teaser comparing 2D motion guidance with projected particles and surface depth when a ball passes through a tunnel.">
</p>

ParForce generates video from a first frame and a scene and action prompt. The teacher also takes projected particle states and surface depth to guide motion and occlusion. The distilled student generates video from the first frame and prompt in four denoiser calls.

## Code and model weights

The [ParForce model repository](https://huggingface.co/BRADY99/ParForce) is the download location for the adapters. The inference CLI in this checkout supports the H3 Teacher and the four-step H3 Student.

| Path | Contents |
| --- | --- |
| `infer.py`, `parforce/h3/` | H3 inference, particle and depth conditioning, and the Ascend runtime integration |
| `parforce/conditions/` | Rigid-body simulation and particle/depth rendering from prepared masks, depth and action inputs |
| `config/inference.json`, `examples/teacher/` | Inference settings and a paired example with a first frame, prompt and both controls |
| [`examples/showcase/`](examples/showcase/) | Archived ParForce inputs and a local preview for simple dominoes, curtain, milk pour and red wine pour |

The Wan and CogVideoX adapters use separate backbones and require the matching inference implementations.

The [showcase manifest](examples/showcase/manifest.json) records the original Teacher and seed, together with checksums and the native sizes and timings of the inputs. The two pour prompts share the same first frame, particles, and depth. The dominoes inputs use different raster aspect ratios for the first frame and condition videos. Keep their archived framing when reproducing the demo. To preview all four cases locally, run `python -m http.server --directory examples/showcase 8000` and open `http://localhost:8000`.

## Inference on Ascend

The reviewed runtime uses Ascend 910B, CANN 9.0.1 and PyTorch 2.7.1 with matching `torch_npu`. Initialize your CANN shell environment. The `--runtime` argument points to the parent of the reviewed `videox_fun` package. Use the matching Diffusers and Transformers dependencies from your installation. The `--dependencies` argument can supply a versioned Python dependency overlay. Install FFmpeg with libx264 and the Python dependencies used in the entry points. `infer.py --help` lists the inputs.

Download the official [MiniMax-H3 FL2VA](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/main/FL2VA) base, the [ControlNet Union 2.0](https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union-2.0) checkpoint and the ParForce adapters. The teacher samples the two registered videos on a common 24 fps timeline and maps all inputs to the first frame's viewport. The input camera framing should match across the first frame and both controls.

```bash
python infer.py teacher --base /path/to/FL2VA \
  --runtime ../runtime --adapter ../weights/parforce_h3_teacher_5000.safetensors \
  --control /path/to/MiniMax-H3-Fun-Controlnet-Union-2.0.safetensors \
  --first-frame examples/teacher/first_frame.png --prompt-file examples/teacher/prompt.txt \
  --particle examples/teacher/particles.mp4 --depth examples/teacher/depth.mp4 \
  --work-dir runs/teacher --devices 8 --seed 42

python infer.py student --base /path/to/FL2VA \
  --runtime ../runtime --adapter ../weights/parforce_h3_student_g200.safetensors \
  --first-frame examples/teacher/first_frame.png --prompt-file examples/teacher/prompt.txt \
  --work-dir runs/student --devices 8 --seed 42
```

The teacher writes `outputs/result.mp4` inside the work directory. The student writes `outputs/student/result.mp4`. Both write inference receipts next to the videos. [Exact settings](config/inference.json)

## Physical condition rendering

The first-frame constructor is designed to take an image and prompt. Its perception and planning stages generate object masks, camera-space depth and action fields. The bundled `parforce/conditions` entry point starts from those intermediate files, builds rigid-body geometry, runs the simulation, and renders the projected particles and surface depth with automated checks. Inspect the input contract with `python -m parforce.conditions.run --help`. The Teacher can also consume the finished controls in the examples directly.

The constructor uses MuJoCo, OpenCV, NumPy, SciPy, trimesh, and Numba. It encodes the two condition videos with FFmpeg and libx264. Set `PARFORCE_FFMPEG` if the desired binary is outside your shell path.

```bash
python -m parforce.conditions.run --manifest /path/to/cases.json --output runs/conditions
```

## Terms

The upstream MiniMax-H3 base and Union 2.0 have their own model terms. The reviewed adapters and code require the authors' release terms when published. [MiniMax-H3 license](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE) · [VideoX-Fun license](https://github.com/aigc-apps/VideoX-Fun/blob/main/LICENSE)
