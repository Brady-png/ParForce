<h1 align="center">Particle Forcing</h1>
<p align="center"><strong>Guiding Spatio-Temporal Layouts, Learning Physics-Aware Generation</strong></p>

<p align="center">
  Liang Dong<sup>1</sup>, Haowen Xu<sup>2</sup>, Tieyue Yin<sup>3</sup>, Yunfeng Wu<sup>4</sup>,<br>
  Jun Sun<sup>3</sup>, Songhua Liu<sup>2,*</sup>, Yingtian Zou<sup>2,*</sup>
</p>

<p align="center">
  <sup>1</sup> Carnegie Mellon University · <sup>2</sup> Shanghai Jiao Tong University<br>
  <sup>3</sup> Nanjing University · <sup>4</sup> The University of Hong Kong<br>
  <small>* Corresponding authors</small>
</p>

<p align="center">
  <a href="https://brady-png.github.io/parforce-project/"><strong>Project page</strong></a> ·
  <a href="https://brady-png.github.io/parforce-project/assets/paper.pdf"><strong>Paper</strong></a> ·
  <a href="https://huggingface.co/BRADY99/ParForce"><strong>Model weights</strong></a> ·
  <a href="#quick-start"><strong>Quick start</strong></a> ·
  <a href="#examples"><strong>Examples</strong></a>
</p>

<p align="center">
  <img src="assets/teaser.webp" width="960" alt="ParForce-generated videos with their projected particles and surface-depth sequences.">
</p>

ParForce uses projected particles and surface depth to guide motion and occlusion. A distilled four-step student generates video from a first frame and text.

## Method

<img src="assets/method.webp" width="960" alt="ParForce teacher and student architecture, including the physical conditions and the distillation path.">

| Component | Inputs | Output |
| --- | --- | --- |
| Physical condition constructor | First frame and action instruction | Projected particle states and visible surface depth |
| ParForce Teacher | First frame, prompt, particles, and depth | Motion-controlled RGB video |
| ParForce Student | First frame and prompt | RGB video in four denoiser calls |

During distillation, the teacher supplies guidance from particle and depth sequences while the student learns to generate from the first frame and text. This repository includes the rigid-body simulation and rendering stage, which starts from prepared masks, depth, and action fields.

## Quick start

### 1. Prepare the environment

Use an Ascend NPU environment with Python 3.12, CANN 9.0.1, PyTorch 2.7.1 and matching `torch_npu`. The inference entry points were smoke-tested on four Ascend 910B devices. Initialize CANN and install FFmpeg with libx264. The [bundled runtime](runtime/README.md) provides the VideoX-Fun source used by the launchers.

```bash
git clone https://github.com/Brady-png/ParForce.git
cd ParForce

python -m pip install "huggingface_hub[cli]" pillow numpy opencv-python-headless \
  diffusers==0.40.0 transformers==5.14.1 peft==0.20.0 \
  safetensors==0.8.0 tokenizers==0.22.2
```

### 2. Download the base and adapters

| Model | Download | Inputs |
| --- | --- | --- |
| MiniMax-H3 FL2VA | [Upstream base](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/main/FL2VA) | Shared base for both commands |
| ParForce H3 Teacher | [Teacher adapter](https://huggingface.co/BRADY99/ParForce/blob/main/conditioned_models/parforce_h3_conditioned.safetensors) · [Union 2.0](https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union-2.0) | First frame, prompt, particles, depth |
| ParForce H3 Student | [Student adapter](https://huggingface.co/BRADY99/ParForce/blob/main/parforce_h3_student.safetensors) | First frame and prompt |

```bash
hf download MiniMaxAI/MiniMax-H3 --include 'FL2VA/*' --local-dir weights/minimax-h3
hf download BRADY99/ParForce \
  conditioned_models/parforce_h3_conditioned.safetensors \
  parforce_h3_student.safetensors --local-dir weights/parforce
hf download alibaba-pai/MiniMax-H3-Fun-Controlnet-Union-2.0 \
  MiniMax-H3-Fun-Controlnet-Union-2.0.safetensors --local-dir weights/union
```

### 3. Run Teacher inference

```bash
python infer.py teacher --base weights/minimax-h3/FL2VA --runtime runtime \
  --adapter weights/parforce/conditioned_models/parforce_h3_conditioned.safetensors \
  --control weights/union/MiniMax-H3-Fun-Controlnet-Union-2.0.safetensors \
  --first-frame examples/teacher/first_frame.png --prompt-file examples/teacher/prompt.txt \
  --particle examples/teacher/particles.mp4 --depth examples/teacher/depth.mp4 \
  --work-dir runs/teacher --devices 4 --seed 42
```

Output is `runs/teacher/outputs/result.mp4`. Keep the first frame and both control videos registered to the same camera view.

### 4. Run four-step Student inference

The student takes only the first frame and the official prompt for **Roll into box**. [Watch the archived comparison](https://brady-png.github.io/parforce-project/#student-comparisons/roll-into-box).

```bash
case_dir=examples/student/roll-into-box
python infer.py student --base weights/minimax-h3/FL2VA --runtime runtime \
  --adapter weights/parforce/parforce_h3_student.safetensors \
  --first-frame "$case_dir/first_frame.png" --prompt-file "$case_dir/prompt.txt" \
  --work-dir runs/roll-into-box --devices 4 --seed 42150
```

Output is `runs/roll-into-box/outputs/student/result.mp4`. The quick start uses a 90-frame clip at 24 fps. The archived benchmark run used its own duration and output crop, recorded in the [example manifest](examples/student/roll-into-box/manifest.json). Both launchers write inference receipts beside the video. Use a fresh work directory per run.

## Examples

| Example | What is provided |
| --- | --- |
| [Teacher showcase inputs](examples/showcase/) | Simple dominoes, curtain, milk pour, and red wine pour. Each has its original first frame, prompt, particles, and depth. [Input hashes and timings](examples/showcase/manifest.json) |
| [Roll into box](examples/student/roll-into-box/) | First frame and official benchmark prompt for view 0151 |
| [Wan2.1-Fun Sim-to-Real](https://huggingface.co/BRADY99/ParForce/tree/main/examples/sim2real) | Water pouring and floating lantern, with original inputs and reproduction settings |

## Condition rendering

With a prepared manifest containing masks, camera-space depth, and action fields, run the rigid-body renderer below. It uses MuJoCo, OpenCV, NumPy, SciPy, trimesh, Numba, and FFmpeg with libx264.

```bash
python -m parforce.conditions.run --manifest /path/to/cases.json --output runs/conditions
```

## Model terms

The upstream models and runtime retain their own terms. [MiniMax-H3 license](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE) · [Union 2.0 license](https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union-2.0/blob/main/LICENSE) · [VideoX-Fun license](runtime/licenses/VideoX-Fun-LICENSE.txt)
