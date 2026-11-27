"""Synchronized physical conditioning clips and globally balanced distributed sampling."""
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


def frame_indices(timestamps, fps, length):
    if length < 1 or fps <= 0:
        raise ValueError(f'Invalid source length/fps: {length}/{fps}')
    return np.minimum(np.floor(np.asarray(timestamps) * fps + 1e-6).astype(np.int64), length - 1)


def normalize_depth(depth, encoding='metric'):
    """Invalid depth is black. Valid metric depth is robustly normalized inverse depth."""
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    out = np.zeros_like(depth)
    if not valid.any():
        return out
    if encoding in ('normalized_inverse', 'inverse_normalized', 'normalized_inverse_depth', 'inverse_depth_normalized'):
        out[valid] = np.clip(depth[valid], 0, 1)
    elif encoding in ('inverse', 'inverse_depth'):
        vals = depth[valid]
        lo, hi = np.percentile(vals, [2, 98])
        out[valid] = np.clip((vals - lo) / max(float(hi - lo), 1e-8), 0, 1) if hi > lo else 1
    elif encoding in ('metric', 'depth', 'metric_depth', 'z', 'camera_z', 'metric_camera_z_m'):
        vals = 1.0 / depth[valid]
        lo, hi = np.percentile(vals, [2, 98])
        out[valid] = np.clip((vals - lo) / max(float(hi - lo), 1e-8), 0, 1) if hi > lo else 1
    else:
        raise ValueError(f'Unknown depth encoding: {encoding}')
    return out


def resize_crop(frames, height, width, nearest=False):
    import cv2
    h, w = frames.shape[1:3]
    scale = max(height / h, width / w)
    nh, nw = max(height, round(h * scale)), max(width, round(w * scale))
    y, x = (nh - height) // 2, (nw - width) // 2
    interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return np.stack([cv2.resize(f, (nw, nh), interpolation=interpolation)[y:y + height, x:x + width] for f in frames])


def read_video(path, timestamps, fallback_fps=24):
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open video: {path}')
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        fps = fps if math.isfinite(fps) and fps > 0 else fallback_fps
        indices = frame_indices(timestamps, fps, count)
        needed = set(indices.tolist())
        decoded = {}
        # Sequential decoding avoids inaccurate compressed-video frame seeking.
        for i in range(int(indices[-1]) + 1):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f'Corrupt/truncated video {path} at frame {i}/{count}')
            if i in needed:
                decoded[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return np.stack([decoded[int(i)] for i in indices])
    finally:
        cap.release()


class PhysicalVideoDataset(Dataset):
    def __init__(self, manifest_path, num_frames=90, height=384, width=640, target_fps=24):
        self.manifest_path = Path(manifest_path)
        with self.manifest_path.open() as f:
            self.records = [json.loads(line) for line in f if line.strip()]
        if num_frames < 5 or (num_frames - 5) % 17:
            raise ValueError('H3 clip length must be 17*n+5')
        if target_fps <= 0:
            raise ValueError('target_fps must be positive')
        self.num_frames, self.height, self.width, self.target_fps = num_frames, height, width, target_fps
        self.real_indices = [i for i, r in enumerate(self.records) if r['domain'] == 'real']
        self.sim_indices = [i for i, r in enumerate(self.records) if r['domain'] == 'sim']

    def __len__(self):
        return len(self.records)

    def _path(self, value):
        p = Path(value)
        return p if p.is_absolute() else self.manifest_path.parent / p

    def __getitem__(self, index):
        row = self.records[index]
        timestamps = np.arange(self.num_frames, dtype=np.float64) / self.target_fps
        fps = float(row.get('fps', 24))
        try:
            rgb = read_video(self._path(row['rgb_path']), timestamps, fps)
            has_particle = bool(row.get('particle_path'))
            particle = read_video(self._path(row['particle_path']), timestamps, fps) if has_particle else None
            # Both spellings denote already visualized near-to-far video
            # pixels. Do not pass MP4 files to np.load or normalize them twice.
            if row.get('depth_encoding') in ('normalized_inverse_depth_video', 'visualized_depth_video'):
                depth = read_video(self._path(row['depth_path']), timestamps, fps).astype(np.float32) / 255
            else:
                # np.load returns an ndarray/memmap for NPY, and an NpzFile for NPZ.
                # Detect the returned container rather than assuming every file is a ZIP archive.
                archive = np.load(self._path(row['depth_path']), allow_pickle=False, mmap_mode='r')
                depth_fps = float(row.get('depth_fps', fps))
                if isinstance(archive, np.lib.npyio.NpzFile):
                    with archive:
                        key = row.get('depth_key')
                        if not key:
                            keys = [k for k in archive.files if archive[k].ndim >= 3 and k not in ('rgb', 'particles')]
                            if len(keys) != 1:
                                raise ValueError(f'Ambiguous depth keys {keys}; set depth_key')
                            key = keys[0]
                        depth = archive[key]
                        if 'fps' in archive.files and 'depth_fps' not in row:
                            depth_fps = float(np.asarray(archive['fps']).reshape(-1)[0])
                else:
                    depth = archive
                if depth.ndim == 4 and depth.shape[-1] == 1:
                    depth = depth[..., 0]
                if depth.ndim != 3:
                    raise ValueError(f'Expected depth [T,H,W], got {depth.shape}')
                depth = normalize_depth(depth, row.get('depth_encoding', 'metric'))
                depth = depth[frame_indices(timestamps, depth_fps, len(depth))]
            def tensor(a, scale=255):
                a = resize_crop(a, self.height, self.width)
                if a.ndim == 3:
                    a = np.repeat(a[..., None], 3, axis=-1)
                return torch.from_numpy(np.ascontiguousarray(a.transpose(3, 0, 1, 2))).float().div_(scale)
            rgb = tensor(rgb)
            particle = tensor(particle) if has_particle else torch.zeros_like(rgb)
            return dict(rgb=rgb, particle=particle, depth=tensor(depth, 1), has_particle=has_particle,
                        prompt=row['prompt'], id=row['id'], domain=row['domain'], timestamps=torch.from_numpy(timestamps))
        except Exception as exc:
            raise RuntimeError(f'Sample {row.get("id", index)} failed: {exc}') from exc


class BalancedGlobalBatchSampler(Sampler):
    """Each iteration yields one local microbatch. A global update has exact 1:1 domains."""
    def __init__(self, real_indices, sim_indices, world_size=32, rank=0, accumulation_steps=4,
                 micro_batch_size=1, seed=0, num_steps=15625, start_step=0):
        self.domains = (np.asarray(real_indices, dtype=np.int64), np.asarray(sim_indices, dtype=np.int64))
        if any(len(x) == 0 for x in self.domains):
            raise ValueError('Both domains must contain samples')
        if not 0 <= rank < world_size or min(world_size, accumulation_steps, micro_batch_size) < 1:
            raise ValueError('Invalid distributed batch configuration')
        self.world_size, self.rank, self.accumulation_steps = world_size, rank, accumulation_steps
        self.micro_batch_size, self.seed = micro_batch_size, seed
        self.num_steps, self.start_step = num_steps, start_step
        self.global_batch_size = world_size * accumulation_steps * micro_batch_size
        if self.global_batch_size % 2:
            raise ValueError('Global batch size must be even')
        self._cache = {}

    def _domain_index(self, domain, position):
        items = self.domains[domain]
        epoch, offset = divmod(position, len(items))
        key = (domain, epoch)
        if key not in self._cache:
            # Keep memory bounded even over millions of samples.
            self._cache = {k: v for k, v in self._cache.items() if k[0] != domain}
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, domain, epoch]))
            self._cache[key] = rng.permutation(items)
        return int(self._cache[key][offset])

    def index_for(self, step, micro, rank=None, local=0):
        rank = self.rank if rank is None else rank
        if not (0 <= micro < self.accumulation_steps and 0 <= rank < self.world_size and 0 <= local < self.micro_batch_size):
            raise ValueError('Invalid microbatch coordinate')
        position = step * self.global_batch_size + (micro * self.world_size + rank) * self.micro_batch_size + local
        return self._domain_index(position % 2, position // 2)

    def __iter__(self):
        for step in range(self.start_step, self.num_steps):
            for micro in range(self.accumulation_steps):
                yield [self.index_for(step, micro, local=i) for i in range(self.micro_batch_size)]

    def __len__(self):
        return max(0, self.num_steps - self.start_step) * self.accumulation_steps
