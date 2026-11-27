"""Minimal device abstraction for the local Ascend H3 ablation run."""

import os

import torch
import torch_npu  # noqa: F401 -- registers the ``npu`` device with PyTorch.


DEVICE_TYPE = "npu"
DIST_BACKEND = "hccl"


def local_device() -> torch.device:
    return torch.device(DEVICE_TYPE, int(os.environ["LOCAL_RANK"]))


def set_device(device: torch.device) -> None:
    torch.npu.set_device(device)


def manual_seed(seed: int) -> None:
    torch.npu.manual_seed(seed)


def get_rng_state() -> torch.Tensor:
    return torch.npu.get_rng_state()


def set_rng_state(state: torch.Tensor) -> None:
    torch.npu.set_rng_state(state)


def empty_cache() -> None:
    torch.npu.empty_cache()


def memory_allocated() -> int:
    return torch.npu.memory_allocated()


def max_memory_allocated() -> int:
    return torch.npu.max_memory_allocated()
