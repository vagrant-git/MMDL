from __future__ import annotations

import os
import sys

import torch


def log_runtime_environment(expected_conda_env: str = "dl") -> None:
    current_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    torch_version = torch.__version__
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "cpu"

    print(
        f"[runtime] python={sys.executable} conda_env={current_env or 'unknown'} "
        f"torch={torch_version} cuda_available={cuda_available} device={device_name}",
        flush=True,
    )
    if current_env and current_env != expected_conda_env:
        print(
            f"[runtime][warning] Expected conda env `{expected_conda_env}`, but current env is `{current_env}`.",
            flush=True,
        )
    if not cuda_available:
        print(
            "[runtime][warning] CUDA is unavailable in the current Python environment. "
            "If this machine has a GPU, switch to the `dl` environment before running experiments.",
            flush=True,
        )
