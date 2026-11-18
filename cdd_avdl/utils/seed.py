"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np

__all__ = ["set_seed", "worker_init_fn"]


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed python, numpy and torch and optionally force deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    """Give every dataloader worker a distinct but reproducible seed."""
    base = int(os.environ.get("CDD_AVDL_SEED", "42"))
    seed = base + worker_id
    random.seed(seed)
    np.random.seed(seed)
