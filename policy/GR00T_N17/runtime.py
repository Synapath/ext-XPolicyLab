from __future__ import annotations

from pathlib import Path
import random

import numpy as np


_DIRECT_CHECKPOINT_FILES = (
    "config.json",
    "model.safetensors.index.json",
    "processor_config.json",
)


def is_direct_checkpoint(path: str | Path) -> bool:
    root = Path(path)
    return root.is_dir() and all((root / name).is_file() for name in _DIRECT_CHECKPOINT_FILES)


def seed_inference(seed: int) -> None:
    """Reset all inference RNGs at every episode boundary."""

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
