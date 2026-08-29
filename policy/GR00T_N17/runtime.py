from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import random
import tempfile
from typing import Iterator

import numpy as np


_DIRECT_CHECKPOINT_FILES = (
    "config.json",
    "model.safetensors.index.json",
    "processor_config.json",
)


def is_direct_checkpoint(path: str | Path) -> bool:
    root = Path(path)
    return root.is_dir() and all((root / name).is_file() for name in _DIRECT_CHECKPOINT_FILES)


@contextmanager
def processor_checkpoint_view(
    checkpoint_dir: str | Path, cosmos_model: str
) -> Iterator[Path]:
    """Expose a processor override without mutating a frozen checkpoint."""

    checkpoint_dir = Path(checkpoint_dir)
    config_path = checkpoint_dir / "processor_config.json"
    if not config_path.is_file():
        yield checkpoint_dir
        return
    data = json.loads(config_path.read_text(encoding="utf-8"))
    processor_kwargs = data.setdefault("processor_kwargs", {})
    if processor_kwargs.get("model_name") == cosmos_model:
        yield checkpoint_dir
        return
    processor_kwargs["model_name"] = cosmos_model
    with tempfile.TemporaryDirectory(prefix="groot-checkpoint-view-") as temporary:
        view = Path(temporary)
        for child in checkpoint_dir.iterdir():
            if child.name == config_path.name:
                continue
            (view / child.name).symlink_to(child, target_is_directory=child.is_dir())
        (view / config_path.name).write_text(
            json.dumps(data, indent=2) + "\n", encoding="utf-8"
        )
        yield view


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
