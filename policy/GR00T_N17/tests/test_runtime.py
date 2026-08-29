from __future__ import annotations

import random
from pathlib import Path
import tempfile
import unittest

import numpy as np

from policy.GR00T_N17.runtime import is_direct_checkpoint, seed_inference


class RuntimeTest(unittest.TestCase):
    def test_direct_checkpoint_requires_all_inference_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (
                "config.json",
                "model.safetensors.index.json",
                "processor_config.json",
            ):
                (root / name).write_text("{}\n", encoding="utf-8")
            self.assertTrue(is_direct_checkpoint(root))
            (root / "processor_config.json").unlink()
            self.assertFalse(is_direct_checkpoint(root))

    def test_seed_inference_resets_python_and_numpy(self):
        seed_inference(17)
        first = (random.random(), np.random.random())
        seed_inference(17)
        second = (random.random(), np.random.random())
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
