from __future__ import annotations

import json
import random
from pathlib import Path
import tempfile
import unittest

import numpy as np

from policy.GR00T_N17.runtime import (
    is_direct_checkpoint,
    processor_checkpoint_view,
    seed_inference,
)


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

    def test_processor_view_does_not_mutate_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = {
                "processor_kwargs": {"model_name": "/old/cosmos"},
                "other": "preserved",
            }
            config = root / "processor_config.json"
            config.write_text(json.dumps(original), encoding="utf-8")
            model_config = root / "config.json"
            model_config.write_text(
                json.dumps({"model_name": "nvidia/Cosmos-Reason2-2B"}),
                encoding="utf-8",
            )
            weights = root / "model.safetensors.index.json"
            weights.write_text("{}\n", encoding="utf-8")
            before = config.read_bytes()
            model_before = model_config.read_bytes()
            with processor_checkpoint_view(root, "/new/cosmos") as view:
                self.assertNotEqual(view, root)
                payload = json.loads(
                    (view / "processor_config.json").read_text(encoding="utf-8")
                )
                self.assertEqual(payload["processor_kwargs"]["model_name"], "/new/cosmos")
                model_payload = json.loads(
                    (view / "config.json").read_text(encoding="utf-8")
                )
                self.assertEqual(model_payload["model_name"], "/new/cosmos")
                self.assertEqual((view / weights.name).resolve(), weights.resolve())
                self.assertEqual(config.read_bytes(), before)
                self.assertEqual(model_config.read_bytes(), model_before)
            self.assertEqual(config.read_bytes(), before)
            self.assertEqual(model_config.read_bytes(), model_before)


if __name__ == "__main__":
    unittest.main()
