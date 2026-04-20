from __future__ import annotations

import unittest

from biorag.config import resolve_model_source


class ModelModeTests(unittest.TestCase):
    def test_pretrained_mode_uses_model_name(self) -> None:
        self.assertEqual(resolve_model_source("model/base"), "model/base")

    def test_finetuned_mode_prefers_checkpoint(self) -> None:
        self.assertEqual(
            resolve_model_source("model/base", checkpoint_path="/tmp/checkpoint"),
            "/tmp/checkpoint",
        )

    def test_runtime_override_has_highest_priority(self) -> None:
        self.assertEqual(
            resolve_model_source(
                "model/base",
                checkpoint_path="/tmp/checkpoint",
                override_path="/tmp/runtime-model",
            ),
            "/tmp/runtime-model",
        )


if __name__ == "__main__":
    unittest.main()
