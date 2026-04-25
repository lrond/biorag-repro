from __future__ import annotations

import unittest
from types import SimpleNamespace

from biorag.config import resolve_model_source
from biorag.modeling import model_load_kwargs, resolve_torch_dtype


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

    def test_torch_dtype_aliases_are_resolved(self) -> None:
        fake_torch = SimpleNamespace(bfloat16="bf16", float16="fp16")
        self.assertEqual(resolve_torch_dtype("auto", fake_torch), "auto")
        self.assertEqual(resolve_torch_dtype("bfloat16", fake_torch), "bf16")
        self.assertEqual(resolve_torch_dtype("fp16", fake_torch), "fp16")
        self.assertIsNone(resolve_torch_dtype("default", fake_torch))

    def test_model_load_kwargs_include_memory_options(self) -> None:
        fake_torch = SimpleNamespace(bfloat16="bf16", float16="fp16")
        model_config = SimpleNamespace(
            torch_dtype="bf16",
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        self.assertEqual(
            model_load_kwargs(model_config, fake_torch),
            {
                "torch_dtype": "bf16",
                "low_cpu_mem_usage": True,
                "attn_implementation": "sdpa",
            },
        )


if __name__ == "__main__":
    unittest.main()
