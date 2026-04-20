from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from biorag.io import load_jsonl, read_json
from biorag.pipeline import run_full_pipeline
from tests.test_pipeline_contracts import make_toy_config


class SmokeTests(unittest.TestCase):
    def test_full_toy_pipeline_generates_predictions_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = make_toy_config(tmpdir)
            artifacts = run_full_pipeline(config)
            predictions = load_jsonl(artifacts["predictions"])
            report = read_json(artifacts["report"])
            self.assertEqual(len(predictions), 4)
            self.assertIn("metrics", report)
            self.assertTrue(Path(artifacts["summary"]).exists())


if __name__ == "__main__":
    unittest.main()
