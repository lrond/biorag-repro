from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from biorag.io import read_json, write_json
from biorag.reporting import build_report_comparison, write_report_comparison


class ReportingTests(unittest.TestCase):
    def test_build_report_comparison_formats_report_style_gain(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline = Path(tmpdir) / "baseline.json"
            ours = Path(tmpdir) / "ours.json"
            write_json(
                baseline,
                {
                    "metrics": {
                        "yesno_accuracy": 0.6364,
                        "factoid_strict_accuracy": 0.3869,
                        "list_recall": 0.3242,
                        "bert_score": 0.8643,
                        "average_latency_seconds": 2.60,
                    }
                },
            )
            write_json(
                ours,
                {
                    "metrics": {
                        "yesno_accuracy": 0.7692,
                        "factoid_strict_accuracy": 0.4526,
                        "list_recall": 0.4133,
                        "bert_score": 0.8679,
                        "average_latency_seconds": 3.31,
                    }
                },
            )
            rows, markdown = build_report_comparison(baseline, ours)
            self.assertEqual(rows[0]["gain"], "+20.9%")
            self.assertIn("| Yes/No | Accuracy | 0.6364 | 0.7692 | +20.9% |", markdown)
            self.assertIn("| Overall | Latency (s) | 2.6000 | 3.3100 | +0.71s |", markdown)

    def test_write_report_comparison_creates_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline = Path(tmpdir) / "baseline.json"
            ours = Path(tmpdir) / "ours.json"
            write_json(baseline, {"metrics": {"yesno_accuracy": 0.5}})
            write_json(ours, {"metrics": {"yesno_accuracy": 0.75}})
            artifacts = write_report_comparison(baseline, ours, Path(tmpdir) / "comparison")
            payload = read_json(artifacts["json"])
            self.assertTrue(Path(artifacts["markdown"]).exists())
            self.assertEqual(payload["rows"][0]["gain"], "+50.0%")


if __name__ == "__main__":
    unittest.main()
