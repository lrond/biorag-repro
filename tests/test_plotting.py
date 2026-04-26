from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from biorag.io import read_json, write_json
from biorag.plotting import _moving_average, write_training_curve


class PlottingTests(unittest.TestCase):
    def test_moving_average_uses_trailing_window(self) -> None:
        self.assertEqual(_moving_average([3.0, 1.0, 5.0], 2), [3.0, 2.0, 3.0])

    def test_write_training_curve_from_step_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = Path(tmpdir) / "training_metrics.json"
            write_json(
                metrics_path,
                {
                    "step_history": [
                        {"step": 1, "epoch": 1, "batch": 1, "optimizer_step": 1, "loss": 3.0},
                        {"step": 2, "epoch": 1, "batch": 2, "optimizer_step": 2, "loss": 1.0},
                        {"step": 3, "epoch": 2, "batch": 1, "optimizer_step": 3, "loss": 5.0},
                    ]
                },
            )

            artifacts = write_training_curve(
                metrics_path,
                Path(tmpdir) / "plots",
                smoothing_window=2,
                render_image=False,
            )

            self.assertTrue(artifacts["csv"].exists())
            self.assertTrue(artifacts["json"].exists())
            with artifacts["csv"].open("r", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[-1]["smoothed_loss"], "3.0")
            summary = read_json(artifacts["json"])
            self.assertEqual(summary["points"], 3)
            self.assertEqual(summary["final_loss"], 5.0)

    def test_write_training_curve_falls_back_to_epoch_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = Path(tmpdir) / "training_metrics.json"
            write_json(
                metrics_path,
                {
                    "history": [
                        {"epoch": 1, "loss": 2.0, "optimizer_steps": 10},
                        {"epoch": 2, "loss": 1.5, "optimizer_steps": 10},
                    ]
                },
            )

            artifacts = write_training_curve(
                metrics_path,
                Path(tmpdir) / "plots",
                render_image=False,
            )

            with artifacts["csv"].open("r", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["step"], "1")


if __name__ == "__main__":
    unittest.main()
