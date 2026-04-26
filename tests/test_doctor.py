from __future__ import annotations

import tempfile
import unittest

from biorag.doctor import collect_diagnostics, render_diagnostics
from tests.test_pipeline_contracts import make_toy_config


class DoctorTests(unittest.TestCase):
    def test_toy_config_has_no_required_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = make_toy_config(tmpdir)
            checks = collect_diagnostics(config)
            failures = [check for check in checks if check.status == "FAIL"]
            self.assertEqual(failures, [])

    def test_render_diagnostics_contains_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = make_toy_config(tmpdir)
            text = render_diagnostics(collect_diagnostics(config))
            self.assertIn("BioRAG doctor", text)
            self.assertIn("Summary:", text)


if __name__ == "__main__":
    unittest.main()
