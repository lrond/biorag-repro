from __future__ import annotations

import unittest

from biorag.cli import QUICKSTART_PROFILES, _default_quickstart_config


class CliTests(unittest.TestCase):
    def test_quickstart_profiles_point_to_existing_configs(self) -> None:
        self.assertEqual(set(QUICKSTART_PROFILES), {"baseline", "full"})
        for profile in QUICKSTART_PROFILES:
            self.assertTrue(_default_quickstart_config(profile).exists())


if __name__ == "__main__":
    unittest.main()
