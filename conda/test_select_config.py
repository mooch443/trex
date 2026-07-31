#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import unittest

import select_config


class SelectConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.recipe_dir = Path(self.temporary_directory.name)
        source_dir = Path(__file__).resolve().parent
        for filename in select_config.PROFILE_FILES.values():
            shutil.copy2(source_dir / filename, self.recipe_dir / filename)
        shutil.copy2(
            self.recipe_dir / select_config.PROFILE_FILES["buildall"],
            self.recipe_dir / "meta.yaml",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_switches_and_restores_profiles(self) -> None:
        self.assertEqual(select_config.recipe_state(self.recipe_dir), "buildall")
        select_config.select_profile(self.recipe_dir, "minimal")
        self.assertEqual(select_config.recipe_state(self.recipe_dir), "minimal")
        select_config.select_profile(self.recipe_dir, "buildall")
        self.assertEqual(select_config.recipe_state(self.recipe_dir), "buildall")

    def test_modified_recipe_requires_force(self) -> None:
        active = self.recipe_dir / "meta.yaml"
        active.write_text(active.read_text(encoding="utf-8") + "# local edit\n")
        self.assertEqual(select_config.recipe_state(self.recipe_dir), "modified")
        with self.assertRaises(RuntimeError):
            select_config.select_profile(self.recipe_dir, "minimal")
        select_config.select_profile(self.recipe_dir, "minimal", force=True)
        self.assertEqual(select_config.recipe_state(self.recipe_dir), "minimal")

    def test_repository_default_is_exact_buildall_recipe(self) -> None:
        source_dir = Path(__file__).resolve().parent
        self.assertEqual(
            (source_dir / "meta.yaml").read_bytes(),
            (source_dir / select_config.PROFILE_FILES["buildall"]).read_bytes(),
        )


if __name__ == "__main__":
    unittest.main()
