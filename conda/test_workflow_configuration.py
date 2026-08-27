#!/usr/bin/env python3
"""Guard workflow runner labels and SDK paths against retired macOS images."""

from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
SUPPORTED_INTEL_MACOS_RUNNERS = {"macos-15-intel", "macos-26-intel"}
CONDA_RECIPES = (
    ROOT / "conda" / ".meta.minimal.yaml",
    ROOT / "conda" / ".meta.buildall.yaml",
    ROOT / "conda" / "meta.yaml",
)


class WorkflowConfigurationTests(unittest.TestCase):
    def test_retired_macos_13_runner_is_not_referenced(self) -> None:
        offenders = [
            path.name
            for path in WORKFLOWS.glob("*.yml")
            if "macos-13" in path.read_text(encoding="utf-8")
        ]

        self.assertEqual(offenders, [])

    def test_intel_build_uses_a_supported_runner(self) -> None:
        workflow = (WORKFLOWS / "cmake-macos-intel.yml").read_text(encoding="utf-8")

        self.assertTrue(
            any(label in workflow for label in SUPPORTED_INTEL_MACOS_RUNNERS),
            f"expected one of {sorted(SUPPORTED_INTEL_MACOS_RUNNERS)}",
        )

    def test_intel_build_uses_the_stable_sdk_symlink(self) -> None:
        workflow = (WORKFLOWS / "cmake-macos-intel.yml").read_text(encoding="utf-8")

        self.assertIn("Developer/SDKs/MacOSX.sdk", workflow)
        self.assertNotIn("Developer/SDKs/MacOSX14.2.sdk", workflow)

    def test_windows_build_preserves_the_miniforge_base_python_abi(self) -> None:
        workflow = (WORKFLOWS / "cmake-windows.yml").read_text(encoding="utf-8")
        install_step = workflow.split(
            "- name: Install packaging tools into base", maxsplit=1
        )[1].split("- name: Fix Meta", maxsplit=1)[0]

        self.assertNotIn("python=", install_step)
        self.assertIn("from rpds import HashTrieMap", workflow)
        self.assertLess(
            workflow.index("from rpds import HashTrieMap"),
            workflow.index("conda-build . --override-channels"),
        )

    def test_post_link_resolver_runs_in_the_four_native_targets(self) -> None:
        workflow = (WORKFLOWS / "post-link-simulation.yml").read_text(encoding="utf-8")

        for runner in (
            "ubuntu-24.04",
            "windows-2025-vs2026",
            "macos-26",
            "macos-15-intel",
        ):
            self.assertIn(runner, workflow)
        self.assertIn("workflow_dispatch:", workflow)
        self.assertIn("python conda/test_post_link_real_resolver.py", workflow)
        self.assertNotIn("real-offline-resolution:", workflow)

    def test_macos_recipes_use_pthread_openblas(self) -> None:
        requirement = "libopenblas * *pthreads* # [osx]"
        for recipe in CONDA_RECIPES:
            with self.subTest(recipe=recipe.name):
                contents = recipe.read_text(encoding="utf-8")
                self.assertEqual(contents.count(requirement), 2)
                self.assertNotIn("pin_compatible('openblas'", contents)


if __name__ == "__main__":
    unittest.main(verbosity=2)
