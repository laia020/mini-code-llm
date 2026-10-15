"""Launcher regressions: reuse, failed setup and preservation of existing environments."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import iniciar


class LauncherTests(unittest.TestCase):
    def fixture(self, folder):
        root = Path(folder) / "project with spaces"
        (root / ".venv").mkdir(parents=True)
        (root / "requirements.txt").write_text("torch>=2.5\n", encoding="utf-8")
        return root, iniciar.environment_python(root / ".venv")

    def test_cached_setup_does_not_run_pip(self):
        with tempfile.TemporaryDirectory() as folder:
            root, python = self.fixture(folder)
            versions = {name: "1.0" for name in iniciar.PACKAGES}
            stamp = {"requirements_sha256": hashlib.sha256((root / "requirements.txt").read_bytes()).hexdigest(), "versions": versions}
            (root / ".venv/mini-code-llm-setup.json").write_text(json.dumps(stamp), encoding="utf-8")
            with patch.object(iniciar, "installed_versions", return_value=versions), patch.object(iniciar.subprocess, "run") as run:
                iniciar.ensure_dependencies(root, python)
            run.assert_not_called()

    def test_new_windows_linux_setup_uses_cpu_wheels(self):
        with tempfile.TemporaryDirectory() as folder:
            root, python = self.fixture(folder)
            with patch.object(iniciar.sys, "platform", "win32"), patch.object(iniciar, "installed_versions", side_effect=[None, {"torch": "2.14.1"}]), patch.object(iniciar, "compatible_torch", return_value=False), patch.object(iniciar.subprocess, "run") as run:
                iniciar.ensure_dependencies(root, python)
            commands = [call.args[0] for call in run.call_args_list]
            self.assertIn("https://download.pytorch.org/whl/cpu", commands[0])
            self.assertTrue((root / ".venv/mini-code-llm-setup.json").is_file())

    def test_existing_compatible_cuda_install_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            root, python = self.fixture(folder)
            versions = {"torch": "2.13.0+cu130"}
            with patch.object(iniciar, "installed_versions", return_value=versions), patch.object(iniciar, "compatible_torch", return_value=True), patch.object(iniciar.subprocess, "run") as run:
                iniciar.ensure_dependencies(root, python)
            self.assertFalse(any("https://download.pytorch.org/whl/cpu" in call.args[0] for call in run.call_args_list))

    def test_failed_install_is_not_marked_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            root, python = self.fixture(folder)
            with patch.object(iniciar, "installed_versions", return_value=None), patch.object(iniciar, "compatible_torch", return_value=True), patch.object(iniciar.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "pip")):
                with self.assertRaises(subprocess.CalledProcessError):
                    iniciar.ensure_dependencies(root, python)
            self.assertFalse((root / ".venv/mini-code-llm-setup.json").exists())

    def test_incomplete_existing_environment_is_not_deleted(self):
        with tempfile.TemporaryDirectory() as folder:
            root, _ = self.fixture(folder)
            original = root / ".venv/user-file.txt"
            original.write_text("preserve")
            with self.assertRaises(RuntimeError):
                iniciar.ensure_environment(root)
            self.assertEqual(original.read_text(), "preserve")

    def test_browser_opens_by_default(self):
        with patch.object(iniciar, "ensure_environment", return_value=Path(sys.executable)), patch.object(iniciar, "ensure_dependencies"), patch.object(iniciar.subprocess, "call", return_value=0) as call:
            self.assertEqual(iniciar.main([]), 0)
        self.assertIn("--open-browser", call.call_args.args[0])

    def test_no_browser_flag_supports_server_checks(self):
        with patch.object(iniciar, "ensure_environment", return_value=Path(sys.executable)), patch.object(iniciar, "ensure_dependencies"), patch.object(iniciar.subprocess, "call", return_value=0) as call:
            self.assertEqual(iniciar.main(["--no-browser", "--port", "7890"]), 0)
        self.assertNotIn("--open-browser", call.call_args.args[0])
        self.assertEqual(call.call_args.args[0][-1], "7890")


if __name__ == "__main__":
    unittest.main()
