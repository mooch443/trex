import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest


LAUNCHER = Path(sys.argv.pop(1)).resolve()


class LauncherTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="trex-launcher-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prefix = self.root / "prefix"
        self.app = self.prefix / "bin/TRex.app/Contents/MacOS/TRex"
        self.app.parent.mkdir(parents=True)
        self.app.write_text(
            '#!/bin/sh\n'
            'printf \'%s\\0\' "$@" > "$TREX_LAUNCHER_ARGUMENTS"\n'
            'if [ "$TREX_LAUNCHER_EXIT" = signal ]; then\n'
            '    kill -TERM "$$"\n'
            'fi\n'
            'exit "$TREX_LAUNCHER_EXIT"\n'
        )
        self.app.chmod(0o755)
        self.arguments = self.root / "arguments"

    def run_launcher(self, args, exit_code=0):
        return subprocess.run(
            [str(LAUNCHER), *args],
            env={
                **os.environ,
                "CONDA_PREFIX": str(self.prefix),
                "TREX_LAUNCHER_ARGUMENTS": str(self.arguments),
                "TREX_LAUNCHER_EXIT": str(exit_code),
            },
            capture_output=True,
            text=True,
            timeout=10,
        )

    def test_preserves_arguments_and_exit_status(self):
        args = [
            "-detect_classes", '{0:"fish",7:"other"}',
            "-i", "video folder/guppies video.mp4",
            "[0,3]", "", "single'quote", 'double"quote', "back\\slash",
            "$HOME;$(printf expanded)`printf expanded` *",
        ]
        for exit_code in (0, 37):
            with self.subTest(exit_code=exit_code):
                result = self.run_launcher(args, exit_code)
                self.assertEqual(result.returncode, exit_code, result.stdout + result.stderr)
                self.assertEqual(
                    self.arguments.read_bytes(),
                    b"".join(arg.encode() + b"\0" for arg in args),
                )

    def test_install_path_with_spaces_and_quotes(self):
        prefix = self.root / "prefix with spaces and 'quotes'"
        self.prefix.rename(prefix)
        self.prefix = prefix
        result = self.run_launcher(["-help"], 37)
        self.assertEqual(result.returncode, 37, result.stdout + result.stderr)
        self.assertEqual(self.arguments.read_bytes(), b"-help\0")

    def test_preserves_signal_termination(self):
        result = self.run_launcher(["-help"], "signal")
        self.assertEqual(result.returncode, -signal.SIGTERM, result.stdout + result.stderr)

    def test_missing_executable_reports_failure(self):
        self.app.unlink()
        result = self.run_launcher(["-help"])
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(str(self.app), result.stderr)


if __name__ == "__main__":
    unittest.main()
