import contextlib
import io
import json
from pathlib import Path
import runpy
import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET
import zipfile


class PackageTest(unittest.TestCase):
    def test_changed_packages_advance_internal_version_and_keep_one_vsix(self):
        source = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory(prefix="package-test-", dir=source / "test") as directory:
            root = Path(directory)
            for name in ["package.py", "package.json", "extension.js", "tokens.js", "fields.js", "README.md", "LICENSE"]:
                shutil.copyfile(source / name, root / name)
            major, minor, patch = map(int, json.loads((root / "package.json").read_text())["version"].split("."))
            for revision in [1, 2]:
                with (root / "extension.js").open("a") as extension:
                    extension.write(f"\n// Package test revision {revision}.\n")
                with contextlib.redirect_stdout(io.StringIO()):
                    runpy.run_path(str(root / "package.py"))
                expected = f"{major}.{minor}.{patch + revision}"
                self.assertEqual(json.loads((root / "package.json").read_text())["version"], expected)
                self.assertEqual([path.name for path in root.glob("*.vsix")], ["commons-layout.vsix"])
                with zipfile.ZipFile(root / "commons-layout.vsix") as archive:
                    self.assertEqual(json.loads(archive.read("extension/package.json"))["version"], expected)
                    identity = ET.fromstring(archive.read("extension.vsixmanifest")).find(".//{*}Identity")
                    self.assertEqual(identity.attrib["Version"], expected)
                    self.assertEqual(archive.read("extension/extension.js"), (root / "extension.js").read_bytes())


if __name__ == "__main__":
    unittest.main()
