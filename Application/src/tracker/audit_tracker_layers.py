#!/usr/bin/env python3

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "TrackerSources.cmake"

SET_INFO = {
    "TREX_CORE_PUBLIC_HEADERS": ("core", "public_header"),
    "TREX_CORE_PRIVATE_HEADERS": ("core", "private_header"),
    "TREX_CORE_SOURCES": ("core", "source"),
    "TREX_DATA_PUBLIC_HEADERS": ("data", "public_header"),
    "TREX_DATA_PRIVATE_HEADERS": ("data", "private_header"),
    "TREX_DATA_SOURCES": ("data", "source"),
    "TREX_TRACKING_PUBLIC_HEADERS": ("tracking", "public_header"),
    "TREX_TRACKING_PRIVATE_HEADERS": ("tracking", "private_header"),
    "TREX_TRACKING_SOURCES": ("tracking", "source"),
    "TREX_ML_PUBLIC_HEADERS": ("ml", "public_header"),
    "TREX_ML_PRIVATE_HEADERS": ("ml", "private_header"),
    "TREX_ML_SOURCES": ("ml", "source"),
    "TREX_UI_PUBLIC_HEADERS": ("ui", "public_header"),
    "TREX_UI_PRIVATE_HEADERS": ("ui", "private_header"),
    "TREX_UI_SOURCES": ("ui", "source"),
    "TRACKER_PYTHON_PUBLIC_HEADERS": ("python", "public_header"),
    "TRACKER_PYTHON_PRIVATE_HEADERS": ("python", "private_header"),
    "TRACKER_PYTHON_SOURCES": ("python", "source"),
}

INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]')
SET_RE = re.compile(r"set\((?P<name>[A-Z0-9_]+)\s*(?P<body>.*?)\)", re.S)


def parse_manifest():
    text = MANIFEST.read_text(encoding="utf-8")
    files = {}
    prefix = "${CMAKE_CURRENT_LIST_DIR}/"

    for match in SET_RE.finditer(text):
        set_name = match.group("name")
        info = SET_INFO.get(set_name)
        if not info:
            continue

        layer, kind = info
        for raw_line in match.group("body").splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line or not line.startswith(prefix):
                continue
            rel_path = line[len(prefix):]
            files[rel_path] = {"layer": layer, "kind": kind}

    return files


def include_prefix(include_path: str) -> str | None:
    if "/" not in include_path:
        return None
    return include_path.split("/", 1)[0]


def iter_violations(rel_path, info):
    path = ROOT / rel_path
    if not path.exists():
        yield f"{rel_path}: file listed in TrackerSources.cmake does not exist"
        return

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin-1")

    owner = info["layer"]
    kind = info["kind"]

    for lineno, line in enumerate(text.splitlines(), start=1):
        match = INCLUDE_RE.match(line)
        if not match:
            continue

        include_path = match.group(1)
        prefix = include_prefix(include_path)
        if not prefix:
            continue

        if owner == "core" and prefix in {"data", "tracking", "ml", "ui", "python"}:
            yield f"{rel_path}:{lineno}: core must not include {include_path}"

        if owner == "data" and prefix in {"tracking", "ml", "ui", "python"}:
            yield f"{rel_path}:{lineno}: data must not include {include_path}"

        if owner == "tracking" and prefix in {"ml", "ui", "python"}:
            yield f"{rel_path}:{lineno}: tracking must not include {include_path}"

        if owner == "ml" and kind == "public_header" and prefix == "python":
            yield f"{rel_path}:{lineno}: ml public headers must not include {include_path}"

        if owner == "python" and prefix == "ui":
            yield f"{rel_path}:{lineno}: python must not include {include_path}"


def main():
    files = parse_manifest()
    violations = []

    for rel_path, info in sorted(files.items()):
        violations.extend(iter_violations(rel_path, info))

    if violations:
        print("Tracker layer audit failed:")
        for violation in violations:
            print(f"  {violation}")
        return 1

    print("Tracker layer audit passed.")
    print("  enforced: core/data/detect/tracking boundaries, ml public-header Python ban, python->ui ban")
    return 0


if __name__ == "__main__":
    sys.exit(main())
