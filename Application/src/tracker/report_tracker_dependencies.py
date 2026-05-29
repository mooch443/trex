#!/usr/bin/env python3

import argparse
import fnmatch
import re
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "TrackerSources.cmake"

LAYER_ORDER = {
    "trex.core": 0,
    "trex.data": 1,
    "trex.tracking": 2,
    "trex.ml": 3,
    "trex.ui": 4,
}

SET_LAYER = {
    "TREX_CORE_PUBLIC_HEADERS": ("trex.core", "public_header"),
    "TREX_CORE_PRIVATE_HEADERS": ("trex.core", "private_header"),
    "TREX_CORE_SOURCES": ("trex.core", "source"),
    "TREX_DATA_PUBLIC_HEADERS": ("trex.data", "public_header"),
    "TREX_DATA_PRIVATE_HEADERS": ("trex.data", "private_header"),
    "TREX_DATA_SOURCES": ("trex.data", "source"),
    "TREX_TRACKING_PUBLIC_HEADERS": ("trex.tracking", "public_header"),
    "TREX_TRACKING_PRIVATE_HEADERS": ("trex.tracking", "private_header"),
    "TREX_TRACKING_SOURCES": ("trex.tracking", "source"),
    "TREX_ML_PUBLIC_HEADERS": ("trex.ml", "public_header"),
    "TREX_ML_PRIVATE_HEADERS": ("trex.ml", "private_header"),
    "TREX_ML_SOURCES": ("trex.ml", "source"),
    "TREX_UI_PUBLIC_HEADERS": ("trex.ui", "public_header"),
    "TREX_UI_PRIVATE_HEADERS": ("trex.ui", "private_header"),
    "TREX_UI_SOURCES": ("trex.ui", "source"),
}

SET_RE = re.compile(r"set\((?P<name>[A-Z0-9_]+)\s*(?P<body>.*?)\)", re.S)
INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]')


def parse_manifest():
    text = MANIFEST.read_text(encoding="utf-8")
    file_info = {}

    for match in SET_RE.finditer(text):
        set_name = match.group("name")
        if set_name not in SET_LAYER:
            continue

        layer, kind = SET_LAYER[set_name]
        for line in match.group("body").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            line = line.split("#", 1)[0].strip()
            if not line:
                continue

            prefix = "${CMAKE_CURRENT_LIST_DIR}/"
            if not line.startswith(prefix):
                continue

            rel_path = line[len(prefix):]
            file_info[rel_path] = {
                "layer": layer,
                "kind": kind,
            }

    return file_info


def resolve_include(rel_path, include_path, file_info):
    if include_path in file_info:
        return include_path

    base_candidate = (Path(rel_path).parent / include_path).resolve().relative_to(ROOT)
    base_candidate_str = base_candidate.as_posix()
    if base_candidate_str in file_info:
        return base_candidate_str

    return None


def iter_connections(rel_path, file_info):
    owner = file_info[rel_path]["layer"]
    path = ROOT / rel_path
    if not path.exists():
        return

    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        match = INCLUDE_RE.match(line)
        if not match:
            continue

        include_path = match.group(1)
        try:
            resolved = resolve_include(rel_path, include_path, file_info)
        except ValueError:
            resolved = None

        if not resolved:
            continue

        target = file_info[resolved]["layer"]
        if target == owner:
            continue

        yield {
            "line": lineno,
            "include": include_path,
            "resolved": resolved,
            "target": target,
        }


def matches_filters(rel_path, info, args):
    if args.package and info["layer"] not in args.package:
        return False

    if args.glob and not any(fnmatch.fnmatch(rel_path, pattern) for pattern in args.glob):
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Report cross-package tracker include edges for manifest-owned headers and sources."
    )
    parser.add_argument(
        "--package",
        action="append",
        choices=sorted(LAYER_ORDER.keys(), key=LAYER_ORDER.get),
        help="Restrict output to files owned by the given package. May be repeated.",
    )
    parser.add_argument(
        "--glob",
        action="append",
        help="Restrict output to files matching the given glob relative to src/tracker. May be repeated.",
    )
    parser.add_argument(
        "--files",
        action="store_true",
        help="Show the exact include files and line numbers for each outgoing package edge.",
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Also print files that do not connect to any other tracker package.",
    )
    args = parser.parse_args()

    file_info = parse_manifest()
    grouped = []

    for rel_path in sorted(
        file_info,
        key=lambda path: (LAYER_ORDER[file_info[path]["layer"]], path),
    ):
        info = file_info[rel_path]
        if not matches_filters(rel_path, info, args):
            continue

        connections = list(iter_connections(rel_path, file_info))
        targets = sorted({connection["target"] for connection in connections}, key=LAYER_ORDER.get)

        if not targets and not args.include_empty:
            continue

        grouped.append((rel_path, info, targets, connections))

    if not grouped:
        print("No matching files.")
        return 0

    for rel_path, info, targets, connections in grouped:
        target_text = ", ".join(targets) if targets else "(none)"
        print(f"{rel_path} [{info['kind']} {info['layer']}] -> {target_text}")

        if args.files:
            by_target = defaultdict(list)
            for connection in connections:
                by_target[connection["target"]].append(connection)

            for target in sorted(by_target, key=LAYER_ORDER.get):
                for connection in sorted(by_target[target], key=lambda item: (item["line"], item["resolved"])):
                    print(
                        f"  {connection['line']}: {connection['include']} -> "
                        f"{connection['resolved']} [{target}]"
                    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
