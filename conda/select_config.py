#!/usr/bin/env python3
"""Select the native-dependency profile used by the Conda recipe."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import stat
import sys
import tempfile


PROFILE_FILES = {
    "buildall": ".meta.buildall.yaml",
    "minimal": ".meta.minimal.yaml",
}


def recipe_state(recipe_dir: Path) -> str:
    active = recipe_dir / "meta.yaml"
    if not active.exists():
        return "missing"

    active_bytes = active.read_bytes()
    for profile, filename in PROFILE_FILES.items():
        if active_bytes == (recipe_dir / filename).read_bytes():
            return profile
    return "modified"


def select_profile(recipe_dir: Path, profile: str, *, force: bool = False) -> None:
    active = recipe_dir / "meta.yaml"
    source = recipe_dir / PROFILE_FILES[profile]
    current_state = recipe_state(recipe_dir)

    if current_state == "modified" and not force:
        raise RuntimeError(
            "meta.yaml does not match either canonical profile. Refusing to overwrite it; "
            "review the changes or rerun with --force."
        )

    source_bytes = source.read_bytes()
    if active.exists() and active.read_bytes() == source_bytes:
        print(f"Conda dependency profile is already '{profile}'.")
        return

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=".meta.yaml.", dir=recipe_dir, delete=False
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(source_bytes)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.chmod(temporary_name, stat.S_IMODE(source.stat().st_mode))
        os.replace(temporary_name, active)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)

    print(f"Selected Conda dependency profile '{profile}'.")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select or report the Conda native-dependency profile."
    )
    parser.add_argument("action", choices=(*PROFILE_FILES, "status"))
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an unrecognized modified meta.yaml",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    recipe_dir = Path(__file__).resolve().parent

    if args.action == "status":
        state = recipe_state(recipe_dir)
        print(f"Conda dependency profile: {state}")
        return 0 if state in PROFILE_FILES else 1

    try:
        select_profile(recipe_dir, args.action, force=args.force)
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
