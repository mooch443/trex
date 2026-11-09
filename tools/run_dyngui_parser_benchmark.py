#!/usr/bin/env python3
"""Run parser and string-decoder comparisons, then save CSV summaries and plots/.

Decoder results live in string_decoding/ beside the existing parser results.
Use --section strings to measure only the pre-unification/current decoders.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import subprocess

from plot_dyngui_parser_benchmark import DECODERS, PARSERS, plot, read_results


def run_string_decoders(executable, output_dir, configuration):
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = output_dir / "samples.csv"
    stderr = output_dir / "benchmark.stderr"
    with raw.open("w") as output, stderr.open("w") as errors:
        result = subprocess.run([str(executable.resolve()), "--strings-only"], stdout=output, stderr=errors)
    if result.returncode:
        raise ValueError(f"String-decoder benchmark failed ({result.returncode}); see {stderr}")

    versions = {"previous": "old", "current": "new"}
    samples = {}
    details = {}
    with raw.open(newline="") as source:
        for row in csv.DictReader(source):
            version = versions[row["version"]]
            key = (row["parser"], row["group"], row["case"])
            rounds = samples.setdefault(key, {}).setdefault(version, {})
            round_number = int(row["round"])
            timing = float(row["ns_per_parse"])
            if round_number in rounds or not math.isfinite(timing) or timing <= 0:
                raise ValueError(f"Duplicate round or invalid timing: {key}, {version}, {round_number}")
            rounds[round_number] = timing
            detail = tuple(row[field] for field in ("iterations", "input_bytes", "output_bytes", "buffer", "checksum"))
            if details.setdefault((key, version), detail) != detail:
                raise ValueError(f"Decoder workload changed across rounds: {key}, {version}")
    if not samples:
        raise ValueError("String-decoder benchmark produced no samples")

    repetitions = 10
    source = output_dir / "comparison.csv"
    with source.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["parser", "group", "case"] + [f"{version}_{suffix}" for version in versions.values()
                        for suffix in ("ns", "min_ns", "max_ns", "prepared_parts")]
                        + ["change_percent", "iterations", "input_bytes", "old_output_bytes", "new_output_bytes", "buffer"])
        for key, values in sorted(samples.items()):
            row = list(key)
            medians = {}
            for version in versions.values():
                if set(values.get(version, {})) != set(range(repetitions)):
                    raise ValueError(f"Missing decoder rounds: {key}, {version}")
                timings = list(values[version].values())
                medians[version] = statistics.median(timings)
                row.extend([medians[version], min(timings), max(timings), 0])
            old, new = details[(key, "old")], details[(key, "new")]
            if old[:2] != new[:2] or old[3] != new[3] or int(old[0]) <= 0:
                raise ValueError(f"Versions measured different workloads: {key}")
            if key[1] == "equivalent" and (old[2] != new[2] or old[4] != new[4]):
                raise ValueError(f"Equivalent-output checksums differ: {key}")
            row.extend([(medians["new"] / medians["old"] - 1) * 100, old[0], old[1], old[2], new[2], old[3]])
            writer.writerow(row)
    metadata = {
        "benchmark": "string_decoding",
        "primary_results": "comparison.csv",
        "previous_source_blob": "dae330fcbb6a3e459ccc3c635ad83de216bcdfab",
        "old_label": "Pre-unification (staged blob dae330f)",
        "current_label": f"Unified decoder ({configuration})",
        "comparison_label": "Δ vs previous",
        "snapshot_note": "Frozen staged fast_fromstr/fast_fromstr_N bodies in a separate translation unit; current calls production helpers. Benchmark translation units disable LTO.",
        "method": {
            "repetitions_per_version": repetitions,
            "target_sample_ms": 5,
            "maximum_iterations_per_sample": 2000000,
            "iteration_counts": "Calibrated per case using the faster version; identical for both versions and all rounds. Counts are in comparison.csv.",
            "order": "Cases shuffled with seed 1337 each round; each version runs first in five of ten paired rounds.",
            "buffers": "fresh includes allocation/destruction; reuse reserves input size before timing and clears between calls.",
            "validation": "Each decoder is checked against its version-specific expected output before timing. Changed JSON/Unicode semantics are plotted separately.",
            "scope": "Decoder calls only, not whole-parser or application throughput. Prefix limits count decoded UTF-8 bytes."
        }
    }
    (output_dir / "methodology.json").write_text(json.dumps(metadata, indent=2) + "\n")
    plot(read_results(source, DECODERS), source, output_dir / "plots" / "comparison", metadata, DECODERS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--configuration", default="unspecified")
    parser.add_argument("--section", choices=("all", "parsers", "strings"), default="all")
    parser.add_argument("--current-only", action="store_true", help="Run only current parser implementations")
    parser.add_argument("--case", action="append", default=[], help="Parser case to measure; repeat to select several")
    parser.add_argument("--parser", dest="parsers", action="append", choices=PARSERS, default=[],
                        help="Parser path to measure; repeat to select several")
    parser.add_argument("--iterations", type=int, help="Timed calls per parser case in each of six rounds")
    args = parser.parse_args()
    if args.iterations is not None and args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.current_only or args.case or args.parsers or args.iterations is not None:
        if args.section == "strings":
            parser.error("Parser filters cannot be combined with --section strings")
        args.section = "parsers"
    try:
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.section in ("all", "strings"):
            run_string_decoders(args.executable, output_dir / "string_decoding", args.configuration)
        if args.section == "strings":
            return
        raw = output_dir / "samples.csv"
        stderr = output_dir / "benchmark.stderr"
        command = [str(args.executable.resolve()), "--parser-only"]
        if args.current_only:
            command.append("--current-only")
        for name in args.case:
            command.extend(["--case", name])
        for name in args.parsers:
            command.extend(["--parser", name])
        if args.iterations is not None:
            command.extend(["--iterations", str(args.iterations)])
        with raw.open("w") as output, stderr.open("w") as errors:
            result = subprocess.run(command, stdout=output, stderr=errors)
        if result.returncode:
            raise ValueError(f"Benchmark failed ({result.returncode}); see {stderr}")

        versions = {"current": "new"} if args.current_only else {"older": "older", "head": "old", "current": "new"}
        samples = {}
        parts = {}
        with raw.open(newline="") as source:
            rows = list(csv.DictReader(source))
        for row in rows:
            version = versions[row["version"]]
            if row["parser"] == "PrepareAndRealizeStringPattern":
                row["parser"] = "SingleUsePattern"
            key = (row["parser"], row["group"], row["case"])
            samples.setdefault(key, {}).setdefault(version, []).append(float(row["ns_per_parse"]))
            count = int(row["prepared_parts"])
            if parts.setdefault((key, version), count) != count:
                raise ValueError(f"Prepared segment count changed across runs: {key}, {version}")
        repetitions = len(next(iter(samples.values()))["new"]) if samples else 0
        if not repetitions:
            raise ValueError("Benchmark produced no samples")
        source = output_dir / "comparison.csv"
        with source.open("w", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(["parser", "group", "case"] + [f"{version}_{suffix}" for version in versions.values()
                            for suffix in ("ns", "min_ns", "max_ns", "prepared_parts")])
            for key, values in samples.items():
                row = list(key)
                for version in versions.values():
                    if len(values[version]) != repetitions:
                        raise ValueError(f"Missing samples: {key}, {version}")
                    row.extend([statistics.median(values[version]), min(values[version]), max(values[version]), parts[(key, version)]])
                writer.writerow(row)
        metadata = {
            "primary_results": "comparison.csv",
            "command": command,
            "selected_cases": args.case,
            "selected_parsers": args.parsers or list(PARSERS),
            "old_revision": "d02acf307f5dca62c1b63368c907ac24ff32c858",
            "old_label": "HEAD snapshot (d02acf3)",
            "older_revision": "e936e32a95169db7df18a6a0bf7af79d9cfae029",
            "older_label": "Oct 17 snapshot (e936e32)",
            "current_label": f"Current ({args.configuration})",
            "snapshot_note": "Historical parsers include their string helpers and measured built-ins; Context/State data types are shared. Terminal logging is quiet; historical escape-warning formatting remains timed.",
            "method": {
                "repetitions_per_version": repetitions,
                "timed_iterations_per_case": int(rows[0]["iterations"]),
                "parse_text": "Calls the current direct parser, including scanning and decoding, on every iteration.",
                "RealizeStringPattern": "Prepares once outside timing; times realization only.",
                "SingleUsePattern": "Prepares, realizes, and destroys the pattern on every iteration; includes decoded-literal ownership costs.",
                "cache": "Clears handler variable values before every call in all three paths.",
                "validation": "Checks each path against version-specific expected output before timing; quote/escape changes are plotted separately."
            }
        }
        if args.current_only:
            for key in ("old_revision", "old_label", "older_revision", "older_label"):
                metadata.pop(key)
            metadata["snapshot_note"] = "Only current parser implementations were executed; no historical comparison is implied."
        (output_dir / "methodology.json").write_text(json.dumps(metadata, indent=2) + "\n")
        parsers = tuple(name for name in PARSERS if not args.parsers or name in args.parsers)
        plot(read_results(source, parsers), source, output_dir / "plots" / "comparison", metadata, parsers)
    except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
        parser.exit(1, f"Cannot run benchmark: {error}\n")


if __name__ == "__main__":
    main()
