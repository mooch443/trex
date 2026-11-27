#!/usr/bin/env python3
"""Plot historical/current parser measurements without rebuilding or rerunning them.

From the repository root:
  KMP_DUPLICATE_LIB_OK=TRUE conda run -n trex python tools/plot_dyngui_parser_benchmark.py Application/build/benchmarks/dyngui_parser/RelWithDebInfo/comparison.csv

Writes equivalent-output timings, changed-semantics timings and prepared-part
counts as PNG/SVG in a plots/ folder beside the CSV. Use --output to set another
filename prefix. Reads matching methodology.json to identify the revisions.
String-decoder comparisons are detected from their metadata; --strings also
selects that format for a standalone comparison.csv without metadata.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path


PARSERS = ("parse_text", "RealizeStringPattern", "SingleUsePattern")
DECODERS = ("fast_fromstr", "fast_fromstr_N_1", "fast_fromstr_N_16")
GROUPS = ("equivalent", "policy_change")
TITLES = {"equivalent": "Equivalent output", "policy_change": "Changed semantics"}


def read_results(path, parsers=None):
    results = {}
    required = {"parser", "group", "case", "new_ns", "new_min_ns", "new_max_ns", "new_prepared_parts"}
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing))}")
        versions = tuple(version for version in ("older", "old", "new") if f"{version}_ns" in reader.fieldnames)
        for version in versions:
            missing = {f"{version}_{suffix}" for suffix in ("ns", "min_ns", "max_ns", "prepared_parts")} - set(reader.fieldnames)
            if missing:
                raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing))}")
        for row in reader:
            if row["parser"] == "PrepareAndRealizeStringPattern":
                row["parser"] = "SingleUsePattern"
            key = (row["group"], row["parser"], row["case"])
            if key in results or key[0] not in GROUPS or key[1] not in (parsers or PARSERS) or not key[2]:
                raise ValueError(f"Duplicate or unsupported benchmark case: {key}")
            values = {}
            for version in versions:
                for suffix in ("ns", "min_ns", "max_ns"):
                    column = f"{version}_{suffix}"
                    values[column] = float(row[column])
                    if not math.isfinite(values[column]) or values[column] <= 0:
                        raise ValueError(f"Invalid timing for {key}: {column}")
                if not values[f"{version}_min_ns"] <= values[f"{version}_ns"] <= values[f"{version}_max_ns"]:
                    raise ValueError(f"Median outside min/max range for {key}: {version}")
                column = f"{version}_prepared_parts"
                values[column] = int(row[column])
                if values[column] < 0:
                    raise ValueError(f"Negative prepared-part count for {key}")
            results[key] = values
    if not results:
        raise ValueError("CSV contains no benchmark cases")
    if parsers is None:
        parsers = tuple(p for p in PARSERS if any(key[1] == p for key in results))
    for group in GROUPS:
        cases = [{case for g, p, case in results if g == group and p == parser} for parser in parsers]
        if any(case_set != cases[0] for case_set in cases[1:]):
            raise ValueError(f"The implementations have different case sets in {group}")
    return results


def read_methodology(source, explicit):
    candidates = [explicit] if explicit else [source.parent / "methodology.json", source.parent.parent / "methodology.json"]
    for candidate in candidates:
        if not candidate.is_file():
            if explicit:
                raise ValueError(f"Methodology file does not exist: {candidate}")
            continue
        metadata = json.loads(candidate.read_text())
        primary = metadata.get("primary_results")
        if primary and (candidate.parent / primary).resolve() == source.resolve():
            return metadata
        if explicit:
            raise ValueError("Methodology primary_results does not identify the input CSV")
    return {}


def plot(results, source, prefix, metadata, parsers=None):
    if parsers is None:
        parsers = tuple(p for p in PARSERS if any(key[1] == p for key in results))
    string_decoders = parsers == DECODERS
    prefix.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(prefix.parent.resolve() / ".matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(prefix.parent.resolve() / ".cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
    labels = (metadata.get("current_label", "Current edits" if metadata.get("old_revision") else "CSV new"),)
    colors = ("#008577",)
    markers = ("D",)
    versions = ("new",)
    if "old_ns" in next(iter(results.values())):
        versions = ("old", *versions)
        labels = (metadata.get("old_label", f"Repository HEAD ({metadata['old_revision'][:7]})" if metadata.get("old_revision") else "CSV old"), *labels)
        colors = ("#67768b", *colors)
        markers = ("o", *markers)
    if "older_ns" in next(iter(results.values())):
        versions = ("older", *versions)
        labels = (metadata.get("older_label", "CSV older"), *labels)
        colors = ("#bd7621", *colors)
        markers = ("^", *markers)
    legend = [Line2D([], [], color=color, marker=marker, linestyle="none", markersize=6)
              for color, marker in zip(colors, markers)]
    method = metadata.get("method", {})
    provenance = " vs ".join(labels)
    if method.get("repetitions_per_version") and method.get("timed_iterations_per_case"):
        provenance += f" · {method['repetitions_per_version']} runs × {method['timed_iterations_per_case']:,} calls/case"
    elif method.get("repetitions_per_version") and method.get("target_sample_ms"):
        provenance += f" · {method['repetitions_per_version']} paired runs · target ~{method['target_sample_ms']} ms/sample"
    if metadata.get("new_source_diff_sha256"):
        provenance += f" · current patch {metadata['new_source_diff_sha256'][:7]}"

    def save(fig, kind):
        for extension in ("png", "svg"):
            destination = Path(f"{prefix}_{kind}.{extension}")
            fig.savefig(destination, dpi=170, facecolor="white")
            print(f"Saved {destination.resolve()}")
        plt.close(fig)

    def style(axis, cases):
        axis.set_yticks(range(len(cases)), cases)
        axis.set_ylim(len(cases) - 0.5, -0.5)
        axis.grid(axis="x", which="major", alpha=0.18)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0, pad=8)

    for group in GROUPS:
        cases = [case for g, p, case in results if g == group and p == parsers[0]]
        if not cases:
            continue
        height = max(5.2, len(cases) * 0.43 + 2.6)
        fig, axes = plt.subplots(1, len(parsers), figsize=(8 * len(parsers), height), sharex=True, sharey=True, squeeze=False)
        values = [row for (g, _, _), row in results.items() if g == group]
        lower = min(row[f"{version}_min_ns"] for row in values for version in versions) / 1500
        upper = max(row[f"{version}_max_ns"] for row in values for version in versions) / 650
        for axis, parser in zip(axes[0], parsers):
            rows = [results[(group, parser, case)] for case in cases]
            for index, version in enumerate(versions):
                medians = [row[f"{version}_ns"] / 1000 for row in rows]
                errors = ([value - row[f"{version}_min_ns"] / 1000 for value, row in zip(medians, rows)],
                          [row[f"{version}_max_ns"] / 1000 - value for value, row in zip(medians, rows)])
                positions = [i + (index - (len(versions) - 1) / 2) * 0.23 for i in range(len(cases))]
                axis.errorbar(medians, positions, xerr=errors, fmt=markers[index], color=colors[index],
                              markersize=5, capsize=3, linewidth=1.3)
            if "old" in versions:
                for i, row in enumerate(rows):
                    change = (row["new_ns"] / row["old_ns"] - 1) * 100
                    axis.text(1.025, i, f"{change:+.1f}%", transform=axis.get_yaxis_transform(), va="center",
                              fontsize=10, color="#a33b36" if change > 0 else "#007365", fontfamily="DejaVu Sans Mono")
                axis.text(1.025, 1.035, metadata.get("comparison_label", "Δ vs HEAD" if metadata.get("old_revision") else "Δ vs old"),
                          transform=axis.transAxes, fontsize=10, color="#52606d")
            axis.set_xscale("log")
            axis.set_xlim(lower, upper)
            axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            axis.set_title(parser, fontsize=13, pad=18)
            axis.set_xlabel("Time per call (µs, log scale) · lower is faster", labelpad=10)
            style(axis, cases)
        subject = "String decoding" if string_decoders else "Dynamic text parsers"
        fig.suptitle(f"{subject} · {TITLES[group].lower()}", fontsize=18, weight="bold", y=1 - 0.20 / height)
        fig.text(0.5, 1 - 0.62 / height, provenance, ha="center", va="top", fontsize=10, color="#52606d")
        fig.legend(legend, labels, loc="upper center", bbox_to_anchor=(0.5, 1 - 0.92 / height), ncol=len(versions), frameon=False)
        note = ("Outputs match between versions. All panels use the same time scale."
                if group == "equivalent" else
                "Outputs differ because of quote/escape semantics; these are not equivalent-work comparisons.")
        if string_decoders:
            note = ("Outputs match between versions. All three panels use the same time scale."
                    if group == "equivalent" else
                    "Outputs differ: the new decoder interprets JSON control/Unicode escapes; these are not equivalent-work comparisons.")
        if len(versions) == 1:
            note = ("Single executable measured. Historical case groups are kept separate; no version comparison is implied.")
        detail = ("October 17 parsers use HEAD support code with JSON compatibility adaptations."
                  if metadata.get("older_build") else "No timings from intermediate edits are mixed into these plots.")
        detail = metadata.get("snapshot_note", detail)
        preparation_note = ("RealizeStringPattern excludes preparation; SingleUsePattern includes preparation, realization and destruction. "
                            if "SingleUsePattern" in parsers else
                            "Prepared patterns are constructed outside timing. ")
        footer = (note + "\nMarkers: medians; whiskers: min–max, not confidence intervals. "
                 "Δ time = (current / HEAD − 1) × 100%; positive means slower.\n"
                 + preparation_note + detail
                 if metadata.get("old_revision") else
                 note + "\nMarkers: medians; whiskers: min–max, not confidence intervals. "
                 + ("Δ time = (new / old − 1) × 100%; positive means slower." if "old" in versions
                    else preparation_note))
        if string_decoders:
            footer = (note + "\nMarkers: medians; whiskers: min–max. Δ time = (new / previous − 1) × 100%; positive means slower.\n"
                      "fresh includes string allocation/destruction; reuse retains capacity. Prefix limits are decoded UTF-8 bytes. "
                      "Identical per-case iteration counts; alternating version order.")
        fig.text(0.5, 0.14 / height, footer, ha="center", fontsize=9, linespacing=1.6)
        fig.subplots_adjust(left=0.20, right=0.90, bottom=1.45 / height, top=1 - 1.8 / height, wspace=0.28)
        save(fig, group)

    if string_decoders:
        return

    prepared_parser = next((name for name in PARSERS[1:] if name in parsers), None)
    if prepared_parser is None:
        return

    groups = [group for group in GROUPS if any(g == group for g, _, _ in results)]
    fig, axes = plt.subplots(1, len(groups), squeeze=False, figsize=(16, 7.7))
    for axis, group in zip(axes[0], groups):
        cases = [case for g, p, case in results if g == group and p == prepared_parser]
        rows = [results[(group, prepared_parser, case)] for case in cases]
        for index, version in enumerate(versions):
            axis.scatter([row[f"{version}_prepared_parts"] for row in rows],
                         [i + (index - (len(versions) - 1) / 2) * 0.23 for i in range(len(cases))],
                         marker=markers[index], color=colors[index], s=30, zorder=3)
        for i, row in enumerate(rows):
            axis.text(1.025, i, " / ".join(str(row[f"{version}_prepared_parts"]) for version in versions),
                      transform=axis.get_yaxis_transform(), va="center", fontsize=10)
        maximum = max(row[f"{version}_prepared_parts"] for row in rows for version in versions)
        axis.set_xscale("symlog", linthresh=1)
        axis.set_xlim(0, max(2, maximum * 1.5))
        axis.xaxis.set_major_locator(FixedLocator([tick for tick in (0, 1, 2, 5, 10, 50, 100, 200, 500, 1000) if tick <= maximum * 1.5]))
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        axis.set_title(TITLES[group], fontsize=13, pad=20)
        axis.set_xlabel("Prepared segments (log scale above 1)", labelpad=10)
        style(axis, cases)
    fig.suptitle("Prepared parser · stored segment counts", fontsize=18, weight="bold", y=0.97)
    fig.text(0.5, 0.92, "Counts in legend order: " + " / ".join(labels), ha="center", fontsize=10)
    fig.legend(legend, labels, loc="upper center", bbox_to_anchor=(0.5, 0.89), ncol=len(versions), frameon=False)
    fig.text(0.5, 0.035, "Counts describe the stored prepared pattern, not execution time. "
             "parse_text has no prepared segments and is omitted.\n"
             "Each panel shows its own case group; changed-semantics cases can do different work.",
             ha="center", fontsize=9, linespacing=1.6)
    fig.subplots_adjust(left=0.19, right=0.90, bottom=0.15, top=0.79, wspace=1.0)
    save(fig, "prepared_parts")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, help="Output filename prefix; defaults to plots/<CSV stem> beside the CSV")
    parser.add_argument("--metadata", type=Path, help="Matching methodology.json; otherwise discovered beside or above the CSV")
    parser.add_argument("--strings", action="store_true", help="Plot the previous/current string-decoder comparison")
    args = parser.parse_args()
    try:
        metadata = read_methodology(args.csv, args.metadata)
        parsers = DECODERS if args.strings or metadata.get("benchmark") == "string_decoding" else None
        plot(read_results(args.csv, parsers), args.csv, args.output or args.csv.parent / "plots" / args.csv.stem, metadata, parsers)
    except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
        parser.exit(1, f"Cannot plot benchmark: {error}\n")


if __name__ == "__main__":
    main()
