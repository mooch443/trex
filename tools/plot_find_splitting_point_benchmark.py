#!/usr/bin/env python3
"""Plot raw samples from benchmark_find_splitting_point without rerunning it.

From the repository root:
    KMP_DUPLICATE_LIB_OK=TRUE conda run -n trex python tools/plot_find_splitting_point_benchmark.py Application/build/benchmarks/find_splitting_point_benchmark.csv

Writes comparison and timing figures as PNG/SVG under benchmark_results/.
Use --output to select a different filename prefix. The CSV is read only.
"""

import argparse
import csv
import math
import os
from pathlib import Path
from statistics import median


def read_results(path):
    groups = {}
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        current = "new_commons_ns" in (reader.fieldnames or [])
        columns = ("old_local_ns", "new_commons_ns") if current else ("commons_ns", "local_ns")
        names = ("old local", "new commons") if current else ("old commons", "new local")
        for row in reader:
            key = (row["case"], int(row["bytes"]), int(row["width_percent"]))
            metadata = (int(row["inputs"]), int(row["calls"]))
            sample = int(row["sample"])
            values = tuple(float(row[column]) for column in columns)
            if key[1] <= 0 or not 0 < key[2] <= 100 or min(metadata) <= 0 or sample < 0:
                raise ValueError(f"Invalid measurement metadata: {key}")
            if not all(math.isfinite(value) and value > 0 for value in values):
                raise ValueError(f"Invalid timing: {key}, sample {sample}")
            operation = row.get("operation", "find_splitting_point")
            group = groups.setdefault(key, {"metadata": metadata, "samples": {}, "names": names,
                                             "operation": operation})
            if group["metadata"] != metadata or sample in group["samples"]:
                raise ValueError(f"Inconsistent metadata or duplicate sample: {key}")
            if group["operation"] != operation:
                raise ValueError(f"Mixed benchmark operations: {key}")
            group["samples"][sample] = values
    if not groups:
        raise ValueError("CSV contains no samples")
    if len({group["operation"] for group in groups.values()}) != 1:
        raise ValueError("CSV contains multiple benchmark operations")
    for group in groups.values():
        group["medians"] = tuple(median(v[i] for v in group["samples"].values()) for i in (0, 1))
    return groups


def plot(groups, source, prefix):
    prefix.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(prefix.parent / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
    order = ("words", "paths", "punctuation", "unbroken")
    cases = sorted({k[0] for k in groups}, key=lambda name: (order.index(name) if name in order else len(order), name))
    sizes = sorted({k[1] for k in groups})
    widths = sorted({k[2] for k in groups})
    rows = [(case, size) for case in cases for size in sizes]
    if any((case, size, width) not in groups for case, size in rows for width in widths):
        raise ValueError("CSV is incomplete: not every case/size/width combination is present")
    sample_counts = sorted({len(g["samples"]) for g in groups.values()})
    calls = sorted({g["metadata"][1] for g in groups.values()})
    provenance = f"{source.name} · {', '.join(map(str, sample_counts))} samples/case · {', '.join(map(str, calls))} calls/sample"
    medians = [[groups[(case, size, width)]["medians"] for width in widths] for case, size in rows]
    speedups = [[old / new for old, new in row] for row in medians]
    savings = [[(old - new) / 1000 for old, new in row] for row in medians]
    log_speedups = [[math.log2(value) for value in row] for row in speedups]
    old_total = sum(g["medians"][0] for g in groups.values()) / 1000
    new_total = sum(g["medians"][1] for g in groups.values()) / 1000
    names = next(iter(groups.values()))["names"]
    operation = next(iter(groups.values()))["operation"]
    old_name, new_name = names

    def save(fig, kind):
        for extension in ("png", "svg"):
            destination = Path(f"{prefix}_{kind}.{extension}")
            fig.savefig(destination, dpi=170, facecolor="white")
            print(f"Saved {destination.resolve()}")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, max(7, len(rows) * 0.43 + 2.7)), layout="constrained")
    fig.suptitle(f"{operation}: relative gain and absolute impact", fontsize=17, weight="bold")
    limits = (
        max(1, max(abs(v) for row in log_speedups for v in row)),
        max(0.001, max(abs(v) for row in savings for v in row)),
    )
    labels = [f"{width}%\n(already fits)" if width == 100 else f"{width}%" for width in widths]
    for panel, axis in enumerate(axes):
        data = log_speedups if panel == 0 else savings
        axis.imshow(data, cmap="RdBu", norm=TwoSlopeNorm(vmin=-limits[panel], vcenter=0, vmax=limits[panel]), aspect="auto")
        axis.set_xticks(range(len(widths)), labels)
        axis.set_yticks(range(len(rows)), [f"{case} · {size:,} B" for case, size in rows])
        axis.set_xlabel("Available width / full measured string width")
        axis.set_title(f"Speedup: {old_name} / {new_name}\n>1× favors {new_name}; color uses log₂ scale" if panel == 0
                       else f"Time saved per call (µs)\nPositive favors {new_name}; color uses linear scale", fontsize=12, pad=12)
        for row in range(len(rows)):
            for col in range(len(widths)):
                if panel == 0:
                    label = f"{speedups[row][col]:.2f}×"
                else:
                    value = savings[row][col]
                    label = f"{value:+.2f}" if abs(value) >= 0.01 else f"{value:+.4f}"
                axis.text(col, row, label, ha="center", va="center",
                          color="white" if abs(data[row][col]) > limits[panel] * 0.55 else "#17212b", fontsize=10)
        for row in range(len(sizes), len(rows), len(sizes)):
            axis.axhline(row - 0.5, color="white", linewidth=2)
        axis.spines[:].set_visible(False)
    fig.supxlabel(
        f"{provenance}\n"
        f"Sum of case medians: {old_name} {old_total:,.1f} µs → {new_name} {new_total:,.1f} µs ({old_total / new_total:.2f}×).\n"
        "Synthetic mix, one call per case. Tiny absolute differences can produce large ratios.\n"
        f"Blue favors {new_name}; red favors {old_name}. The two panels use different color scales.",
        fontsize=10,
    )
    save(fig, "comparison")

    colors = ("#64748b", "#007f91")
    fig, axes = plt.subplots(len(cases), len(sizes), squeeze=False,
                             figsize=(max(8, 4.2 * len(sizes)), max(5, 2.7 * len(cases) + 1.5)))
    for row, case in enumerate(cases):
        for col, size in enumerate(sizes):
            axis = axes[row, col]
            for method, (name, color) in enumerate(zip(names, colors)):
                values = [[v[method] / 1000 for v in groups[(case, size, w)]["samples"].values()] for w in widths]
                centers = [median(v) for v in values]
                positions = [i + (method - 0.5) * 0.12 for i in range(len(widths))]
                errors = ([m - min(v) for m, v in zip(centers, values)],
                          [max(v) - m for m, v in zip(centers, values)])
                axis.errorbar(positions, centers, yerr=errors, fmt="o-" if method == 0 else "s-",
                              color=color, capsize=3, markersize=4, linewidth=1.2)
                for position, samples in zip(positions, values):
                    axis.scatter([position] * len(samples), samples, color=color, s=12, alpha=0.3)
            axis.set_yscale("log")
            axis.set_xticks(range(len(widths)), [f"{w}%" for w in widths])
            axis.set_title(f"{case} · {size:,} bytes")
            axis.grid(axis="y", which="both", alpha=0.18)
            axis.spines[["top", "right"]].set_visible(False)
            if col == 0:
                axis.set_ylabel("µs / call (log scale)")
            if row == len(cases) - 1:
                axis.set_xlabel("Available width / full width")
    fig.suptitle(f"{operation}: measured time per call", fontsize=17, weight="bold", y=0.995)
    fig.legend([Line2D([], [], color=color, marker=marker) for color, marker in zip(colors, ("o", "s"))],
               names, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=2, frameon=False)
    fit_note = ("100% width measures the whole string and emits one segment without wrapping."
                if operation == "add_string" else
                "100% width returns immediately; its nanosecond-scale differences may be noise.")
    fig.text(0.5, 0.015, f"{provenance}\n"
             "Markers: medians. Whiskers: min–max, not confidence intervals. Faint dots: raw samples.\n"
             f"{fit_note}",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    save(fig, "timings")

    print(f"Sum of medians: {old_name} {old_total:.3f} us, {new_name} {new_total:.3f} us, speedup {old_total / new_total:.3f}x")
    for case in cases:
        ratios = [g["medians"][0] / g["medians"][1] for (name, size, width), g in groups.items() if name == case and width < 100]
        print(f"{case}, splitting cases: {min(ratios):.3f}x to {max(ratios):.3f}x")
    smallest = [g["medians"] for (name, size, width), g in groups.items() if size == sizes[0]]
    print(f"Largest {new_name} regression at {sizes[0]} bytes: {max(new - old for old, new in smallest) / 1000:.3f} us/call")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, help="Output filename prefix")
    args = parser.parse_args()
    try:
        plot(read_results(args.csv), args.csv,
             args.output if args.output is not None else Path("benchmark_results") / args.csv.stem)
    except (OSError, ValueError, KeyError, ImportError) as error:
        parser.exit(1, f"Cannot plot benchmark: {error}\n")


if __name__ == "__main__":
    main()
