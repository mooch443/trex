#!/usr/bin/env python3
"""Plot the CSV produced by benchmark_image_zero.cpp; requires matplotlib.

From the repository root:
    conda run -n trex python tools/plot_image_zero_benchmark.py
An optional CSV path and --output prefix support saved runs. Separate timing and
speedup figures are saved as PNG and SVG. Output files with that prefix are
replaced; the input CSV is never modified.
"""

import argparse
import csv
import math
from pathlib import Path


METHODS = ("old malloc+fill", "exact malloc+fill", "word calloc", "exact calloc", "reuse+fill")
COLORS = ("#64748b", "#e69f00", "#cc79a7", "#0072b2", "#009e73")


def read_results(path):
    results = {}
    shapes = []
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            shape = tuple(int(row[key]) for key in ("rows", "cols", "channels"))
            if any(value <= 0 for value in shape):
                raise ValueError("Image dimensions must be positive")
            if shape not in shapes:
                shapes.append(shape)
            blob_percent, hole_percent = int(row["blob_percent"]), int(row["hole_percent"])
            if not 1 <= blob_percent <= 100 or not 0 <= hole_percent <= 100:
                raise ValueError("Invalid footprint or hole percentage")
            key = (shape, blob_percent, hole_percent, row["method"])
            if key in results:
                raise ValueError(f"Duplicate result: {key}")
            if row["method"] not in METHODS:
                raise ValueError(f"Unknown method: {key}")
            values = {name: float(row[name]) for name in ("median_us", "min_us", "max_us")}
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError(f"Non-finite timing: {key}")
            if not 0 < values["min_us"] <= values["median_us"] <= values["max_us"]:
                raise ValueError(f"Invalid or zero timing for {key}; rerun with more iterations")
            values.update({name: int(row[name]) for name in (
                "bytes", "iterations", "batch", "samples", "footprint_pixels", "filled_pixels", "spans"
            )})
            if values["bytes"] != math.prod(shape) or any(
                values[name] <= 0 for name in ("iterations", "batch", "samples", "spans")
            ):
                raise ValueError(f"Invalid measurement metadata: {key}")
            if not 0 < values["filled_pixels"] <= values["footprint_pixels"] <= shape[0] * shape[1]:
                raise ValueError(f"Invalid blob pixel counts: {key}")
            results[key] = values
    if not results:
        raise ValueError("The CSV contains no measurements")
    footprints = sorted({key[1] for key in results})
    holes = sorted({key[2] for key in results})
    methods = [method for method in METHODS
               if method != "word calloc" or any(key[3] == method for key in results)]
    for shape in shapes:
        for footprint in footprints:
            for hole in holes:
                metadata = None
                for method in methods:
                    key = (shape, footprint, hole, method)
                    if key not in results:
                        raise ValueError(f"Missing result: {key}")
                    current = tuple(results[key][name] for name in (
                        "bytes", "iterations", "batch", "samples", "footprint_pixels", "filled_pixels", "spans"
                    ))
                    if metadata is not None and current != metadata:
                        raise ValueError(f"Inconsistent measurement metadata for {key}")
                    metadata = current
    return shapes, results


def plot(shapes, results, prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogFormatter

    footprints = sorted({key[1] for key in results})
    holes = sorted({key[2] for key in results})
    samples = sorted({value["samples"] for value in results.values()})
    methods = [(method, color) for method, color in zip(METHODS, COLORS)
               if any(key[3] == method for key in results)]
    for kind in ("timings", "speedups"):
        fig, axes = plt.subplots(len(shapes), len(holes), squeeze=False, sharey="row",
                                 figsize=(max(8, 5 * len(holes)), max(5, 2.8 * len(shapes) + 1.5)))
        for row, shape in enumerate(shapes):
            for col, hole in enumerate(holes):
                axis = axes[row, col]
                for method, color in methods:
                    values = [results[(shape, footprint, hole, method)] for footprint in footprints]
                    medians = [value["median_us"] for value in values]
                    if kind == "timings":
                        errors = ([value["median_us"] - value["min_us"] for value in values],
                                  [value["max_us"] - value["median_us"] for value in values])
                        axis.errorbar(footprints, medians, yerr=errors, color=color, label=method,
                                      marker="o", markersize=4, capsize=2, linewidth=1.3)
                    else:
                        speedups = [results[(shape, footprint, hole, METHODS[0])]["median_us"] / median
                                    for footprint, median in zip(footprints, medians)]
                        axis.plot(footprints, speedups, color=color, label=method,
                                  marker="o", markersize=4, linewidth=1.3)
                axis.set_yscale("log")
                axis.yaxis.set_major_formatter(LogFormatter())
                axis.yaxis.set_minor_formatter(LogFormatter())
                axis.set_xticks(footprints, [str(value) for value in footprints], fontsize=9)
                axis.set_xlabel("Blob footprint target (% of image)", fontsize=9)
                axis.grid(axis="y", which="both", alpha=0.2)
                axis.set_axisbelow(True)
                axis.spines[["top", "right"]].set_visible(False)
                if row == 0:
                    axis.set_title(f"Holes target: {hole}% of footprint", fontsize=11)
                if col == 0:
                    units = "Median µs/image (log)" if kind == "timings" else "Speedup over old (×, log)"
                    axis.set_ylabel("×".join(map(str, shape)) + f"\n{units}", fontsize=10)
                if kind == "speedups":
                    axis.axhline(1, color="#111827", linestyle="--", linewidth=0.8)

        handles, legend_labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.95),
                   ncol=(len(methods) + 1) // 2 if len(holes) == 1 else len(methods), frameon=False)
        comparison = "timings (lower is better)" if kind == "timings" else "speedups (above 1× is faster)"
        fig.suptitle(f"Full pixel pipeline: {comparison}", fontsize=16, fontweight="bold", y=0.99)
        fig.text(0.5, 0.02,
                 "Initialize → draw row spans → checksum every byte → release. Reuse excludes allocation/free.\n"
                 f"Medians of {', '.join(map(str, samples))} warmed samples. Timing whiskers: min/max, not confidence intervals.\n"
                 "Footprint/hole targets are rounded to pixels with a retained outer rim; actual counts are in the CSV.\n"
                 "Shape generation is untimed; checksum arithmetic is timed. Small differences may be noise.",
                 ha="center", va="bottom", fontsize=9, color="#475569")
        fig.tight_layout(rect=(0, 0.11 if len(shapes) == 1 else 0.075, 1, 0.85 if len(shapes) == 1 else 0.92),
                         h_pad=2)
        for extension in ("png", "svg"):
            destination = Path(f"{prefix}_{kind}.{extension}")
            fig.savefig(destination, dpi=180, facecolor="white")
            print(f"Saved {destination.resolve()}")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", type=Path, default=Path("image_zero_benchmark.csv"))
    parser.add_argument("--output", type=Path, help="Output filename prefix (default: CSV path without .csv)")
    args = parser.parse_args()
    try:
        shapes, results = read_results(args.csv)
        plot(shapes, results, args.output if args.output is not None else args.csv.with_suffix(""))
    except (OSError, ValueError, KeyError, ImportError) as error:
        parser.exit(1, f"Cannot plot benchmark: {error}\n")


if __name__ == "__main__":
    main()
