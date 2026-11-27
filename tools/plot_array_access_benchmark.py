"""Plot concept variants, complete accessors, and HEAD versus new full parsers.

From the repository root:
  KMP_DUPLICATE_LIB_OK=TRUE conda run -n trex python tools/plot_array_access_benchmark.py array_access.csv

Writes <input_stem>_gains.png and <input_stem>_summary.csv beside the input CSV.
The complete function comparisons include helper optimizations on the new side.
Concept variants share the original ICU trimming and measure lookup, not frame times.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median, quantiles

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()
    samples = defaultdict(dict)
    cases = defaultdict(list)
    with args.csv.open(newline="") as source:
        for row in csv.DictReader(source):
            case = (row["case"], int(row["index"]))
            if case not in cases[row["mode"]]:
                cases[row["mode"]].append(case)
            key = (row["mode"], case, row["variant"])
            samples[key][int(row["round"])] = float(row["ns_per_lookup"])
    if not cases:
        parser.error("No benchmark samples found")

    variants = [
        ("vector_string", "Full vector + string", "#C08430"),
        ("accessor_stack", "Accessor + stack", "#367DA2"),
        ("accessor_string", "Accessor + string", "#388566"),
    ]
    panels = [
        ("selection", "Token selection only", "head", variants),
        ("copies", "Input copy + selection + output copy", "head", variants),
    ]
    if any(mode == "access_array_part" for mode, _, _ in samples):
        panels.append(("access_array_part", "Complete access_array_part: old vs. new", "old",
                       [("new", "New accessor + optimized helpers", "#388566")]))
    if "parse_array_parts" in cases:
        panels.append(("parse_array_parts", "Complete parse_array_parts: HEAD vs. new", "head",
                       [("new", "New parser + optimized helpers", "#388566")]))
    panels = [panel for panel in panels if panel[0] in cases]
    if not panels:
        parser.error("No supported benchmark modes found")
    fig, axes = plt.subplots(1, len(panels),
                             figsize=(7.5 * len(panels), max(6, max(len(cases[p[0]]) for p in panels) * 0.38)),
                             squeeze=False)
    axes = axes[0]
    summary = []
    for ax, (mode, title, baseline_name, comparisons) in zip(axes, panels):
        for variant_index, (variant, label, color) in enumerate(comparisons):
            centers, lows, highs = [], [], []
            for case in cases[mode]:
                baseline = samples[(mode, case, baseline_name)]
                candidate = samples[(mode, case, variant)]
                if len(baseline) < 3 or candidate.keys() != baseline.keys():
                    parser.error(f"Incomplete samples for {mode}/{case}/{variant}")
                ratios = [baseline[r] / candidate[r] for r in sorted(baseline)]
                center = median(ratios)
                lower, _, upper = quantiles(ratios, n=4, method="inclusive")
                centers.append(center)
                lows.append(center - lower)
                highs.append(upper - center)
                summary.append({
                    "mode": mode, "case": case[0], "index": case[1], "variant": variant,
                    "baseline": baseline_name, "baseline_ns": median(baseline.values()),
                    "variant_ns": median(candidate.values()),
                    "saved_ns": median([baseline[r] - candidate[r] for r in baseline]),
                    "speedup": center, "speedup_p25": lower, "speedup_p75": upper,
                })
            positions = [i + (variant_index - (len(comparisons) - 1) / 2) * 0.22 for i in range(len(cases[mode]))]
            ax.errorbar(centers, positions, xerr=[lows, highs], fmt="o", markersize=4,
                        capsize=2, color=color, label=label)
        baseline_label = ("HEAD parser + original helpers" if mode == "parse_array_parts"
                          else "HEAD model" if baseline_name == "head" else "Old accessor + original helpers")
        ax.axvline(1, color="#555555", linestyle="--", linewidth=1, label=baseline_label + " (1x)")
        ax.set_xscale("log")
        ax.set_title(title)
        ax.set_xlabel("Speedup versus " + baseline_label + " (higher is faster)")
        ax.grid(axis="x", which="both", alpha=0.2)
        ax.legend(loc="best", fontsize=8)
        ax.set_yticks(range(len(cases[mode])),
                     [name if index == -1 else f"{name} / index {index}" for name, index in cases[mode]])
        ax.invert_yaxis()
    fig.suptitle("Array parsing: complete functions and concept comparisons", fontsize=15)
    fig.text(0.5, 0.01, "Dots: median paired speedup. Whiskers: middle 50% of rounds, not confidence intervals.\n"
             "View-based token lookup and full-array parsing; GUI resolution, decoding and HTML excluded.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.055, 1, 0.96))
    image_path = args.csv.with_name(args.csv.stem + "_gains.png")
    fig.savefig(image_path, dpi=170)
    plt.close(fig)
    summary_path = args.csv.with_name(args.csv.stem + "_summary.csv")
    with summary_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"Chart: {image_path}\nTimings and speedups: {summary_path}")


if __name__ == "__main__":
    main()
