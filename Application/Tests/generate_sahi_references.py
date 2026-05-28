#!/usr/bin/env python3
"""Generate golden reference values for the YOLO tile-postprocessing tests.

The tile-dedup logic in ``compute_tile_merge_groups`` / ``compute_tile_nms_indices``
(``Application/src/tracker/python/YOLO.cpp``) mirrors SAHI's sliced-prediction
postprocess. For the ``YoloTileMergeGroupsTest`` SAHI tests to be meaningful,
their expected values must come from *real* SAHI -- not from a re-implementation
living in the test file (that would only compare our code against our code).

This script runs the actual SAHI postprocess functions on the exact box layouts
used by those tests and prints C++-ready golden values. Run it once whenever a
layout changes or the SAHI version is bumped, then paste the printed blocks into
the matching tests in ``test_tiling.cpp`` (look for the ``TODO(sahi-ref)`` markers).

Usage
-----
    pip install sahi torch          # any recent SAHI release works
    python Application/Tests/generate_sahi_references.py

The box layouts below MUST stay in sync with the ``makeBoxes({...})`` rows in
``Application/Tests/test_tiling.cpp``. Each row is (x0, y0, x1, y1, score, class).
"""

import sys

try:
    import torch
    import sahi
    # greedy_nmm / nms and their batched (per-class) variants live here in
    # SAHI 0.11.x. If a future SAHI moves them, update this import.
    from sahi.postprocess.combine import batched_greedy_nmm, batched_nms
except ImportError as exc:  # pragma: no cover - environment hint only
    sys.exit(
        f"Missing dependency ({exc}).\n"
        "Install with:  pip install sahi torch"
    )


# --- GreedyNMM (IOS) cases: compared against compute_tile_merge_groups --------
# class_agnostic=False -> per-class grouping (SAHI's batched_greedy_nmm).
MERGE_CASES = {
    # TEST(YoloTileMergeGroupsTest, MatchesSahiGreedyNmmIosReferenceForFourWayOverlap)
    "four_way_overlap": dict(
        ios_threshold=0.5,
        boxes=[
            (100.0, 100.0, 200.0, 200.0, 0.95, 1),
            (110.0, 100.0, 210.0, 200.0, 0.90, 1),
            (100.0, 110.0, 200.0, 210.0, 0.85, 1),
            (110.0, 110.0, 210.0, 210.0, 0.80, 1),
            (110.0, 110.0, 210.0, 210.0, 0.75, 2),
        ],
    ),
    # TEST(YoloTileMergeGroupsTest, MatchesSahiGreedyNmmIosReferenceAtThresholdBoundary)
    "threshold_boundary": dict(
        ios_threshold=0.5,
        boxes=[
            (0.0, 0.0, 100.0, 100.0, 0.9, 1),
            (50.0, 0.0, 150.0, 100.0, 0.8, 1),
            (151.0, 0.0, 251.0, 100.0, 0.7, 1),
        ],
    ),
    # TEST(YoloTileMergeGroupsTest, MatchesSahiGreedyNmmIosReferenceForRepresentativeChain)
    "representative_chain": dict(
        ios_threshold=0.5,
        boxes=[
            (0.0, 0.0, 100.0, 100.0, 0.9, 1),
            (40.0, 0.0, 140.0, 100.0, 0.8, 1),
            (80.0, 0.0, 180.0, 100.0, 0.7, 1),
            (220.0, 0.0, 320.0, 100.0, 0.6, 1),
        ],
    ),
}

# --- NMS (IOU) cases: compared against compute_tile_nms_indices --------------
NMS_CASES = {
    # TEST(YoloTileMergeGroupsTest, PoseBboxFallbackMatchesActualSahiNmsGoldenOutput)
    "pose_bbox_fallback": dict(
        iou_threshold=0.55,
        boxes=[
            (40.0, 40.0, 80.0, 80.0, 0.9, 1),
            (45.0, 40.0, 85.0, 80.0, 0.8, 1),
        ],
    ),
}


def _tensor(boxes):
    """SAHI expects [x1, y1, x2, y2, score, category_id] rows."""
    return torch.tensor([list(b) for b in boxes], dtype=torch.float32)


def emit_merge_case(name, case):
    boxes = case["boxes"]
    threshold = case["ios_threshold"]

    # keep_to_merge: {representative_index: [suppressed_index, ...]}, all
    # referring to original row indices.
    keep_to_merge = batched_greedy_nmm(
        _tensor(boxes), match_metric="IOS", match_threshold=threshold
    )

    groups = []
    for keep, merged in keep_to_merge.items():
        sources = sorted([int(keep)] + [int(m) for m in merged])
        groups.append((int(keep), sources))
    groups.sort(key=lambda g: g[0])  # compute_tile_merge_groups sorts by rep index

    print(f"// --- {name}: SAHI {sahi.__version__} GreedyNMM "
          f'match_metric="IOS" match_threshold={threshold} ---')
    print(f"    ASSERT_EQ(groups.size(), {len(groups)}u);")
    for i, (rep, sources) in enumerate(groups):
        src = ", ".join(f"{s}u" for s in sources)
        print(f"    EXPECT_EQ(groups[{i}].representative_index, {rep}u);")
        print(f"    EXPECT_EQ(groups[{i}].source_indices, "
              f"DetectionRowSelection({src}));")
    print()


def emit_nms_case(name, case):
    boxes = case["boxes"]
    threshold = case["iou_threshold"]

    keep = sorted(int(k) for k in batched_nms(
        _tensor(boxes), match_metric="IOU", match_threshold=threshold
    ))

    print(f"// --- {name}: SAHI {sahi.__version__} NMS "
          f'match_metric="IOU" match_threshold={threshold} ---')
    src = ", ".join(f"{k}u" for k in keep)
    print(f"    EXPECT_EQ(indices, DetectionRowSelection({src}));")
    print()


def main():
    print(f"# SAHI {sahi.__version__} -- paste each block into the matching "
          f"test in Application/Tests/test_tiling.cpp\n")
    for name, case in MERGE_CASES.items():
        emit_merge_case(name, case)
    for name, case in NMS_CASES.items():
        emit_nms_case(name, case)


if __name__ == "__main__":
    main()
