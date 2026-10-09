"""TEST-ONLY capture helpers for the full detector application parity test.

This module is intentionally kept outside ``src/tracker/python`` so the
production detector interface remains a minimal example for new backends.
``test_headless_convert_exit`` adds this directory to ``PYTHONPATH`` only while
the opt-in end-to-end test is running.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import trex_utils


def capture_input(input_value, destination: str) -> Optional[Path]:
    output = Path(destination)
    if output.exists():
        return None

    images = input_value.images()
    if len(images) != 1:
        raise AssertionError(
            "RF-DETR application parity capture expects exactly one detector "
            f"image, got {len(images)}."
        )

    input_path = output.with_suffix(".input.npy")
    input_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(
        input_path,
        trex_utils.asarray(images[0], copy=True),
        allow_pickle=False,
    )
    return input_path


def write_result_dump(
    results,
    destination: str,
    input_path: Optional[Path],
    confidence_threshold: float,
    iou_threshold: Optional[float],
) -> None:
    output = Path(destination)
    if output.exists():
        return

    serialized_results = []
    for result in results:
        boxes = result.boxes_and_scores()
        keypoint_data = result.keypoints()
        serialized_boxes = []
        serialized_keypoints = []
        for index in range(boxes.num_rows()):
            row = boxes.row(index)
            serialized_boxes.append([
                float(row.box.x0),
                float(row.box.y0),
                float(row.box.x1),
                float(row.box.y1),
                float(row.conf),
                int(row.clid),
            ])
            if keypoint_data.num_bones() > 0:
                serialized_keypoints.append([
                    [float(bone.x), float(bone.y)]
                    for bone in keypoint_data.at(index).bones
                ])

        serialized_results.append({
            "index": int(result.index()),
            "boxes": serialized_boxes,
            "keypoints": serialized_keypoints,
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "input_path": str(input_path) if input_path is not None else None,
                "confidence_threshold": confidence_threshold,
                "iou_threshold": iou_threshold,
                "results": serialized_results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
