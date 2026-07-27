#!/usr/bin/env python3
"""Real-model detector-framework to TRex coordinate parity tests.

Set TREX_RUN_DETECTOR_E2E=1 to run every registered lightweight backend.
Individual backends can also be enabled with TREX_RUN_RFDETR_E2E=1 or
TREX_RUN_YOLO_E2E=1. Cached checkpoints can be selected with
TREX_RFDETR_E2E_MODEL and TREX_YOLO_E2E_MODEL.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
import sys
import types
import unittest
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from PIL import Image


TEST_PYTHON_DIR = Path(__file__).resolve().parent
TESTS_DIR = TEST_PYTHON_DIR.parent
RUNTIME_PYTHON_DIR = TESTS_DIR.parent / "src" / "tracker" / "python"
if str(RUNTIME_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(RUNTIME_PYTHON_DIR))

FIXTURE = TESTS_DIR / "data" / "detector_e2e" / "chelsea.png"
FIXTURE_SHA256 = "596aa1e7cb875eb79f437e310381d26b338a81c2da23439704a73c4651e8c4bb"


class _DetectResolution:
    def __init__(self, width: int = 0, height: int = 0):
        self.width = width
        self.height = height


class _KeypointFormat:
    def __init__(self, n_points: int = 0, n_dims: int = 0):
        self.n_points = n_points
        self.n_dims = n_dims


class _ModelTaskType:
    detect = "detect"
    region = "region"


class _ObjectDetectionFormat:
    none = "none"
    boxes = "boxes"
    masks = "masks"
    poses = "poses"
    obb = "obb"
    points = "points"


class _ModelConfig:
    def __init__(self, model_path: str):
        self.task = _ModelTaskType.detect
        self.use_tracking = False
        self.try_optimize = False
        self.model_path = model_path
        self.trained_resolution = _DetectResolution()
        self.output_format = _ObjectDetectionFormat.none
        self.keypoint_format = _KeypointFormat()
        self.classes = {}
        self.requires_exact_input_size = False


class _ArrayData:
    def __init__(self, values):
        self.values = values


class _Result:
    def __init__(self, index, boxes, masks, keypoints, obbs, points):
        self.index = index
        self.boxes = boxes
        self.masks = masks
        self.keypoints = keypoints
        self.obbs = obbs
        self.points = points


class _TRexModule(types.ModuleType):
    settings = {
        "gpu_torch_device": os.environ.get("TREX_DETECTOR_E2E_DEVICE", "cpu"),
        "gpu_torch_device_index": "-1",
        "detect_keypoint_threshold": 0.1,
        "detect_point_radii": "{}",
    }

    def __init__(self):
        super().__init__("TRex")
        self.ModelTaskType = _ModelTaskType
        self.ObjectDetectionFormat = _ObjectDetectionFormat
        self.DetectResolution = _DetectResolution
        self.KeypointFormat = _KeypointFormat
        self.ModelConfig = _ModelConfig
        self.Boxes = _ArrayData
        self.KeypointData = _ArrayData
        self.ObbData = _ArrayData
        self.PointData = _ArrayData
        self.Result = _Result

    @staticmethod
    def log(message):
        del message

    @staticmethod
    def warn(message):
        del message

    @classmethod
    def setting(cls, name):
        return cls.settings[name]

    @staticmethod
    def tile_affines(geometries):
        scales = [np.asarray(geometry.scale, dtype=np.float32) for geometry in geometries]
        offsets = [np.asarray(geometry.offset, dtype=np.float32) for geometry in geometries]
        return scales, offsets


class _Input:
    def __init__(self, bgr_image, scale, offset):
        self._images = [bgr_image]
        self._geometries = [
            types.SimpleNamespace(
                scale=np.asarray(scale, dtype=np.float32),
                offset=np.asarray(offset, dtype=np.float32),
            )
        ]

    def images(self):
        return self._images

    def tile_geometries(self):
        return self._geometries

    def orig_id(self):
        return np.array([0], dtype=np.uint64)


@dataclass(frozen=True)
class _Backend:
    name: str
    enable_env: str
    run: Callable[[np.ndarray], None]


def _load_fixture() -> np.ndarray:
    with Image.open(FIXTURE) as image:
        return np.asarray(image.convert("RGB"))


def _load_e2e_image() -> np.ndarray:
    video_path = os.environ.get("TREX_DETECTOR_E2E_VIDEO")
    if video_path:
        frame_index = int(os.environ.get("TREX_DETECTOR_E2E_FRAME", "12"))
        capture = cv2.VideoCapture(video_path)
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, bgr = capture.read()
        finally:
            capture.release()
        if not ok or bgr is None:
            raise ValueError(
                f"Could not read frame {frame_index} from {video_path}."
            )
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    image_path = os.environ.get("TREX_DETECTOR_E2E_IMAGE")
    if image_path:
        with Image.open(image_path) as image:
            return np.asarray(image.convert("RGB"))

    return _load_fixture()


def _map_boxes_to_source(
    boxes: np.ndarray,
    scale: np.ndarray,
    offset: np.ndarray,
) -> np.ndarray:
    mapped = np.ascontiguousarray(boxes, dtype=np.float32).copy()
    mapped[:, [0, 2]] = (mapped[:, [0, 2]] + offset[0]) * scale[0]
    mapped[:, [1, 3]] = (mapped[:, [1, 3]] + offset[1]) * scale[1]
    return mapped


def _pair_iou(left: np.ndarray, right: np.ndarray) -> float:
    top_left = np.maximum(left[:2], right[:2])
    bottom_right = np.minimum(left[2:4], right[2:4])
    extents = np.maximum(0.0, bottom_right - top_left)
    intersection = float(extents[0] * extents[1])
    left_area = max(0.0, float(left[2] - left[0])) * max(
        0.0,
        float(left[3] - left[1]),
    )
    right_area = max(0.0, float(right[2] - right[0])) * max(
        0.0,
        float(right[3] - right[1]),
    )
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


@dataclass
class _Predictions:
    boxes: np.ndarray
    scores: np.ndarray
    classes: np.ndarray
    keypoints: np.ndarray

    @classmethod
    def empty(cls) -> "_Predictions":
        return cls(
            boxes=np.empty((0, 4), dtype=np.float32),
            scores=np.empty((0,), dtype=np.float32),
            classes=np.empty((0,), dtype=np.int64),
            keypoints=np.empty((0, 0, 2), dtype=np.float32),
        )

    def packed_boxes(self) -> np.ndarray:
        if self.boxes.shape[0] == 0:
            return np.empty((0, 6), dtype=np.float32)
        return np.ascontiguousarray(
            np.column_stack(
                (
                    self.boxes,
                    self.scores,
                    self.classes.astype(np.float32),
                )
            ),
            dtype=np.float32,
        )


def _prediction_arrays(
    prediction,
    keypoint_threshold: float,
) -> _Predictions:
    if hasattr(prediction, "detection_confidence"):
        boxes = np.asarray(prediction.data["xyxy"], dtype=np.float32)
        scores = np.asarray(
            prediction.detection_confidence,
            dtype=np.float32,
        )
    else:
        boxes = np.asarray(prediction.xyxy, dtype=np.float32)
        scores = np.asarray(prediction.confidence, dtype=np.float32)

    classes = np.asarray(prediction.class_id, dtype=np.int64)
    if hasattr(prediction, "keypoint_confidence"):
        keypoints = np.asarray(prediction.xy, dtype=np.float32).copy()
        visible = (
            np.asarray(prediction.keypoint_confidence, dtype=np.float32)
            >= keypoint_threshold
        )
        keypoints[~visible] = 0.0
    else:
        keypoints = np.empty((boxes.shape[0], 0, 2), dtype=np.float32)
    return _Predictions(boxes, scores, classes, keypoints)


def _tensor_result_arrays(
    result: dict[str, torch.Tensor],
) -> _Predictions:
    boxes = np.asarray(
        result["boxes"].detach().cpu().numpy(),
        dtype=np.float32,
    )
    scores = np.asarray(
        result["scores"].detach().cpu().numpy(),
        dtype=np.float32,
    )
    classes = np.asarray(
        result["labels"].detach().cpu().numpy(),
        dtype=np.int64,
    )
    if "keypoints" in result:
        keypoints = np.asarray(
            result["keypoints"][..., :2].detach().cpu().numpy(),
            dtype=np.float32,
        )
    else:
        keypoints = np.empty((boxes.shape[0], 0, 2), dtype=np.float32)
    return _Predictions(boxes, scores, classes, keypoints)


def _trex_result_arrays(result) -> _Predictions:
    packed = np.asarray(result.boxes.values, dtype=np.float32)
    if packed.size == 0:
        return _Predictions.empty()
    packed = packed.reshape((-1, 6))
    keypoints = np.asarray(result.keypoints.values, dtype=np.float32)
    if keypoints.size == 0:
        keypoints = np.empty((packed.shape[0], 0, 2), dtype=np.float32)
    return _Predictions(
        boxes=np.ascontiguousarray(packed[:, :4], dtype=np.float32),
        scores=np.ascontiguousarray(packed[:, 4], dtype=np.float32),
        classes=np.ascontiguousarray(packed[:, 5], dtype=np.int64),
        keypoints=np.ascontiguousarray(keypoints, dtype=np.float32),
    )


def _app_dump_arrays(path: Path) -> _Predictions:
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("results", [])
    if len(results) != 1:
        raise AssertionError(
            f"Expected one result in C++ application dump {path}, got {len(results)}."
        )
    packed = np.asarray(results[0].get("boxes", []), dtype=np.float32)
    if packed.size == 0:
        return _Predictions.empty()
    packed = packed.reshape((-1, 6))
    keypoints = np.asarray(results[0].get("keypoints", []), dtype=np.float32)
    if keypoints.size == 0:
        keypoints = np.empty((packed.shape[0], 0, 2), dtype=np.float32)
    return _Predictions(
        boxes=np.ascontiguousarray(packed[:, :4], dtype=np.float32),
        scores=np.ascontiguousarray(packed[:, 4], dtype=np.float32),
        classes=np.ascontiguousarray(packed[:, 5], dtype=np.int64),
        keypoints=np.ascontiguousarray(keypoints, dtype=np.float32),
    )


def _clone_tensor_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {
            key: _clone_tensor_tree(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_clone_tensor_tree(item) for item in value)
    if isinstance(value, list):
        return [_clone_tensor_tree(item) for item in value]
    return value


def _strict_prediction_error(
    expected: _Predictions,
    actual: _Predictions,
    context: str,
) -> str | None:
    try:
        np.testing.assert_array_equal(actual.classes, expected.classes)
        np.testing.assert_allclose(
            actual.scores,
            expected.scores,
            rtol=1e-4,
            atol=1e-3,
        )
        np.testing.assert_allclose(
            actual.boxes,
            expected.boxes,
            rtol=1e-4,
            atol=1e-3,
        )
        np.testing.assert_allclose(
            actual.keypoints,
            expected.keypoints,
            rtol=1e-4,
            atol=1e-3,
        )
    except AssertionError as error:
        return f"{context}: {error}"
    return None


def _with_bounded_fused_scores(predictions: _Predictions) -> _Predictions:
    return _Predictions(
        boxes=predictions.boxes,
        scores=np.ascontiguousarray(
            predictions.scores / (1.0 + predictions.scores),
            dtype=np.float32,
        ),
        classes=predictions.classes,
        keypoints=predictions.keypoints,
    )


def _match_predictions(
    native: _Predictions,
    trex: _Predictions,
) -> tuple[list[dict[str, object]], list[int], list[int]]:
    candidates = sorted(
        (
            (
                _pair_iou(native.boxes[native_index], trex.boxes[trex_index]),
                native_index,
                trex_index,
            )
            for native_index in range(native.boxes.shape[0])
            for trex_index in range(trex.boxes.shape[0])
            if native.classes[native_index] == trex.classes[trex_index]
        ),
        reverse=True,
    )
    used_native: set[int] = set()
    used_trex: set[int] = set()
    matches: list[dict[str, object]] = []
    for iou, native_index, trex_index in candidates:
        if native_index in used_native or trex_index in used_trex:
            continue
        used_native.add(native_index)
        used_trex.add(trex_index)
        if (
            native.keypoints.shape[1:] == trex.keypoints.shape[1:]
            and native.keypoints.shape[1] > 0
        ):
            keypoint_delta = float(
                np.max(
                    np.abs(
                        native.keypoints[native_index]
                        - trex.keypoints[trex_index]
                    )
                )
            )
        else:
            keypoint_delta = None
        matches.append(
            {
                "native_index": native_index,
                "trex_index": trex_index,
                "class_id": int(native.classes[native_index]),
                "iou": iou,
                "native_score": float(native.scores[native_index]),
                "trex_score": float(trex.scores[trex_index]),
                "score_delta": float(
                    trex.scores[trex_index] - native.scores[native_index]
                ),
                "native_box": native.boxes[native_index].tolist(),
                "trex_box": trex.boxes[trex_index].tolist(),
                "max_box_delta": float(
                    np.max(
                        np.abs(
                            native.boxes[native_index]
                            - trex.boxes[trex_index]
                        )
                    )
                ),
                "max_keypoint_delta": keypoint_delta,
            }
        )
    matches.sort(key=lambda match: int(match["native_index"]))
    unmatched_native = sorted(
        set(range(native.boxes.shape[0])) - used_native
    )
    unmatched_trex = sorted(
        set(range(trex.boxes.shape[0])) - used_trex
    )
    return matches, unmatched_native, unmatched_trex


def _operational_prediction_error(
    expected: _Predictions,
    actual: _Predictions,
    source_shape: tuple[int, ...],
    iou_threshold: float,
) -> str | None:
    errors: list[str] = []
    if not bool(np.all(np.isfinite(actual.scores))):
        errors.append("TRex returned a non-finite confidence.")
    if not bool(np.all((actual.scores >= 0.0) & (actual.scores < 1.0))):
        errors.append(
            "TRex RF-DETR confidences must be bounded to [0,1)."
        )

    for left in range(actual.boxes.shape[0]):
        for right in range(left + 1, actual.boxes.shape[0]):
            overlap = _pair_iou(actual.boxes[left], actual.boxes[right])
            if overlap > iou_threshold + 1e-5:
                errors.append(
                    "TRex retained overlapping rows "
                    f"{left}/{right} with IoU {overlap:.6f} after "
                    f"class-agnostic NMS at {iou_threshold:.3f}."
                )

    matches, unmatched_expected, unmatched_actual = _match_predictions(
        expected,
        actual,
    )
    matches_by_expected = {
        int(match["native_index"]): match
        for match in matches
    }
    matches_by_actual = {
        int(match["trex_index"]): match
        for match in matches
    }
    parity_floor = 0.5
    coordinate_tolerance = max(4.0, max(source_shape[:2]) * 0.002)
    for expected_index in np.flatnonzero(expected.scores >= parity_floor):
        match = matches_by_expected.get(int(expected_index))
        if match is None:
            errors.append(
                "No TRex match for native-derived high-confidence row "
                f"{int(expected_index)}."
            )
            continue
        if float(match["iou"]) < 0.95:
            errors.append(
                f"Source-coordinate IoU for native-derived row "
                f"{int(expected_index)} is {float(match['iou']):.6f}, "
                "below 0.95."
            )
        if float(match["max_box_delta"]) > coordinate_tolerance:
            errors.append(
                f"Source-coordinate box delta for native-derived row "
                f"{int(expected_index)} is "
                f"{float(match['max_box_delta']):.3f}px, above "
                f"{coordinate_tolerance:.3f}px."
            )
        if abs(float(match["score_delta"])) > 0.05:
            errors.append(
                f"Confidence delta for native-derived row "
                f"{int(expected_index)} is "
                f"{float(match['score_delta']):+.6f}, above 0.05."
            )
        keypoint_delta = match["max_keypoint_delta"]
        if (
            keypoint_delta is not None
            and float(keypoint_delta) > coordinate_tolerance
        ):
            errors.append(
                f"Source-coordinate keypoint delta for native-derived row "
                f"{int(expected_index)} is {float(keypoint_delta):.3f}px, "
                f"above {coordinate_tolerance:.3f}px."
            )

    for actual_index in np.flatnonzero(actual.scores >= parity_floor):
        if int(actual_index) not in matches_by_actual:
            errors.append(
                "No native-derived match for TRex high-confidence row "
                f"{int(actual_index)}."
            )

    if errors:
        return (
            "Native-derived operational policy versus complete TRex "
            "inference:\n"
            + "\n".join(f"  {error}" for error in errors)
            + f"\n  unmatched native-derived rows: {unmatched_expected}"
            + f"\n  unmatched TRex rows: {unmatched_actual}"
        )
    return None


def _draw_predictions(
    image_bgr: np.ndarray,
    predictions: _Predictions,
    color: tuple[int, int, int],
    keypoint_shape: str,
) -> None:
    for index, (box, score) in enumerate(
        zip(predictions.boxes, predictions.scores)
    ):
        x0, y0, x1, y1 = np.rint(box).astype(int)
        cv2.rectangle(image_bgr, (x0, y0), (x1, y1), color, 4)
        cv2.putText(
            image_bgr,
            f"{index}: {float(score):.4f}",
            (x0, max(32, y0 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            3,
            cv2.LINE_AA,
        )
        if index >= predictions.keypoints.shape[0]:
            continue
        for x, y in predictions.keypoints[index]:
            if x == 0 and y == 0:
                continue
            center = (int(round(float(x))), int(round(float(y))))
            if keypoint_shape == "circle":
                cv2.circle(image_bgr, center, 7, color, 3, cv2.LINE_AA)
            else:
                cv2.drawMarker(
                    image_bgr,
                    center,
                    color,
                    cv2.MARKER_TILTED_CROSS,
                    18,
                    3,
                    cv2.LINE_AA,
                )


def _write_visualization(
    path: Path,
    original_rgb: np.ndarray,
    native: _Predictions,
    trex: _Predictions,
    matches: list[dict[str, object]],
    unmatched_native: list[int],
    unmatched_trex: list[int],
) -> None:
    source_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
    native_view = source_bgr.copy()
    trex_view = source_bgr.copy()
    combined_view = source_bgr.copy()
    native_color = (40, 210, 40)
    trex_color = (220, 40, 220)
    _draw_predictions(native_view, native, native_color, "circle")
    _draw_predictions(trex_view, trex, trex_color, "cross")
    _draw_predictions(combined_view, native, native_color, "circle")
    _draw_predictions(combined_view, trex, trex_color, "cross")

    panel_width = 1120
    panel_height = max(
        1,
        round(original_rgb.shape[0] * panel_width / original_rgb.shape[1]),
    )
    panels = [
        cv2.resize(
            view,
            (panel_width, panel_height),
            interpolation=cv2.INTER_AREA,
        )
        for view in (native_view, trex_view, combined_view)
    ]
    image_row = np.concatenate(panels, axis=1)
    cv2.putText(
        image_row,
        "Native: green boxes + circles",
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        native_color,
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image_row,
        "TRex: magenta boxes + crosses",
        (panel_width + 20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        trex_color,
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image_row,
        "Combined overlay",
        (panel_width * 2 + 20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (235, 235, 235),
        3,
        cv2.LINE_AA,
    )

    visible_matches = matches[:24]
    table_height = 82 + 31 * (
        len(visible_matches)
        + int(bool(unmatched_native or unmatched_trex))
    )
    table = np.full(
        (max(table_height, 120), image_row.shape[1], 3),
        28,
        dtype=np.uint8,
    )
    headers = (
        "native",
        "TRex",
        "IoU",
        "native score",
        "TRex score",
        "score delta",
        "max box delta",
        "max keypoint delta",
    )
    columns = (20, 180, 340, 520, 760, 1000, 1280, 1580)
    for x, header in zip(columns, headers):
        cv2.putText(
            table,
            header,
            (x, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (225, 225, 225),
            2,
            cv2.LINE_AA,
        )
    for row_index, match in enumerate(visible_matches):
        y = 68 + 31 * row_index
        values = (
            str(match["native_index"]),
            str(match["trex_index"]),
            f'{float(match["iou"]):.5f}',
            f'{float(match["native_score"]):.6f}',
            f'{float(match["trex_score"]):.6f}',
            f'{float(match["score_delta"]):+.6f}',
            f'{float(match["max_box_delta"]):.3f}px',
            (
                "n/a"
                if match["max_keypoint_delta"] is None
                else f'{float(match["max_keypoint_delta"]):.3f}px'
            ),
        )
        for x, value in zip(columns, values):
            cv2.putText(
                table,
                value,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (210, 210, 210),
                1,
                cv2.LINE_AA,
            )
    if unmatched_native or unmatched_trex:
        y = 68 + 31 * len(visible_matches)
        cv2.putText(
            table,
            (
                f"Unmatched native rows: {unmatched_native}; "
                f"unmatched TRex rows: {unmatched_trex}"
            ),
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (70, 170, 255),
            2,
            cv2.LINE_AA,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.concatenate((image_row, table), axis=0)):
        raise OSError(f"Could not write RF-DETR visualization to {path}.")


def _write_diagnostics(
    visualization_path: Path,
    original_rgb: np.ndarray,
    native: _Predictions,
    trex: _Predictions,
    input_metrics: dict[str, object],
    postprocess_error: str | None,
    full_error: str | None,
) -> None:
    matches, unmatched_native, unmatched_trex = _match_predictions(native, trex)
    _write_visualization(
        visualization_path,
        original_rgb,
        native,
        trex,
        matches,
        unmatched_native,
        unmatched_trex,
    )
    report_path = visualization_path.with_suffix(".json")
    report_path.write_text(
        json.dumps(
            {
                "input_tensor": input_metrics,
                "same_raw_postprocess_error": postprocess_error,
                "full_pipeline_error": full_error,
                "native": {
                    "boxes": native.boxes.tolist(),
                    "scores": native.scores.tolist(),
                    "classes": native.classes.tolist(),
                    "keypoints": native.keypoints.tolist(),
                },
                "trex": {
                    "boxes": trex.boxes.tolist(),
                    "scores": trex.scores.tolist(),
                    "classes": trex.classes.tolist(),
                    "keypoints": trex.keypoints.tolist(),
                },
                "matches": matches,
                "unmatched_native": unmatched_native,
                "unmatched_trex": unmatched_trex,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"RF-DETR parity visualization: {visualization_path}")
    print(f"RF-DETR parity report: {report_path}")
    for match in matches:
        print(
            "RF-DETR match "
            f"native={match['native_index']} trex={match['trex_index']} "
            f"iou={float(match['iou']):.6f} "
            f"scores={float(match['native_score']):.6f}/"
            f"{float(match['trex_score']):.6f} "
            f"box_delta={float(match['max_box_delta']):.3f}px "
            f"keypoint_delta={match['max_keypoint_delta']}"
        )
    print(
        "RF-DETR unmatched rows: "
        f"native={unmatched_native}, trex={unmatched_trex}"
    )


def _letterbox_rgb(
    image: np.ndarray,
    resolution: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    gain = min(resolution / width, resolution / height)
    resized_width = max(1, round(width * gain))
    resized_height = max(1, round(height * gain))
    pad_x = max(0, int((resolution - resized_width) / 2))
    pad_y = max(0, int((resolution - resized_height) / 2))
    prepared = np.full((resolution, resolution, 3), 114, dtype=np.uint8)
    prepared[
        pad_y:pad_y + resized_height,
        pad_x:pad_x + resized_width,
    ] = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    scale = np.array(
        [width / resized_width, height / resized_height],
        dtype=np.float32,
    )
    offset = np.array([-pad_x, -pad_y], dtype=np.float32)
    return prepared, scale, offset


def _with_trex_modules(callback: Callable[[types.ModuleType], None]) -> None:
    saved_modules = {
        name: sys.modules.get(name)
        for name in (
            "TRex",
            "trex_utils",
            "trex_detection_model",
            "trex_rfdetr",
            "trex_yolo",
        )
    }
    try:
        fake_trex = _TRexModule()
        sys.modules["TRex"] = fake_trex
        for name in ("trex_utils", "trex_detection_model", "trex_rfdetr", "trex_yolo"):
            sys.modules.pop(name, None)
        callback(fake_trex)
    finally:
        for name, previous in saved_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def _load_rfdetr_wrapper():
    from rfdetr import RFDETR, RFDETRNano

    device = torch.device(
        os.environ.get("TREX_DETECTOR_E2E_DEVICE", "cpu")
    )
    checkpoint = os.environ.get("TREX_RFDETR_E2E_MODEL")
    if checkpoint:
        loader_parameters = inspect.signature(RFDETR.from_checkpoint).parameters
        loader_kwargs = (
            {"trust_checkpoint": True}
            if "trust_checkpoint" in loader_parameters
            else {}
        )
        wrapper = RFDETR.from_checkpoint(checkpoint, **loader_kwargs)
        model_path = checkpoint
    else:
        wrapper = RFDETRNano()
        model_path = str(
            getattr(wrapper.model_config, "pretrain_weights", "rf-detr-nano.pth")
        )

    wrapper.model.model = wrapper.model.model.to(device)
    wrapper.model.device = device
    return wrapper, model_path


def _run_rfdetr(original_rgb: np.ndarray) -> None:
    wrapper, model_path = _load_rfdetr_wrapper()
    resolution = int(wrapper.model.resolution)
    prepared_rgb = cv2.resize(
        original_rgb,
        (resolution, resolution),
        interpolation=cv2.INTER_LINEAR,
    )
    scale = np.array(
        [
            original_rgb.shape[1] / resolution,
            original_rgb.shape[0] / resolution,
        ],
        dtype=np.float32,
    )
    offset = np.zeros(2, dtype=np.float32)
    native_notebook_threshold = 0.25
    operational_confidence = 0.1
    operational_iou = 0.5
    keypoint_threshold = 0.1
    uses_fused_scores = bool(
        getattr(wrapper.model_config, "use_grouppose_keypoints", False)
    )
    notebook_trex_threshold = (
        native_notebook_threshold / (1.0 + native_notebook_threshold)
        if uses_fused_scores
        else native_notebook_threshold
    )

    captured: dict[str, object] = {}

    def capture_input(_module, arguments):
        captured["batch"] = _clone_tensor_tree(arguments[0])

    def capture_output(_module, _arguments, output):
        captured["raw"] = _clone_tensor_tree(output)

    input_hook = wrapper.model.model.register_forward_pre_hook(capture_input)
    output_hook = wrapper.model.model.register_forward_hook(capture_output)
    try:
        with torch.inference_mode():
            reference = wrapper.predict(
                original_rgb,
                threshold=native_notebook_threshold,
                include_source_image=False,
            )
    finally:
        input_hook.remove()
        output_hook.remove()

    if "batch" not in captured or "raw" not in captured:
        raise AssertionError(
            "RF-DETR native prediction hooks did not capture the normalized "
            "input tensor and raw network output."
        )
    native_raw = _prediction_arrays(reference, keypoint_threshold)
    native = (
        _with_bounded_fused_scores(native_raw)
        if uses_fused_scores
        else native_raw
    )

    def compare(fake_trex):
        del fake_trex
        detection_module = importlib.import_module("trex_detection_model")
        adapter_module = importlib.import_module("trex_rfdetr")
        config = _ModelConfig(model_path)
        adapter = adapter_module.RFDETRModel(config)
        adapter._finish_load(wrapper)
        adapter.load = lambda: None
        trex_inference = detection_module.TRexDetection([adapter])

        app_dump = os.environ.get("TREX_RFDETR_E2E_APP_DUMP")
        if app_dump:
            app_payload = json.loads(
                Path(app_dump).read_text(encoding="utf-8")
            )
            if not np.isclose(
                float(app_payload.get("confidence_threshold", -1.0)),
                operational_confidence,
            ):
                raise AssertionError(
                    "C++ application dump used an unexpected confidence "
                    f"threshold: {app_payload.get('confidence_threshold')}."
                )
            if not np.isclose(
                float(app_payload.get("iou_threshold", -1.0)),
                operational_iou,
            ):
                raise AssertionError(
                    "C++ application dump used an unexpected IoU threshold: "
                    f"{app_payload.get('iou_threshold')}."
                )
            app_input_value = app_payload.get("input_path")
            if not app_input_value:
                raise AssertionError(
                    "C++ application dump does not identify its captured "
                    "detector input. Rebuild TRex so bbx_saved_model.py writes "
                    "the first callback input alongside its predictions."
                )
            app_input_path = Path(app_input_value)
            if not app_input_path.is_file():
                raise AssertionError(
                    f"C++ detector input capture does not exist: {app_input_path}."
                )
            app_input = np.load(app_input_path, allow_pickle=False)
            prepared_for_trex = trex_inference.preprocess([app_input])[0]
        else:
            app_input = None
            prepared_for_trex = prepared_rgb

        trex_batch, _ = adapter._prepare_batch([prepared_for_trex])
        native_batch = np.asarray(
            captured["batch"].detach().cpu().numpy(),
            dtype=np.float32,
        )
        trex_batch_array = np.asarray(
            trex_batch.detach().cpu().numpy(),
            dtype=np.float32,
        )
        if native_batch.shape != trex_batch_array.shape:
            raise AssertionError(
                "RF-DETR normalized input shapes differ: "
                f"native={native_batch.shape}, TRex={trex_batch_array.shape}."
            )
        input_delta = np.abs(native_batch - trex_batch_array)
        input_metrics = {
            "shape": list(native_batch.shape),
            "different_values": int(np.count_nonzero(input_delta)),
            "total_values": int(input_delta.size),
            "maximum_absolute_delta": float(np.max(input_delta)),
            "mean_absolute_delta": float(np.mean(input_delta)),
        }
        if app_input is not None:
            input_metrics["cpp_input_shape"] = list(app_input.shape)
            input_metrics["cpp_input_dtype"] = str(app_input.dtype)
        print(f"RF-DETR normalized input comparison: {input_metrics}")

        raw = captured["raw"]
        if not isinstance(raw, dict):
            raise AssertionError(
                "RF-DETR native raw output was expected to be a dictionary, "
                f"got {type(raw).__name__}."
            )
        target_sizes = torch.tensor(
            [[original_rgb.shape[0], original_rgb.shape[1]]],
            dtype=torch.int64,
        )
        postprocessed = adapter.postprocess(
            raw,
            target_sizes=target_sizes,
        )
        postprocessed_with_scores = (
            adapter_module._with_bounded_keypoint_scores(postprocessed[0])
        )
        same_raw_result = adapter_module._filter_result(
            postprocessed_with_scores,
            notebook_trex_threshold,
            1000,
            None,
            adapter._valid_class_ids(),
            None,
        )
        same_raw_result = adapter._zero_low_confidence_keypoints(
            same_raw_result,
            keypoint_threshold,
        )
        same_raw = _tensor_result_arrays(same_raw_result)
        postprocess_error = _strict_prediction_error(
            native,
            same_raw,
            "Native API versus TRex postprocessing of identical raw output",
        )

        operational_result = adapter_module._filter_result(
            postprocessed_with_scores,
            operational_confidence,
            1000,
            None,
            adapter._valid_class_ids(),
            operational_iou,
        )
        operational_result = adapter._zero_low_confidence_keypoints(
            operational_result,
            keypoint_threshold,
        )
        operational_expected = _tensor_result_arrays(operational_result)

        with torch.inference_mode():
            if app_dump:
                actual = _app_dump_arrays(Path(app_dump))
            else:
                actual_result = trex_inference.inference(
                    _Input(
                        prepared_rgb,
                        scale,
                        offset,
                    ),
                    conf_threshold=operational_confidence,
                    iou_threshold=operational_iou,
                )[0]
                actual = _trex_result_arrays(actual_result)

        full_error = _operational_prediction_error(
            operational_expected,
            actual,
            original_rgb.shape,
            operational_iou,
        )
        visualization_value = os.environ.get(
            "TREX_RFDETR_E2E_VISUALIZATION"
        )
        if visualization_value:
            _write_diagnostics(
                Path(visualization_value),
                original_rgb,
                operational_expected,
                actual,
                input_metrics,
                postprocess_error,
                full_error,
            )

        errors = [
            error
            for error in (postprocess_error, full_error)
            if error is not None
        ]
        if errors:
            matches, unmatched_native, unmatched_trex = _match_predictions(
                operational_expected,
                actual,
            )
            diagnostic_lines = [
                *errors,
                "Matched predictions:",
                *[
                    (
                        f"  native={match['native_index']} "
                        f"trex={match['trex_index']} "
                        f"class={match['class_id']} "
                        f"iou={float(match['iou']):.6f} "
                        f"scores={float(match['native_score']):.7f}/"
                        f"{float(match['trex_score']):.7f} "
                        f"score_delta={float(match['score_delta']):+.7f} "
                        f"native_box={match['native_box']} "
                        f"trex_box={match['trex_box']} "
                        f"max_box_delta={float(match['max_box_delta']):.4f} "
                        f"max_keypoint_delta={match['max_keypoint_delta']}"
                    )
                    for match in matches
                ],
                f"Unmatched native rows: {unmatched_native}",
                f"Unmatched TRex rows: {unmatched_trex}",
            ]
            raise AssertionError("\n".join(diagnostic_lines))

        if not bool(np.all((actual.scores >= 0.0) & (actual.scores < 1.0))):
            raise AssertionError(
                "RF-DETR packed TRex confidences escaped [0,1)."
            )

    _with_trex_modules(compare)


def _rfdetr_label_names(wrapper) -> dict[int, str]:
    names = wrapper.class_names
    if isinstance(names, Mapping):
        return {
            int(class_id): str(name)
            for class_id, name in names.items()
        }

    names = [str(name) for name in names]
    keypoint_counts = list(
        getattr(wrapper.model.args, "num_keypoints_per_class", []) or []
    )
    if (
        bool(getattr(wrapper.model_config, "use_grouppose_keypoints", False))
        and len(keypoint_counts) > len(names)
        and keypoint_counts[0] == 0
    ):
        active_slots = [
            index
            for index, count in enumerate(keypoint_counts)
            if count > 0
        ]
        return {
            slot: names[index]
            for index, slot in enumerate(active_slots)
            if index < len(names)
        }
    return dict(enumerate(names))


def _run_rfdetr_annotated_evaluator_parity(
    annotation_path: Path,
    image_directory: Path,
) -> None:
    try:
        from faster_coco_eval import COCO
        try:
            from faster_coco_eval import COCOeval_faster as COCOeval
        except ImportError:
            from faster_coco_eval import COCOeval
    except ImportError as error:
        raise unittest.SkipTest(
            f"faster-coco-eval is required for annotated parity: {error}"
        ) from error

    wrapper, _ = _load_rfdetr_wrapper()
    coco_ground_truth = COCO(str(annotation_path))
    categories = {
        int(category["id"]): str(category["name"])
        for category in coco_ground_truth.dataset["categories"]
    }
    category_by_name = {
        name: category_id
        for category_id, name in categories.items()
    }
    label_to_category = {
        label: category_by_name[name]
        for label, name in _rfdetr_label_names(wrapper).items()
        if name in category_by_name
    }
    if not label_to_category:
        raise AssertionError(
            "RF-DETR checkpoint classes do not match any annotated "
            f"categories in {annotation_path}."
        )

    native_bbox: list[dict[str, object]] = []
    bounded_bbox: list[dict[str, object]] = []
    native_keypoints: list[dict[str, object]] = []
    bounded_keypoints: list[dict[str, object]] = []
    uses_fused_scores = bool(
        getattr(wrapper.model_config, "use_grouppose_keypoints", False)
    )
    image_ids: list[int] = []
    for image_record in sorted(
        coco_ground_truth.dataset["images"],
        key=lambda record: int(record["id"]),
    ):
        image_id = int(image_record["id"])
        image_ids.append(image_id)
        image_path = image_directory / str(image_record["file_name"])
        with Image.open(image_path) as image:
            original_rgb = np.asarray(image.convert("RGB"))
        with torch.inference_mode():
            prediction = wrapper.predict(
                original_rgb,
                threshold=0.0,
                include_source_image=False,
            )

        native = _prediction_arrays(prediction, keypoint_threshold=0.0)
        if hasattr(prediction, "keypoint_confidence"):
            keypoint_confidences = np.asarray(
                prediction.keypoint_confidence,
                dtype=np.float32,
            )
        else:
            keypoint_confidences = None
        for row_index in range(native.boxes.shape[0]):
            label = int(native.classes[row_index])
            if label not in label_to_category:
                continue
            category_id = label_to_category[label]
            x0, y0, x1, y1 = (
                float(value)
                for value in native.boxes[row_index]
            )
            raw_score = float(native.scores[row_index])
            confidence = (
                raw_score / (1.0 + raw_score)
                if uses_fused_scores
                else raw_score
            )
            common = {
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x0, y0, x1 - x0, y1 - y0],
            }
            native_bbox.append({**common, "score": raw_score})
            bounded_bbox.append({**common, "score": confidence})

            if keypoint_confidences is not None:
                xy = native.keypoints[row_index]
                points = np.column_stack(
                    (xy, keypoint_confidences[row_index])
                )
                keypoint_common = {
                    "image_id": image_id,
                    "category_id": category_id,
                    "keypoints": points.reshape(-1).tolist(),
                }
                native_keypoints.append(
                    {**keypoint_common, "score": raw_score}
                )
                bounded_keypoints.append(
                    {**keypoint_common, "score": confidence}
                )

    if not native_bbox:
        raise AssertionError(
            "RF-DETR produced no annotated-category candidates at threshold 0."
        )

    def evaluate(
        results: list[dict[str, object]],
        iou_type: str,
    ) -> np.ndarray:
        detections = coco_ground_truth.loadRes(results)
        evaluator = COCOeval(
            coco_ground_truth,
            detections,
            iou_type,
        )
        evaluator.params.imgIds = image_ids
        if iou_type == "keypoints":
            keypoint_counts = {
                len(category.get("keypoints", []))
                for category in coco_ground_truth.dataset["categories"]
                if category.get("keypoints")
            }
            if len(keypoint_counts) != 1:
                raise AssertionError(
                    "Annotated keypoint parity requires one shared keypoint "
                    f"count, got {sorted(keypoint_counts)}."
                )
            count = next(iter(keypoint_counts))
            configured_sigmas = getattr(
                wrapper.model.args,
                "keypoint_oks_sigmas",
                None,
            )
            if configured_sigmas is not None and len(configured_sigmas) == count:
                evaluator.params.kpt_oks_sigmas = np.asarray(
                    configured_sigmas,
                    dtype=np.float32,
                )
            else:
                evaluator.params.kpt_oks_sigmas = np.full(
                    count,
                    0.05,
                    dtype=np.float32,
                )
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        return np.asarray(evaluator.stats, dtype=np.float64)

    np.testing.assert_allclose(
        evaluate(native_bbox, "bbox"),
        evaluate(bounded_bbox, "bbox"),
        rtol=0,
        atol=0,
    )
    if native_keypoints:
        np.testing.assert_allclose(
            evaluate(native_keypoints, "keypoints"),
            evaluate(bounded_keypoints, "keypoints"),
            rtol=0,
            atol=0,
        )


def _run_yolo(original_rgb: np.ndarray) -> None:
    from ultralytics import YOLO

    model_path = os.environ.get("TREX_YOLO_E2E_MODEL", "yolo26n.pt")
    wrapper = YOLO(model_path).to("cpu")
    resolution = 640
    prepared_rgb, scale, offset = _letterbox_rgb(original_rgb, resolution)
    confidence = 0.25

    def compare(fake_trex):
        del fake_trex
        detection_module = importlib.import_module("trex_detection_model")
        adapter_module = importlib.import_module("trex_yolo")
        config = _ModelConfig(model_path)
        config.trained_resolution = _DetectResolution(resolution, resolution)
        config.output_format = _ObjectDetectionFormat.boxes
        config.classes = dict(wrapper.names)
        adapter = adapter_module.YOLOModel(config)
        adapter.ptr = wrapper
        adapter.device = torch.device("cpu")
        adapter.load = lambda: None
        trex_inference = detection_module.TRexDetection([adapter])

        prediction_kwargs = {
            "conf": confidence,
            "imgsz": [resolution, resolution],
            "classes": None,
            "verbose": False,
            "max_det": 1000,
            "device": torch.device("cpu"),
        }
        reference = wrapper.predict(prepared_rgb, **prediction_kwargs)[0]
        actual = trex_inference.inference(
            _Input(np.ascontiguousarray(prepared_rgb[..., ::-1]), scale, offset),
            conf_threshold=confidence,
            iou_threshold=None,
        )[0].boxes.values
        expected = np.asarray(reference.boxes.data.cpu(), dtype=np.float32)
        expected = _map_boxes_to_source(expected, scale, offset)
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-3)

    _with_trex_modules(compare)


BACKENDS = (
    _Backend("rfdetr", "TREX_RUN_RFDETR_E2E", _run_rfdetr),
    _Backend("yolo", "TREX_RUN_YOLO_E2E", _run_yolo),
)


class DetectorFixtureTest(unittest.TestCase):
    def test_cc0_fixture_hash_and_shape(self):
        self.assertEqual(
            hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
            FIXTURE_SHA256,
        )
        self.assertEqual(_load_fixture().shape, (300, 451, 3))


class DetectorRealModelEndToEndTest(unittest.TestCase):
    def test_registered_backends_match_trex_coordinates(self):
        run_all = os.environ.get("TREX_RUN_DETECTOR_E2E") == "1"
        enabled = [
            backend
            for backend in BACKENDS
            if run_all or os.environ.get(backend.enable_env) == "1"
        ]
        if not enabled:
            self.skipTest(
                "Set TREX_RUN_DETECTOR_E2E=1, TREX_RUN_RFDETR_E2E=1, "
                "or TREX_RUN_YOLO_E2E=1 to run real detector parity tests."
            )

        image = _load_e2e_image()
        for backend in enabled:
            with self.subTest(backend=backend.name):
                backend.run(image)

    def test_rfdetr_annotated_evaluator_score_parity(self):
        annotation_value = os.environ.get(
            "TREX_RFDETR_E2E_ANNOTATIONS"
        )
        if not annotation_value:
            self.skipTest(
                "Set TREX_RFDETR_E2E_ANNOTATIONS to an RF-DETR COCO "
                "validation annotation file to run real evaluator parity."
            )
        annotation_path = Path(annotation_value)
        image_directory = Path(
            os.environ.get(
                "TREX_RFDETR_E2E_IMAGES",
                str(annotation_path.parent),
            )
        )
        _run_rfdetr_annotated_evaluator_parity(
            annotation_path,
            image_directory,
        )


if __name__ == "__main__":
    unittest.main()
