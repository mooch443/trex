#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for TRex YOLO runtime-policy selection."""

from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path

import numpy as np


PYTHON_DIR = Path(__file__).resolve().parent
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))


class FakeDetectResolution:
    def __init__(self, width: int = 0, height: int = 0):
        self.width = width
        self.height = height

    def __repr__(self) -> str:
        return f"DetectResolution({self.width}, {self.height})"


class FakeKeypointFormat:
    def __init__(self, n_points: int = 0, n_dims: int = 0):
        self.n_points = n_points
        self.n_dims = n_dims

    def __bool__(self) -> bool:
        return bool(self.n_points or self.n_dims)


class FakeModelTaskType:
    detect = "detect"
    region = "region"


class FakeObjectDetectionFormat:
    none = "none"
    boxes = "boxes"
    masks = "masks"
    poses = "poses"
    obb = "obb"
    points = "points"


class FakeModelConfig:
    def __init__(
        self,
        task: str = FakeModelTaskType.detect,
        use_tracking: bool = False,
        model_path: str = "fake.pt",
        trained_resolution: FakeDetectResolution | None = None,
        output_format: str = FakeObjectDetectionFormat.none,
        keypoint_format: FakeKeypointFormat | None = None,
    ):
        self.task = task
        self.use_tracking = use_tracking
        self.model_path = model_path
        self.trained_resolution = trained_resolution or FakeDetectResolution()
        self.output_format = output_format
        self.keypoint_format = keypoint_format or FakeKeypointFormat()
        self.classes = {}

    def __repr__(self) -> str:
        return f"FakeModelConfig(task={self.task}, model_path={self.model_path})"


class FakeDetectionModel:
    def __init__(self, config):
        self.config = config
        self.ptr = None
        self.device = types.SimpleNamespace(type="cpu")

    def reinit_device(self):
        self.device = types.SimpleNamespace(type="cpu")

    def load(self):
        return None


class FakeTRexDetection:
    def __init__(self, models):
        self.models = models


class FakeStrippedResults:
    def __init__(self, scale, offset):
        self.scale = scale
        self.offset = offset
        self.boxes = None
        self.keypoints = None
        self.masks = None
        self.orig_shape = None
        self.obb = None
        self.points = None


class FakeTRexModule(types.ModuleType):
    def __init__(self):
        super().__init__("TRex")
        self.ModelTaskType = FakeModelTaskType
        self.DetectResolution = FakeDetectResolution
        self.ObjectDetectionFormat = FakeObjectDetectionFormat
        self.KeypointFormat = FakeKeypointFormat
        self.ModelConfig = FakeModelConfig
        self.YoloInput = object
        self.Result = object
        self.settings = {
            "gpu_torch_no_fixes": "true",
            "detect_conf_threshold": 0.1,
            "detect_iou_threshold": None,
            "detect_point_radii": "{}",
            "detect_keypoint_threshold": 0.1,
        }

    def setting(self, name: str):
        return self.settings[name]

    @staticmethod
    def log(message: str) -> None:
        del message

    @staticmethod
    def warn(message: str) -> None:
        del message


class FakeTensorLike:
    def __init__(self, array):
        self._array = np.asarray(array, dtype=np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self._array


class FakeBoxes:
    def __init__(self, data=None):
        array = np.zeros((0, 6), dtype=np.float32) if data is None else data
        self.data = FakeTensorLike(array)
        self.xyxy = FakeTensorLike(np.asarray(array, dtype=np.float32)[:, :4])


class FakePredictionResult:
    def __init__(self):
        self.boxes = FakeBoxes()
        self.masks = None
        self.keypoints = None
        self.obb = None
        self.locations = None
        self.orig_shape = (480, 800)


class FakeHead:
    def __init__(self, *, end2end: bool, modern: bool):
        self._end2end = end2end
        self.cv2 = object()
        self.cv3 = object()
        if modern:
            self.one2one = {"box_head": object(), "cls_head": object()}

    @property
    def end2end(self):
        return self._end2end

    @end2end.setter
    def end2end(self, value):
        self._end2end = bool(value)

    def fuse(self):
        self.cv2 = None
        self.cv3 = None


class FakeInnerModel:
    def __init__(self, head: FakeHead):
        self._head = head

    def modules(self):
        return [self._head]


class FakeYOLOScenario:
    def __init__(self, *, modern: bool, end2end: bool, imgsz=(640, 640), task="detect"):
        self.modern = modern
        self.end2end = end2end
        self.imgsz = list(imgsz)
        self.task = task


class FakeYOLO:
    active_scenario = FakeYOLOScenario(modern=True, end2end=True)
    instances = []

    def __init__(self, model_path: str):
        self.model_path = model_path
        scenario = type(self).active_scenario
        self.head = FakeHead(end2end=scenario.end2end, modern=scenario.modern)
        self.model = FakeInnerModel(self.head)
        self.task = scenario.task
        self.names = {0: "fish"}
        self.ckpt = {"train_args": {"imgsz": list(scenario.imgsz)}}
        self.predict_calls = []
        self.track_calls = []
        self.fuse_called = False
        self.half_called = False
        self.last_device = None
        type(self).instances.append(self)

    def to(self, device):
        self.last_device = device
        return self

    def fuse(self):
        self.fuse_called = True
        if self.head.end2end:
            self.head.fuse()
        return self

    def half(self):
        self.half_called = True
        return self

    def predict(self, images, stream=True, **kwargs):
        self.predict_calls.append({"images": images, "stream": stream, "kwargs": dict(kwargs)})
        return [FakePredictionResult() for _ in images]

    def track(self, image, tracker="bytetrack.yaml", persist=True, device=None, **kwargs):
        self.track_calls.append(
            {
                "image": image,
                "tracker": tracker,
                "persist": persist,
                "device": device,
                "kwargs": dict(kwargs),
            }
        )
        return [FakePredictionResult()]


class RuntimePolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fake_trex = FakeTRexModule()
        fake_detection_module = types.ModuleType("trex_detection_model")
        fake_detection_module.DetectionModel = FakeDetectionModel
        fake_detection_module.StrippedResults = FakeStrippedResults
        fake_detection_module.TRexDetection = FakeTRexDetection
        fake_detection_module.BBox = np.ndarray

        sys.modules["TRex"] = cls.fake_trex
        sys.modules["trex_detection_model"] = fake_detection_module
        sys.modules.pop("trex_yolo", None)
        sys.modules.pop("bbx_saved_model", None)

        cls.trex_yolo = importlib.import_module("trex_yolo")
        cls.stripped_yolo_results = cls.trex_yolo.StrippedYoloResults
        cls.trex_yolo.YOLO = FakeYOLO
        cls.trex_yolo.StrippedYoloResults = lambda result, scale, offset: types.SimpleNamespace(
            result=result, scale=scale, offset=offset
        )
        cls.bbx_saved_model = importlib.import_module("bbx_saved_model")

    def setUp(self):
        FakeYOLO.instances.clear()
        self.fake_trex.settings = {
            "gpu_torch_no_fixes": "true",
            "detect_conf_threshold": 0.1,
            "detect_iou_threshold": None,
            "detect_point_radii": "{}",
            "detect_keypoint_threshold": 0.1,
        }

    def _make_model(self, *, modern: bool, end2end: bool, use_tracking: bool = False):
        FakeYOLO.active_scenario = FakeYOLOScenario(modern=modern, end2end=end2end)
        config = FakeModelConfig(use_tracking=use_tracking, model_path="fake.pt")
        model = self.trex_yolo.YOLOModel(config)
        model.load()
        return model, FakeYOLO.instances[-1]

    def test_modern_end2end_profile_preserved_without_iou_override(self):
        model, backend = self._make_model(modern=True, end2end=True)
        self.assertEqual(model.runtime_profile, self.trex_yolo.RUNTIME_PROFILE_MODERN_END2END)
        self.assertTrue(backend.head.end2end)
        self.assertIsNone(backend.head.cv2)
        self.assertIsNone(backend.head.cv3)

        image = np.zeros((32, 48, 3), dtype=np.uint8)
        model.predict_boxes([image], imgsz=[512, 768], iou=0.7, agnostic_nms=True)
        call = backend.predict_calls[-1]["kwargs"]
        self.assertEqual(call["imgsz"], [512, 768])
        self.assertNotIn("iou", call)
        self.assertNotIn("agnostic_nms", call)

    def test_modern_end2end_forced_nms_when_iou_setting_present(self):
        self.fake_trex.settings["detect_iou_threshold"] = 0.4
        model, backend = self._make_model(modern=True, end2end=True)
        self.assertEqual(model.runtime_profile, self.trex_yolo.RUNTIME_PROFILE_MODERN_END2END_FORCED_NMS)
        self.assertFalse(backend.head.end2end)
        self.assertIsNotNone(backend.head.cv2)
        self.assertIsNotNone(backend.head.cv3)

        image = np.zeros((32, 48, 3), dtype=np.uint8)
        model.predict_boxes([image], imgsz=[512, 768], iou=0.4, agnostic_nms=True)
        call = backend.predict_calls[-1]["kwargs"]
        self.assertEqual(call["iou"], 0.4)
        self.assertTrue(call["agnostic_nms"])

    def test_legacy_head_keeps_classic_path(self):
        model, backend = self._make_model(modern=False, end2end=False)
        self.assertEqual(model.runtime_profile, self.trex_yolo.RUNTIME_PROFILE_LEGACY)
        self.assertFalse(backend.head.end2end)
        self.assertIsNotNone(backend.head.cv2)
        self.assertIsNotNone(backend.head.cv3)

        image = np.zeros((32, 48, 3), dtype=np.uint8)
        model.predict_boxes([image], iou=0.7, agnostic_nms=True)
        call = backend.predict_calls[-1]["kwargs"]
        self.assertEqual(call["iou"], 0.7)
        self.assertTrue(call["agnostic_nms"])

    def test_tracking_path_uses_same_policy(self):
        model, backend = self._make_model(modern=True, end2end=True, use_tracking=True)
        image = np.zeros((32, 48, 3), dtype=np.uint8)
        results = model.predict_boxes([image], iou=0.6, agnostic_nms=True)

        self.assertEqual(len(results), 1)
        call = backend.track_calls[-1]["kwargs"]
        self.assertNotIn("iou", call)
        self.assertNotIn("agnostic_nms", call)

    def test_bbx_saved_model_only_passes_iou_when_explicitly_set(self):
        calls = []

        class CaptureModel:
            def inference(self, input_value, **kwargs):
                calls.append({"input": input_value, "kwargs": dict(kwargs)})
                return ["ok"]

        self.bbx_saved_model.model = CaptureModel()

        self.fake_trex.settings["detect_iou_threshold"] = None
        result = self.bbx_saved_model.predict("frame")
        self.assertEqual(result, ["ok"])
        self.assertIsNone(calls[-1]["kwargs"]["iou_threshold"])

        self.fake_trex.settings["detect_iou_threshold"] = 0.55
        self.bbx_saved_model.predict("frame")
        self.assertEqual(calls[-1]["kwargs"]["iou_threshold"], 0.55)

    def test_keypoint_threshold_uses_missing_sentinel_and_keeps_float32_layout(self):
        torch = self.trex_yolo.torch

        class FakeKeypoints:
            def __init__(self, data):
                self.data = data

            def cpu(self):
                return self

        result = FakePredictionResult()
        result.keypoints = FakeKeypoints(torch.tensor(
            [[[1.0, 2.0, 0.2], [3.0, 4.0, 0.8], [0.0, 0.0, 0.9]]],
            dtype=torch.float32,
        ))
        self.fake_trex.settings["detect_keypoint_threshold"] = 0.5

        stripped = self.stripped_yolo_results(
            result,
            scale=np.array([2.0, 3.0], dtype=np.float32),
            offset=np.array([10.0, 20.0], dtype=np.float32),
        )

        self.assertIsNotNone(stripped.keypoints)
        self.assertEqual(len(stripped.keypoints), 1)
        keypoints = stripped.keypoints[0]
        self.assertEqual(keypoints.shape, (1, 3, 2))
        self.assertEqual(keypoints.dtype, np.float32)
        self.assertTrue(keypoints.flags["C_CONTIGUOUS"])
        np.testing.assert_array_equal(keypoints[0, 0], [0.0, 0.0])
        np.testing.assert_array_equal(keypoints[0, 1], [26.0, 72.0])
        np.testing.assert_array_equal(keypoints[0, 2], [0.0, 0.0])

    def test_mask_box_uses_floor_ceil_xyxy_and_exact_raster_size(self):
        torch = self.trex_yolo.torch
        result = FakePredictionResult()
        result.orig_shape = (8, 8)
        result.boxes = FakeBoxes(np.asarray(
            [[1.2, 1.4, 4.1, 5.2, 0.9, 1.0]],
            dtype=np.float32,
        ))
        result.masks = types.SimpleNamespace(
            data=torch.ones((1, 8, 8), dtype=torch.float32)
        )

        stripped = self.stripped_yolo_results(
            result,
            scale=np.array([1.0, 1.0], dtype=np.float32),
            offset=np.array([0.0, 0.0], dtype=np.float32),
        )

        np.testing.assert_array_equal(
            stripped.boxes[0, :4],
            np.asarray([1.0, 1.0, 5.0, 6.0], dtype=np.float32),
        )
        self.assertEqual(stripped.masks[0].shape, (5, 4))
        self.assertEqual(stripped.boxes.dtype, np.float32)
        self.assertTrue(stripped.boxes.flags["C_CONTIGUOUS"])


if __name__ == "__main__":
    unittest.main()
