#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Edge-case tests for trex_detection_model aggregation behavior."""

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
    def __init__(self, width: int = 32, height: int = 32):
        self.width = width
        self.height = height


class FakeModelTaskType:
    detect = "detect"
    region = "region"


class FakeObjectDetectionFormat:
    boxes = "boxes"


class FakeModelConfig:
    def __init__(self, task: str = FakeModelTaskType.detect):
        self.task = task
        self.trained_resolution = FakeDetectResolution()
        self.output_format = FakeObjectDetectionFormat.boxes
        self.classes = {}


class CapturedBoxes:
    def __init__(self, data):
        self.data = data


class CapturedKeypoints:
    def __init__(self, data):
        self.data = data


class CapturedObbs:
    def __init__(self, data):
        self.data = data


class CapturedPoints:
    def __init__(self, data):
        self.data = data


class CapturedResult:
    def __init__(self, index, boxes, masks, keypoints, obbs=None, points=None):
        self.index = index
        self.boxes = boxes
        self.masks = masks
        self.keypoints = keypoints
        self.obbs = obbs if obbs is not None else CapturedObbs(np.empty((0, 7), dtype=np.float32))
        self.points = points if points is not None else CapturedPoints(np.empty((0, 5), dtype=np.float32))


class FakeTRexModule(types.ModuleType):
    def __init__(self):
        super().__init__("TRex")
        self.ModelTaskType = FakeModelTaskType
        self.DetectResolution = FakeDetectResolution
        self.ObjectDetectionFormat = FakeObjectDetectionFormat
        self.ModelConfig = FakeModelConfig
        self.Result = CapturedResult
        self.Boxes = CapturedBoxes
        self.KeypointData = CapturedKeypoints
        self.ObbData = CapturedObbs
        self.PointData = CapturedPoints
        self.settings = {
            "gpu_torch_device": "",
            "gpu_torch_device_index": "-1",
            "detect_point_radii": "{}",
        }

    def setting(self, name: str):
        return self.settings[name]

    @staticmethod
    def log(message: str) -> None:
        del message

    @staticmethod
    def warn(message: str) -> None:
        del message

    @staticmethod
    def tile_affines(geometries):
        scales = [np.array([1.0, 1.0], dtype=np.float32) for _ in geometries]
        offsets = [np.array([0.0, 0.0], dtype=np.float32) for _ in geometries]
        return scales, offsets


class FakeTorchModule(types.ModuleType):
    def __init__(self):
        super().__init__("torch")
        self.float32 = "float32"
        self.int32 = "int32"
        self.float64 = "float64"
        self.int64 = "int64"
        self.float16 = "float16"
        self.int16 = "int16"
        self.uint8 = "uint8"
        self.cuda = types.SimpleNamespace(is_available=lambda: False)
        self.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))

    @staticmethod
    def device(name: str):
        return types.SimpleNamespace(type=str(name).split(":", maxsplit=1)[0])


class FakeCv2Module(types.ModuleType):
    COLOR_BGR2RGB = 1

    def __init__(self):
        super().__init__("cv2")

    @staticmethod
    def cvtColor(image, code):
        if code != FakeCv2Module.COLOR_BGR2RGB:
            raise ValueError(f"Unsupported conversion code {code}.")

        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 image, got {array.shape}.")
        if array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError(f"Expected non-empty image, got {array.shape}.")
        return array[..., ::-1]


class FakeTrexUtilsModule(types.ModuleType):
    def __init__(self):
        super().__init__("trex_utils")

    @staticmethod
    def asarray(value, copy=False, dtype=None):
        if copy or dtype is not None:
            return np.array(value, copy=copy, dtype=dtype)
        return np.asarray(value)


class FakeYoloInput:
    def __init__(self, images, orig_ids=None):
        self._images = list(images)
        self._geometries = [object() for _ in self._images]
        self._orig_ids = list(range(len(self._images))) if orig_ids is None else list(orig_ids)

    def tile_geometries(self):
        return self._geometries

    def orig_id(self):
        return self._orig_ids

    def images(self):
        return self._images


class FakeDetectionModel:
    def __init__(self, responses, *, task=FakeModelTaskType.detect):
        self.responses = list(responses)
        self.task = task
        self.config = FakeModelConfig(task=task)
        self.device = types.SimpleNamespace(type="cpu")
        self.predict_calls = []

    def load(self):
        return None

    def predict(self, images, scales, offsets, **kwargs):
        self.predict_calls.append(
            {
                "images": list(images),
                "scales": list(scales),
                "offsets": list(offsets),
                "kwargs": dict(kwargs),
            }
        )
        return list(self.responses)

    def predict_boxes(self, images, **kwargs):
        del kwargs
        return [np.zeros((0, 4), dtype=np.float32) for _ in images]


def install_fakes():
    sys.modules["TRex"] = FakeTRexModule()
    sys.modules["torch"] = FakeTorchModule()
    sys.modules["cv2"] = FakeCv2Module()
    sys.modules["trex_utils"] = FakeTrexUtilsModule()
    sys.modules.pop("trex_detection_model", None)
    return importlib.import_module("trex_detection_model")


def make_image(height: int, width: int):
    return np.zeros((height, width, 3), dtype=np.uint8)


class TrexDetectionModelEdgeCaseTest(unittest.TestCase):
    def setUp(self):
        # install_fakes() overwrites these entries in the global sys.modules; snapshot
        # the originals and restore them afterwards so the fakes cannot leak into any
        # other test module that runs in the same interpreter.
        fake_names = ("TRex", "torch", "cv2", "trex_utils", "trex_detection_model")
        saved = {name: sys.modules.get(name) for name in fake_names}

        def restore():
            for name, module in saved.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

        self.addCleanup(restore)
        self.module = install_fakes()

    def stripped(self, *, boxes=None, masks=None, keypoints=None, obb=None, points=None):
        result = self.module.StrippedResults(
            scale=np.array([1.0, 1.0], dtype=np.float32),
            offset=np.array([0.0, 0.0], dtype=np.float32),
        )
        result.boxes = boxes
        result.masks = masks
        result.keypoints = keypoints
        result.obb = obb
        result.points = points
        return result

    def make_detection(self, responses):
        return self.module.TRexDetection([FakeDetectionModel(responses)])

    def test_device_index_accepts_typed_setting_from_cpp(self):
        fake_trex = sys.modules["TRex"]
        fake_trex.settings["gpu_torch_device"] = "cpu"
        fake_trex.settings["gpu_torch_device_index"] = -1

        model = self.module.DetectionModel(FakeModelConfig())

        self.assertEqual(model.device.type, "cpu")

    def assert_downstream_ready_empty_results(self, results, expected_count: int):
        self.assertEqual(
            len(results),
            expected_count,
            "TRexDetection must return exactly one TRex.Result per original image "
            "after tile grouping; mismatched counts make C++ receive() index the wrong frame.",
        )
        for result_index, result in enumerate(results):
            self.assertIsInstance(
                result.boxes.data,
                np.ndarray,
                f"Result {result_index} boxes must be a numpy array before TRex.Boxes reaches C++.",
            )
            self.assertEqual(
                result.boxes.data.ndim,
                2,
                f"Result {result_index} boxes must be 2D with shape (N, 6).",
            )
            self.assertEqual(
                result.boxes.data.shape,
                (0, 6),
                f"Result {result_index} empty detections must be encoded as shape (0, 6), "
                "not None, [], or a one-dimensional empty array.",
            )
            self.assertEqual(
                result.boxes.data.dtype,
                np.float32,
                f"Result {result_index} boxes must be float32 for downstream C++ conversion.",
            )
            self.assertTrue(
                result.boxes.data.flags["C_CONTIGUOUS"],
                f"Result {result_index} boxes must be C-contiguous for downstream conversion.",
            )
            self.assertIsInstance(
                result.masks,
                list,
                f"Result {result_index} masks must be a list even when no masks are present.",
            )
            self.assertIsNotNone(
                result.keypoints,
                f"Result {result_index} must carry a keypoint payload wrapper.",
            )
            self.assertIsNotNone(
                result.obbs,
                f"Result {result_index} must carry an OBB payload wrapper.",
            )
            self.assertIsNotNone(
                result.points,
                f"Result {result_index} must carry a point payload wrapper.",
            )

    def test_empty_detections_return_empty_result_for_image(self):
        detections = self.make_detection(
            [self.stripped(boxes=np.zeros((0, 6), dtype=np.float32))]
        )

        results = detections.inference(FakeYoloInput([make_image(4, 4)]))

        self.assert_downstream_ready_empty_results(results, expected_count=1)

    def test_missing_boxes_are_treated_as_empty_detections(self):
        detections = self.make_detection([self.stripped()])

        results = detections.inference(FakeYoloInput([make_image(4, 4)]))

        self.assert_downstream_ready_empty_results(results, expected_count=1)

    def test_empty_image_does_not_leak_preprocessing_crash(self):
        detections = self.make_detection(
            [self.stripped(boxes=np.zeros((0, 6), dtype=np.float32))]
        )

        with self.assertRaisesRegex(ValueError, "TRexDetection\\.inference received empty image"):
            detections.inference(FakeYoloInput([make_image(0, 0)]))

    def test_very_small_images_return_one_result_per_image(self):
        images = [make_image(1, 1), make_image(1, 2), make_image(2, 1)]
        detections = self.make_detection(
            [self.stripped(boxes=np.zeros((0, 6), dtype=np.float32)) for _ in images]
        )

        results = detections.inference(FakeYoloInput(images))

        self.assert_downstream_ready_empty_results(results, expected_count=len(images))

    def test_multiple_tiles_for_one_image_return_one_downstream_ready_result(self):
        images = [make_image(4, 4), make_image(4, 4)]
        detections = self.make_detection(
            [self.stripped(boxes=np.zeros((0, 6), dtype=np.float32)) for _ in images]
        )

        results = detections.inference(FakeYoloInput(images, orig_ids=[0, 0]))

        self.assert_downstream_ready_empty_results(results, expected_count=1)

    def test_empty_tile_does_not_drop_nonempty_tile_boxes(self):
        images = [make_image(4, 4), make_image(4, 4)]
        nonempty_boxes = np.array([[1.0, 2.0, 3.0, 4.0, 0.9, 1.0]], dtype=np.float32)
        detections = self.make_detection(
            [
                self.stripped(boxes=np.zeros((0, 6), dtype=np.float32)),
                self.stripped(boxes=nonempty_boxes),
            ]
        )

        results = detections.inference(FakeYoloInput(images, orig_ids=[0, 0]))

        self.assertEqual(
            len(results),
            1,
            "TRexDetection must return one grouped result for two tiles with the same orig_id.",
        )
        self.assertEqual(
            results[0].boxes.data.shape,
            (1, 6),
            "Empty tile boxes must not cause non-empty tile boxes in the same image group to be dropped.",
        )
        self.assertEqual(results[0].boxes.data.dtype, np.float32)
        self.assertTrue(results[0].boxes.data.flags["C_CONTIGUOUS"])
        np.testing.assert_array_equal(results[0].boxes.data, nonempty_boxes)

    def test_missing_tile_boxes_do_not_drop_neighboring_valid_boxes(self):
        images = [make_image(4, 4) for _ in range(4)]
        expected_boxes = np.array(
            [
                [1.0, 2.0, 3.0, 4.0, 0.95, 1.0],
                [5.0, 6.0, 7.0, 8.0, 0.85, 2.0],
                [9.0, 10.0, 11.0, 12.0, 0.75, 3.0],
            ],
            dtype=np.float32,
        )
        detections = self.make_detection(
            [
                self.stripped(boxes=expected_boxes[0:1]),
                self.stripped(boxes=expected_boxes[1:2]),
                self.stripped(boxes=None),
                self.stripped(boxes=expected_boxes[2:3]),
            ]
        )

        results = detections.inference(FakeYoloInput(images, orig_ids=[0, 0, 0, 0]))

        self.assertEqual(
            len(results),
            1,
            "TRexDetection must return one grouped result for tile results sharing one orig_id.",
        )
        self.assertEqual(
            results[0].boxes.data.shape,
            (3, 6),
            "A tile result with boxes=None must be treated as zero detections without "
            "dropping valid neighboring tile boxes.",
        )
        self.assertEqual(results[0].boxes.data.dtype, np.float32)
        self.assertTrue(results[0].boxes.data.flags["C_CONTIGUOUS"])
        np.testing.assert_array_equal(results[0].boxes.data, expected_boxes)

    def test_too_few_model_outputs_is_explicit_error(self):
        detections = self.make_detection(
            [self.stripped(boxes=np.zeros((0, 6), dtype=np.float32))]
        )

        with self.assertRaisesRegex(ValueError, "one model result per input tile/image"):
            detections.inference(FakeYoloInput([make_image(4, 4), make_image(4, 4)]))

    def test_too_many_model_outputs_is_explicit_error(self):
        detections = self.make_detection(
            [
                self.stripped(boxes=np.zeros((0, 6), dtype=np.float32)),
                self.stripped(boxes=np.zeros((0, 6), dtype=np.float32)),
            ]
        )

        with self.assertRaisesRegex(ValueError, "one model result per input tile/image"):
            detections.inference(FakeYoloInput([make_image(4, 4)]))

    def test_malformed_box_shape_is_explicit_error(self):
        detections = self.make_detection(
            [self.stripped(boxes=np.zeros((0, 4), dtype=np.float32))]
        )

        with self.assertRaisesRegex(ValueError, "shape \\(N, 6\\)"):
            detections.inference(FakeYoloInput([make_image(4, 4)]))

    def test_region_proposal_empty_boxes_return_empty_result_per_image(self):
        region_model = FakeDetectionModel([], task=FakeModelTaskType.region)
        detect_model = FakeDetectionModel([], task=FakeModelTaskType.detect)
        detections = self.module.TRexDetection([region_model, detect_model])

        results = detections.inference(FakeYoloInput([make_image(4, 4), make_image(5, 5)]))

        self.assert_downstream_ready_empty_results(results, expected_count=2)


if __name__ == "__main__":
    unittest.main()
