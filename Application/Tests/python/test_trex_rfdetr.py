#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused tests for the RF-DETR adapter used by TRex."""

from __future__ import annotations

import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import torch


TEST_PYTHON_DIR = Path(__file__).resolve().parent
RUNTIME_PYTHON_DIR = (
    TEST_PYTHON_DIR.parent.parent / "src" / "tracker" / "python"
)
if str(RUNTIME_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(RUNTIME_PYTHON_DIR))


class FakeDetectResolution:
    def __init__(self, width: int = 0, height: int = 0):
        self.width = width
        self.height = height


class FakeKeypointFormat:
    def __init__(self, n_points: int = 0, n_dims: int = 0):
        self.n_points = n_points
        self.n_dims = n_dims


class FakeModelTaskType:
    detect = "detect"
    region = "region"


class FakeObjectDetectionFormat:
    none = "none"
    boxes = "boxes"
    masks = "masks"
    poses = "poses"


class FakeModelConfig:
    def __init__(self, model_path: str = "rf-detr-small.pth"):
        self.task = FakeModelTaskType.detect
        self.use_tracking = False
        self.try_optimize = False
        self.model_path = model_path
        self.trained_resolution = FakeDetectResolution()
        self.output_format = FakeObjectDetectionFormat.none
        self.keypoint_format = FakeKeypointFormat()
        self.classes = {}
        self.requires_exact_input_size = False


class FakeTRexModule(types.ModuleType):
    def __init__(self):
        super().__init__("TRex")
        self.ModelTaskType = FakeModelTaskType
        self.DetectResolution = FakeDetectResolution
        self.KeypointFormat = FakeKeypointFormat
        self.ObjectDetectionFormat = FakeObjectDetectionFormat
        self.ModelConfig = FakeModelConfig
        self.YoloInput = object
        self.Result = object
        self.settings = {
            "gpu_torch_device": "cpu",
            "gpu_torch_device_index": "-1",
            "detect_keypoint_threshold": 0.1,
        }

    @staticmethod
    def log(message: str) -> None:
        del message

    @staticmethod
    def warn(message: str) -> None:
        del message

    def setting(self, name: str):
        return self.settings[name]


class FakeDetectionModel:
    def __init__(self, config):
        self.config = config
        self.ptr = None
        self.device = torch.device("cpu")

    def reinit_device(self):
        self.device = torch.device("cpu")

    def load(self):
        return None


class FakeStrippedResults:
    def __init__(self, scale, offset):
        self.boxes = None
        self.keypoints = None
        self.masks = None
        self.orig_shape = None
        self.scale = np.asarray(scale, dtype=np.float32)
        self.offset = np.asarray(offset, dtype=np.float32)
        self.obb = None
        self.points = None
        self.locations = None


class FakeTRexDetection:
    def __init__(self, models):
        self.models = models


class FakeNetwork(torch.nn.Module):
    def forward(self, batch):
        batch_size = batch.shape[0]
        return {
            "pred_logits": torch.zeros((batch_size, 1, 2), device=batch.device),
            "pred_boxes": torch.zeros((batch_size, 1, 4), device=batch.device),
        }


class FakeSegmentationNetwork(torch.nn.Module):
    def forward(self, batch):
        batch_size = batch.shape[0]
        logits = torch.tensor(
            [[[4.0, -4.0], [3.0, -3.0]]],
            device=batch.device,
        ).expand(batch_size, -1, -1)
        boxes = torch.zeros((batch_size, 2, 4), device=batch.device)
        masks = torch.stack(
            (
                torch.ones((2, 2), device=batch.device),
                -torch.ones((2, 2), device=batch.device),
            )
        ).expand(batch_size, -1, -1, -1)
        return {
            "pred_logits": logits,
            "pred_boxes": boxes,
            "pred_masks": masks,
        }


class FakePostprocess:
    num_select = 300

    def __init__(self, results):
        self.results = results
        self.target_sizes = None
        self.raw = None

    def __call__(self, raw, target_sizes):
        self.raw = raw
        self.target_sizes = target_sizes.detach().cpu()
        return self.results


class FakeWrapper:
    def __init__(
        self,
        *,
        postprocess,
        resolution=8,
        segmentation=False,
        pose=False,
        keypoints=(0, 3),
    ):
        self.model_config = types.SimpleNamespace(
            resolution=resolution,
            patch_size=2,
            num_windows=2,
            num_channels=3,
            num_classes=2,
            segmentation_head=segmentation,
            use_grouppose_keypoints=pose,
            num_keypoints_per_class=list(keypoints),
        )
        self.model = types.SimpleNamespace(
            resolution=resolution,
            model=FakeNetwork(),
            inference_model=None,
            postprocess=postprocess,
            device=torch.device("cpu"),
            args=types.SimpleNamespace(
                num_classes=2,
                num_keypoints_per_class=list(keypoints),
            ),
            class_names=["fish", "crab"],
        )
        self.class_names = ["fish", "crab"]
        self.optimization_calls = []

    def optimize_for_inference(
        self,
        compile=True,
        batch_size=1,
        dtype=torch.float32,
        *,
        inplace=False,
    ):
        self.optimization_calls.append(
            {
                "compile": compile,
                "batch_size": batch_size,
                "dtype": dtype,
                "inplace": inplace,
            }
        )
        self.model.inference_model = self.model.model
        if inplace:
            self.model.model = None


class RFDETRAdapterTest(unittest.TestCase):
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
        sys.modules.pop("trex_rfdetr", None)
        cls.module = importlib.import_module("trex_rfdetr")

    def setUp(self):
        self.fake_trex.settings.update(
            {
                "gpu_torch_device": "cpu",
                "gpu_torch_device_index": "-1",
                "detect_keypoint_threshold": 0.1,
            }
        )

    def test_default_optimization_uses_memory_path(self):
        wrapper = FakeWrapper(postprocess=FakePostprocess([]))
        model = self.module.RFDETRModel(FakeModelConfig())

        model._finish_load(wrapper)

        self.assertEqual(model.optimization_mode, "memory")
        self.assertEqual(
            wrapper.optimization_calls,
            [
                {
                    "compile": False,
                    "batch_size": 1,
                    "dtype": torch.float32,
                    "inplace": True,
                }
            ],
        )
        self.assertIsNone(wrapper.model.model)
        self.assertIs(model.network, wrapper.model.inference_model)

    def test_try_optimize_uses_torchscript_and_splits_batches(self):
        class RecordingNetwork(FakeNetwork):
            def __init__(self):
                super().__init__()
                self.batch_sizes = []

            def forward(self, batch):
                self.batch_sizes.append(int(batch.shape[0]))
                output = super().forward(batch)
                return output["pred_boxes"], output["pred_logits"]

        result = {
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([0]),
            "boxes": torch.tensor([[1.0, 1.0, 2.0, 2.0]]),
        }
        wrapper = FakeWrapper(postprocess=FakePostprocess([result]))
        network = RecordingNetwork()
        wrapper.model.model = network
        config = FakeModelConfig()
        config.try_optimize = True
        model = self.module.RFDETRModel(config)

        model._finish_load(wrapper)
        outputs = model._predict_tile_results(
            [
                np.zeros((8, 8, 3), dtype=np.uint8),
                np.zeros((8, 8, 3), dtype=np.uint8),
            ],
            conf=0.0,
        )

        self.assertEqual(model.optimization_mode, "torchscript")
        self.assertEqual(
            wrapper.optimization_calls,
            [
                {
                    "compile": True,
                    "batch_size": 1,
                    "dtype": torch.float32,
                    "inplace": False,
                }
            ],
        )
        self.assertEqual(network.batch_sizes, [1, 1])
        self.assertEqual(len(outputs), 2)
        self.assertEqual(
            set(wrapper.model.postprocess.raw),
            {"pred_boxes", "pred_logits"},
        )

    def test_exported_optional_output_maps_to_keypoints_or_masks(self):
        boxes = torch.zeros((1, 1, 4))
        logits = torch.zeros((1, 1, 2))
        optional = torch.ones((1, 1, 3))

        pose = self.module.RFDETRModel(FakeModelConfig())
        pose.model_config = types.SimpleNamespace(use_grouppose_keypoints=True)
        pose_output = pose._normalize_network_output(
            (boxes, logits, optional)
        )
        self.assertEqual(
            set(pose_output),
            {"pred_boxes", "pred_logits", "pred_keypoints"},
        )
        self.assertIs(pose_output["pred_keypoints"], optional)

        segmentation = self.module.RFDETRModel(FakeModelConfig())
        segmentation.model_config = types.SimpleNamespace(
            use_grouppose_keypoints=False
        )
        segmentation_output = segmentation._normalize_network_output(
            (boxes, logits, optional)
        )
        self.assertEqual(
            set(segmentation_output),
            {"pred_boxes", "pred_logits", "pred_masks"},
        )
        self.assertIs(segmentation_output["pred_masks"], optional)

    def test_torchscript_exception_falls_back_to_memory_path(self):
        class FailingCompileWrapper(FakeWrapper):
            def optimize_for_inference(
                self,
                compile=True,
                batch_size=1,
                dtype=torch.float32,
                *,
                inplace=False,
            ):
                if compile:
                    self.optimization_calls.append(
                        {
                            "compile": compile,
                            "batch_size": batch_size,
                            "dtype": dtype,
                            "inplace": inplace,
                        }
                    )
                    raise RuntimeError("trace ran out of memory")
                super().optimize_for_inference(
                    compile=compile,
                    batch_size=batch_size,
                    dtype=dtype,
                    inplace=inplace,
                )

        wrapper = FailingCompileWrapper(postprocess=FakePostprocess([]))
        config = FakeModelConfig()
        config.try_optimize = True
        model = self.module.RFDETRModel(config)

        model._finish_load(wrapper)

        self.assertEqual(model.optimization_mode, "memory")
        self.assertEqual(
            [call["compile"] for call in wrapper.optimization_calls],
            [True, False],
        )
        self.assertTrue(wrapper.optimization_calls[-1]["inplace"])

    def test_checkpoint_probe_recognizes_rfdetr_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            rfdetr_path = Path(directory) / "checkpoint.pt"
            yolo_path = Path(directory) / "yolo.pt"
            torch.save(
                {
                    "model_name": "RFDETRSmall",
                    "args": {"pretrain_weights": "rf-detr-small.pth"},
                },
                rfdetr_path,
            )
            torch.save({"train_args": {"imgsz": 640}}, yolo_path)

            self.assertTrue(self.module.is_rfdetr_checkpoint(str(rfdetr_path)))
            self.assertFalse(self.module.is_rfdetr_checkpoint(str(yolo_path)))

    def test_loader_dispatch_keeps_yolo_and_rfdetr_paths_separate(self):
        fake_yolo = types.ModuleType("trex_yolo")

        class FakeYOLOModel(FakeDetectionModel):
            pass

        fake_yolo.YOLOModel = FakeYOLOModel
        previous_yolo = sys.modules.get("trex_yolo")
        previous_loader = sys.modules.get("bbx_saved_model")
        previous_rfdetr_package = sys.modules.get("rfdetr")
        sys.modules["trex_yolo"] = fake_yolo
        sys.modules.pop("bbx_saved_model", None)
        try:
            loader = importlib.import_module("bbx_saved_model")
            configs = [
                FakeModelConfig("ordinary-yolo.pt"),
                FakeModelConfig("rf-detr-nano.pth"),
            ]
            loaded = loader.load_yolo(configs)
            self.assertEqual(loaded, configs)
            self.assertIsInstance(loader.model.models[0], FakeYOLOModel)
            self.assertIsInstance(loader.model.models[1], self.module.RFDETRModel)
            self.assertIs(sys.modules.get("rfdetr"), previous_rfdetr_package)
        finally:
            if previous_yolo is None:
                sys.modules.pop("trex_yolo", None)
            else:
                sys.modules["trex_yolo"] = previous_yolo
            if previous_loader is None:
                sys.modules.pop("bbx_saved_model", None)
            else:
                sys.modules["bbx_saved_model"] = previous_loader

    def test_load_trusts_checkpoint_and_sets_existing_metadata(self):
        postprocess = FakePostprocess([])
        wrapper = FakeWrapper(postprocess=postprocess, resolution=8)
        calls = []

        fake_rfdetr = types.ModuleType("rfdetr")

        class FakeRFDETR:
            @staticmethod
            def from_checkpoint(path, trust_checkpoint):
                calls.append((path, trust_checkpoint))
                return wrapper

        fake_rfdetr.RFDETR = FakeRFDETR
        previous = sys.modules.get("rfdetr")
        sys.modules["rfdetr"] = fake_rfdetr
        try:
            config = FakeModelConfig("model.pth")
            model = self.module.RFDETRModel(config)
            model.load()
        finally:
            if previous is None:
                sys.modules.pop("rfdetr", None)
            else:
                sys.modules["rfdetr"] = previous

        self.assertEqual(calls, [("model.pth", True)])
        self.assertEqual(config.trained_resolution.width, 8)
        self.assertEqual(config.trained_resolution.height, 8)
        self.assertEqual(config.output_format, FakeObjectDetectionFormat.boxes)
        self.assertEqual(config.classes, {0: "fish", 1: "crab"})
        self.assertTrue(config.requires_exact_input_size)

    def test_load_supports_rfdetr_1_8_3_unrestricted_loader_signature(self):
        wrapper = FakeWrapper(postprocess=FakePostprocess([]), resolution=8)
        calls = []
        fake_rfdetr = types.ModuleType("rfdetr")

        class FakeRFDETR:
            @staticmethod
            def from_checkpoint(path):
                calls.append(path)
                return wrapper

        fake_rfdetr.RFDETR = FakeRFDETR
        previous = sys.modules.get("rfdetr")
        sys.modules["rfdetr"] = fake_rfdetr
        try:
            model = self.module.RFDETRModel(FakeModelConfig("model.pth"))
            model.load()
        finally:
            if previous is None:
                sys.modules.pop("rfdetr", None)
            else:
                sys.modules["rfdetr"] = previous

        self.assertEqual(calls, ["model.pth"])

    def test_load_recovers_missing_keypoint_resolution_from_checkpoint_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint_best_total.pth"
            torch.save(
                {
                    "model_name": "RFDETRKeypointPreview",
                    "args": {"square_resize_div_64": True},
                    "model": {
                        (
                            "backbone.0.encoder.encoder.embeddings."
                            "position_embeddings"
                        ): torch.zeros((1, 40001, 1)),
                        (
                            "backbone.0.encoder.encoder.embeddings."
                            "patch_embeddings.projection.weight"
                        ): torch.zeros((1, 3, 12, 12)),
                    },
                },
                checkpoint_path,
            )

            wrapper = FakeWrapper(
                postprocess=FakePostprocess([]),
                resolution=2400,
                pose=True,
                keypoints=(8,),
            )
            calls = []
            fake_rfdetr = types.ModuleType("rfdetr")

            class FakeRFDETR:
                @staticmethod
                def from_checkpoint(path, **kwargs):
                    calls.append((path, kwargs))
                    return wrapper

            fake_rfdetr.RFDETR = FakeRFDETR
            previous = sys.modules.get("rfdetr")
            sys.modules["rfdetr"] = fake_rfdetr
            try:
                config = FakeModelConfig(str(checkpoint_path))
                model = self.module.RFDETRModel(config)
                model.load()
            finally:
                if previous is None:
                    sys.modules.pop("rfdetr", None)
                else:
                    sys.modules["rfdetr"] = previous

        self.assertEqual(
            calls,
            [
                (
                    str(checkpoint_path),
                    {
                        "resolution": 2400,
                        "positional_encoding_size": 200,
                        "patch_size": 12,
                    },
                )
            ],
        )
        self.assertEqual(config.trained_resolution.width, 2400)
        self.assertEqual(config.trained_resolution.height, 2400)

    def test_preprocessing_preserves_cpp_sized_shape_and_normalizes_rgb(self):
        model = self.module.RFDETRModel(FakeModelConfig())
        model.device = torch.device("cpu")
        model.input_height = 2
        model.input_width = 2
        model.input_channels = 3
        model.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        model.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        rgb[..., 0] = 255
        batch, sizes = model._prepare_batch([rgb])

        self.assertEqual(tuple(batch.shape), (1, 3, 2, 2))
        self.assertEqual(sizes, [(2, 2)])
        self.assertAlmostEqual(float(batch[0, 0, 0, 0]), (1.0 - 0.485) / 0.229, places=5)
        self.assertAlmostEqual(float(batch[0, 2, 0, 0]), (0.0 - 0.406) / 0.225, places=5)

        with self.assertRaisesRegex(ValueError, "C\\+\\+ detector image size"):
            model._prepare_batch([np.zeros((2, 3, 3), dtype=np.uint8)])

    def test_preprocessing_preserves_cpp_rgb_and_drops_alpha(self):
        model = self.module.RFDETRModel(FakeModelConfig())
        rgba = np.array([[[11, 22, 33, 255]]], dtype=np.uint8)

        rgb = model.preprocess_image(rgba)

        np.testing.assert_array_equal(
            rgb,
            np.array([[[11, 22, 33]]], dtype=np.uint8),
        )
        self.assertTrue(rgb.flags.c_contiguous)

    def test_segmentation_pose_and_public_class_metadata(self):
        segmentation_config = FakeModelConfig()
        segmentation = self.module.RFDETRModel(segmentation_config)
        segmentation._finish_load(
            FakeWrapper(
                postprocess=FakePostprocess([]),
                segmentation=True,
            )
        )
        self.assertEqual(
            segmentation_config.output_format,
            FakeObjectDetectionFormat.masks,
        )

        pose_config = FakeModelConfig()
        pose = self.module.RFDETRModel(pose_config)
        pose._finish_load(
            FakeWrapper(
                postprocess=FakePostprocess([]),
                pose=True,
                keypoints=(0, 3, 2),
            )
        )
        self.assertEqual(pose_config.output_format, FakeObjectDetectionFormat.poses)
        self.assertEqual(pose_config.keypoint_format.n_points, 3)
        self.assertEqual(
            pose_config.classes,
            {0: "__background__", 1: "fish", 2: "crab"},
        )

        mapped_config = FakeModelConfig()
        mapped_wrapper = FakeWrapper(postprocess=FakePostprocess([]))
        mapped_wrapper.class_names = {1: "person", 18: "dog"}
        mapped = self.module.RFDETRModel(mapped_config)
        mapped._finish_load(mapped_wrapper)
        self.assertEqual(mapped_config.classes, {1: "person", 18: "dog"})

    def test_detection_coordinates_follow_trex_tile_affine(self):
        result = {
            "scores": torch.tensor([0.9, 0.1]),
            "labels": torch.tensor([1, 0]),
            "boxes": torch.tensor([[1.0, 2.0, 4.0, 6.0], [0.0, 0.0, 1.0, 1.0]]),
        }
        postprocess = FakePostprocess([result])
        wrapper = FakeWrapper(postprocess=postprocess, resolution=8)
        model = self.module.RFDETRModel(FakeModelConfig())
        model._finish_load(wrapper)

        image = np.zeros((8, 8, 3), dtype=np.uint8)
        output = model.predict(
            [image],
            scales=[np.array([2.0, 3.0])],
            offsets=[np.array([5.0, 7.0])],
            conf=0.5,
        )

        self.assertEqual(len(output), 1)
        np.testing.assert_allclose(
            output[0].boxes,
            np.array([[12.0, 27.0, 18.0, 39.0, 0.9, 1.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(postprocess.target_sizes, np.array([[8, 8]]))

    def test_fused_score_transform_is_bounded_monotonic_and_invertible(self):
        raw_scores = torch.tensor([0.0, 0.1, 0.21, 0.25, 1.0, 4.1])

        confidences = self.module._bounded_fused_confidence(raw_scores)

        self.assertTrue(bool(torch.all(confidences >= 0)))
        self.assertTrue(bool(torch.all(confidences < 1)))
        self.assertTrue(bool(torch.all(confidences[1:] > confidences[:-1])))
        torch.testing.assert_close(
            confidences,
            raw_scores / (1.0 + raw_scores),
        )
        torch.testing.assert_close(
            confidences / (1.0 - confidences),
            raw_scores,
        )
        self.assertAlmostEqual(
            float(self.module._bounded_fused_confidence(torch.tensor(0.25))),
            0.2,
        )
        example_confidences = self.module._bounded_fused_confidence(
            torch.tensor([4.1, 0.21, 0.10])
        )
        torch.testing.assert_close(
            example_confidences,
            torch.tensor([4.1 / 5.1, 0.21 / 1.21, 0.10 / 1.10]),
        )

        threshold_scores = torch.tensor([0.25, 0.2501])
        threshold_result = {
            "scores": self.module._bounded_fused_confidence(threshold_scores),
            "raw_scores": threshold_scores,
            "labels": torch.zeros(2, dtype=torch.int64),
            "boxes": torch.tensor(
                [[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0]]
            ),
        }
        filtered = self.module._filter_result(
            threshold_result,
            threshold=0.2,
            max_det=0,
        )
        torch.testing.assert_close(
            filtered["raw_scores"],
            torch.tensor([0.2501]),
        )

    def test_filter_applies_foreground_confidence_nms_then_max_det(self):
        raw_scores = torch.tensor([4.0, 3.0, 2.0, 1.0, 0.05])
        result = {
            "scores": self.module._bounded_fused_confidence(raw_scores),
            "raw_scores": raw_scores,
            "labels": torch.tensor([2, 0, 1, 0, 0]),
            "boxes": torch.tensor(
                [
                    [0.0, 0.0, 8.0, 8.0],
                    [0.0, 0.0, 4.0, 4.0],
                    [0.1, 0.1, 4.1, 4.1],
                    [5.0, 5.0, 8.0, 8.0],
                    [1.0, 1.0, 2.0, 2.0],
                ]
            ),
            "queries": torch.tensor([10, 11, 12, 13, 14]),
            "keypoints": torch.arange(30, dtype=torch.float32).reshape(5, 2, 3),
            "masks": torch.arange(20, dtype=torch.float32).reshape(5, 2, 2),
        }

        native = self.module._filter_result(
            result,
            threshold=0.1,
            max_det=0,
            valid_class_ids=[0, 1],
        )
        self.assertEqual(native["queries"].tolist(), [11, 12, 13])

        filtered = self.module._filter_result(
            result,
            threshold=0.1,
            max_det=2,
            valid_class_ids=[0, 1],
            iou_threshold=0.5,
        )
        bounded_rank_result = dict(result)
        bounded_rank_result.pop("raw_scores")
        bounded_rank_filtered = self.module._filter_result(
            bounded_rank_result,
            threshold=0.1,
            max_det=2,
            valid_class_ids=[0, 1],
            iou_threshold=0.5,
        )

        # The raw-score 2.0 box is suppressed despite having a different class,
        # proving that the explicit TRex override is class agnostic. max_det is
        # applied afterwards, so the distant raw-score 1.0 box is retained.
        self.assertEqual(filtered["queries"].tolist(), [11, 13])
        self.assertEqual(
            bounded_rank_filtered["queries"].tolist(),
            filtered["queries"].tolist(),
        )
        self.assertEqual(filtered["labels"].tolist(), [0, 0])
        torch.testing.assert_close(
            filtered["scores"],
            self.module._bounded_fused_confidence(torch.tensor([3.0, 1.0])),
        )
        torch.testing.assert_close(
            filtered["raw_scores"],
            torch.tensor([3.0, 1.0]),
        )
        torch.testing.assert_close(filtered["keypoints"], result["keypoints"][[1, 3]])
        torch.testing.assert_close(filtered["masks"], result["masks"][[1, 3]])

    def test_max_det_uses_native_fused_score_order_without_nms(self):
        raw_scores = torch.tensor([0.2, 4.0, 1.0])
        result = {
            "scores": self.module._bounded_fused_confidence(raw_scores),
            "raw_scores": raw_scores,
            "labels": torch.tensor([0, 0, 0]),
            "boxes": torch.tensor(
                [
                    [0.0, 0.0, 1.0, 1.0],
                    [2.0, 2.0, 3.0, 3.0],
                    [4.0, 4.0, 5.0, 5.0],
                ]
            ),
            "queries": torch.tensor([10, 11, 12]),
        }

        filtered = self.module._filter_result(
            result,
            threshold=0.0,
            max_det=2,
            valid_class_ids=[0],
            iou_threshold=None,
        )

        self.assertEqual(filtered["queries"].tolist(), [11, 12])
        torch.testing.assert_close(
            filtered["raw_scores"],
            torch.tensor([4.0, 1.0]),
        )

    def test_bounded_scores_preserve_keypoint_top20(self):
        raw_scores = torch.tensor(
            [4.0, 3.0, 2.0, 1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
            + [0.2 - index * 0.005 for index in range(15)],
            dtype=torch.float32,
        )
        bounded_scores = self.module._bounded_fused_confidence(raw_scores)
        self.assertEqual(
            torch.argsort(raw_scores, descending=True)[:20].tolist(),
            torch.argsort(bounded_scores, descending=True)[:20].tolist(),
        )

    def test_bounded_scores_preserve_bbox_map(self):
        try:
            from torchmetrics.detection import MeanAveragePrecision
        except ImportError as error:
            self.skipTest(f"torchmetrics detection evaluation unavailable: {error}")

        boxes = torch.tensor(
            [
                [10.0, 10.0, 30.0, 30.0],
                [11.0, 11.0, 31.0, 31.0],
                [60.0, 60.0, 75.0, 75.0],
            ],
            dtype=torch.float32,
        )
        labels = torch.zeros(3, dtype=torch.int64)
        targets = [{
            "boxes": torch.tensor(
                [[10.0, 10.0, 30.0, 30.0]],
                dtype=torch.float32,
            ),
            "labels": torch.zeros(1, dtype=torch.int64),
        }]

        def evaluate(scores):
            metric = MeanAveragePrecision(
                backend="faster_coco_eval",
                max_detection_thresholds=[1, 10, 100],
            )
            metric.update(
                [{"boxes": boxes, "scores": scores, "labels": labels}],
                targets,
            )
            return metric.compute()

        try:
            native_metrics = evaluate(torch.tensor([4.0, 0.2, 0.1]))
            bounded_metrics = evaluate(
                self.module._bounded_fused_confidence(
                    torch.tensor([4.0, 0.2, 0.1])
                )
            )
        except ModuleNotFoundError as error:
            self.skipTest(f"COCO evaluation backend unavailable: {error}")
        for key in ("map", "map_50", "map_75", "mar_100"):
            torch.testing.assert_close(
                bounded_metrics[key],
                native_metrics[key],
                rtol=0,
                atol=0,
            )

    def test_model_filters_no_object_sentinel_from_public_results(self):
        result = {
            "scores": torch.tensor([0.9, 0.8]),
            "labels": torch.tensor([0, 1]),
            "boxes": torch.tensor(
                [[1.0, 1.0, 3.0, 3.0], [4.0, 4.0, 6.0, 6.0]]
            ),
        }
        postprocess = FakePostprocess([result])
        wrapper = FakeWrapper(postprocess=postprocess, resolution=8)
        wrapper.class_names = ["fish"]
        wrapper.model.class_names = ["fish"]
        wrapper.model.args.num_classes = 1
        wrapper.model_config.num_classes = 1
        model = self.module.RFDETRModel(FakeModelConfig())
        model._finish_load(wrapper)

        output = model._predict_tile_results(
            [np.zeros((8, 8, 3), dtype=np.uint8)],
            conf=0.0,
        )

        self.assertEqual(output[0]["labels"].tolist(), [0])

    def test_keypoint_threshold_is_independent_from_detection_threshold(self):
        result = {
            "scores": torch.tensor([2.5]),
            "labels": torch.tensor([1]),
            "boxes": torch.tensor([[1.0, 1.0, 5.0, 5.0]]),
            "keypoints": torch.tensor(
                [[[2.0, 3.0, 0.09], [4.0, 5.0, 0.10], [6.0, 7.0, 0.75]]]
            ),
        }
        postprocess = FakePostprocess([result])
        wrapper = FakeWrapper(
            postprocess=postprocess,
            resolution=8,
            pose=True,
            keypoints=(0, 3),
        )
        model = self.module.RFDETRModel(FakeModelConfig())
        model._finish_load(wrapper)

        output = model._predict_tile_results(
            [np.zeros((8, 8, 3), dtype=np.uint8)],
            conf=0.7,
        )[0]

        self.assertAlmostEqual(float(output["raw_scores"][0]), 2.5)
        self.assertAlmostEqual(float(output["scores"][0]), 2.5 / 3.5)
        torch.testing.assert_close(
            output["keypoints"],
            torch.tensor([[[0.0, 0.0, 0.0], [4.0, 5.0, 0.10], [6.0, 7.0, 0.75]]]),
        )

    def test_segmentation_filters_queries_before_mask_upsampling(self):
        result = {
            "scores": torch.sigmoid(torch.tensor([4.0, 3.0, -3.0, -4.0])),
            "labels": torch.tensor([0, 0, 1, 1]),
            "boxes": torch.tensor(
                [
                    [1.0, 1.0, 7.0, 7.0],
                    [2.0, 2.0, 6.0, 6.0],
                    [2.0, 2.0, 6.0, 6.0],
                    [1.0, 1.0, 7.0, 7.0],
                ]
            ),
        }
        postprocess = FakePostprocess([result])
        wrapper = FakeWrapper(
            postprocess=postprocess,
            resolution=8,
            segmentation=True,
        )
        wrapper.model.model = FakeSegmentationNetwork()
        model = self.module.RFDETRModel(FakeModelConfig())
        model._finish_load(wrapper)

        output = model._predict_tile_results(
            [np.zeros((8, 8, 3), dtype=np.uint8)],
            conf=0.9,
            max_det=1,
        )

        self.assertEqual(output[0]["boxes"].shape[0], 1)
        self.assertEqual(tuple(output[0]["masks"].shape), (1, 1, 8, 8))
        self.assertTrue(bool(output[0]["masks"].all()))

    def test_mask_is_bbox_local_uint8_and_pose_keeps_invalid_zero(self):
        mask = torch.zeros((1, 1, 8, 8), dtype=torch.bool)
        mask[0, 0, 1:4, 1:5] = True
        mask_result = {
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([0]),
            "boxes": torch.tensor([[1.0, 1.0, 5.0, 4.0]]),
            "masks": mask,
        }
        stripped_mask = self.module.StrippedRFDETRResults(
            mask_result,
            scale=np.array([1.0, 1.0]),
            offset=np.array([0.0, 0.0]),
            image_shape=(8, 8),
        )
        self.assertEqual(stripped_mask.masks[0].shape, (3, 4))
        self.assertEqual(stripped_mask.masks[0].dtype, np.uint8)
        self.assertTrue(np.all(stripped_mask.masks[0] == 255))

        pose_result = {
            "scores": torch.tensor([0.8]),
            "labels": torch.tensor([0]),
            "boxes": torch.tensor([[1.0, 1.0, 5.0, 5.0]]),
            "keypoints": torch.tensor([[[2.0, 3.0, 0.9], [0.0, 0.0, 0.0]]]),
        }
        stripped_pose = self.module.StrippedRFDETRResults(
            pose_result,
            scale=np.array([2.0, 3.0]),
            offset=np.array([5.0, 7.0]),
            image_shape=(8, 8),
        )
        np.testing.assert_allclose(stripped_pose.keypoints[0][0, 0], [14.0, 30.0])
        np.testing.assert_array_equal(stripped_pose.keypoints[0][0, 1], [0.0, 0.0])

    def test_mask_box_uses_floor_ceil_source_bounds_and_exact_raster_size(self):
        result = {
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([1]),
            "boxes": torch.tensor([[1.2, 1.4, 4.1, 5.2]]),
            "masks": torch.ones((1, 1, 8, 8), dtype=torch.bool),
        }

        stripped = self.module.StrippedRFDETRResults(
            result,
            scale=np.array([2.0, 3.0], dtype=np.float32),
            offset=np.array([5.0, 7.0], dtype=np.float32),
            image_shape=(8, 8),
        )

        np.testing.assert_array_equal(
            stripped.boxes[0, :4],
            np.array([12.0, 25.0, 19.0, 37.0], dtype=np.float32),
        )
        self.assertEqual(stripped.masks[0].shape, (12, 7))
        self.assertEqual(stripped.boxes.dtype, np.float32)
        self.assertTrue(stripped.boxes.flags["C_CONTIGUOUS"])


if __name__ == "__main__":
    unittest.main()
