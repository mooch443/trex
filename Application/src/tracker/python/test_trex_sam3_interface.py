#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the almost-stateless Python SAM3 adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import torch

import trex_sam3_interface as sam3


class FakeResult:
    def __init__(self, frame_index, boxes, masks, keypoints, obb, points):
        self.frame_index = frame_index
        self.boxes = np.asarray(boxes, dtype=np.float32)
        self.masks = list(masks)
        self.keypoints = keypoints
        self.obb = obb
        self.points = points


class FakeTRex:
    settings: dict[str, float] = {"detect_conf_threshold": 0.25, "detect_iou_threshold": 0.5}
    Result = FakeResult
    Boxes = staticmethod(lambda value: np.asarray(value, dtype=np.float32))
    KeypointData = staticmethod(lambda value: value)
    ObbData = staticmethod(lambda value: value)
    PointData = staticmethod(lambda value: value)

    @staticmethod
    def choose_device() -> str:
        return "cpu"

    @staticmethod
    def log(message: str) -> None:
        del message

    @staticmethod
    def setting(name: str) -> float:
        return FakeTRex.settings[name]


class FakeTrackerModel:
    class _MemoryEncoder:
        class _MaskDownsampler:
            interpol_size = 4

        mask_downsampler = _MaskDownsampler()

    memory_encoder = _MemoryEncoder()

    def set_imgsz(self, imgsz):
        self.imgsz = imgsz


class FakeTracker:
    def __init__(self):
        self.model = FakeTrackerModel()
        self.imgsz = None
        self._bb_feat_sizes = None


class FakePredictor:
    box_detection_score_overrides: dict[tuple[int, int], float] = {}
    box_tracker_score_overrides: dict[tuple[int, int], float | None] = {}

    def __init__(self, overrides):
        self.overrides = dict(overrides)
        self.args = SimpleNamespace(conf=float(overrides.get("conf", 0.25)))
        self.device = "cpu"
        self.stride = 14
        self.model = SimpleNamespace(fp16=False, set_imgsz=lambda imgsz: None)
        self.tracker = FakeTracker()
        self.imgsz = (640, 640)
        self.dataset = None
        self.batch = None
        self.im = None
        self.inference_state = {}
        self.shutdown_called = False

    def setup_model(self, verbose=True):
        del verbose

    @staticmethod
    def init_state(predictor):
        if predictor.inference_state:
            return
        frames = int(predictor.dataset.frames)
        predictor.inference_state = {
            "num_frames": frames,
            "tracker_inference_states": [],
            "tracker_metadata": {},
            "text_prompt": None,
            "per_frame_geometric_prompt": [None] * frames,
        }

    def preprocess(self, images):
        return images[0]

    def _apply_object_wise_non_overlapping_constraints(self, masks, tracker_scores, background_value=0):
        del tracker_scores, background_value
        return masks

    def add_prompt(self, frame_idx, text=None, bboxes=None, labels=None, inference_state=None):
        del labels
        state = inference_state or self.inference_state
        if text is not None:
            state["text_prompt"] = text
            count = len(text) if isinstance(text, list) else 1
            state["text_ids"] = np.arange(count, dtype=np.int32)
        elif "text_ids" not in state:
            state["text_ids"] = np.arange(1, dtype=np.int32)
        if bboxes is not None:
            state["per_frame_geometric_prompt"][frame_idx] = np.asarray(bboxes, dtype=np.float32)
        if "tracker_capacity" not in state:
            state["tracker_capacity"] = int(state["num_frames"])
        return frame_idx, self._run_single_frame_inference(frame_idx, inference_state=state)

    def _run_single_frame_inference(self, frame_idx, reverse=False, inference_state=None):
        del reverse
        state = inference_state or self.inference_state
        obj_id_to_mask: dict[int, torch.Tensor] = {}
        obj_id_to_score: dict[int, float] = {}
        obj_id_to_cls: dict[int, float] = {}
        tracker_scores: dict[int, float] = {}

        if state.get("text_prompt") not in (None, [], ""):
            mask = torch.zeros((1, 2, 2), dtype=torch.bool)
            mask[0, 0, 0] = True
            obj_id_to_mask[100] = mask
            obj_id_to_score[100] = 0.95
            obj_id_to_cls[100] = 1.0
            tracker_scores[100] = 0.95

        tracker_capacity = int(state.get("tracker_capacity", state.get("num_frames", 0)))
        if frame_idx >= tracker_capacity:
            empty_mask = torch.zeros((1, 2, 2), dtype=torch.float32)
            obj_id_to_mask[0] = empty_mask
            obj_id_to_score[0] = 0.3
            obj_id_to_cls[0] = 2.0
            tracker_scores[0] = 0.9
            return {
                "obj_id_to_mask": obj_id_to_mask,
                "obj_id_to_score": obj_id_to_score,
                "obj_id_to_cls": obj_id_to_cls,
                "obj_id_to_tracker_score": tracker_scores,
            }

        for prompt_frame, prompt in enumerate(state.get("per_frame_geometric_prompt", [])[: frame_idx + 1]):
            if prompt is None:
                continue
            obj_id = 1000 + prompt_frame
            mask = torch.zeros((1, 2, 2), dtype=torch.bool)
            mask[0, prompt_frame % 2, 1 - (prompt_frame % 2)] = True
            score_key = (prompt_frame, frame_idx)
            det_score = type(self).box_detection_score_overrides.get(score_key, 0.9)
            tracker_score = type(self).box_tracker_score_overrides.get(score_key, 0.9)
            obj_id_to_mask[obj_id] = mask
            obj_id_to_score[obj_id] = det_score
            obj_id_to_cls[obj_id] = float(prompt_frame + 2)
            if tracker_score is not None:
                tracker_scores[obj_id] = tracker_score

        return {
            "obj_id_to_mask": obj_id_to_mask,
            "obj_id_to_score": obj_id_to_score,
            "obj_id_to_cls": obj_id_to_cls,
            "obj_id_to_tracker_score": tracker_scores,
        }

    def shutdown(self):
        self.shutdown_called = True


@dataclass
class FakeScale:
    x: float = 1.0
    y: float = 1.0


@dataclass
class FakeOffset:
    x: float = 0.0
    y: float = 0.0


class FakeBaseInput:
    def __init__(self, images, orig_ids, offsets, scales):
        self._images = images
        self._orig_ids = orig_ids
        self._offsets = offsets
        self._scales = scales

    def images(self):
        return self._images

    def orig_id(self):
        return self._orig_ids

    def offsets(self):
        return self._offsets

    def scales(self):
        return self._scales


class FakeSam3Input:
    def __init__(self, images, orig_ids, offsets, scales, prompts_per_image):
        self._base = FakeBaseInput(images, orig_ids, offsets, scales)
        self._prompts_per_image = prompts_per_image

    def base(self):
        return self._base

    def prompts_per_image(self):
        return self._prompts_per_image


def text_prompt(text: str):
    return SimpleNamespace(type="text", text=text)


def box_prompt(x0: float, y0: float, x1: float, y1: float):
    return SimpleNamespace(type="boxes", boxes=[[x0, y0, x1, y1]])


class Sam3InterfaceTest(unittest.TestCase):
    def setUp(self):
        self.prev_trex = sam3.TRex
        self.prev_predictor = sam3.SAM3VideoSemanticPredictor
        self.prev_check_imgsz = sam3.check_imgsz
        sam3.TRex = FakeTRex
        sam3.SAM3VideoSemanticPredictor = FakePredictor
        sam3.check_imgsz = lambda imgsz, stride, min_dim, max_dim: imgsz
        FakeTRex.settings = {"detect_conf_threshold": 0.25, "detect_iou_threshold": 0.5}
        FakePredictor.box_detection_score_overrides = {}
        FakePredictor.box_tracker_score_overrides = {}
        sam3.shutdown()

        self.temp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        self.weights_path = Path(self.temp.name)
        self.temp.close()

    def tearDown(self):
        sam3.shutdown()
        sam3.TRex = self.prev_trex
        sam3.SAM3VideoSemanticPredictor = self.prev_predictor
        sam3.check_imgsz = self.prev_check_imgsz
        self.weights_path.unlink(missing_ok=True)

    def create_session(self, **overrides):
        request = {"weights_path": str(self.weights_path), "imgsz": 64, "conf": 0.25}
        request.update(overrides)
        response = sam3.create_session(request)
        self.assertTrue(response["ok"])

    def frame_input(self, frame_index: int, prompts, *, image_shape=(8, 8, 3), offset=None, scale=None):
        image = np.zeros(image_shape, dtype=np.uint8)
        return FakeSam3Input(
            [image],
            [frame_index],
            [offset or FakeOffset()],
            [scale or FakeScale()],
            [prompts],
        )

    def test_global_text_persists_across_frames(self):
        self.create_session()

        result0 = sam3.predict_frame(self.frame_input(0, [text_prompt("fish")]))[0]
        result1 = sam3.predict_frame(self.frame_input(1, []))[0]

        self.assertEqual(result0.frame_index, 0)
        self.assertEqual(result1.frame_index, 1)
        self.assertEqual(len(result0.masks), 1)
        self.assertEqual(len(result1.masks), 1)

    def test_frame_local_bbox_persists_forward(self):
        self.create_session()

        sam3.predict_frame(self.frame_input(0, [box_prompt(0.1, 0.1, 0.4, 0.4)]))
        result1 = sam3.predict_frame(self.frame_input(1, []))[0]

        self.assertEqual(result1.frame_index, 1)
        self.assertEqual(len(result1.masks), 1)

    def test_session_initializes_with_capacity_large_enough_for_frame1_propagation(self):
        self.create_session(video_capacity=8)

        result0 = sam3.predict_frame(self.frame_input(0, [box_prompt(0.1, 0.1, 0.4, 0.4)]))[0]
        result1 = sam3.predict_frame(self.frame_input(1, []))[0]

        self.assertEqual(len(result0.masks), 1)
        self.assertEqual(len(result1.masks), 1)

    def test_adaptive_default_video_capacity_grows_only_as_needed(self):
        self.create_session()

        result0 = sam3.predict_frame(self.frame_input(0, [box_prompt(0.1, 0.1, 0.4, 0.4)]))[0]
        result30 = None
        for frame_index in range(1, 31):
            result30 = sam3.predict_frame(self.frame_input(frame_index, []))[0]

        current = sam3._require_session()

        self.assertEqual(len(result0.masks), 1)
        self.assertIsNotNone(result30)
        self.assertEqual(result30.frame_index, 30)
        self.assertEqual(len(result30.masks), 1)
        self.assertLess(current.predictor.dataset.frames, 128)
        self.assertGreaterEqual(current.predictor.dataset.frames, 31)

    def test_create_session_accepts_non_square_imgsz_pair(self):
        self.create_session(imgsz=(96, 64))

        current = sam3._require_session()

        self.assertEqual(current.predictor.imgsz, (96, 64))

    def test_session_applies_keep_alive_directly_to_predictor_and_survives_frame30(self):
        self.create_session()

        current = sam3._require_session()
        self.assertEqual(current.predictor.init_trk_keep_alive, 300)
        self.assertEqual(current.predictor.max_trk_keep_alive, 300)
        self.assertFalse(current.predictor.decrease_trk_keep_alive_for_empty_masklets)

        for frame_index in range(31):
            prompts = [box_prompt(0.1, 0.1, 0.4, 0.4)] if frame_index == 0 else []
            result = sam3.predict_frame(self.frame_input(frame_index, prompts))[0]

        self.assertEqual(result.frame_index, 30)
        self.assertEqual(len(result.masks), 1)

    def test_propagated_bbox_uses_tracker_score_when_detection_score_goes_stale(self):
        FakePredictor.box_detection_score_overrides[(0, 0)] = 0.3
        FakePredictor.box_tracker_score_overrides[(0, 0)] = None
        FakePredictor.box_detection_score_overrides[(0, 1)] = 0.1
        FakePredictor.box_tracker_score_overrides[(0, 1)] = 0.9
        self.create_session()

        result0 = sam3.predict_frame(self.frame_input(0, [box_prompt(0.1, 0.1, 0.4, 0.4)]))[0]
        result1 = sam3.predict_frame(self.frame_input(1, []))[0]

        self.assertEqual(len(result0.masks), 1)
        self.assertEqual(len(result1.masks), 1)
        self.assertAlmostEqual(float(result0.boxes[0, 4]), 0.3, places=5)
        self.assertAlmostEqual(float(result1.boxes[0, 4]), 0.9, places=5)

    def test_propagated_tracker_masks_fall_back_to_positive_logits(self):
        self.create_session()
        session = sam3._require_session()
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        output = {
            "obj_id_to_mask": {7: torch.full((1, 2, 2), 0.1, dtype=torch.float32)},
            "obj_id_to_score": {7: 0.1},
            "obj_id_to_cls": {7: 5.0},
            "obj_id_to_tracker_score": {7: 0.9},
        }

        masks_np, conf_np, cls_np = sam3._postprocess_video_output(session, output, image)

        self.assertEqual(masks_np.shape[0], 1)
        self.assertTrue(bool(masks_np[0].any()))
        self.assertAlmostEqual(float(conf_np[0]), 0.9, places=5)
        self.assertAlmostEqual(float(cls_np[0]), 5.0, places=5)

    def test_duplicate_masks_are_suppressed(self):
        pred_masks = torch.tensor(
            [
                [[True, True], [False, False]],
                [[True, True], [False, False]],
            ],
            dtype=torch.bool,
        )
        pred_scores = torch.tensor([0.9, 0.9], dtype=torch.float32)

        keep = sam3._suppress_near_duplicate_masks(pred_masks, pred_scores, 0.95)

        self.assertEqual(keep.tolist(), [True, False])

    def test_build_result_inverse_maps_letterboxed_masks(self):
        result = sam3._build_result(
            frame_index=0,
            scale=FakeScale(2.0, 2.0),
            offset=FakeOffset(-2.0, 0.0),
            image_shape=(6, 8),
            masks_np=np.asarray(
                [[[0, 0, 1, 1, 1, 1, 0, 0],
                  [0, 0, 1, 1, 1, 1, 0, 0],
                  [0, 0, 1, 1, 1, 1, 0, 0],
                  [0, 0, 1, 1, 1, 1, 0, 0],
                  [0, 0, 1, 1, 1, 1, 0, 0],
                  [0, 0, 1, 1, 1, 1, 0, 0]]],
                dtype=np.uint8,
            ),
            conf_np=np.asarray([0.9], dtype=np.float32),
            cls_np=np.asarray([3.0], dtype=np.float32),
        )

        self.assertEqual(result.boxes.shape, (1, 6))
        self.assertAlmostEqual(float(result.boxes[0, 0]), 0.0, places=5)
        self.assertAlmostEqual(float(result.boxes[0, 2]), 7.0, places=5)

    def test_predict_frame_uses_live_thresholds_from_trex_settings(self):
        FakePredictor.box_detection_score_overrides[(0, 0)] = 0.6
        FakePredictor.box_tracker_score_overrides[(0, 0)] = None
        FakeTRex.settings["detect_conf_threshold"] = 0.7
        self.create_session()

        sam3.reset_runtime({"max_frame_index": 0})
        rejected = sam3.predict_frame(self.frame_input(0, [box_prompt(0.1, 0.1, 0.4, 0.4)]))[0]

        FakeTRex.settings["detect_conf_threshold"] = 0.5
        sam3.reset_runtime({"max_frame_index": 0})
        accepted = sam3.predict_frame(self.frame_input(0, [box_prompt(0.1, 0.1, 0.4, 0.4)]))[0]

        self.assertEqual(len(rejected.masks), 0)
        self.assertEqual(len(accepted.masks), 1)

    def test_reset_runtime_clears_previous_frame_prompts(self):
        self.create_session()

        prompted = sam3.predict_frame(self.frame_input(0, [box_prompt(0.1, 0.1, 0.4, 0.4)]))[0]
        sam3.reset_runtime({"max_frame_index": 0})
        cleared = sam3.predict_frame(self.frame_input(0, []))[0]

        self.assertEqual(len(prompted.masks), 1)
        self.assertEqual(len(cleared.masks), 0)

    def test_public_api_does_not_expose_legacy_replay_surface(self):
        self.assertFalse(hasattr(sam3, "predict"))
        self.assertFalse(hasattr(sam3, "snapshot_runtime"))
        self.assertFalse(hasattr(sam3, "restore_runtime"))
        self.assertFalse(hasattr(sam3, "_RUNTIME_BLOBS"))

    def test_shutdown_clears_session(self):
        self.create_session()

        sam3.predict_frame(self.frame_input(0, [text_prompt("fish")]))
        current = sam3._require_session()
        predictor = current.predictor

        response = sam3.shutdown()

        self.assertTrue(response["ok"])
        self.assertTrue(predictor.shutdown_called)
        self.assertIsNone(sam3._SESSION)

    def test_predict_frame_rejects_mismatched_offsets_length(self):
        self.create_session()

        with self.assertRaisesRegex(ValueError, "offsets"):
            sam3.predict_frame(FakeSam3Input(
                [np.zeros((8, 8, 3), dtype=np.uint8)],
                [0],
                [],
                [FakeScale()],
                [[]],
            ))


if __name__ == "__main__":
    unittest.main()
