#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the Python SAM3 video-semantic adapter."""

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
    settings: dict[str, float] = {"detect_iou_threshold": 0.5}
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
    def imshow(name: str, image) -> None:
        del name, image

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
        self.model = SimpleNamespace(fp16=False)
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
        if state["per_frame_geometric_prompt"][frame_idx] is None and bboxes is not None:
            state["per_frame_geometric_prompt"][frame_idx] = np.asarray(bboxes, dtype=np.float32)
        elif bboxes is not None:
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
                "removed_obj_ids": set(),
                "frame_stats": {},
                "unconfirmed_obj_ids": [],
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
            "removed_obj_ids": set(),
            "frame_stats": {},
            "unconfirmed_obj_ids": [],
        }

    def shutdown(self):
        self.shutdown_called = True


@dataclass
class FakeScale:
    x: float = 1.0
    y: float = 1.0


class FakeBaseInput:
    def __init__(self, images, orig_ids, scales):
        self._images = images
        self._orig_ids = orig_ids
        self._scales = scales

    def images(self):
        return self._images

    def orig_id(self):
        return self._orig_ids

    def scales(self):
        return self._scales


class FakeSam3Input:
    def __init__(self, images, orig_ids, scales, prompts_per_image):
        self._base = FakeBaseInput(images, orig_ids, scales)
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
        sam3.check_imgsz = lambda imgsz, stride, min_dim, max_dim: int(imgsz)
        FakeTRex.settings = {"detect_iou_threshold": 0.5}
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

    def frame_input(self, frame_index: int, prompts):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        return FakeSam3Input([image], [frame_index], [FakeScale()], [prompts])

    def test_global_text_persists_across_frames(self):
        self.create_session()

        result0 = sam3.predict(self.frame_input(0, [text_prompt("fish")]))[0]
        result1 = sam3.predict(self.frame_input(1, []))[0]

        self.assertEqual(result0.frame_index, 0)
        self.assertEqual(result1.frame_index, 1)
        self.assertEqual(len(result0.masks), 1)
        self.assertEqual(len(result1.masks), 1)

    def test_frame_local_bbox_persists_forward(self):
        self.create_session()

        sam3.predict(self.frame_input(0, [box_prompt(0.1, 0.1, 0.4, 0.4)]))
        result1 = sam3.predict(self.frame_input(1, []))[0]

        self.assertEqual(result1.frame_index, 1)
        self.assertEqual(len(result1.masks), 1)

    def test_session_initializes_with_capacity_large_enough_for_frame1_propagation(self):
        self.create_session(video_capacity=8)

        result0 = sam3.predict(self.frame_input(0, [box_prompt(0.1, 0.1, 0.4, 0.4)]))[0]
        result1 = sam3.predict(self.frame_input(1, []))[0]

        self.assertEqual(len(result0.masks), 1)
        self.assertEqual(len(result1.masks), 1)

    def test_propagated_bbox_uses_tracker_score_when_detection_score_goes_stale(self):
        FakePredictor.box_detection_score_overrides[(0, 0)] = 0.3
        FakePredictor.box_tracker_score_overrides[(0, 0)] = None
        FakePredictor.box_detection_score_overrides[(0, 1)] = 0.1
        FakePredictor.box_tracker_score_overrides[(0, 1)] = 0.9
        self.create_session()

        result0 = sam3.predict(self.frame_input(0, [box_prompt(0.1, 0.1, 0.4, 0.4)]))[0]
        result1 = sam3.predict(self.frame_input(1, []))[0]

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

    def test_session_uses_detect_iou_threshold_from_request(self):
        self.create_session(detect_iou_threshold=0.7)

        current = sam3._require_session()

        self.assertAlmostEqual(current.duplicate_mask_iou, 0.7, places=6)

    def test_set_iou_threshold_updates_session_threshold(self):
        self.create_session(detect_iou_threshold=0.6)

        response = sam3.set_iou_threshold({"iou": 0.3})

        self.assertTrue(response["ok"])
        self.assertAlmostEqual(response["iou"], 0.3, places=6)
        self.assertAlmostEqual(sam3._require_session().duplicate_mask_iou, 0.3, places=6)

    def test_later_frame_prompt_remains_active(self):
        self.create_session()

        sam3.predict(self.frame_input(0, [text_prompt("fish")]))
        sam3.predict(self.frame_input(1, []))
        result2 = sam3.predict(self.frame_input(2, [box_prompt(0.2, 0.2, 0.6, 0.6)]))[0]
        result3 = sam3.predict(self.frame_input(3, []))[0]

        self.assertEqual(len(result2.masks), 2)
        self.assertEqual(len(result3.masks), 2)

    def test_repeated_prompt_snapshot_does_not_duplicate(self):
        self.create_session()

        result0 = sam3.predict(self.frame_input(0, [box_prompt(0.1, 0.1, 0.4, 0.4)]))[0]
        result0_repeat = sam3.predict(self.frame_input(0, [box_prompt(0.1, 0.1, 0.4, 0.4)]))[0]

        self.assertEqual(len(result0.masks), 1)
        self.assertEqual(len(result0_repeat.masks), 1)

    def test_out_of_order_frame_replays_cached_state(self):
        self.create_session()

        sam3.predict(self.frame_input(0, [box_prompt(0.1, 0.1, 0.4, 0.4)]))
        sam3.predict(self.frame_input(1, [box_prompt(0.5, 0.5, 0.8, 0.8)]))
        sam3.predict(self.frame_input(2, []))
        result1 = sam3.predict(self.frame_input(1, []))[0]

        self.assertEqual(result1.frame_index, 1)
        self.assertEqual(len(result1.masks), 1)

    def test_old_frame_can_be_replayed_from_new_input_without_crashing(self):
        self.create_session()

        sam3.predict(self.frame_input(0, [text_prompt("fish")]))
        sam3.predict(self.frame_input(1, []))
        sam3.predict(self.frame_input(2, []))
        result0 = sam3.predict(self.frame_input(0, []))[0]

        self.assertEqual(result0.frame_index, 0)

    def test_missing_replay_context_is_deferred_instead_of_raising(self):
        self.create_session()

        sam3.predict(self.frame_input(0, [text_prompt("fish")]))
        sam3.predict(self.frame_input(1, []))
        sam3.predict(self.frame_input(2, []))
        deferred = sam3.predict(self.frame_input(5, []))[0]

        self.assertEqual(deferred.frame_index, 5)
        self.assertEqual(len(deferred.masks), 0)
        self.assertIsNone(sam3._require_session().last_processed_frame)

    def test_shutdown_clears_session(self):
        self.create_session()

        sam3.predict(self.frame_input(0, [text_prompt("fish")]))
        current = sam3._require_session()
        predictor = current.predictor

        response = sam3.shutdown()

        self.assertTrue(response["ok"])
        self.assertTrue(predictor.shutdown_called)
        self.assertIsNone(sam3._SESSION)

    def test_session_syncs_ultralytics_detection_gate_with_conf(self):
        self.create_session()

        current = sam3._require_session()

        self.assertEqual(current.conf, 0.25)
        self.assertEqual(current.predictor.score_threshold_detection, 0.25)


if __name__ == "__main__":
    unittest.main()
