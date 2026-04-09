#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the Python SAM3 video-semantic adapter."""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
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

    def frame_input(self, frame_index: int, prompts):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        return FakeSam3Input([image], [frame_index], [FakeScale()], [prompts])

    def frame_batch_input(self, frame_indices, prompts_per_image):
        images = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in frame_indices]
        scales = [FakeScale() for _ in frame_indices]
        return FakeSam3Input(images, list(frame_indices), scales, list(prompts_per_image))

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

    def test_session_defaults_to_checkpoints_enabled(self):
        self.create_session()

        current = sam3._require_session()

        self.assertTrue(current.runtime_checkpoints_enabled)

    def test_adaptive_default_video_capacity_grows_only_as_needed(self):
        self.create_session()

        result0 = sam3.predict(self.frame_input(0, [box_prompt(0.1, 0.1, 0.4, 0.4)]))[0]
        result30 = None
        for frame_index in range(1, 31):
            result30 = sam3.predict(self.frame_input(frame_index, []))[0]

        current = sam3._require_session()

        self.assertEqual(len(result0.masks), 1)
        self.assertIsNotNone(result30)
        self.assertEqual(result30.frame_index, 30)
        self.assertEqual(len(result30.masks), 1)
        self.assertLess(current.predictor.dataset.frames, 128)
        self.assertGreaterEqual(current.predictor.dataset.frames, 31)

    def test_session_applies_keep_alive_directly_to_predictor_and_survives_frame30(self):
        self.create_session()

        current = sam3._require_session()
        self.assertEqual(current.predictor.init_trk_keep_alive, 300)
        self.assertEqual(current.predictor.max_trk_keep_alive, 300)
        self.assertFalse(current.predictor.decrease_trk_keep_alive_for_empty_masklets)

        for frame_index in range(31):
            prompts = [box_prompt(0.1, 0.1, 0.4, 0.4)] if frame_index == 0 else []
            result = sam3.predict(self.frame_input(frame_index, prompts))[0]

        self.assertEqual(result.frame_index, 30)
        self.assertEqual(len(result.masks), 1)

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

    def test_predict_frame_uses_live_thresholds_from_trex_settings(self):
        FakePredictor.box_detection_score_overrides[(0, 0)] = 0.6
        FakePredictor.box_tracker_score_overrides[(0, 0)] = None
        FakeTRex.settings["detect_conf_threshold"] = 0.7
        self.create_session()

        sam3.reset_runtime({"max_frame_index": 0})
        before = sam3.snapshot_runtime()["state"]
        rejected = sam3.predict_frame(self.frame_input(0, [box_prompt(0.1, 0.1, 0.4, 0.4)]))[0]

        FakeTRex.settings["detect_conf_threshold"] = 0.5
        sam3.restore_runtime({"state": before})
        accepted = sam3.predict_frame(self.frame_input(0, [box_prompt(0.1, 0.1, 0.4, 0.4)]))[0]

        self.assertEqual(len(rejected.masks), 0)
        self.assertEqual(len(accepted.masks), 1)

    def test_snapshot_and_restore_runtime_round_trip_predict_frame(self):
        self.create_session()

        sam3.reset_runtime({"max_frame_index": 0})
        before = sam3.snapshot_runtime()["state"]
        first = sam3.predict_frame(self.frame_input(0, [box_prompt(0.1, 0.1, 0.4, 0.4)]))[0]
        after = sam3.snapshot_runtime()["state"]

        sam3.restore_runtime({"state": before})
        rerun = sam3.predict_frame(self.frame_input(0, [box_prompt(0.1, 0.1, 0.4, 0.4)]))[0]
        sam3.restore_runtime({"state": after})

        self.assertEqual(len(first.masks), 1)
        self.assertEqual(len(rerun.masks), 1)
        self.assertEqual(sam3._require_session().last_processed_frame, 0)

    def test_predict_frame_does_not_resurrect_empty_frame_prompts(self):
        self.create_session()

        sam3.reset_runtime({"max_frame_index": 0})
        before = sam3.snapshot_runtime()["state"]
        prompted = sam3.predict_frame(self.frame_input(0, [box_prompt(0.1, 0.1, 0.4, 0.4)]))[0]

        sam3.restore_runtime({"state": before})
        cleared = sam3.predict_frame(self.frame_input(0, []))[0]

        self.assertEqual(len(prompted.masks), 1)
        self.assertEqual(len(cleared.masks), 0)

    def test_predict_frame_skips_python_replay_bookkeeping(self):
        self.create_session(runtime_checkpoints_enabled=True)

        sam3.reset_runtime({"max_frame_index": 0})
        result = sam3.predict_frame(self.frame_input(0, [box_prompt(0.1, 0.1, 0.4, 0.4)]))[0]
        session = sam3._require_session()

        self.assertEqual(result.frame_index, 0)
        self.assertEqual(session.prompt_history, {})
        self.assertEqual(session.checkpoint_history, {})

    def test_checkpoint_restore_preserves_defaultdict_metadata(self):
        self.create_session(runtime_checkpoints_enabled=True)
        session = sam3._require_session()
        session.predictor.inference_state = {
            "num_frames": 8,
            "tracker_inference_states": [],
            "tracker_metadata": {
                "metadata": {
                    "obj_first_frame_idx": {},
                    "unmatched_frame_inds": defaultdict(list),
                    "trk_keep_alive": defaultdict(int),
                    "overlap_pair_to_frame_inds": defaultdict(list),
                    "removed_obj_ids": set(),
                },
                "obj_id_to_tracker_score_frame_wise": defaultdict(dict),
                "obj_ids": np.array([0], dtype=np.int32),
            },
            "text_prompt": None,
            "per_frame_geometric_prompt": [None] * 8,
        }
        session.last_processed_frame = 0
        session.snapshot_runtime_checkpoint(0, sam3._empty_result(0))

        session.predictor.inference_state = {}
        session.restore_runtime_checkpoint(0)

        metadata = session.predictor.inference_state["tracker_metadata"]["metadata"]
        metadata["unmatched_frame_inds"][np.int64(0)].append(1)
        metadata["trk_keep_alive"][np.int64(0)] += 1

        self.assertEqual(metadata["unmatched_frame_inds"][np.int64(0)], [1])
        self.assertEqual(metadata["trk_keep_alive"][np.int64(0)], 1)

    def test_checkpoint_snapshot_omits_im_tensor(self):
        self.create_session(runtime_checkpoints_enabled=True)
        session = sam3._require_session()
        session.predictor.inference_state = {
            "num_frames": 8,
            "tracker_inference_states": [SimpleNamespace(frame=0)],
            "tracker_metadata": {"metadata": {}},
            "text_prompt": None,
            "im": torch.ones((1, 3, 8, 8), dtype=torch.float32),
            "per_frame_geometric_prompt": [np.ones((1, 4), dtype=np.float32)] + [None] * 7,
        }
        session.last_processed_frame = 0

        session.snapshot_runtime_checkpoint(0, sam3._empty_result(0))

        checkpoint = session.checkpoint_history[0]
        self.assertNotIn("im", checkpoint.inference_state)
        self.assertEqual(checkpoint.inference_state["num_frames"], 1)
        self.assertEqual(list(checkpoint.inference_state["per_frame_geometric_prompt"].keys()), [0])

    def test_later_frame_prompt_remains_active(self):
        self.create_session()

        sam3.predict(self.frame_input(0, [text_prompt("fish")]))
        sam3.predict(self.frame_input(1, []))
        result2 = sam3.predict(self.frame_input(2, [box_prompt(0.2, 0.2, 0.6, 0.6)]))[0]
        result3 = sam3.predict(self.frame_input(3, []))[0]

        self.assertEqual(len(result2.masks), 2)
        self.assertEqual(len(result3.masks), 2)

    def test_repeated_prompt_snapshot_does_not_duplicate(self):
        self.create_session(runtime_checkpoints_enabled=True)

        result0 = sam3.predict(self.frame_input(0, [box_prompt(0.1, 0.1, 0.4, 0.4)]))[0]
        result0_repeat = sam3.predict(self.frame_input(0, [box_prompt(0.1, 0.1, 0.4, 0.4)]))[0]

        self.assertEqual(len(result0.masks), 1)
        self.assertEqual(len(result0_repeat.masks), 1)

    def test_out_of_order_frame_replays_cached_state(self):
        self.create_session(runtime_checkpoints_enabled=True)

        sam3.predict(self.frame_input(0, [box_prompt(0.1, 0.1, 0.4, 0.4)]))
        sam3.predict(self.frame_input(1, [box_prompt(0.5, 0.5, 0.8, 0.8)]))
        sam3.predict(self.frame_input(2, []))
        result1 = sam3.predict(self.frame_input(1, []))[0]

        self.assertEqual(result1.frame_index, 1)
        self.assertEqual(len(result1.masks), 2)

    def test_exact_checkpoint_restore_recovers_frame_after_image_eviction(self):
        self.create_session(runtime_checkpoints_enabled=True)

        for frame_index in range(41):
            prompts = [box_prompt(0.1, 0.1, 0.4, 0.4)] if frame_index == 0 else []
            sam3.predict(self.frame_input(frame_index, prompts))

        result20 = sam3.predict(self.frame_input(20, []))[0]

        self.assertEqual(result20.frame_index, 20)
        self.assertEqual(len(result20.masks), 1)
        self.assertIn(0, sam3._require_session().prompt_history)
        self.assertEqual(sam3._require_session().last_processed_frame, 20)

    def test_checkpoint_replay_uses_current_input_frames_to_bridge_gap(self):
        self.create_session(runtime_checkpoints_enabled=True)

        for frame_index in range(21):
            prompts = [box_prompt(0.1, 0.1, 0.4, 0.4)] if frame_index == 0 else []
            sam3.predict(self.frame_input(frame_index, prompts))

        session = sam3._require_session()
        incoming_frames = {
            frame_index: sam3.ReplayFrame(
                frame_index=frame_index,
                image=np.zeros((8, 8, 3), dtype=np.uint8),
                prompt_state=session.prompt_history.get(frame_index, sam3.PromptState()),
            )
            for frame_index in range(11, 15)
        }

        result14 = sam3._replay_to_frame(session, 14, FakeScale(), incoming_frames)

        self.assertEqual(result14.frame_index, 14)
        self.assertEqual(len(result14.masks), 1)
        self.assertEqual(session.last_processed_frame, 14)

    def test_old_frame_can_be_replayed_from_new_input_without_crashing(self):
        self.create_session(runtime_checkpoints_enabled=True)

        sam3.predict(self.frame_input(0, [text_prompt("fish")]))
        sam3.predict(self.frame_input(1, []))
        sam3.predict(self.frame_input(2, []))
        result0 = sam3.predict(self.frame_input(0, []))[0]

        self.assertEqual(result0.frame_index, 0)

    def test_missing_replay_context_runs_current_frame_when_prompted(self):
        self.create_session(runtime_checkpoints_enabled=False)

        sam3.predict(self.frame_input(0, [box_prompt(0.1, 0.1, 0.4, 0.4)]))
        sam3.predict(self.frame_input(1, []))
        sam3.predict(self.frame_input(2, []))
        recovered = sam3.predict(self.frame_input(5, [box_prompt(0.5, 0.5, 0.8, 0.8)]))[0]

        self.assertEqual(recovered.frame_index, 5)
        self.assertEqual(len(recovered.masks), 1)
        self.assertEqual(sam3._require_session().last_processed_frame, 5)

    def test_out_of_order_requests_run_current_frame_when_checkpoints_disabled(self):
        self.create_session(runtime_checkpoints_enabled=False)

        sam3.predict(self.frame_input(0, [text_prompt("fish")]))
        sam3.predict(self.frame_input(1, []))
        recovered = sam3.predict(self.frame_input(0, [text_prompt("fish")]))[0]

        self.assertEqual(recovered.frame_index, 0)
        self.assertEqual(len(recovered.masks), 1)
        self.assertEqual(sam3._require_session().last_processed_frame, 0)

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
