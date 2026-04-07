# -*- coding: utf-8 -*-
"""Stateful SAM3 video-semantic adapter used by the C++ SAM3 backend.

The contract remains intentionally small:
- C++ owns frame lookup, prompt lookup, batching, and prompt authority.
- Python owns the loaded SAM3 video session, timeline state, and replay logic.
- Public entrypoints are `create_session`, `set_conf_threshold`, `shutdown`,
  and `predict`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import gc
from pathlib import Path
from types import SimpleNamespace
from typing import TypedDict, cast

import numpy as np
import numpy.typing as npt
import torch
from torch.nn import functional as F
from ultralytics.models.sam.predict import SAM3VideoSemanticPredictor
from ultralytics.utils.checks import check_imgsz

try:
    import TRex  # type: ignore
except Exception:  # pragma: no cover
    TRex = None


class GenericOkResponse(TypedDict):
    ok: bool


class SetConfThresholdResponse(TypedDict):
    ok: bool
    conf: float


class CreateSessionResponse(TypedDict):
    ok: bool
    device: str


@dataclass
class PromptState:
    """Normalized prompt payload for one frame as received from C++."""

    texts: list[str] = field(default_factory=list)
    boxes: npt.NDArray[np.float32] | None = None
    points: npt.NDArray[np.float32] | None = None

    def has_prompts(self) -> bool:
        return bool(self.texts) or self.boxes is not None or self.points is not None

    def has_video_prompts(self) -> bool:
        return bool(self.texts) or self.boxes is not None


@dataclass
class ReplayFrame:
    """Frame payload cached inside the active Python session for replay."""

    frame_index: int
    image: npt.NDArray[np.uint8]
    prompt_state: PromptState


@dataclass
class _VideoDatasetShim:
    """Small dataset shim that satisfies Ultralytics' video predictor state."""

    frames: int
    frame: int = 1
    mode: str = "video"


@dataclass
class Sam3VideoSession:
    """Loaded SAM3 video-semantic predictor plus replayable timeline state."""

    predictor: SAM3VideoSemanticPredictor
    device: str
    conf: float
    timeline_capacity: int = 65536
    frame_cache: dict[int, ReplayFrame] = field(default_factory=dict)
    prompt_history: dict[int, PromptState] = field(default_factory=dict)
    last_processed_frame: int | None = None
    active_texts: list[str] = field(default_factory=list)
    max_cached_frames: int = 2

    def set_conf_threshold(self, conf: float) -> float:
        """Update the confidence threshold used for future predictions."""
        value = float(np.clip(float(conf), 0.0, 1.0))
        self.conf = value

        overrides = getattr(self.predictor, "overrides", None)
        if isinstance(overrides, dict):
            overrides["conf"] = value

        args = getattr(self.predictor, "args", None)
        if isinstance(args, dict):
            args["conf"] = value
        elif args is not None:
            try:
                setattr(args, "conf", value)
            except Exception:
                pass

        return value

    def store_frame(self, frame_index: int, image: npt.NDArray[np.uint8], prompt_state: PromptState) -> None:
        """Cache the latest authoritative frame payload received from C++."""
        if not prompt_state.has_prompts() and frame_index in self.prompt_history:
            prompt_state = self.prompt_history[frame_index]
        self.prompt_history[frame_index] = prompt_state
        self.frame_cache[frame_index] = ReplayFrame(frame_index, image, prompt_state)
        if len(self.frame_cache) > self.max_cached_frames:
            oldest = min(self.frame_cache)
            del self.frame_cache[oldest]

    def reset_runtime(self, max_frame_index: int) -> None:
        """Reset tracker state while keeping the loaded model and cached frames."""
        self.active_texts = []
        self.last_processed_frame = None

        inference_state = getattr(self.predictor, "inference_state", None)
        if isinstance(inference_state, dict):
            inference_state.clear()
        self.predictor.inference_state = {}
        self.predictor.im = None
        self.predictor.batch = None

        reset_tracking = getattr(self.predictor, "_reset_tracking_results", None)
        if callable(reset_tracking):
            try:
                reset_tracking()
            except Exception:
                pass

        self._ensure_runtime_capacity(max_frame_index)

    def close(self) -> None:
        """Fully discard runtime state, predictor references, and cached replay data."""
        try:
            self.reset_runtime(max(self.frame_cache, default=0))
        except Exception:
            pass

        shutdown_fn = getattr(self.predictor, "shutdown", None)
        if callable(shutdown_fn):
            try:
                shutdown_fn()
            except Exception:
                pass

        self.frame_cache.clear()
        self.prompt_history.clear()
        self.active_texts = []
        self.last_processed_frame = None
        self.predictor.inference_state = {}
        self.predictor.batch = None
        self.predictor.im = None
        self.predictor.dataset = None
        _cleanup_device_caches(self.device)

    def _ensure_runtime_capacity(self, max_frame_index: int) -> None:
        """Ensure the predictor has enough timeline capacity for `max_frame_index`."""
        required_frames = max(1, self.timeline_capacity, max_frame_index + 1)
        dataset = getattr(self.predictor, "dataset", None)
        if dataset is None or getattr(dataset, "mode", None) != "video":
            self.predictor.dataset = _VideoDatasetShim(frames=required_frames)
        else:
            dataset.frames = max(required_frames, int(getattr(dataset, "frames", 0)))

        self.predictor.inference_state = cast(dict[str, object], getattr(self.predictor, "inference_state", {}))
        if not self.predictor.inference_state:
            SAM3VideoSemanticPredictor.init_state(self.predictor)
        else:
            self.predictor.inference_state["num_frames"] = int(self.predictor.dataset.frames)
            prompts = self.predictor.inference_state.get("per_frame_geometric_prompt")
            if isinstance(prompts, list) and len(prompts) < int(self.predictor.dataset.frames):
                prompts.extend([None] * (int(self.predictor.dataset.frames) - len(prompts)))

        tracker = getattr(self.predictor, "tracker", None)
        if tracker is None:
            return

        tracker.imgsz = self.predictor.imgsz
        tracker_model = getattr(tracker, "model", None)
        if tracker_model is not None and hasattr(tracker_model, "set_imgsz"):
            tracker_model.set_imgsz(self.predictor.imgsz)

        if hasattr(self.predictor, "stride"):
            tracker._bb_feat_sizes = [
                [int(x / (self.predictor.stride * i)) for x in self.predictor.imgsz]
                for i in [1 / 4, 1 / 2, 1]
            ]

        mask_downsampler = getattr(getattr(tracker_model, "memory_encoder", None), "mask_downsampler", None)
        interpol_size = getattr(mask_downsampler, "interpol_size", None)
        if interpol_size is not None:
            self.predictor.interpol_size = interpol_size


_SESSION: Sam3VideoSession | None = None


def _choose_device() -> str:
    """Resolve the torch device from TRex first, then local availability."""
    if TRex is not None and hasattr(TRex, "choose_device"):
        return str(TRex.choose_device())
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _cleanup_device_caches(device: str) -> None:
    """Best-effort Python-side device cleanup after session shutdown."""
    try:
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif device.startswith("mps") and hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass
    gc.collect()


def _require_session() -> Sam3VideoSession:
    """Return the active SAM3 session or fail with a clear error."""
    if _SESSION is None:
        raise RuntimeError("SAM3 session is not created. Call create_session(...) first.")
    return _SESSION


def _normalize_frame(image: npt.ArrayLike) -> npt.NDArray[np.uint8]:
    """Normalize incoming image data to contiguous HxWx3 uint8 BGR."""
    frame = np.asarray(image)
    if frame.ndim == 2:
        frame = np.repeat(frame[:, :, None], 3, axis=2)
    elif frame.ndim == 3 and frame.shape[2] == 1:
        frame = np.repeat(frame, 3, axis=2)
    elif frame.ndim == 3 and frame.shape[2] >= 3:
        frame = frame[:, :, :3]
    else:
        raise ValueError("SAM3 predict expects HxW, HxWx1, HxWx3, or HxWx4 image data.")

    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8, copy=False)
    return np.ascontiguousarray(frame)


def _empty_result(frame_index: int) -> "TRex.Result":
    """Construct an empty TRex result for an image with no prompts/output."""
    if TRex is None:
        raise RuntimeError("TRex module is required for SAM3 predict().")

    return TRex.Result(  # type: ignore[attr-defined]
        int(frame_index),
        TRex.Boxes(np.empty((0, 6), dtype=np.float32)),  # type: ignore[attr-defined]
        [],
        TRex.KeypointData(np.empty((0, 1, 2), dtype=np.float32)),  # type: ignore[attr-defined]
        TRex.ObbData(np.empty((0, 7), dtype=np.float32)),  # type: ignore[attr-defined]
        TRex.PointData(np.empty((0, 5), dtype=np.float32)),  # type: ignore[attr-defined]
    )


def _resize_mask(mask: npt.NDArray[np.uint8], height: int, width: int) -> npt.NDArray[np.uint8]:
    """Resize a binary mask with nearest-neighbor sampling."""
    if mask.shape == (height, width):
        return np.ascontiguousarray(mask)

    tensor = torch.from_numpy(mask.astype(np.float32, copy=False)).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor, size=(height, width), mode="nearest")[0, 0]
    return np.ascontiguousarray(resized.to(torch.uint8).cpu().numpy())


def _denormalize_points(
    points: npt.NDArray[np.float32],
    image_shape: tuple[int, int],
) -> npt.NDArray[np.float32]:
    """Convert normalized point prompts into image-space pixel coordinates."""
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float32)

    height, width = image_shape
    points_px = points.astype(np.float32, copy=True)
    points_px[:, 0] = np.clip(points_px[:, 0], 0.0, 1.0) * max(0, width - 1)
    points_px[:, 1] = np.clip(points_px[:, 1], 0.0, 1.0) * max(0, height - 1)
    return points_px


def _denormalize_boxes(
    boxes: npt.NDArray[np.float32],
    image_shape: tuple[int, int],
) -> npt.NDArray[np.float32]:
    """Convert normalized XYXY box prompts into image-space pixel coordinates."""
    if boxes.size == 0:
        return np.empty((0, 4), dtype=np.float32)

    height, width = image_shape
    boxes_px = boxes.astype(np.float32, copy=True)
    boxes_px[:, 0] = np.clip(boxes_px[:, 0], 0.0, 1.0) * width
    boxes_px[:, 1] = np.clip(boxes_px[:, 1], 0.0, 1.0) * height
    boxes_px[:, 2] = np.clip(boxes_px[:, 2], 0.0, 1.0) * width
    boxes_px[:, 3] = np.clip(boxes_px[:, 3], 0.0, 1.0) * height
    return boxes_px


def _mask_contains_point(mask: npt.NDArray[np.uint8], x: float, y: float, radius: int = 2) -> bool:
    """Return whether a resized mask contains the given point within a small tolerance."""
    xi = int(round(float(x)))
    yi = int(round(float(y)))
    if xi < 0 or yi < 0 or xi >= mask.shape[1] or yi >= mask.shape[0]:
        return False

    x0 = max(0, xi - radius)
    x1 = min(mask.shape[1], xi + radius + 1)
    y0 = max(0, yi - radius)
    y1 = min(mask.shape[0], yi + radius + 1)
    return bool(np.any(mask[y0:y1, x0:x1] > 0))


def _collect_prompt_state(prompt_list: Sequence[object]) -> PromptState:
    """Merge one image's normalized prompt list into typed Python data."""
    texts: list[str] = []
    boxes: list[list[float]] = []
    points: list[list[float]] = []

    for prompt in prompt_list:
        ptype = str(getattr(prompt, "type")).split(".")[-1].lower()
        if ptype == "none":
            continue
        if ptype == "text":
            text = getattr(prompt, "text", None)
            if text is not None:
                texts.append(str(text))
            continue
        if ptype in {"box", "boxes"}:
            for box in getattr(prompt, "boxes", []):
                boxes.append([float(v) for v in box])
            continue
        if ptype in {"point", "points"}:
            for point in getattr(prompt, "points", []):
                points.append([float(point.x), float(point.y)])
            continue
        raise ValueError(f"Unsupported SAM3 prompt type: {ptype}")

    TRex.log(f"Received SAM3 prompts: {len(texts)} text(s), {boxes} box(es), {len(points)} point(s)")  # type: ignore[union-attr]

    return PromptState(
        texts=texts,
        boxes=np.asarray(boxes, dtype=np.float32) if boxes else None,
        points=np.asarray(points, dtype=np.float32) if points else None,
    )


def _select_masks_matching_points(
    masks_np: npt.NDArray[np.uint8],
    scale: object,
    points_normalized: npt.NDArray[np.float32],
) -> npt.NDArray[np.intp]:
    """Return indices of masks that contain all requested point prompts."""
    if masks_np.size == 0 or points_normalized.size == 0:
        return np.empty((0,), dtype=np.intp)

    scale_x = float(getattr(scale, "x", 1.0))
    scale_y = float(getattr(scale, "y", 1.0))
    matches: list[int] = []

    for idx, mask in enumerate(masks_np):
        target_h = max(1, int(round(mask.shape[0] * scale_y)))
        target_w = max(1, int(round(mask.shape[1] * scale_x)))
        resized_mask = _resize_mask(mask * np.uint8(255), target_h, target_w)
        points_px = _denormalize_points(points_normalized, resized_mask.shape)
        if all(_mask_contains_point(resized_mask, point[0], point[1]) for point in points_px):
            matches.append(idx)

    return np.asarray(matches, dtype=np.intp)


def _build_result(
    frame_index: int,
    scale: object,
    image_shape: tuple[int, int],
    masks_np: npt.NDArray[np.uint8],
    conf_np: npt.NDArray[np.float32],
    cls_np: npt.NDArray[np.float32],
    *,
    pred_boxes_np: npt.NDArray[np.float32] | None = None,
    keep_indices: npt.NDArray[np.intp] | None = None,
) -> "TRex.Result":
    """Build a TRex.Result from mask/score arrays."""
    if TRex is None:
        raise RuntimeError("TRex module is required for SAM3 predict().")

    if keep_indices is not None:
        masks_np = masks_np[keep_indices]
        conf_np = conf_np[keep_indices]
        cls_np = cls_np[keep_indices]
        if pred_boxes_np is not None:
            pred_boxes_np = pred_boxes_np[keep_indices]

    scale_x = float(getattr(scale, "x", 1.0))
    scale_y = float(getattr(scale, "y", 1.0))
    target_h = max(1, int(round(image_shape[0] * scale_y)))
    target_w = max(1, int(round(image_shape[1] * scale_x)))

    n = masks_np.shape[0]
    trex_masks: list[npt.NDArray[np.uint8]] = []
    boxes = np.zeros((n, 6), dtype=np.float32)

    if TRex is not None and hasattr(TRex, "log"):
        TRex.log(  # type: ignore[union-attr]
            f"SAM3 prediction for frame={frame_index} has {n} mask(s) before resizing to "
            f"{target_w}x{target_h} with scale=({scale_x:.3f},{scale_y:.3f})"
        )

    for idx, mask in enumerate(masks_np):
        resized_mask = _resize_mask(mask * np.uint8(255), target_h, target_w)
        if TRex is not None and hasattr(TRex, "imshow"):
            TRex.imshow("SAM3 full frmae", resized_mask)  # type: ignore[union-attr]

        if pred_boxes_np is not None and idx < len(pred_boxes_np):
            boxes[idx, 0] = float(pred_boxes_np[idx, 0] * scale_x)
            boxes[idx, 1] = float(pred_boxes_np[idx, 1] * scale_y)
            boxes[idx, 2] = float(pred_boxes_np[idx, 2] * scale_x)
            boxes[idx, 3] = float(pred_boxes_np[idx, 3] * scale_y)
        else:
            ys, xs = np.where(resized_mask > 0)
            if xs.size and ys.size:
                boxes[idx, 0] = float(xs.min())
                boxes[idx, 1] = float(ys.min())
                boxes[idx, 2] = (float(xs.max() + 1) - boxes[idx, 0])
                boxes[idx, 3] = (float(ys.max() + 1) - boxes[idx, 1])
                if TRex is not None and hasattr(TRex, "log"):
                    TRex.log(  # type: ignore[union-attr]
                        f" * found object at ({boxes[idx, 0]:.1f}, {boxes[idx, 1]:.1f}, "
                        f"{boxes[idx, 2]:.1f}, {boxes[idx, 3]:.1f})"
                    )
                if TRex is not None and hasattr(TRex, "imshow"):
                    copied = resized_mask.copy()
                    TRex.imshow("SAM3 mask", copied)  # type: ignore[union-attr]
                    cropped = copied[int(ys.min()) : int(ys.max()) + 1, int(xs.min()) : int(xs.max()) + 1].copy()
                    TRex.imshow("SAM3 mask cropped", cropped)  # type: ignore[union-attr]
                    resized_mask = cropped
                    boxes[idx, 2] = cropped.shape[1] - 1
                    boxes[idx, 3] = cropped.shape[0] - 1
            else:
                boxes[idx, 0] = 0
                boxes[idx, 1] = 0
                boxes[idx, 2] = target_w - 1
                boxes[idx, 3] = target_h - 1

        trex_masks.append(resized_mask)

        boxes[idx, 4] = float(conf_np[idx]) if idx < len(conf_np) else 0.0
        boxes[idx, 5] = float(cls_np[idx]) if idx < len(cls_np) else 0.0

    if TRex is not None and hasattr(TRex, "log"):
        TRex.log(f"Boxes detected: {boxes}")  # type: ignore[union-attr]

    return TRex.Result(  # type: ignore[attr-defined]
        int(frame_index),
        TRex.Boxes(boxes),  # type: ignore[attr-defined]
        trex_masks,
        TRex.KeypointData(np.empty((0, 1, 2), dtype=np.float32)),  # type: ignore[attr-defined]
        TRex.ObbData(np.empty((0, 7), dtype=np.float32)),  # type: ignore[attr-defined]
        TRex.PointData(np.empty((0, 5), dtype=np.float32)),  # type: ignore[attr-defined]
    )


def _label_array(boxes: npt.NDArray[np.float32] | None) -> npt.NDArray[np.int32] | None:
    if boxes is None:
        return None
    return np.ones((len(boxes),), dtype=np.int32)


def _postprocess_video_output(
    session: Sam3VideoSession,
    output: Mapping[str, object],
    image: npt.NDArray[np.uint8],
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Convert one video predictor output dict into resized masks and score arrays."""
    obj_id_to_mask = cast(Mapping[int, torch.Tensor], output.get("obj_id_to_mask", {}))
    obj_id_to_score = cast(Mapping[int, float], output.get("obj_id_to_score", {}))
    obj_id_to_cls = cast(Mapping[int, float], output.get("obj_id_to_cls", {}))
    obj_id_to_tracker_score = cast(Mapping[int, float], output.get("obj_id_to_tracker_score", {}))

    curr_obj_ids = sorted(obj_id_to_mask.keys())
    if not curr_obj_ids:
        return (
            np.empty((0, image.shape[0], image.shape[1]), dtype=np.uint8),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    pred_mask_logits = torch.cat([obj_id_to_mask[obj_id] for obj_id in curr_obj_ids], dim=0)
    pred_mask_logits = F.interpolate(pred_mask_logits.float()[None], image.shape[:2], mode="bilinear")[0]
    pred_masks = pred_mask_logits > 0.5
    det_scores = torch.tensor(
        [float(obj_id_to_score.get(obj_id, 0.0)) for obj_id in curr_obj_ids],
        device=pred_masks.device,
        dtype=torch.float32,
    )
    tracker_scores = torch.tensor(
        [
            float(obj_id_to_tracker_score[obj_id]) if obj_id in obj_id_to_tracker_score else float("nan")
            for obj_id in curr_obj_ids
        ],
        device=pred_masks.device,
        dtype=torch.float32,
    )
    pred_scores = torch.where(torch.isnan(tracker_scores), det_scores, tracker_scores)
    pred_cls = torch.tensor(
        [float(obj_id_to_cls.get(obj_id, 0.0)) for obj_id in curr_obj_ids],
        device=pred_masks.device,
        dtype=torch.float32,
    )

    if TRex is not None and hasattr(TRex, "log"):
        debug_rows: list[str] = []
        for idx, obj_id in enumerate(curr_obj_ids):
            tracker_score_text = (
                f"{float(tracker_scores[idx]):.3f}" if not torch.isnan(tracker_scores[idx]) else "None"
            )
            debug_rows.append(
                f"id={obj_id}:det={float(det_scores[idx]):.3f},tracker={tracker_score_text},keep={float(pred_scores[idx]):.3f}"
            )
        score_debug = ", ".join(debug_rows)
        TRex.log(f"SAM3 score selection: {score_debug}")  # type: ignore[union-attr]

    weak_positive_masks = pred_mask_logits > 0
    fallback_mask_rows = (~pred_masks.any(dim=(1, 2))) & weak_positive_masks.any(dim=(1, 2)) & ~torch.isnan(tracker_scores)
    if fallback_mask_rows.any():
        pred_masks = pred_masks.clone()
        pred_masks[fallback_mask_rows] = weak_positive_masks[fallback_mask_rows]
        if TRex is not None and hasattr(TRex, "log"):
            fallback_ids = [str(curr_obj_ids[idx]) for idx, enabled in enumerate(fallback_mask_rows.tolist()) if enabled]
            TRex.log(  # type: ignore[union-attr]
                "SAM3 mask threshold fallback (>0) for propagated ids: " + ", ".join(fallback_ids)
            )

    keep = (pred_scores > session.conf) & pred_masks.any(dim=(1, 2))
    pred_masks = pred_masks[keep]
    pred_scores = pred_scores[keep]
    pred_cls = pred_cls[keep]

    if pred_masks.shape[0] > 1:
        pred_masks = (
            session.predictor._apply_object_wise_non_overlapping_constraints(
                pred_masks.unsqueeze(1),
                pred_scores.unsqueeze(1),
                background_value=0,
            ).squeeze(1)
            > 0
        )

    return (
        pred_masks.detach().to(torch.uint8).cpu().numpy(),
        pred_scores.detach().cpu().numpy().astype(np.float32, copy=False),
        pred_cls.detach().cpu().numpy().astype(np.float32, copy=False),
    )


def _run_one_frame(
    session: Sam3VideoSession,
    replay_frame: ReplayFrame,
    scale: object,
) -> "TRex.Result":
    """Run one cached frame through the stateful SAM3 video predictor."""
    frame_index = replay_frame.frame_index
    image = replay_frame.image
    prompt_state = replay_frame.prompt_state

    session._ensure_runtime_capacity(frame_index)
    session.predictor.dataset.frame = frame_index + 1
    session.predictor.batch = ([f"frame_{frame_index}"], [image], None)
    session.predictor.im = None

    with torch.inference_mode():
        session.predictor.inference_state["im"] = session.predictor.preprocess([image])

        effective_texts = session.active_texts
        if prompt_state.texts:
            effective_texts = list(prompt_state.texts)
            session.active_texts = list(prompt_state.texts)

        output: Mapping[str, object] | None = None
        if prompt_state.has_video_prompts():
            boxes = (
                _denormalize_boxes(prompt_state.boxes, image.shape[:2])
                if prompt_state.boxes is not None
                else None
            )
            _, output = session.predictor.add_prompt(
                frame_idx=frame_index,
                text=effective_texts or None,
                bboxes=boxes,
                labels=_label_array(boxes),
            )
        elif "text_ids" in session.predictor.inference_state:
            output = session.predictor._run_single_frame_inference(
                frame_index,
                reverse=False,
                inference_state=session.predictor.inference_state,
            )
        else:
            session.last_processed_frame = frame_index
            return _empty_result(frame_index)

    if output is None:
        session.last_processed_frame = frame_index
        return _empty_result(frame_index)

    masks_np, conf_np, cls_np = _postprocess_video_output(session, output, image)
    keep_indices = None
    if prompt_state.points is not None and prompt_state.points.size:
        keep_indices = _select_masks_matching_points(masks_np, scale, prompt_state.points)
        if keep_indices.size == 0:
            session.last_processed_frame = frame_index
            return _empty_result(frame_index)

    session.last_processed_frame = frame_index
    return _build_result(
        frame_index=frame_index,
        scale=scale,
        image_shape=image.shape[:2],
        masks_np=masks_np,
        conf_np=conf_np,
        cls_np=cls_np,
        keep_indices=keep_indices,
    )


def _replay_to_frame(
    session: Sam3VideoSession,
    target_frame_index: int,
    scale: object,
) -> "TRex.Result":
    """Reset runtime state and replay cached frames up to `target_frame_index`."""
    if target_frame_index not in session.frame_cache:
        raise RuntimeError(
            f"SAM3 cannot replay frame {target_frame_index} because its image is no longer cached."
        )

    replay_indices = sorted(frame for frame in session.frame_cache if frame <= target_frame_index)
    if not replay_indices:
        raise RuntimeError(f"SAM3 replay has no cached frames for target frame {target_frame_index}.")

    expected = replay_indices[0]
    for frame_index in replay_indices:
        if frame_index != expected:
            raise RuntimeError(
                "SAM3 cannot rebuild tracker state because cached replay images are missing between "
                f"{replay_indices[0]} and {target_frame_index}."
            )
        expected += 1

    session.reset_runtime(target_frame_index)

    result = _empty_result(target_frame_index)
    for frame_index in replay_indices:
        replay_frame = session.frame_cache[frame_index]
        result = _run_one_frame(session, replay_frame, scale if frame_index == target_frame_index else SimpleNamespace(x=1.0, y=1.0))
    return result


def create_session(request: Mapping[str, object]) -> CreateSessionResponse:
    """Create or replace the singleton SAM3 model session."""
    global _SESSION

    if "weights_path" not in request:
        raise ValueError("create_session requires 'weights_path'.")

    weights_path = Path(str(cast(str | Path, request["weights_path"])))
    if not weights_path.exists():
        raise ValueError(f"SAM3 weights_path does not exist: {weights_path}")

    if _SESSION is not None:
        shutdown()

    device = _choose_device()
    overrides: dict[str, object] = {
        "task": "segment",
        "mode": "predict",
        "model": str(weights_path),
        "imgsz": int(request.get("imgsz", 640)),
        "conf": float(request.get("conf", 0.25)),
        "half": bool(request.get("half", True)),
        "device": device,
        "verbose": bool(request.get("verbose", False)),
    }

    predictor_kwargs = request.get("predictor_kwargs")
    if isinstance(predictor_kwargs, Mapping):
        overrides.update(cast(Mapping[str, object], predictor_kwargs))

    if str(device).startswith("mps"):
        overrides["half"] = False

    predictor = SAM3VideoSemanticPredictor(overrides=overrides)
    predictor.setup_model(verbose=bool(overrides["verbose"]))
    predictor.score_threshold_detection = float(overrides["conf"])

    imgsz_checked = check_imgsz(
        overrides.get("imgsz", 640),
        stride=int(getattr(predictor, "stride", 14)),
        min_dim=1,
        max_dim=2,
    )
    if isinstance(imgsz_checked, int):
        predictor.imgsz = (int(imgsz_checked), int(imgsz_checked))
    else:
        dims = [int(x) for x in imgsz_checked]
        predictor.imgsz = (dims[0], dims[0]) if len(dims) == 1 else (dims[0], dims[1])
    if hasattr(predictor.model, "set_imgsz"):
        predictor.model.set_imgsz(predictor.imgsz)

    _SESSION = Sam3VideoSession(
        predictor=predictor,
        device=device,
        conf=float(overrides["conf"]),
        timeline_capacity=max(1, int(request.get("video_capacity", 65536))),
    )
    return {"ok": True, "device": device}


def set_conf_threshold(request: Mapping[str, object]) -> SetConfThresholdResponse:
    """Update the active SAM3 confidence threshold."""
    if "conf" not in request:
        raise ValueError("set_conf_threshold requires 'conf'.")
    value = _require_session().set_conf_threshold(float(request["conf"]))
    return {"ok": True, "conf": value}


def shutdown() -> GenericOkResponse:
    """Shutdown and discard the active SAM3 session."""
    global _SESSION
    if _SESSION is not None:
        _SESSION.close()
        _SESSION = None
    return {"ok": True}


def predict(input: object) -> list["TRex.Result"]:
    """Run stateful SAM3 video-semantic inference for a C++-provided `TRex.Sam3Input`."""
    session = _require_session()
    if TRex is None:
        raise RuntimeError("TRex module is required for SAM3 predict().")

    base = input.base()
    images = list(base.images())
    orig_ids = list(base.orig_id())
    scales = list(base.scales())
    prompts_per_image = list(input.prompts_per_image()) if hasattr(input, "prompts_per_image") else []

    if len(images) != len(orig_ids) or len(images) != len(scales):
        raise ValueError("SAM3 predict received mismatched images/orig_id/scales lengths.")
    if prompts_per_image and len(prompts_per_image) != len(images):
        raise ValueError("SAM3 predict received prompts_per_image with a different length than images().")

    if not prompts_per_image:
        prompts_per_image = [[] for _ in images]

    results: list["TRex.Result"] = []
    for frame_index, image, scale, prompt_list in zip(orig_ids, images, scales, prompts_per_image):
        idx = int(frame_index)
        normalized_image = _normalize_frame(image)
        prompt_state = _collect_prompt_state(cast(Sequence[object], prompt_list))
        session.store_frame(idx, normalized_image, prompt_state)

        if session.last_processed_frame is None:
            session.reset_runtime(idx)
            result = _run_one_frame(session, session.frame_cache[idx], scale)
        elif idx == session.last_processed_frame + 1:
            result = _run_one_frame(session, session.frame_cache[idx], scale)
        else:
            result = _replay_to_frame(session, idx, scale)

        results.append(result)

    return results
