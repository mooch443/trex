# -*- coding: utf-8 -*-
"""Almost-stateless SAM3 adapter used by the C++ interactive backend.

The contract is intentionally narrow:
- C++ owns prompt history, keyframe selection, and replay orchestration.
- Python owns only the loaded SAM3 model plus the currently active mutable
  runtime for the branch being processed right now.
- Public entrypoints are `create_session`, `reset_runtime`, `predict_frame`,
  and `shutdown`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import gc
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import numpy.typing as npt
import torch
from torch.nn import functional as F
from tqdm import tqdm
from ultralytics.models.sam.predict import SAM3VideoSemanticPredictor
from ultralytics.utils.checks import check_imgsz

try:
    import TRex  # type: ignore
except Exception:  # pragma: no cover
    TRex = None


class GenericOkResponse(TypedDict):
    ok: bool


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
class _VideoDatasetShim:
    """Small dataset shim that satisfies Ultralytics' video predictor state."""

    frames: int
    frame: int = 1
    mode: str = "video"


@dataclass
class Sam3VideoSession:
    """Loaded SAM3 video predictor plus the current mutable runtime only."""

    predictor: SAM3VideoSemanticPredictor
    device: str
    conf: float
    timeline_capacity: int | None = None
    last_processed_frame: int | None = None
    active_texts: list[str] = field(default_factory=list)
    duplicate_mask_iou: float | None = 0.95
    timeline_slack: int = 10

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

    def set_iou_threshold(self, iou: float | None) -> float | None:
        """Update the duplicate-mask IoU suppression threshold."""
        if iou is None:
            self.duplicate_mask_iou = None
            return None

        value = float(np.clip(float(iou), 0.0, 1.0))
        self.duplicate_mask_iou = value
        return value

    def reset_runtime(self, max_frame_index: int) -> None:
        """Reset predictor state while keeping the loaded model."""
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
        """Fully discard runtime state and model references for the active session."""
        try:
            self.reset_runtime(max(self.last_processed_frame or 0, 0))
        except Exception:
            pass

        shutdown_fn = getattr(self.predictor, "shutdown", None)
        if callable(shutdown_fn):
            try:
                shutdown_fn()
            except Exception:
                pass

        self.active_texts = []
        self.last_processed_frame = None
        self.predictor.inference_state = {}
        self.predictor.batch = None
        self.predictor.im = None
        self.predictor.dataset = None
        _cleanup_device_caches(self.device)

    def _ensure_runtime_capacity(self, max_frame_index: int) -> None:
        """Ensure the predictor can represent frames up to `max_frame_index`."""
        required_frames = max(1, max_frame_index + 1)
        if self.timeline_capacity is not None:
            required_frames = max(required_frames, int(self.timeline_capacity))
        else:
            required_frames += max(0, int(self.timeline_slack))

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
            tracker_capacity = self.predictor.inference_state.get("tracker_capacity")
            if tracker_capacity is not None:
                try:
                    self.predictor.inference_state["tracker_capacity"] = max(
                        int(tracker_capacity),
                        int(self.predictor.dataset.frames),
                    )
                except Exception:
                    pass

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
_SESSION_LOG_PREFIX = "[py][sam3-session]"
_REPLAY_PROGRESS: tqdm | None = None


def _choose_device() -> str:
    """Resolve the torch device from TRex first, then local availability."""
    if TRex is not None and hasattr(TRex, "choose_device"):
        return str(TRex.choose_device())
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _session_log(message: str) -> None:
    """Emit filterable session diagnostics even when TRex logging is unavailable."""
    formatted = f"{_SESSION_LOG_PREFIX} {message}"
    if TRex is not None and hasattr(TRex, "log"):
        TRex.log(formatted)  # type: ignore[union-attr]
    else:  # pragma: no cover
        print(formatted)


def _close_replay_progress() -> None:
    global _REPLAY_PROGRESS
    if _REPLAY_PROGRESS is not None:
        try:
            _REPLAY_PROGRESS.close()
        except Exception:
            pass
        _REPLAY_PROGRESS = None


def _coerce_optional_threshold(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() == "null":
        return None
    return float(np.clip(float(value), 0.0, 1.0))


def _resolve_iou_threshold(request: Mapping[str, object]) -> float | None:
    """Resolve the IoU threshold from request payload first, then TRex settings."""
    for key in ("detect_iou_threshold", "iou_threshold", "iou", "duplicate_mask_iou"):
        if key in request:
            return _coerce_optional_threshold(request[key])

    if TRex is not None and hasattr(TRex, "setting"):
        try:
            return _coerce_optional_threshold(TRex.setting("detect_iou_threshold"))  # type: ignore[union-attr]
        except Exception:
            pass

    return None


def _resolve_conf_threshold(default: float) -> float:
    """Resolve the active confidence threshold from TRex settings when available."""
    if TRex is not None and hasattr(TRex, "setting"):
        try:
            return float(np.clip(float(TRex.setting("detect_conf_threshold")), 0.0, 1.0))  # type: ignore[union-attr]
        except Exception:
            pass
    return float(np.clip(default, 0.0, 1.0))


def _sync_runtime_settings(session: Sam3VideoSession) -> None:
    """Refresh runtime thresholds from TRex settings before executing a frame step."""
    conf = _resolve_conf_threshold(session.conf)
    iou = _resolve_iou_threshold({})
    session.set_conf_threshold(conf)
    session.set_iou_threshold(iou)
    predictor = session.predictor
    try:
        predictor.score_threshold_detection = conf
    except Exception:
        pass


def _set_optional_predictor_attr(predictor: object, name: str, value: object) -> None:
    """Best-effort runtime attribute assignment for SAM3 predictor knobs."""
    try:
        setattr(predictor, name, value)
    except Exception:
        pass


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
        raise RuntimeError("TRex module is required for SAM3 predict_frame().")

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


def _scale_components(scale: object) -> tuple[float, float]:
    return float(getattr(scale, "x", 1.0)), float(getattr(scale, "y", 1.0))


def _offset_components(offset: object) -> tuple[float, float]:
    return float(getattr(offset, "x", 0.0)), float(getattr(offset, "y", 0.0))


def _estimate_original_image_shape(
    model_shape: tuple[int, int],
    scale: object,
    offset: object,
) -> tuple[int, int]:
    scale_x, scale_y = _scale_components(scale)
    offset_x, offset_y = _offset_components(offset)
    content_w = max(1, int(round(float(model_shape[1]) + 2.0 * offset_x)))
    content_h = max(1, int(round(float(model_shape[0]) + 2.0 * offset_y)))
    original_w = max(1, int(round(content_w * scale_x)))
    original_h = max(1, int(round(content_h * scale_y)))
    return original_h, original_w


def _restore_mask_to_original(
    mask: npt.NDArray[np.uint8],
    model_shape: tuple[int, int],
    scale: object,
    offset: object,
) -> npt.NDArray[np.uint8]:
    original_h, original_w = _estimate_original_image_shape(model_shape, scale, offset)
    scale_x, scale_y = _scale_components(scale)
    offset_x, offset_y = _offset_components(offset)
    pad_left = max(0, int(round(-offset_x)))
    pad_top = max(0, int(round(-offset_y)))
    content_w = max(1, int(round(original_w / scale_x)))
    content_h = max(1, int(round(original_h / scale_y)))
    pad_right = min(mask.shape[1], pad_left + content_w)
    pad_bottom = min(mask.shape[0], pad_top + content_h)
    cropped = np.ascontiguousarray(mask[pad_top:pad_bottom, pad_left:pad_right])
    if cropped.size == 0:
        return np.zeros((original_h, original_w), dtype=np.uint8)
    if cropped.shape != (content_h, content_w):
        cropped = _resize_mask(cropped, content_h, content_w)
    return _resize_mask(cropped, original_h, original_w)


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

    _session_log(
        f"received_prompts texts={len(texts)} boxes={len(boxes)} points={len(points)}"
    )

    return PromptState(
        texts=texts,
        boxes=np.asarray(boxes, dtype=np.float32) if boxes else None,
        points=np.asarray(points, dtype=np.float32) if points else None,
    )


def _select_masks_matching_points(
    masks_np: npt.NDArray[np.uint8],
    image_shape: tuple[int, int],
    points_normalized: npt.NDArray[np.float32],
) -> npt.NDArray[np.intp]:
    """Return indices of masks that contain all requested point prompts."""
    if masks_np.size == 0 or points_normalized.size == 0:
        return np.empty((0,), dtype=np.intp)

    points_px = _denormalize_points(points_normalized, image_shape)
    matches: list[int] = []

    for idx, mask in enumerate(masks_np):
        if all(_mask_contains_point(mask, point[0], point[1]) for point in points_px):
            matches.append(idx)

    return np.asarray(matches, dtype=np.intp)


def _build_result(
    frame_index: int,
    scale: object,
    offset: object,
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
        raise RuntimeError("TRex module is required for SAM3 predict_frame().")

    if keep_indices is not None:
        masks_np = masks_np[keep_indices]
        conf_np = conf_np[keep_indices]
        cls_np = cls_np[keep_indices]
        if pred_boxes_np is not None:
            pred_boxes_np = pred_boxes_np[keep_indices]

    scale_x = float(getattr(scale, "x", 1.0))
    scale_y = float(getattr(scale, "y", 1.0))
    offset_x, offset_y = _offset_components(offset)
    target_h, target_w = _estimate_original_image_shape(image_shape, scale, offset)

    n = masks_np.shape[0]
    trex_masks: list[npt.NDArray[np.uint8]] = []
    boxes = np.zeros((n, 6), dtype=np.float32)

    _session_log(
        f"frame={frame_index} masks={n} target={target_w}x{target_h} "
        f"scale=({scale_x:.3f},{scale_y:.3f}) offset=({offset_x:.3f},{offset_y:.3f})"
    )

    for idx, mask in enumerate(masks_np):
        resized_mask = _restore_mask_to_original(mask * np.uint8(255), image_shape, scale, offset)

        if pred_boxes_np is not None and idx < len(pred_boxes_np):
            x0 = float((pred_boxes_np[idx, 0] + offset_x) * scale_x)
            y0 = float((pred_boxes_np[idx, 1] + offset_y) * scale_y)
            x1 = float((pred_boxes_np[idx, 2] + offset_x) * scale_x)
            y1 = float((pred_boxes_np[idx, 3] + offset_y) * scale_y)
            boxes[idx, 0] = x0
            boxes[idx, 1] = y0
            boxes[idx, 2] = max(0.0, x1 - x0)
            boxes[idx, 3] = max(0.0, y1 - y0)
        else:
            ys, xs = np.where(resized_mask > 0)
            if xs.size and ys.size:
                boxes[idx, 0] = float(xs.min())
                boxes[idx, 1] = float(ys.min())
                boxes[idx, 2] = float(xs.max() + 1) - boxes[idx, 0]
                boxes[idx, 3] = float(ys.max() + 1) - boxes[idx, 1]
                cropped = resized_mask[int(ys.min()) : int(ys.max()) + 1, int(xs.min()) : int(xs.max()) + 1].copy()
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


def _prepare_prompt_batch(prompt_state: PromptState) -> tuple[str | list[str] | None, npt.NDArray[np.float32] | None]:
    """Prepare one shared concept prompt plus optional multi-box geometry."""
    boxes = prompt_state.boxes
    texts = [str(text) for text in prompt_state.texts]

    if boxes is None or len(boxes) == 0:
        if not texts:
            return None, None
        return texts[0] if len(texts) == 1 else texts, None

    if not texts:
        # Let SAM3 use its built-in visual prompt path for box-only prompting.
        return None, boxes

    # Multiple positive boxes on one frame are one shared concept with many objects.
    # Ultralytics' add_prompt() expects the concept batch size to stay at 1 here.
    return texts[0], boxes


def _add_multi_object_prompt(
    session: Sam3VideoSession,
    frame_index: int,
    texts: Sequence[str],
    boxes: npt.NDArray[np.float32],
) -> Mapping[str, object]:
    """Dispatch one box at a time so each new seed enters SAM3 as a separate addition."""
    predictor = session.predictor
    concept_text = next((str(text) for text in texts if str(text)), None)
    output: Mapping[str, object] | None = None

    _session_log(
        f"dispatch_multi_object_prompt frame={frame_index} objects={len(boxes)} "
        f"mode=sequential concept={'yes' if concept_text else 'no'}"
    )

    for box_index, box in enumerate(boxes):
        _, output = predictor.add_prompt(
            frame_idx=frame_index,
            text=concept_text,
            bboxes=np.asarray([box], dtype=np.float32),
            labels=np.ones((1,), dtype=np.int32),
        )
        _session_log(
            f"dispatch_multi_object_prompt frame={frame_index} step={box_index + 1}/{len(boxes)}"
        )

    if output is None:
        raise RuntimeError("Sequential SAM3 multi-object prompt dispatch produced no output.")
    return output


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

    weak_positive_masks = pred_mask_logits > 0
    fallback_mask_rows = (~pred_masks.any(dim=(1, 2))) & weak_positive_masks.any(dim=(1, 2)) & ~torch.isnan(tracker_scores)
    if fallback_mask_rows.any():
        pred_masks = pred_masks.clone()
        pred_masks[fallback_mask_rows] = weak_positive_masks[fallback_mask_rows]

    keep = (pred_scores > session.conf) & pred_masks.any(dim=(1, 2))
    pred_masks = pred_masks[keep]
    pred_scores = pred_scores[keep]
    pred_cls = pred_cls[keep]

    if pred_masks.shape[0] > 1 and session.duplicate_mask_iou is not None:
        duplicate_keep = _suppress_near_duplicate_masks(pred_masks, pred_scores, session.duplicate_mask_iou)
        if duplicate_keep.shape[0] == pred_masks.shape[0] and not torch.all(duplicate_keep):
            pred_masks = pred_masks[duplicate_keep]
            pred_scores = pred_scores[duplicate_keep]
            pred_cls = pred_cls[duplicate_keep]

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


def _suppress_near_duplicate_masks(
    pred_masks: torch.Tensor,
    pred_scores: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    """Greedily suppress near-duplicate masks using mask IoU."""
    count = int(pred_masks.shape[0])
    if count < 2:
        return torch.ones((count,), dtype=torch.bool, device=pred_masks.device)

    pred_masks = pred_masks.to(torch.bool).reshape(count, -1)
    areas = pred_masks.sum(dim=1)
    order = sorted(range(count), key=lambda idx: (-float(pred_scores[idx]), idx))
    keep = torch.ones((count,), dtype=torch.bool, device=pred_masks.device)

    for rank, idx in enumerate(order):
        area_idx = int(areas[idx].item())
        if not bool(keep[idx]):
            continue
        if area_idx == 0:
            keep[idx] = False
            continue
        mask = pred_masks[idx]
        for other_idx in order[rank + 1 :]:
            area_other = int(areas[other_idx].item())
            if not bool(keep[other_idx]):
                continue
            if area_other == 0:
                keep[other_idx] = False
                continue
            intersection = int(torch.logical_and(mask, pred_masks[other_idx]).sum().item())
            union = area_idx + area_other - intersection
            if union > 0 and (intersection / union) >= float(iou_threshold):
                keep[other_idx] = False

    return keep


def _run_one_frame(
    session: Sam3VideoSession,
    frame_index: int,
    image: npt.NDArray[np.uint8],
    prompt_state: PromptState,
    scale: object,
    offset: object,
) -> "TRex.Result":
    """Run one frame through the active SAM3 video predictor runtime."""
    _sync_runtime_settings(session)
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
            if boxes is not None and len(boxes) > 1:
                output = _add_multi_object_prompt(
                    session,
                    frame_index,
                    effective_texts,
                    boxes,
                )
            else:
                prompt_text, prompt_boxes = _prepare_prompt_batch(PromptState(
                    texts=effective_texts,
                    boxes=boxes,
                    points=prompt_state.points,
                ))
                _, output = session.predictor.add_prompt(
                    frame_idx=frame_index,
                    text=prompt_text,
                    bboxes=prompt_boxes,
                    labels=_label_array(prompt_boxes),
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
        keep_indices = _select_masks_matching_points(masks_np, image.shape[:2], prompt_state.points)
        if keep_indices.size == 0:
            session.last_processed_frame = frame_index
            return _empty_result(frame_index)

    session.last_processed_frame = frame_index
    return _build_result(
        frame_index=frame_index,
        scale=scale,
        offset=offset,
        image_shape=image.shape[:2],
        masks_np=masks_np,
        conf_np=conf_np,
        cls_np=cls_np,
        keep_indices=keep_indices,
    )


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
        "imgsz": request.get("imgsz", 640),
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
    _set_optional_predictor_attr(predictor, "init_trk_keep_alive", int(request.get("init_trk_keep_alive", 300)))
    _set_optional_predictor_attr(predictor, "max_trk_keep_alive", int(request.get("max_trk_keep_alive", 300)))
    _set_optional_predictor_attr(
        predictor,
        "decrease_trk_keep_alive_for_empty_masklets",
        bool(request.get("decrease_trk_keep_alive_for_empty_masklets", False)),
    )

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

    explicit_video_capacity = max(1, int(request["video_capacity"])) if "video_capacity" in request else None
    _SESSION = Sam3VideoSession(
        predictor=predictor,
        device=device,
        conf=float(overrides["conf"]),
        timeline_capacity=explicit_video_capacity,
        duplicate_mask_iou=_resolve_iou_threshold(request),
    )

    video_capacity_text = str(explicit_video_capacity) if explicit_video_capacity is not None else f"adaptive(+{_SESSION.timeline_slack})"
    _session_log(
        "create_session "
        f"model={weights_path} "
        f"device={device} "
        f"imgsz={predictor.imgsz} "
        f"conf={float(overrides['conf']):.3f} "
        f"half={bool(overrides['half'])} "
        f"video_capacity={video_capacity_text}"
    )
    return {"ok": True, "device": device}


def reset_runtime(request: Mapping[str, object]) -> GenericOkResponse:
    """Reset the mutable predictor runtime while keeping the loaded model."""
    session = _require_session()
    max_frame_index = int(request.get("max_frame_index", 0))
    _session_log(f"reset_runtime max_frame_index={max_frame_index}")
    _close_replay_progress()
    session.reset_runtime(max_frame_index)
    return {"ok": True}


def begin_replay_progress(request: Mapping[str, object]) -> GenericOkResponse:
    """Create a tqdm progress bar for a replay run driven from C++."""
    global _REPLAY_PROGRESS
    _close_replay_progress()

    total_steps = max(0, int(request.get("total_steps", 0)))
    start_frame = int(request.get("start_frame", 0))
    target_frame = int(request.get("target_frame", start_frame))
    if total_steps <= 0:
        return {"ok": True}

    _REPLAY_PROGRESS = tqdm(
        total=total_steps,
        desc=f"SAM3 replay {start_frame}->{target_frame}",
        unit="frame",
        dynamic_ncols=True,
        leave=False,
    )
    return {"ok": True}


def advance_replay_progress(request: Mapping[str, object]) -> GenericOkResponse:
    """Advance the active replay tqdm progress bar."""
    steps = max(0, int(request.get("steps", 1)))
    if _REPLAY_PROGRESS is not None and steps > 0:
        _REPLAY_PROGRESS.update(steps)
    return {"ok": True}


def finish_replay_progress(_: Mapping[str, object] | None = None) -> GenericOkResponse:
    """Close the active replay tqdm progress bar."""
    _close_replay_progress()
    return {"ok": True}


def shutdown() -> GenericOkResponse:
    """Shutdown and discard the active SAM3 session."""
    global _SESSION
    _close_replay_progress()
    if _SESSION is not None:
        _session_log("shutdown")
        _SESSION.close()
        _SESSION = None
    return {"ok": True}


def predict_frame(input: object) -> list["TRex.Result"]:
    """Run the supplied image-aligned prompts against the current SAM3 runtime."""
    session = _require_session()
    if TRex is None:
        raise RuntimeError("TRex module is required for SAM3 predict_frame().")

    base = input.base()
    images = list(base.images())
    orig_ids = list(base.orig_id())
    offsets = list(base.offsets())
    scales = list(base.scales())
    prompts_per_image = list(input.prompts_per_image()) if hasattr(input, "prompts_per_image") else []

    if len(images) != len(orig_ids) or len(images) != len(scales) or len(images) != len(offsets):
        raise ValueError("SAM3 predict_frame received mismatched images/orig_id/scales/offsets lengths.")
    if prompts_per_image and len(prompts_per_image) != len(images):
        raise ValueError("SAM3 predict_frame received prompts_per_image with a different length than images().")

    if not prompts_per_image:
        prompts_per_image = [[] for _ in images]

    results: list["TRex.Result"] = []
    for frame_index, image, scale, offset, prompt_list in zip(orig_ids, images, scales, offsets, prompts_per_image):
        idx = int(frame_index)
        normalized_image = _normalize_frame(image)
        prompt_state = _collect_prompt_state(cast(Sequence[object], prompt_list))
        results.append(_run_one_frame(session, idx, normalized_image, prompt_state, scale, offset))

    return results
