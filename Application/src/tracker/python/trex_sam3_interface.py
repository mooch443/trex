# -*- coding: utf-8 -*-
"""Minimal SAM3 adapter used by the C++ SAM3 backend.

The contract is intentionally small:
- C++ owns frame lookup, prompt lookup, batching, and post-processing policy.
- Python owns only the loaded SAM3 predictor and direct per-image inference.
- Public entrypoints are `create_session`, `set_conf_threshold`, `shutdown`,
  and `predict`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import numpy.typing as npt
import torch
from torch.nn import functional as F
from ultralytics.models.sam.predict import SAM3SemanticPredictor
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
class Sam3Session:
    """Loaded SAM3 predictor plus the small amount of runtime state it needs."""

    predictor: SAM3SemanticPredictor
    device: str
    conf: float

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


_SESSION: Sam3Session | None = None


def _choose_device() -> str:
    """Resolve the torch device from TRex first, then local availability."""
    if TRex is not None and hasattr(TRex, "choose_device"):
        return str(TRex.choose_device())
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _first_stream_item(results: object) -> object | None:
    """Return the first item from predictor output without retaining the stream."""
    if results is None:
        return None
    if isinstance(results, Sequence):
        return results[0] if results else None
    try:
        iterator = iter(cast(Iterable[object], results))
    except TypeError:
        return results
    first = next(iterator, None)
    close = getattr(iterator, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass
    return first


def _require_session() -> Sam3Session:
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


def _collect_prompt_kwargs(prompt_list: Sequence[object]) -> dict[str, object]:
    """Merge one image's normalized prompt list into Python-native arrays."""
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

    kwargs: dict[str, object] = {}
    if texts:
        kwargs["text"] = texts
    if boxes:
        kwargs["bboxes"] = np.asarray(boxes, dtype=np.float32)
    if points:
        kwargs["points"] = np.asarray(points, dtype=np.float32)
    return kwargs


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
    """Build a TRex.Result from mask/score arrays.

    image_shape: the source image size
    pred_boxes_np: (N, 4) XYXY coords in input space, scaled by scale_x/y.
                   Used by the inference-features (box-prompt) path.
    If neither is given, boxes are derived from each mask's tight bounding rect.
    keep_indices: optional index filter applied before any processing.
    """
    if TRex is None:
        raise RuntimeError("TRex module is required for SAM3 predict().")

    assert image_shape is not None, "image_shape is required for _build_result in this SAM3 interface implementation"

    if keep_indices is not None:
        masks_np = masks_np[keep_indices]
        conf_np = conf_np[keep_indices]
        cls_np = cls_np[keep_indices]
        if pred_boxes_np is not None:
            pred_boxes_np = pred_boxes_np[keep_indices]

    scale_x = float(getattr(scale, "x", 1.0))
    scale_y = float(getattr(scale, "y", 1.0))
    n = masks_np.shape[0]
    trex_masks: list[npt.NDArray[np.uint8]] = []
    boxes = np.zeros((n, 6), dtype=np.float32)

    tgt_h, tgt_w = int(round(image_shape[0] * scale_y)), int(round(image_shape[1] * scale_x))
    #tgt_h, tgt_w = int(image_shape[0]), int(image_shape[1])  # --- IGNORE ---
    TRex.log(f"SAM3 prediction for frame={frame_index} has {n} mask(s) before resizing to {tgt_w}x{tgt_h} with scale=({scale_x:.3f},{scale_y:.3f})")  # type: ignore[union-attr]

    for idx, mask in enumerate(masks_np):
        resized_mask = _resize_mask(mask * np.uint8(255), tgt_h, tgt_w)
        #if image_shape is not None:
        #    TRex.imshow("SAM3 mask", resized_mask)  # type: ignore[union-attr]
        trex_masks.append(resized_mask)

        if image_shape is not None:
            boxes[idx, 0] = 0.0
            boxes[idx, 1] = 0.0
            boxes[idx, 2] = float(tgt_w)
            boxes[idx, 3] = float(tgt_h)
        elif pred_boxes_np is not None and idx < len(pred_boxes_np):
            boxes[idx, 0] = float(pred_boxes_np[idx, 0] * scale_x)
            boxes[idx, 1] = float(pred_boxes_np[idx, 1] * scale_y)
            boxes[idx, 2] = float(pred_boxes_np[idx, 2] * scale_x)
            boxes[idx, 3] = float(pred_boxes_np[idx, 3] * scale_y)
        else:
            ys, xs = np.where(resized_mask > 0)
            if xs.size and ys.size:
                boxes[idx, 0] = float(xs.min())
                boxes[idx, 1] = float(ys.min())
                boxes[idx, 2] = float(xs.max() + 1)
                boxes[idx, 3] = float(ys.max() + 1)

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


def _predict_one(
    session: Sam3Session,
    frame_index: int,
    image: npt.NDArray[np.uint8],
    scale: object,
    prompt_list: Sequence[object],
) -> "TRex.Result":
    """Run SAM3 on one image and convert the output to `TRex.Result`."""
    print(f"[SAM3] _predict_one frame_index={frame_index} image={image.shape} dtype={image.dtype} scale=({getattr(scale,'x',1.0):.3f},{getattr(scale,'y',1.0):.3f}) n_prompts={len(prompt_list)}")

    prompt_kwargs = _collect_prompt_kwargs(prompt_list)
    if not prompt_kwargs:
        print(f"[SAM3] frame={frame_index}: no prompt kwargs — returning empty result")
        return _empty_result(frame_index)

    texts = cast(list[str] | None, prompt_kwargs.get("text"))
    normalized_bboxes = cast(npt.NDArray[np.float32] | None, prompt_kwargs.get("bboxes"))
    normalized_points = cast(npt.NDArray[np.float32] | None, prompt_kwargs.get("points"))
    bboxes = _denormalize_boxes(normalized_bboxes, image.shape[:2]) if normalized_bboxes is not None else None
    points = _denormalize_points(normalized_points, image.shape[:2]) if normalized_points is not None else None

    print(
        f"[SAM3] frame={frame_index} dims={image.shape} scale=({getattr(scale,'x',1.0):.3f},{getattr(scale,'y',1.0):.3f}): texts={texts}"
        f" bboxes_shape={None if bboxes is None else bboxes.shape}"
        f" points_shape={None if points is None else points.shape}"
        f" normalized_boxes={None if normalized_bboxes is None else normalized_bboxes.tolist()}"
    )

    if bboxes is None and points is None:
        print(f"[SAM3] frame={frame_index}: text-only path — calling predictor(source=..., conf={session.conf}, text={texts})")
        results = session.predictor(source=image, stream=True, conf=float(session.conf), text=texts)
        first = _first_stream_item(results)
        if first is None:
            print(f"[SAM3] frame={frame_index}: predictor returned no results — returning empty result")
            return _empty_result(frame_index)
        masks_data = getattr(getattr(first, "masks", None), "data", None)
        if masks_data is None:
            print(f"[SAM3] frame={frame_index}: text-only path got no masks")
            return _empty_result(frame_index)
        masks_np = cast(npt.NDArray[np.uint8], masks_data.detach().to(torch.uint8).cpu().numpy())
        boxes_data = getattr(first, "boxes", None)
        conf_np = np.zeros(masks_np.shape[0], dtype=np.float32)
        cls_np = np.zeros(masks_np.shape[0], dtype=np.float32)
        if boxes_data is not None and getattr(boxes_data, "conf", None) is not None:
            conf_np = boxes_data.conf.detach().cpu().numpy().astype(np.float32, copy=False)
        if boxes_data is not None and getattr(boxes_data, "cls", None) is not None:
            cls_np = boxes_data.cls.detach().cpu().numpy().astype(np.float32, copy=False)
        print(f"[SAM3] frame={frame_index}: text-only path got {masks_np.shape[0]} mask(s)")
        return _build_result(frame_index=frame_index, scale=scale, masks_np=masks_np, conf_np=conf_np, cls_np=cls_np, image_shape=image.shape[:2])

    if points is not None and bboxes is None:
        if texts:
            print(f"[SAM3] frame={frame_index}: text+points safe fallback — running text-only inference then selecting masks by points")
            results = session.predictor(source=image, stream=True, conf=float(session.conf), text=texts)
            first = _first_stream_item(results)
            if first is None:
                print(f"[SAM3] frame={frame_index}: text+points fallback returned no text results")
                return _empty_result(frame_index)

            masks_data = getattr(getattr(first, "masks", None), "data", None)
            if masks_data is None:
                print(f"[SAM3] frame={frame_index}: text+points fallback had no masks to filter")
                return _empty_result(frame_index)

            masks_np = cast(
                npt.NDArray[np.uint8],
                masks_data.detach().to(torch.uint8).cpu().numpy(),
            )
            boxes_data = getattr(first, "boxes", None)
            conf_np = np.zeros(masks_np.shape[0], dtype=np.float32)
            cls_np = np.zeros(masks_np.shape[0], dtype=np.float32)
            if boxes_data is not None and getattr(boxes_data, "conf", None) is not None:
                conf_np = boxes_data.conf.detach().cpu().numpy().astype(np.float32, copy=False)
            if boxes_data is not None and getattr(boxes_data, "cls", None) is not None:
                cls_np = boxes_data.cls.detach().cpu().numpy().astype(np.float32, copy=False)
            keep_indices = _select_masks_matching_points(
                masks_np,
                scale,
                normalized_points if normalized_points is not None else np.empty((0, 2), dtype=np.float32),
            )
            print(f"[SAM3] frame={frame_index}: text+points fallback kept {len(keep_indices)} of {masks_np.shape[0]} mask(s)")
            if keep_indices.size == 0:
                return _empty_result(frame_index)
            return _build_result(frame_index=frame_index, scale=scale, masks_np=masks_np, conf_np=conf_np, cls_np=cls_np, image_shape=image.shape[:2], keep_indices=keep_indices)

        print(f"[SAM3] frame={frame_index}: point-only prompts are disabled for SAM3 in this build to avoid unstable predictor crashes")
        return _empty_result(frame_index)

    if points is not None and bboxes is not None:
        print(f"[SAM3] frame={frame_index}: ignoring point prompts because SAM3 geometric inference is only stable with boxes in this build")

    reset_image = getattr(session.predictor, "reset_image", None)
    if callable(reset_image):
        reset_image()
    session.predictor.set_image(image)

    features = session.predictor.features
    if features is None:
        raise RuntimeError("SAM3 predictor did not expose cached features after set_image().")
    print(f"[SAM3] frame={frame_index}: using features from set_image()")

    print(
        f"[SAM3] frame={frame_index} dims={image.shape}: SAM3 box path — using {len(bboxes) if bboxes is not None else 0} box prompt(s)"
        f" text={texts}"
        f" boxes={bboxes.tolist() if bboxes is not None else None}"
    )
    masks, pred_boxes = session.predictor.inference_features(
        features,
        image.shape[:2],
        bboxes=bboxes,
        text=texts,
    )

    masks_shape = getattr(masks, "shape", None)
    pred_boxes_shape = getattr(pred_boxes, "shape", None)
    print(f"[SAM3] frame={frame_index}: inference done — masks={masks_shape} pred_boxes={pred_boxes_shape}")

    if masks is None:
        return _empty_result(frame_index)
    masks_np = cast(npt.NDArray[np.uint8], cast(torch.Tensor, masks).detach().to(torch.uint8).cpu().numpy())
    boxes_np = cast(npt.NDArray[np.float32], cast(torch.Tensor, pred_boxes).detach().cpu().numpy().astype(np.float32, copy=False))
    return _build_result(frame_index=frame_index, scale=scale, masks_np=masks_np, conf_np=boxes_np[:, 4], cls_np=boxes_np[:, 5], pred_boxes_np=boxes_np[:, :4], image_shape=image.shape[:2])


def create_session(request: Mapping[str, object]) -> CreateSessionResponse:
    """Create or replace the singleton SAM3 model session."""
    global _SESSION

    if "weights_path" not in request:
        raise ValueError("create_session requires 'weights_path'.")

    if _SESSION is not None:
        shutdown()

    device = _choose_device()
    overrides: dict[str, object] = {
        "task": "segment",
        "mode": "predict",
        "model": str(cast(str | Path, request["weights_path"])),
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

    predictor = SAM3SemanticPredictor(overrides=overrides)
    predictor.setup_model(verbose=bool(overrides["verbose"]))

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

    _SESSION = Sam3Session(
        predictor=predictor,
        device=device,
        conf=float(overrides["conf"]),
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
        shutdown_fn = getattr(_SESSION.predictor, "shutdown", None)
        if callable(shutdown_fn):
            shutdown_fn()
        _SESSION = None
    return {"ok": True}


def predict(input: object) -> list["TRex.Result"]:
    """Run direct SAM3 batch inference for a C++-provided `TRex.Sam3Input`."""
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
        results.append(
            _predict_one(
                session,
                int(frame_index),
                _normalize_frame(image),
                scale,
                cast(Sequence[object], prompt_list),
            )
        )

    return results
