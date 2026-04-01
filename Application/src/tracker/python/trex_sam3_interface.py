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


def _collect_prompt_kwargs(prompt_list: Sequence[object]) -> dict[str, object]:
    """Merge one image's ordered prompt list into predictor kwargs."""
    texts: list[str] = []
    boxes: list[list[float]] = []
    points: list[list[float]] = []
    point_labels: list[float] = []

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
                point_labels.append(1.0)
            continue

        raise ValueError(f"Unsupported SAM3 prompt type: {ptype}")

    kwargs: dict[str, object] = {}
    if texts:
        kwargs["text"] = texts
    if boxes:
        kwargs["bboxes"] = np.asarray(boxes, dtype=np.float32)
    if points:
        kwargs["points"] = np.asarray(points, dtype=np.float32)
        kwargs["point_labels"] = np.asarray(point_labels, dtype=np.float32)
    return kwargs


def _result_from_public_prediction(
    frame_index: int,
    prediction: object,
    scale: object,
) -> "TRex.Result":
    """Convert a standard Ultralytics SAM3 `Results` item into `TRex.Result`."""
    if TRex is None:
        raise RuntimeError("TRex module is required for SAM3 predict().")

    masks_data = getattr(getattr(prediction, "masks", None), "data", None)
    boxes_data = getattr(prediction, "boxes", None)
    if masks_data is None:
        return _empty_result(frame_index)

    masks_np = cast(
        npt.NDArray[np.uint8],
        masks_data.detach().to(torch.uint8).cpu().numpy(),
    )

    box_conf = np.zeros((masks_np.shape[0],), dtype=np.float32)
    box_cls = np.zeros((masks_np.shape[0],), dtype=np.float32)
    if boxes_data is not None and getattr(boxes_data, "conf", None) is not None:
        box_conf = boxes_data.conf.detach().cpu().numpy().astype(np.float32, copy=False)
    if boxes_data is not None and getattr(boxes_data, "cls", None) is not None:
        box_cls = boxes_data.cls.detach().cpu().numpy().astype(np.float32, copy=False)

    scale_x = float(getattr(scale, "x", 1.0))
    scale_y = float(getattr(scale, "y", 1.0))
    trex_masks: list[npt.NDArray[np.uint8]] = []
    boxes = np.zeros((masks_np.shape[0], 6), dtype=np.float32)

    for idx, mask in enumerate(masks_np):
        target_h = max(1, int(round(mask.shape[0] * scale_y)))
        target_w = max(1, int(round(mask.shape[1] * scale_x)))
        resized_mask = _resize_mask(mask * np.uint8(255), target_h, target_w)
        trex_masks.append(resized_mask)

        boxes[idx, 0] = 0.0
        boxes[idx, 1] = 0.0
        boxes[idx, 2] = float(resized_mask.shape[1])
        boxes[idx, 3] = float(resized_mask.shape[0])
        boxes[idx, 4] = float(box_conf[idx]) if idx < len(box_conf) else 0.0
        boxes[idx, 5] = float(box_cls[idx]) if idx < len(box_cls) else 0.0

    return TRex.Result(  # type: ignore[attr-defined]
        int(frame_index),
        TRex.Boxes(boxes),  # type: ignore[attr-defined]
        trex_masks,
        TRex.KeypointData(np.empty((0, 1, 2), dtype=np.float32)),  # type: ignore[attr-defined]
        TRex.ObbData(np.empty((0, 7), dtype=np.float32)),  # type: ignore[attr-defined]
        TRex.PointData(np.empty((0, 5), dtype=np.float32)),  # type: ignore[attr-defined]
    )


def _result_from_prompt_masks(
    frame_index: int,
    masks: object,
    scores: object,
    scale: object,
) -> "TRex.Result":
    """Convert prompt-inference tensors into a `TRex.Result`."""
    if TRex is None:
        raise RuntimeError("TRex module is required for SAM3 predict().")

    masks_np = cast(
        npt.NDArray[np.uint8],
        cast(torch.Tensor, masks).detach().to(torch.uint8).cpu().numpy(),
    )
    scores_np = cast(
        npt.NDArray[np.float32],
        cast(torch.Tensor, scores).detach().cpu().numpy().astype(np.float32, copy=False),
    )

    scale_x = float(getattr(scale, "x", 1.0))
    scale_y = float(getattr(scale, "y", 1.0))
    trex_masks: list[npt.NDArray[np.uint8]] = []
    boxes = np.zeros((masks_np.shape[0], 6), dtype=np.float32)

    for idx, mask in enumerate(masks_np):
        target_h = max(1, int(round(mask.shape[0] * scale_y)))
        target_w = max(1, int(round(mask.shape[1] * scale_x)))
        resized_mask = _resize_mask(mask * np.uint8(255), target_h, target_w)
        trex_masks.append(resized_mask)

        ys, xs = np.where(resized_mask > 0)
        if xs.size and ys.size:
            boxes[idx, 0] = float(xs.min())
            boxes[idx, 1] = float(ys.min())
            boxes[idx, 2] = float(xs.max() + 1)
            boxes[idx, 3] = float(ys.max() + 1)
        boxes[idx, 4] = float(scores_np[idx]) if idx < len(scores_np) else 0.0
        boxes[idx, 5] = 0.0

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
    bboxes = cast(npt.NDArray[np.float32] | None, prompt_kwargs.get("bboxes"))
    points = cast(npt.NDArray[np.float32] | None, prompt_kwargs.get("points"))
    point_labels = cast(npt.NDArray[np.float32] | None, prompt_kwargs.get("point_labels"))

    print(f"[SAM3] frame={frame_index}: texts={texts} bboxes_shape={None if bboxes is None else bboxes.shape} points_shape={None if points is None else points.shape}")

    if bboxes is None and points is None:
        print(f"[SAM3] frame={frame_index}: text-only path — calling predictor(source=..., conf={session.conf}, text={texts})")
        results = session.predictor(source=image, stream=True, conf=float(session.conf), text=texts)
        first = _first_stream_item(results)
        if first is None:
            print(f"[SAM3] frame={frame_index}: predictor returned no results — returning empty result")
            return _empty_result(frame_index)
        masks_data = getattr(getattr(first, "masks", None), "data", None)
        n_masks = 0 if masks_data is None else masks_data.shape[0]
        print(f"[SAM3] frame={frame_index}: text-only path got {n_masks} mask(s)")
        return _result_from_public_prediction(frame_index, first, scale)

    reset_image = getattr(session.predictor, "reset_image", None)
    if callable(reset_image):
        reset_image()
    session.predictor.set_image(image)
    batch = getattr(session.predictor, "batch", None)
    if batch is None:
        raise RuntimeError("SAM3 predictor did not expose a prepared batch for prompt inference.")

    model_input = session.predictor.preprocess(batch[1])
    print(f"[SAM3] frame={frame_index}: model_input shape={model_input.shape}")

    features = session.predictor.features
    if features is None:
        print(f"[SAM3] frame={frame_index}: features not cached — calling get_im_features")
        features = session.predictor.get_im_features(model_input)
    else:
        print(f"[SAM3] frame={frame_index}: using cached features")

    if texts:
        print(f"[SAM3] frame={frame_index}: text+prompt path — _prepare_prompts then _inference_features")
        prepared_points, prepared_labels, _ = session.predictor._prepare_prompts(
            model_input.shape[2:],
            batch[1][0].shape[:2],
            bboxes=bboxes,
            points=points,
            labels=point_labels,
            masks=None,
        )
        masks, scores, _ = session.predictor._inference_features(
            features,
            prepared_points,
            prepared_labels,
            text=texts,
        )
    else:
        print(f"[SAM3] frame={frame_index}: prompt_inference path (no text)")
        masks, scores, _ = session.predictor.prompt_inference(
            model_input,
            bboxes=bboxes,
            points=points,
            labels=point_labels,
        )

    masks_shape = getattr(masks, "shape", None)
    scores_val = scores.detach().cpu().numpy() if hasattr(scores, "detach") else scores
    print(f"[SAM3] frame={frame_index}: inference done — masks={masks_shape} scores={scores_val}")

    return _result_from_prompt_masks(frame_index, masks, scores, scale)


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
