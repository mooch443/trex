# -*- coding: utf-8 -*-
"""TRex SAM3 interface for pybind11 ModuleProxy.

This module is intentionally C++-first:
- C++ owns all video decoding and frame access.
- Python owns only SAM3 model state and prompt/inference logic.
- Integration is driven via `ModuleProxy::run(...)` and `ModuleProxy::set_variable(...)`.

Typical call order from C++:
1) `create_session({...})`
2) Per frame: `set_variable("sam3_frame", <np uint8 HxWx3>)`
3) Per frame: `set_frame({"frame_index": i})`
4) Optional prompt updates: `add_prompt({...})`
5) Query output: `get_frame({"frame_index": i})`
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, NotRequired, Tuple, TypedDict, cast, Any

import numpy as np
import numpy.typing as npt
import torch
from torch.nn import functional as F
from ultralytics.utils.checks import check_imgsz

try:
    import TRex  # type: ignore
except Exception:  # pragma: no cover
    TRex = None

from ultralytics.models.sam.predict import SAM3SemanticPredictor, SAM3VideoSemanticPredictor


# Hard, always-on safety cap to prevent unbounded host-memory growth.
_MAX_FRAME_CACHE = 0


# ModuleProxy::set_variable("sam3_frame", ...) writes this module variable.
sam3_frame: npt.NDArray[np.uint8] | None = None


class BoxPromptPayload(TypedDict):
    boxes: npt.NDArray[np.float32]
    labels: npt.NDArray[np.float32]


class Sam3Object(TypedDict):
    obj_id: int
    score: float
    class_id: int
    height: int
    width: int
    foreground_indices: list[int]


class FrameOutput(TypedDict):
    frame_index: int
    num_objects: int
    objects: list[Sam3Object]
    frame_stats: dict[str, object]
    unconfirmed_obj_ids: list[int]
    ok: NotRequired[bool]


class FrameSetResponse(TypedDict):
    frame_index: int
    height: int
    width: int
    channels: int
    ok: bool


class PromptResponse(TypedDict):
    ok: bool
    type: str
    scope: NotRequired[str]
    frame_index: NotRequired[int]
    unchanged: NotRequired[bool]
    count: NotRequired[int]


class RemoveObjectResponse(TypedDict):
    ok: bool
    obj_id: int
    mode: str


class GenericOkResponse(TypedDict):
    ok: bool


class SetConfThresholdResponse(TypedDict):
    ok: bool
    conf: float


class FramesResponse(TypedDict):
    ok: bool
    frames: list[FrameOutput]


class CreateSessionResponse(TypedDict):
    ok: bool
    device: str
    capabilities: dict[str, bool]


class CapabilitiesResponse(TypedDict):
    ok: bool
    capabilities: dict[str, bool]


@dataclass
class FrameCacheEntry:
    frame: npt.NDArray[np.uint8] | None = None
    box_prompt: BoxPromptPayload | None = None
    output: FrameOutput | None = None


def _log(msg: str) -> None:
    if TRex is not None:
        TRex.log(msg)
    else:
        print(f"[sam3] {msg}")


def _choose_device() -> str:
    """Resolve torch device from TRex settings first, then local availability."""
    if TRex is not None and hasattr(TRex, "choose_device"):
        return str(TRex.choose_device())
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _normalize_text_prompt(value: object) -> str | list[str]:
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [str(item) for item in value]
    raise ValueError("Text prompt must be a string or a sequence of strings.")


def _first_stream_item(results: object) -> object | None:
    """Return the first prediction item from list/tuple/generator-like outputs."""
    if results is None:
        return None
    if isinstance(results, Sequence):
        return results[0] if len(results) > 0 else None
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


@dataclass(frozen=True)
class Sam3Capabilities:
    """Runtime capabilities of the active SAM3 backend and adapter."""

    text: bool
    boxes: bool
    points: bool
    masks: bool
    remove_object: bool
    reset_session: bool
    close_session: bool
    frame_ingest: bool

    def as_dict(self) -> dict[str, bool]:
        return {
            "text": self.text,
            "boxes": self.boxes,
            "points": self.points,
            "masks": self.masks,
            "remove_object": self.remove_object,
            "reset_session": self.reset_session,
            "close_session": self.close_session,
            "frame_ingest": self.frame_ingest,
        }


class Sam3VideoSession:
    """Stateful SAM3 session keyed by frame index, without video file access.

    The session stores only:
    - predictor runtime
    - prompt state
    - cached per-frame outputs
    - frames pushed from C++ for inference

    It never opens/reads videos, by design.
    """

    def __init__(
        self,
        weights_path: str | Path,
        *,
        imgsz: int = 640,
        conf: float = 0.25,
        half: bool = True,
        verbose: bool = True,
        predictor_kwargs: Mapping[str, object] | None = None,
    ) -> None:
        """Initialize predictor and empty session state.

        Args:
            weights_path: SAM3 weights path.
            imgsz: Predictor image size override.
            conf: Confidence threshold.
            half: Request fp16 execution where supported.
            verbose: Verbose backend logging.
            predictor_kwargs: Additional Ultralytics predictor overrides.
        """
        self.weights_path = str(weights_path)
        self.device = _choose_device()
        self.verbose = bool(verbose)
        self._conf = float(conf)
        # Keep a single switch for future re-enabling of video internals.
        self._use_video_backend = False

        overrides: dict[str, object] = {
            "task": "segment",
            "mode": "predict",
            "model": self.weights_path,
            "imgsz": int(imgsz),
            "conf": float(self._conf),
            "half": bool(half),
            "device": self.device,
            "verbose": self.verbose,
        }
        if predictor_kwargs:
            overrides.update(predictor_kwargs)
        # SAM3 on MPS is unstable in half precision; force fp32.
        if str(self.device).startswith("mps"):
            overrides["half"] = False

        predictor_cls = SAM3VideoSemanticPredictor if self._use_video_backend else SAM3SemanticPredictor
        self._predictor = predictor_cls(overrides=overrides)
        self._predictor.setup_model(verbose=self.verbose)
        # setup_source() is not used in this C++ frame-ingest path, so ensure imgsz
        # is initialized for preprocessing and aligned to model stride.
        imgsz_cfg = overrides.get("imgsz", 640)
        imgsz_checked = check_imgsz(
            imgsz_cfg,
            stride=int(getattr(self._predictor, "stride", 14)),
            min_dim=1,
            max_dim=2,
        )
        if isinstance(imgsz_checked, int):
            imgsz_hw = (int(imgsz_checked), int(imgsz_checked))
        else:
            vals = [int(x) for x in imgsz_checked]
            imgsz_hw = (vals[0], vals[0]) if len(vals) == 1 else (vals[0], vals[1])
        self._predictor.imgsz = imgsz_hw
        self._log_effective_device()

        # Prompt and inference state.
        self._text_prompt: str | list[str] | None = None
        self._text_frame_index: int = 0
        self._text_session_scope: bool = False
        self._removed_obj_ids: set[int] = set()

        # Unified per-frame cache entry: pixels + frame-level prompts + output.
        self._frame_cache: OrderedDict[int, FrameCacheEntry] = OrderedDict()
        self._max_computed_frame: int = -1

        self._capabilities = Sam3Capabilities(
            text=True,
            boxes=True,
            points=False,
            masks=False,
            remove_object=True,
            reset_session=True,
            close_session=True,
            frame_ingest=True,
        )

        _log(
            f"SAM3 session created (device={self.device}, backend={'video' if self._use_video_backend else 'image'}, "
            f"weights={self.weights_path})"
        )

    def _log_effective_device(self) -> None:
        """Log effective torch devices for predictor components after setup."""
        requested = self.device
        detector_dev = "unknown"
        tracker_dev = "unknown"

        try:
            model = getattr(self._predictor, "model", None)
            if model is not None:
                detector_dev = str(next(model.parameters()).device)
        except Exception:
            pass

        try:
            tracker = getattr(self._predictor, "tracker", None)
            tmodel = getattr(tracker, "model", None) if tracker is not None else None
            if tmodel is not None:
                tracker_dev = str(next(tmodel.parameters()).device)
        except Exception:
            pass

        _log(
            f"SAM3 effective devices: requested={requested}, detector={detector_dev}, "
            f"tracker={tracker_dev}, imgsz={getattr(self._predictor, 'imgsz', 'unknown')}"
        )

    def capabilities(self) -> dict[str, bool]:
        """Return a plain dict of runtime capability flags."""
        return self._capabilities.as_dict()

    def set_conf_threshold(self, conf: float) -> float:
        """Update predictor confidence threshold used for subsequent inference."""
        conf_value = float(conf)
        if not np.isfinite(conf_value):
            raise ValueError("set_conf_threshold requires a finite numeric `conf`.")

        conf_value = float(np.clip(conf_value, 0.0, 1.0))
        self._conf = conf_value

        overrides = getattr(self._predictor, "overrides", None)
        if isinstance(overrides, dict):
            overrides["conf"] = conf_value

        args = getattr(self._predictor, "args", None)
        if isinstance(args, dict):
            args["conf"] = conf_value
        elif args is not None:
            try:
                setattr(args, "conf", conf_value)
            except Exception:
                pass

        _log(f"SAM3 confidence threshold set to {conf_value:.4f}")
        return conf_value

    def _recompute_max_cached_output_index(self) -> None:
        cached = [idx for idx, entry in self._frame_cache.items() if entry.output is not None]
        self._max_computed_frame = max(cached, default=-1)

    def _get_or_create_frame_entry(self, frame_index: int) -> FrameCacheEntry:
        idx = int(frame_index)
        if _MAX_FRAME_CACHE <= 0:
            # Zero-cache mode: keep exactly one in-flight frame entry so
            # set_frame(frame_i) -> get_frame(frame_i) still works.
            entry = self._frame_cache.get(idx)
            if entry is None:
                self._frame_cache.clear()
                entry = FrameCacheEntry()
                self._frame_cache[idx] = entry
            self._recompute_max_cached_output_index()
            return entry

        entry = self._frame_cache.get(idx)
        if entry is None:
            entry = FrameCacheEntry()
            self._frame_cache[idx] = entry
        self._frame_cache.move_to_end(idx)
        while len(self._frame_cache) > _MAX_FRAME_CACHE:
            self._frame_cache.popitem(last=False)
        self._recompute_max_cached_output_index()
        return entry

    def _get_frame_entry(self, frame_index: int) -> FrameCacheEntry | None:
        idx = int(frame_index)
        entry = self._frame_cache.get(idx)
        if entry is not None:
            self._frame_cache.move_to_end(idx)
        return entry

    def set_frame(self, frame_index: int, frame: npt.ArrayLike) -> FrameSetResponse:
        """Ingest one decoded frame from C++.

        Args:
            frame_index: Absolute frame index in the C++ timeline.
            frame: `np.uint8` array with shape `H x W x 3` (BGR).

        Returns:
            Metadata summary for confirmation.
        """
        arr = np.asarray(frame)
        # Accept common TRex tile layouts and normalize to HxWx3 uint8.
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.ndim == 3 and arr.shape[2] >= 3:
            arr = arr[:, :, :3]
        else:
            raise ValueError("set_frame expects HxW, HxWx1, HxWx3, or HxWx4 image data.")

        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8, copy=False)

        idx = int(frame_index)
        entry = self._get_or_create_frame_entry(idx)
        entry.frame = cast(npt.NDArray[np.uint8], np.ascontiguousarray(arr))
        entry.output = None
        self._invalidate_from(idx)
        return {
            "frame_index": idx,
            "height": int(arr.shape[0]),
            "width": int(arr.shape[1]),
            "channels": int(arr.shape[2]),
            "ok": True,
        }

    def reset_session(self, clear_prompts: bool = False) -> None:
        """Reset incremental state and optionally clear prompts."""
        self._predictor.inference_state = {}
        for entry in self._frame_cache.values():
            entry.output = None
        self._max_computed_frame = -1

        if clear_prompts:
            self._text_prompt = None
            self._text_session_scope = False
            for entry in self._frame_cache.values():
                entry.box_prompt = None
            self._removed_obj_ids.clear()

    def close_session(self) -> None:
        """Release cached frames/outputs and reset predictor state."""
        self.reset_session(clear_prompts=True)
        self._frame_cache.clear()

    def shutdown(self) -> None:
        """Shutdown predictor runtime if backend exposes it."""
        self.close_session()
        if hasattr(self._predictor, "shutdown"):
            self._predictor.shutdown()

    def add_prompt(self, prompt: Mapping[str, object]) -> PromptResponse:
        """Add or update prompts.

        Supported prompt payloads in this adapter:
        - Text:
          `{"type":"text", "text":"fish", "frame_index":0}`
          Optional text behavior keys:
          - `text_session_scope` (bool): if true, set/replace global session text.
          - `text_skip_if_unchanged` (bool): if true, no-op when text is unchanged.
        - Single box:
          `{"type":"box", "frame_index":12, "box":[x1,y1,x2,y2], "label":1}`
        - Multiple boxes:
          `{"type":"boxes", "frame_index":12, "boxes":[[...],[...]], "labels":[1,1]}`

        Note:
        - Point/mask prompts require request-style SAM3 APIs not available in the
          current Ultralytics build used by TRex.
        """
        ptype = str(prompt.get("type", "")).strip().lower()
        frame_index = int(prompt.get("frame_index", 0))
        has_session_scope_flag = ("text_session_scope" in prompt) or ("session_scope" in prompt)
        if has_session_scope_flag and ptype != "text":
            raise ValueError("Session-global prompt scope is supported for text prompts only.")

        if ptype == "text":
            text = prompt.get("text")
            if text is None:
                raise ValueError(f"Text prompt requires key 'text', got: {prompt}")
            text_prompt = _normalize_text_prompt(text)
            session_scope = bool(prompt.get("text_session_scope", prompt.get("session_scope", False)))
            skip_if_unchanged = bool(prompt.get("text_skip_if_unchanged", prompt.get("skip_if_unchanged", True)))
            current_same = self._text_prompt == text_prompt

            if skip_if_unchanged and current_same:
                if session_scope:
                    return {"ok": True, "type": "text", "scope": "session", "unchanged": True}
                if self._text_frame_index == frame_index:
                    return {"ok": True, "type": "text", "frame_index": frame_index, "unchanged": True}

            self._text_prompt = text_prompt
            if session_scope:
                self._text_session_scope = True
                self._text_frame_index = 0
                self._invalidate_from(0)
                return {"ok": True, "type": "text", "scope": "session", "frame_index": 0}

            self._text_session_scope = False
            self._text_frame_index = frame_index
            self._invalidate_from(frame_index)
            return {"ok": True, "type": "text", "frame_index": frame_index, "scope": "frame"}

        if ptype in {"box", "boxes"}:
            boxes = prompt.get("boxes")
            labels = prompt.get("labels")
            if ptype == "box":
                single = prompt.get("box")
                if single is None:
                    raise ValueError("Box prompt requires key 'box'.")
                boxes = [single]
                if labels is None:
                    labels = [prompt.get("label", 1)]
            if boxes is None:
                raise ValueError("Boxes prompt requires key 'boxes'.")

            boxes_arr = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
            if boxes_arr.shape[0] == 0:
                entry = self._get_or_create_frame_entry(frame_index)
                entry.box_prompt = None
                self._invalidate_from(frame_index)
                return {
                    "ok": True,
                    "type": "boxes",
                    "frame_index": frame_index,
                    "count": 0,
                }
            if labels is not None:
                labels_arr = np.asarray(labels, dtype=np.float32).reshape(-1)
                if boxes_arr.shape[0] != labels_arr.shape[0]:
                    raise ValueError("Number of labels must match number of boxes.")
            else:
                labels_arr = np.ones((boxes_arr.shape[0],), dtype=np.float32)

            entry = self._get_or_create_frame_entry(frame_index)
            entry.box_prompt = {"boxes": boxes_arr, "labels": labels_arr}
            self._invalidate_from(frame_index)
            return {
                "ok": True,
                "type": "boxes",
                "frame_index": frame_index,
                "count": int(boxes_arr.shape[0]),
            }

        if ptype in {"point", "points", "mask", "masks"}:
            raise NotImplementedError(
                "Point/mask prompts are not supported by the active SAM3 backend adapter."
            )

        raise ValueError(f"Unsupported prompt type: {ptype}")

    def remove_object(self, obj_id: int) -> RemoveObjectResponse:
        """Suppress one object id from returned outputs."""
        oid = int(obj_id)
        self._removed_obj_ids.add(oid)
        for entry in self._frame_cache.values():
            if entry.output is not None:
                entry.output = self._filter_removed(entry.output)
        return {"ok": True, "obj_id": oid, "mode": "post_filter"}

    def get_frame(self, frame_index: int) -> FrameOutput:
        """Return segmentation for one frame index using current prompt state."""
        idx = int(frame_index)
        entry = self._get_frame_entry(idx)
        if entry is not None and entry.output is not None:
            return entry.output
        self._compute_with_native_api(idx)
        out_entry = self._get_frame_entry(idx)
        if out_entry is None or out_entry.output is None:
            raise RuntimeError(f"Frame {idx} output missing after compute.")
        return out_entry.output

    def get_frames(self, frame_indices: Iterable[int]) -> list[FrameOutput]:
        """Batch helper around :meth:`get_frame`."""
        return [self.get_frame(int(i)) for i in frame_indices]
    
    def _color_for_obj_id(self, obj_id: int) -> Tuple[int, int, int]:
        oid = int(obj_id)
        # Deterministic BGR palette for stable visualization across frames.
        return (
            int((37 * oid + 71) % 255),
            int((67 * oid + 131) % 255),
            int((97 * oid + 191) % 255),
        )
    
    def _obj_to_mask(self, obj: Dict[str, Any]) -> np.ndarray:
        h = int(obj["height"])
        w = int(obj["width"])
        flat = np.zeros((h * w,), dtype=np.uint8)
        fg = np.asarray(obj.get("foreground_indices", []), dtype=np.int64)
        if fg.size:
            flat[fg] = 1
        return flat.reshape((h, w))

    def _render_preview_frame(
        self,
        frame: np.ndarray,
        frame_out: Dict[str, Any],
        alpha: float,
        text_prompt: str | list[str] | None,
        text_scope: str,
        box_payload: BoxPromptPayload | None,
    ) -> np.ndarray:
        import cv2

        vis = np.ascontiguousarray(frame.copy())
        a = float(np.clip(alpha, 0.0, 1.0))

        for obj in frame_out.get("objects", []):
            obj_id = int(obj.get("obj_id", 0))
            score = float(obj.get("score", 0.0))
            class_id = int(obj.get("class_id", 0))
            color = np.asarray(self._color_for_obj_id(obj_id), dtype=np.float32)

            mask = self._obj_to_mask(obj) > 0
            if np.any(mask):
                src = vis[mask].astype(np.float32)
                vis[mask] = np.clip((1.0 - a) * src + a * color[None, :], 0, 255).astype(np.uint8)

                ys, xs = np.where(mask)
                x0 = int(xs.min())
                y0 = int(ys.min())
                x1 = int(xs.max() + 1)
                y1 = int(ys.max() + 1)
            else:
                x0 = y0 = x1 = y1 = 0

            cv2.rectangle(vis, (x0, y0), (x1, y1), tuple(int(v) for v in color), 2)
            label = f"id={obj_id} cls={class_id} conf={score:.2f}"
            ty = y0 - 8 if y0 > 20 else y0 + 16
            cv2.putText(vis, label, (x0, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        header = f"frame={int(frame_out.get('frame_index', -1))} objects={int(frame_out.get('num_objects', 0))}"
        cv2.putText(vis, header, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        if text_prompt is None:
            text_display = "text: (none)"
        elif isinstance(text_prompt, list):
            joined = ", ".join(str(item) for item in text_prompt)
            text_display = f"text({text_scope}): {joined}"
        else:
            text_display = f"text({text_scope}): {text_prompt}"
        if len(text_display) > 120:
            text_display = text_display[:117] + "..."
        cv2.putText(vis, text_display, (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        if box_payload is None:
            box_header = "boxes: (none)"
        else:
            box_count = int(box_payload["boxes"].shape[0])
            box_header = f"boxes: {box_count}"
            if box_count > 0:
                for i, box in enumerate(box_payload["boxes"]):
                    x0, y0, x1, y1 = [int(round(v)) for v in box]
                    cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 255, 0), 2)
                    label = "prompt box"
                    if i < len(box_payload["labels"]):
                        label = f"prompt box label={int(box_payload['labels'][i])}"
                    ty = y0 - 6 if y0 > 12 else y0 + 14
                    cv2.putText(vis, label, (x0, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)

        cv2.putText(vis, box_header, (8, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return vis

    def _compute_with_native_api(self, target_frame: int) -> None:
        target_entry = self._get_frame_entry(target_frame)
        if target_entry is None or target_entry.frame is None:
            raise RuntimeError(
                f"Frame {target_frame} is not available (not set or evicted by cache cap={_MAX_FRAME_CACHE}). "
                "Call set_frame(frame_index=...) before get_frame for this index."
            )

        start = self._max_computed_frame + 1
        if target_frame < start:
            frame_sequence = [target_frame]
        else:
            # Sparse/large ids are valid. Never iterate over full numeric ranges.
            # Only process ids that are actually present in frame cache.
            frame_sequence = sorted(
                idx for idx, entry in self._frame_cache.items() if start <= idx <= target_frame and entry.frame is not None
            )
            if not frame_sequence:
                frame_sequence = [target_frame]
            elif frame_sequence[-1] != target_frame:
                frame_sequence.append(target_frame)

        for frame_idx in frame_sequence:
            entry = self._get_frame_entry(frame_idx)
            if entry is None or entry.frame is None:
                raise RuntimeError(
                    f"Frame {frame_idx} is not available (not set or evicted by cache cap={_MAX_FRAME_CACHE}). "
                    "Call set_frame(frame_index=...) before get_frame for this index."
                )

            frame = entry.frame
            self._predictor.features = None
            has_text = self._text_prompt is not None and (self._text_session_scope or frame_idx == self._text_frame_index)
            box_payload = entry.box_prompt
            if not has_text and box_payload is None:
                formatted = {
                    "frame_index": int(frame_idx),
                    "num_objects": 0,
                    "objects": [],
                    "frame_stats": {},
                    "unconfirmed_obj_ids": [],
                }
                entry.output = self._filter_removed(formatted)
                self._frame_cache.move_to_end(frame_idx)
                self._max_computed_frame = frame_idx
                continue
            kwargs: dict[str, object] = {"stream": True, "conf": float(self._conf)}
            if has_text:
                kwargs["text"] = self._text_prompt
            if box_payload is not None:
                if int(box_payload["boxes"].shape[0]) > 0:
                    kwargs["bboxes"] = box_payload["boxes"]
                    kwargs["labels"] = box_payload["labels"]
            results = self._predictor(source=frame, **kwargs)
            first_result = _first_stream_item(results)
            if first_result is None:
                formatted = {
                    "frame_index": int(frame_idx),
                    "num_objects": 0,
                    "objects": [],
                    "frame_stats": {},
                    "unconfirmed_obj_ids": [],
                }
            else:
                formatted = self._format_result_output(frame_idx, first_result, frame.shape[:2])
            entry.output = self._filter_removed(formatted)
            self._frame_cache.move_to_end(frame_idx)
            self._max_computed_frame = frame_idx

            text_scope = "session" if self._text_session_scope else f"frame {int(self._text_frame_index)}"
            preview = self._render_preview_frame(
                frame,
                entry.output,
                alpha=0.5,
                text_prompt=self._text_prompt,
                text_scope=text_scope,
                box_payload=box_payload,
            )
            if TRex is not None:
                TRex.imshow(f"SAM3 Preview", preview)

    def _format_result_output(self, frame_index: int, result: object, frame_shape: tuple[int, int]) -> FrameOutput:
        h, w = int(frame_shape[0]), int(frame_shape[1])
        objects: list[Sam3Object] = []
        masks_data = None
        boxes_conf: npt.NDArray[np.float32] = np.zeros((0,), dtype=np.float32)
        boxes_cls: npt.NDArray[np.int32] = np.zeros((0,), dtype=np.int32)

        if getattr(result, "masks", None) is not None and getattr(result.masks, "data", None) is not None:
            masks_data = result.masks.data
        if getattr(result, "boxes", None) is not None:
            if getattr(result.boxes, "conf", None) is not None:
                boxes_conf = result.boxes.conf.detach().cpu().numpy().astype(np.float32, copy=False)
            if getattr(result.boxes, "cls", None) is not None:
                boxes_cls = result.boxes.cls.detach().cpu().numpy().astype(np.int32, copy=False)

        if masks_data is not None:
            masks_np = cast(npt.NDArray[np.uint8], masks_data.detach().to(torch.uint8).cpu().numpy())
        else:
            masks_np = np.zeros((0, h, w), dtype=np.uint8)

        num = int(masks_np.shape[0])
        for i in range(num):
            mask = masks_np[i]
            if mask.shape != (h, w):
                mask_t = torch.from_numpy(mask.astype(np.float32, copy=False))[None, None]
                mask = cast(
                    npt.NDArray[np.uint8],
                    (F.interpolate(mask_t, size=(h, w), mode="nearest")[0, 0] > 0).to(torch.uint8).cpu().numpy(),
                )
            fg = cast(npt.NDArray[np.int64], np.flatnonzero(mask.reshape(-1) > 0).astype(np.int64, copy=False))
            objects.append(
                {
                    "obj_id": i,
                    "score": float(boxes_conf[i]) if i < len(boxes_conf) else 0.0,
                    "class_id": int(boxes_cls[i]) if i < len(boxes_cls) else 0,
                    "height": h,
                    "width": w,
                    "foreground_indices": fg.tolist(),
                }
            )

        return {
            "frame_index": int(frame_index),
            "num_objects": int(len(objects)),
            "objects": objects,
            "frame_stats": {},
            "unconfirmed_obj_ids": [],
        }

    def _filter_removed(self, frame_out: FrameOutput) -> FrameOutput:
        if not self._removed_obj_ids:
            return frame_out
        kept = [obj for obj in frame_out["objects"] if int(obj["obj_id"]) not in self._removed_obj_ids]
        frame_out = cast(FrameOutput, dict(frame_out))
        frame_out["objects"] = kept
        frame_out["num_objects"] = len(kept)
        return frame_out

    def _invalidate_from(self, frame_index: int) -> None:
        for idx, entry in self._frame_cache.items():
            if idx >= frame_index:
                entry.output = None
        self._max_computed_frame = min(self._max_computed_frame, int(frame_index) - 1)
        self._recompute_max_cached_output_index()

    def evict_frame_cache(self, frame_index: int) -> None:
        """Drop one per-frame cache entry (frame + prompt + output)."""
        idx = int(frame_index)
        self._frame_cache.pop(idx, None)
        self._recompute_max_cached_output_index()


# ---------------------------------------------------------------------------
# Module-level wrappers for C++ ModuleProxy::run(...)
# ---------------------------------------------------------------------------

_SESSION: Sam3VideoSession | None = None


def _require_session() -> Sam3VideoSession:
    if _SESSION is None:
        raise RuntimeError("SAM3 session is not created. Call create_session(...) first.")
    return _SESSION


def create_session(request: Mapping[str, object]) -> CreateSessionResponse:
    """Create or replace the singleton session.

    Args:
        request: Dict with required key `weights_path` and optional keys
            `imgsz`, `conf`, `half`, `verbose`, `predictor_kwargs`.
    """
    global _SESSION
    if _SESSION is not None:
        try:
            _SESSION.shutdown()
        except Exception:
            pass

    if "weights_path" not in request:
        raise ValueError("create_session requires 'weights_path'.")

    _SESSION = Sam3VideoSession(
        weights_path=cast(str | Path, request["weights_path"]),
        imgsz=int(request.get("imgsz", 640)),
        conf=float(request.get("conf", 0.25)),
        half=bool(request.get("half", True)),
        verbose=bool(request.get("verbose", False)),
        predictor_kwargs=cast(Mapping[str, object] | None, request.get("predictor_kwargs")),
    )
    return {"ok": True, "device": _SESSION.device, "capabilities": _SESSION.capabilities()}


def capabilities() -> CapabilitiesResponse:
    """Return capabilities for the active session."""
    s = _require_session()
    return {"ok": True, "capabilities": s.capabilities()}


def set_frame(request: Mapping[str, object]) -> FrameSetResponse:
    """Ingest frame pixels previously injected via `ModuleProxy::set_variable`.

    Expected keys:
    - `frame_index` (required)

    Required module variable:
    - `sam3_frame`: `np.uint8` HxWx3 image (BGR), set by C++ before this call.
    """
    s = _require_session()
    if "frame_index" not in request:
        raise ValueError("set_frame requires 'frame_index'.")

    frame = globals().get("sam3_frame", None)
    if frame is None:
        raise RuntimeError("sam3_frame is not set. Call ModuleProxy::set_variable('sam3_frame', ...) first.")

    return s.set_frame(int(request["frame_index"]), np.asarray(frame))


def reset_session(request: Mapping[str, object] | None = None) -> GenericOkResponse:
    """Reset active session state.

    Optional keys:
    - `clear_prompts` (bool, default False)
    """
    s = _require_session()
    request = request or {}
    s.reset_session(clear_prompts=bool(request.get("clear_prompts", False)))
    return {"ok": True}


def set_conf_threshold(request: Mapping[str, object]) -> SetConfThresholdResponse:
    """Update confidence threshold for the active session.

    Expected keys:
    - `conf` (required float, clamped to [0, 1])
    """
    s = _require_session()
    if "conf" not in request:
        raise ValueError("set_conf_threshold requires 'conf'.")

    updated = s.set_conf_threshold(float(request["conf"]))
    return {"ok": True, "conf": updated}


def add_prompt(request: Mapping[str, object]) -> PromptResponse:
    """Add/update one prompt in the active session."""
    s = _require_session()
    out = s.add_prompt(request)
    out["ok"] = True
    return out


def remove_object(request: Mapping[str, object]) -> RemoveObjectResponse:
    """Suppress one object id in the active session.

    Expected keys:
    - `obj_id` (required)
    """
    s = _require_session()
    if "obj_id" not in request:
        raise ValueError("remove_object requires 'obj_id'.")
    out = s.remove_object(int(request["obj_id"]))
    out["ok"] = True
    return out


def get_frame(request: Mapping[str, object]) -> FrameOutput:
    """Return segmentation output for one frame index.

    Expected keys:
    - `frame_index` (required)
    """
    s = _require_session()
    if "frame_index" not in request:
        raise ValueError("get_frame requires 'frame_index'.")
    out = s.get_frame(int(request["frame_index"]))
    out["ok"] = True
    return out


def get_frames(request: Mapping[str, object]) -> FramesResponse:
    """Return segmentation outputs for multiple frame indices.

    Expected keys:
    - `frame_indices` (required list[int])
    """
    s = _require_session()
    if "frame_indices" not in request:
        raise ValueError("get_frames requires 'frame_indices'.")
    return {"ok": True, "frames": s.get_frames(cast(Iterable[int], request["frame_indices"]))}


def close_session() -> GenericOkResponse:
    """Clear cached frames/prompts/inference state while keeping model loaded."""
    s = _require_session()
    s.close_session()
    return {"ok": True}


def shutdown() -> GenericOkResponse:
    """Shutdown and delete singleton session."""
    global _SESSION
    if _SESSION is not None:
        _SESSION.shutdown()
        _SESSION = None
    return {"ok": True}


def predict(input: object) -> list[object]:
    """Compatibility entrypoint to reuse `TRex.YoloInput` with SAM3.

    This mirrors the YOLO module contract (`predict(input: TRex.YoloInput)`)
    so C++ can reuse the same transport object for image batches.

    Notes:
    - Requires an active session (`create_session(...)` called first).
    - Uses `orig_id()` values as frame indices.
    - Returns `List[TRex.Result]` with masks + box/conf/class placeholders.
    """
    if TRex is None:
        raise RuntimeError("TRex module is required for predict(TRex.YoloInput).")

    s = _require_session()

    # Accept both TRex.YoloInput and TRex.Sam3Input.
    base = input.base() if hasattr(input, "base") else input
    TRex.log(f"Received predict(...) call with input type {type(input).__name__} => base type {type(base).__name__}")
    images = list(base.images())
    orig_ids = list(base.orig_id())
    scales = list(base.scales())

    if not hasattr(base, "scales"):
        raise ValueError("Input missing scales(). Scales are required to map orig_id() to frame indices.")
    
    TRex.log(f"Scales: {scales}")

    if len(images) != len(orig_ids):
        raise ValueError("YoloInput.images() and YoloInput.orig_id() size mismatch.")

    typed_prompts: list[list[object]] = []
    if hasattr(input, "prompts_per_item"):
        typed_prompts = list(input.prompts_per_item())
        if typed_prompts and len(typed_prompts) != len(images):
            raise ValueError("Sam3Input.prompts_per_item must match number of images.")

    results: list[object] = []
    for item_idx, (idx, image, scale) in enumerate(zip(orig_ids, images, scales)):
        frame_index = int(idx)
        TRex.log(f"Processing frame_index={frame_index} with SAM3 predictor ({image.shape}) scale={scale}...")

        # Apply typed prompts if provided by Sam3Input.
        if typed_prompts:
            for p in typed_prompts[item_idx]:
                req: dict[str, object] = {"frame_index": int(getattr(p, "frame_index", frame_index))}
                ptype = str(getattr(p, "type")).split(".")[-1].lower()
                req["type"] = ptype

                if ptype == "text":
                    txt = getattr(p, "text", None)
                    if txt is not None:
                        req["text"] = str(txt)
                    req["text_session_scope"] = bool(getattr(p, "text_session_scope", False))
                    req["text_skip_if_unchanged"] = bool(getattr(p, "text_skip_if_unchanged", True))
                elif ptype in {"box", "boxes"}:
                    boxes = getattr(p, "boxes", [])
                    req["boxes"] = [list(b) for b in boxes]
                    labels = getattr(p, "labels", [])
                    if labels:
                        req["labels"] = list(labels)
                    if ptype == "box" and req["boxes"]:
                        req["box"] = req["boxes"][0]
                        req.pop("boxes", None)
                elif ptype in {"point", "points"}:
                    pts = getattr(p, "points", [])
                    req["points"] = [[float(v.x), float(v.y)] for v in pts]
                    pl = getattr(p, "point_labels", [])
                    if pl:
                        req["point_labels"] = list(pl)
                    oid = getattr(p, "obj_id", None)
                    if oid is not None:
                        req["obj_id"] = int(oid)
                elif ptype in {"remove_object"}:
                    oid = getattr(p, "obj_id", None)
                    if oid is not None:
                        remove_object({"obj_id": int(oid)})
                    continue

                add_prompt(req)

        frame_meta = s.set_frame(frame_index, np.asarray(image))
        TRex.log(
            f"Normalized frame_index={frame_index} -> "
            f"{frame_meta['height']}x{frame_meta['width']}x{frame_meta['channels']}"
        )
        frame_out = s.get_frame(frame_index)

        n = int(frame_out.get("num_objects", 0))
        boxes = np.zeros((n, 6), dtype=np.float32)  # [x0,y0,x1,y1,conf,clid]
        masks: list[npt.NDArray[np.uint8]] = []

        for i, obj in enumerate(frame_out.get("objects", [])):
            h = int(obj["height"])
            w = int(obj["width"])
            fg = np.asarray(obj["foreground_indices"], dtype=np.int64)
            mask = np.zeros((h * w,), dtype=np.uint8)
            if fg.size > 0:
                mask[fg] = 255
            mask = np.ascontiguousarray(mask.reshape((h, w)))

            orig = [0, 0, int(round(w * scale.x)), int(round(h * scale.y))]

            assert w == 640 and h == 640, f"Expected SAM3 output masks to be 640x640, got {w}x{h}. Check predictor output and scaling logic."

            #TRex.imshow(f"SAM3 Mask {i}", mask)

            # Resize valid mask to box's size
            import torch
            tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
            
            ssub = F.interpolate(
                tensor, 
                size=(orig[3], orig[2]),
                mode="nearest",
            ).squeeze(0).squeeze(0)

            TRex.log(f"ssub = {ssub.shape}")
            mask = ssub.cpu().numpy().astype(np.uint8)

            masks.append(mask)

            ys, xs = np.where(mask > 0)
            if xs.size > 0 and ys.size > 0:
                x0 = float(xs.min())
                y0 = float(ys.min())
                x1 = float(xs.max() + 1)
                y1 = float(ys.max() + 1)
            else:
                x0 = y0 = x1 = y1 = 0.0

            boxes[i, 0] = 0#x0 #* float(scale.x)
            boxes[i, 1] = 0#y0 #* float(scale.y)
            boxes[i, 2] = mask.shape[1] #x1 #* float(scale.x)
            boxes[i, 3] = mask.shape[0] #y1 #* float(scale.y)
            boxes[i, 4] = float(obj.get("score", 0.0))
            boxes[i, 5] = float(obj.get("class_id", 0))

            TRex.log(f"Adding box for object {i}: [{boxes[i, 0]}, {boxes[i, 1]}, {boxes[i, 2]}, {boxes[i, 3]}] conf={boxes[i, 4]:.4f} cls={boxes[i, 5]}")

        if len(masks) != n:
            raise RuntimeError("SAM3 predict produced inconsistent boxes/masks counts.")

        # Empty placeholders for keypoints/obb/points.
        keypoints = TRex.KeypointData(np.empty((0, 1, 2), dtype=np.float32))
        obb = TRex.ObbData(np.empty((0, 7), dtype=np.float32))
        points = TRex.PointData(np.empty((0, 5), dtype=np.float32))

        results.append(
            TRex.Result(  # type: ignore[attr-defined]
                frame_index,
                TRex.Boxes(boxes),  # type: ignore[attr-defined]
                masks,
                keypoints,
                obb,
                points,
            )
        )

        # `predict(...)` is the streaming C++ path; retaining all frame caches here
        # causes host-memory growth over long videos.
        s.evict_frame_cache(frame_index)

    return results
