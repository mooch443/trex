"""Production RF-DETR adapter for TRex's shared detector interface.

Everything in this module participates in checkpoint loading, inference, or
result conversion. Test orchestration and diagnostic rendering live in the
``test_trex_*`` modules and ``Application/Tests``.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import isqrt
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.ops import nms

import TRex
from TRex import DetectResolution
from TRex import ObjectDetectionFormat

import trex_utils
from trex_detection_model import BBox, DetectionModel, StrippedResults


# COCO category IDs are sparse in RF-DETR's pretrained classification head.
_COCO_CATEGORY_IDS = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18,
    19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37,
    38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54,
    55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73,
    74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89,
    90,
)

_RFDETR_MEAN = (0.485, 0.456, 0.406)
_RFDETR_STD = (0.229, 0.224, 0.225)
_MASK_CHUNK_SIZE = 32


def _bounded_fused_confidence(raw_scores: torch.Tensor) -> torch.Tensor:
    """
    Map RF-DETR's non-negative keypoint ranking score into ``[0, 1)``.

    The transform is strictly monotonic, so it preserves the ordering used by
    RF-DETR evaluation, NMS, and max-detection selection without saturating
    strong predictions into ties.
    """
    return raw_scores / (1.0 + raw_scores)


def _with_bounded_keypoint_scores(
    result: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    Retain native RF-DETR keypoint ranking scores beside TRex confidences.

    Detection and segmentation checkpoints already expose sigmoid scores in
    ``[0, 1]`` and must not be transformed a second time.
    """
    if "keypoints" not in result:
        return result

    updated = dict(result)
    updated["raw_scores"] = result["scores"]
    updated["scores"] = _bounded_fused_confidence(result["scores"])
    return updated


def _checkpoint_arg(args: object, name: str, default: Any = None) -> Any:
    if isinstance(args, Mapping):
        return args.get(name, default)
    return getattr(args, name, default)


def _load_checkpoint_mapping(model_path: str) -> Optional[Mapping[str, Any]]:
    path = Path(model_path)
    if not path.is_file():
        return None

    try:
        try:
            checkpoint = torch.load(
                str(path),
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
        except (TypeError, RuntimeError):
            checkpoint = torch.load(
                str(path),
                map_location="cpu",
                weights_only=False,
            )
    except Exception:
        return None

    return checkpoint if isinstance(checkpoint, Mapping) else None


def _tensor_ending_with(
    values: object,
    suffix: str,
) -> Optional[torch.Tensor]:
    if not isinstance(values, Mapping):
        return None
    for name, value in values.items():
        if str(name).endswith(suffix) and isinstance(value, torch.Tensor):
            return value
    return None


def _checkpoint_model_overrides(model_path: str) -> dict[str, int]:
    """
    Recover omitted KeypointPreview spatial metadata from its saved weights.

    Early Lightning ``checkpoint_best_*`` files can omit ``model_config`` and
    ``args.resolution`` even though their positional embedding was trained at a
    custom resolution. RF-DETR then falls back to the 576px class default and
    interpolates the learned positional table to that incorrect size.
    """
    checkpoint = _load_checkpoint_mapping(model_path)
    if checkpoint is None:
        return {}

    model_config = checkpoint.get("model_config")
    if _checkpoint_arg(model_config, "resolution") is not None:
        return {}

    args = checkpoint.get("args")
    model_name = str(
        checkpoint.get("model_name")
        or _checkpoint_arg(args, "model_name", "")
    )
    pretrain_weights = str(_checkpoint_arg(args, "pretrain_weights", ""))
    if "keypoint" not in f"{model_name} {pretrain_weights}".lower():
        return {}

    weights = checkpoint.get("model")
    if not isinstance(weights, Mapping):
        weights = checkpoint.get("state_dict")

    position_embeddings = _tensor_ending_with(
        weights,
        "encoder.embeddings.position_embeddings",
    )
    patch_projection = _tensor_ending_with(
        weights,
        "encoder.embeddings.patch_embeddings.projection.weight",
    )
    if (
        position_embeddings is None
        or position_embeddings.ndim != 3
        or position_embeddings.shape[0] != 1
        or position_embeddings.shape[1] <= 1
        or patch_projection is None
        or patch_projection.ndim != 4
        or patch_projection.shape[-2] != patch_projection.shape[-1]
    ):
        return {}

    patch_tokens = int(position_embeddings.shape[1]) - 1
    positional_encoding_size = isqrt(patch_tokens)
    if positional_encoding_size * positional_encoding_size != patch_tokens:
        return {}

    patch_size = int(patch_projection.shape[-1])
    inferred_resolution = positional_encoding_size * patch_size
    explicit_resolution = _checkpoint_arg(args, "resolution")
    resolution = (
        int(explicit_resolution)
        if explicit_resolution is not None
        else inferred_resolution
    )
    if resolution != inferred_resolution:
        raise ValueError(
            "RF-DETR keypoint checkpoint contains conflicting spatial metadata: "
            f"args.resolution={resolution}, but its "
            f"{positional_encoding_size}x{positional_encoding_size} positional "
            f"grid and {patch_size}px patches imply {inferred_resolution}px."
        )

    return {
        "resolution": resolution,
        "positional_encoding_size": positional_encoding_size,
        "patch_size": patch_size,
    }


def is_rfdetr_checkpoint(model_path: str) -> bool:
    """
    Return whether a local checkpoint identifies itself as RF-DETR.

    The probe uses mmap where supported so tensor storages are not eagerly read
    before RF-DETR loads the selected checkpoint for real.
    """
    path = Path(model_path)
    lowered_name = path.name.lower()
    filename_match = "rf-detr" in lowered_name or "rfdetr" in lowered_name
    extension_match = path.suffix.lower() == ".pth"
    if not path.is_file():
        return filename_match or extension_match

    checkpoint = _load_checkpoint_mapping(model_path)
    if checkpoint is None:
        return filename_match or extension_match

    model_name = checkpoint.get("model_name")
    if isinstance(model_name, str) and model_name.strip().upper().startswith("RFDETR"):
        return True

    args = checkpoint.get("args")
    pretrain_weights = str(_checkpoint_arg(args, "pretrain_weights", "")).lower()
    return (
        filename_match
        or extension_match
        or "rf-detr" in pretrain_weights
        or "rfdetr" in pretrain_weights
    )


def _filter_result(
    result: dict[str, torch.Tensor],
    threshold: float,
    max_det: int,
    classes: Optional[List[int]] = None,
    valid_class_ids: Optional[List[int]] = None,
    iou_threshold: Optional[float] = None,
) -> dict[str, torch.Tensor]:
    scores = result["scores"]
    ranking_scores = result.get("raw_scores", scores)
    labels = result["labels"]
    # MPS currently aborts inside small bool-vector kernels used by isin/nonzero
    # (rather than raising a catchable Python exception). Candidate results are
    # at most num_select rows, so select indices on CPU while leaving network
    # inference and large mask tensors on Metal.
    selection_device = scores.device
    select_on_cpu = selection_device.type == "mps"
    selection_scores = scores.detach().cpu() if select_on_cpu else scores
    selection_ranking_scores = (
        ranking_scores.detach().cpu()
        if select_on_cpu
        else ranking_scores
    )
    selection_labels = labels.detach().cpu() if select_on_cpu else labels
    keep = selection_scores > threshold

    if valid_class_ids is not None:
        valid = torch.as_tensor(
            valid_class_ids,
            device=selection_labels.device,
            dtype=selection_labels.dtype,
        )
        keep &= torch.isin(selection_labels, valid)

    if classes is not None:
        allowed = torch.as_tensor(
            classes,
            device=selection_labels.device,
            dtype=selection_labels.dtype,
        )
        keep &= torch.isin(selection_labels, allowed)

    indices = keep.nonzero(as_tuple=True)[0]
    if iou_threshold is not None and indices.numel() > 0:
        selection_boxes = (
            result["boxes"].detach().cpu()
            if select_on_cpu
            else result["boxes"]
        )
        boxes = selection_boxes[indices].to(dtype=torch.float32)
        retained_scores = selection_ranking_scores[indices].to(
            dtype=torch.float32,
        )
        try:
            retained = nms(boxes, retained_scores, float(iou_threshold))
        except (NotImplementedError, RuntimeError):
            retained = nms(
                boxes.detach().cpu(),
                retained_scores.detach().cpu(),
                float(iou_threshold),
            ).to(device=indices.device)
        indices = indices[retained]

    if max_det > 0:
        if iou_threshold is None and indices.numel() > max_det:
            ranking_order = torch.argsort(
                selection_ranking_scores[indices],
                descending=True,
            )
            indices = indices[ranking_order[:max_det]]
        else:
            indices = indices[:max_det]
    gather_indices = (
        indices.to(device=selection_device)
        if select_on_cpu
        else indices
    )

    filtered: dict[str, torch.Tensor] = {}
    for key, value in result.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == scores.shape[0]:
            filtered[key] = value[gather_indices]
        else:
            filtered[key] = value
    return filtered


class StrippedRFDETRResults(StrippedResults):
    """Convert one filtered RF-DETR tile result to TRex's existing arrays."""

    def __init__(
        self,
        result: dict[str, torch.Tensor],
        scale: np.ndarray,
        offset: np.ndarray,
        image_shape: tuple[int, int],
    ) -> None:
        super().__init__(
            np.asarray(scale, dtype=np.float32),
            np.asarray(offset, dtype=np.float32),
        )
        self.orig_shape = image_shape

        tile_boxes = result["boxes"].to(dtype=torch.float32)
        scores = result["scores"].to(dtype=torch.float32)
        labels = result["labels"].to(dtype=torch.float32)

        source_boxes = tile_boxes.clone()
        source_boxes[:, [0, 2]] = (
            source_boxes[:, [0, 2]] + float(self.offset[0])
        ) * float(self.scale[0])
        source_boxes[:, [1, 3]] = (
            source_boxes[:, [1, 3]] + float(self.offset[1])
        ) * float(self.scale[1])

        masks = result.get("masks")
        if masks is not None and tile_boxes.shape[0] > 0:
            tile_height, tile_width = image_shape
            tile_bounds = torch.cat(
                [torch.floor(tile_boxes[:, :2]), torch.ceil(tile_boxes[:, 2:4])],
                dim=1,
            ).to(dtype=torch.int64).cpu()
            tile_bounds[:, [0, 2]].clamp_(0, tile_width)
            tile_bounds[:, [1, 3]].clamp_(0, tile_height)
            source_bounds = torch.cat(
                [torch.floor(source_boxes[:, :2]), torch.ceil(source_boxes[:, 2:4])],
                dim=1,
            ).to(dtype=torch.int64).cpu()
            valid = (
                (tile_bounds[:, 2] > tile_bounds[:, 0])
                & (tile_bounds[:, 3] > tile_bounds[:, 1])
                & (source_bounds[:, 2] > source_bounds[:, 0])
                & (source_bounds[:, 3] > source_bounds[:, 1])
            )
            if not bool(valid.all()):
                valid_device = valid.to(device=tile_boxes.device)
                tile_boxes = tile_boxes[valid_device]
                source_boxes = source_boxes[valid_device]
                scores = scores[valid_device]
                labels = labels[valid_device]
                masks = masks[valid_device]
                tile_bounds = tile_bounds[valid]
                source_bounds = source_bounds[valid]
                if "keypoints" in result:
                    result = dict(result)
                    result["keypoints"] = result["keypoints"][valid_device]

            source_boxes = source_bounds.to(
                device=source_boxes.device,
                dtype=torch.float32,
            )

        if source_boxes.shape[0] == 0:
            self.boxes = np.empty((0, 6), dtype=np.float32)
        else:
            packed = torch.cat(
                [
                    source_boxes,
                    scores.unsqueeze(1),
                    labels.unsqueeze(1),
                ],
                dim=1,
            )
            self.boxes = np.ascontiguousarray(
                packed.detach().cpu().numpy(),
                dtype=np.float32,
            )

        if masks is not None:
            if masks.ndim == 4:
                if masks.shape[1] != 1:
                    raise ValueError(
                        f"Expected RF-DETR masks as [N,1,H,W], got {tuple(masks.shape)}."
                    )
                masks = masks[:, 0]
            if masks.ndim != 3:
                raise ValueError(
                    f"Expected RF-DETR masks as [N,H,W], got {tuple(masks.shape)}."
                )

            self.masks = []
            for index in range(masks.shape[0]):
                x0, y0, x1, y1 = (int(v) for v in tile_bounds[index].tolist())
                sx0, sy0, sx1, sy1 = (int(v) for v in source_bounds[index].tolist())
                cropped = masks[index, y0:y1, x0:x1]
                destination_height = sy1 - sy0
                destination_width = sx1 - sx0
                resized = F.interpolate(
                    cropped[None, None].to(dtype=torch.float32),
                    size=(destination_height, destination_width),
                    mode="nearest",
                )[0, 0]
                mask = (resized > 0.5).to(dtype=torch.uint8).mul_(255)
                self.masks.append(
                    np.ascontiguousarray(mask.cpu().numpy(), dtype=np.uint8)
                )

        keypoints = result.get("keypoints")
        if keypoints is not None:
            keypoints = keypoints.to(dtype=torch.float32)
            if keypoints.ndim != 3 or keypoints.shape[-1] < 2:
                raise ValueError(
                    "Expected RF-DETR keypoints as [N,K,2+] but got "
                    f"{tuple(keypoints.shape)}."
                )

            xy = keypoints[..., :2].clone()
            if keypoints.shape[-1] >= 3:
                valid_points = keypoints[..., 2] > 0
            else:
                valid_points = torch.any(xy != 0, dim=-1)
            valid_points &= torch.any(xy != 0, dim=-1)

            xy[..., 0] = torch.where(
                valid_points,
                (xy[..., 0] + float(self.offset[0])) * float(self.scale[0]),
                torch.zeros_like(xy[..., 0]),
            )
            xy[..., 1] = torch.where(
                valid_points,
                (xy[..., 1] + float(self.offset[1])) * float(self.scale[1]),
                torch.zeros_like(xy[..., 1]),
            )
            keypoint_array = np.ascontiguousarray(
                xy.detach().cpu().numpy(),
                dtype=np.float32,
            )
            self.keypoints = [keypoint_array] if keypoint_array.shape[0] > 0 else []


class RFDETRModel(DetectionModel):
    """RF-DETR implementation of TRex's existing Python detection-model API."""

    def __init__(self, config: TRex.ModelConfig):
        super().__init__(config)
        self.context: Any = None
        self.network: Any = None
        self.postprocess: Any = None
        self.model_config: Any = None
        self.input_height = 0
        self.input_width = 0
        self.input_channels = 0
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None
        self.optimization_mode: Optional[str] = None
        self._torchscript_batch_size: Optional[int] = None

    def _valid_class_ids(self) -> list[int]:
        return [
            int(class_id)
            for class_id, name in self.config.classes.items()
            if str(name) != "__background__"
        ]

    @staticmethod
    def _zero_low_confidence_keypoints(
        result: dict[str, torch.Tensor],
        threshold: float,
    ) -> dict[str, torch.Tensor]:
        keypoints = result.get("keypoints")
        if keypoints is None or keypoints.shape[-1] < 3:
            return result

        if keypoints.device.type == "mps":
            keypoints = keypoints.detach().cpu()
        visible = keypoints[..., 2] >= threshold
        filtered = keypoints.clone()
        filtered[..., :2] = torch.where(
            visible.unsqueeze(-1),
            filtered[..., :2],
            torch.zeros_like(filtered[..., :2]),
        )
        filtered[..., 2] = torch.where(
            visible,
            filtered[..., 2],
            torch.zeros_like(filtered[..., 2]),
        )
        updated = dict(result)
        updated["keypoints"] = filtered
        return updated

    def __str__(self) -> str:
        return f"RFDETRModel<{str(self.config)}>"

    def preprocess_image(self, image: Any) -> np.ndarray:
        """
        Preserve the RGB channel order produced by TRex's video pipeline.

        Unlike Ultralytics, RF-DETR normalizes the supplied numpy array
        directly and does not perform a later BGR-to-RGB swap. FFmpeg-backed
        detector frames arrive as RGB or RGBA, so only the unused alpha channel
        must be removed here.
        """
        array = trex_utils.asarray(image, copy=False)
        if array.ndim != 3 or array.shape[2] not in (3, 4):
            raise ValueError(
                "RF-DETR expects an RGB or RGBA C++ detector image, "
                f"got shape {array.shape}."
            )
        if array.shape[2] == 4:
            array = array[..., :3]
        return np.ascontiguousarray(array)

    def _class_map(self) -> dict[int, str]:
        public_names = self.ptr.class_names
        if isinstance(public_names, Mapping):
            return {
                int(class_id): str(name)
                for class_id, name in public_names.items()
            }

        names = [str(name) for name in public_names]
        args = getattr(self.context, "args", None)
        num_logit_slots = int(_checkpoint_arg(args, "num_classes", len(names)))
        embedded_names = getattr(self.context, "class_names", None)
        keypoint_counts = list(
            _checkpoint_arg(args, "num_keypoints_per_class", []) or []
        )

        if embedded_names is None and num_logit_slots > len(names):
            return {
                category_id: names[index]
                for index, category_id in enumerate(_COCO_CATEGORY_IDS)
                if index < len(names)
            }

        is_background_first = (
            bool(getattr(self.model_config, "use_grouppose_keypoints", False))
            and len(keypoint_counts) > 1
            and keypoint_counts[0] == 0
            and any(count > 0 for count in keypoint_counts[1:])
        )
        if is_background_first:
            foreground_slots = [
                index for index, count in enumerate(keypoint_counts)
                if count > 0
            ]
            result = {0: "__background__"}
            result.update(
                {
                    slot: names[index]
                    for index, slot in enumerate(foreground_slots)
                    if index < len(names)
                }
            )
            return result

        return dict(enumerate(names))

    def _finish_load(self, wrapper: Any) -> None:
        self.ptr = wrapper
        self.context = wrapper.model
        self.network = self.context.model
        self.postprocess = self.context.postprocess
        self.model_config = wrapper.model_config

        resolution = int(self.context.resolution)
        configured_resolution = int(self.model_config.resolution)
        if configured_resolution != resolution:
            raise ValueError(
                "RF-DETR checkpoint resolution mismatch: "
                f"context={resolution}, model_config={configured_resolution}."
            )

        block_size = (
            int(self.model_config.patch_size)
            * int(self.model_config.num_windows)
        )
        if resolution % block_size != 0:
            raise ValueError(
                f"RF-DETR resolution {resolution} must be divisible by "
                f"patch_size * num_windows ({block_size})."
            )

        self.input_height = resolution
        self.input_width = resolution
        self.input_channels = int(self.model_config.num_channels)
        if self.input_channels != 3:
            raise ValueError(
                "TRex currently provides three-channel RGB detector images, "
                f"but this RF-DETR checkpoint expects {self.input_channels} channels."
            )

        self.reinit_device()
        self.network = self.network.to(self.device)
        self.network.eval()
        self.context.model = self.network
        self.context.device = self.device
        self._apply_inference_optimization(wrapper)

        self.mean = torch.tensor(
            _RFDETR_MEAN,
            device=self.device,
            dtype=torch.float32,
        ).view(1, 3, 1, 1)
        self.std = torch.tensor(
            _RFDETR_STD,
            device=self.device,
            dtype=torch.float32,
        ).view(1, 3, 1, 1)

        self.config.trained_resolution = DetectResolution(
            self.input_width,
            self.input_height,
        )
        self.config.requires_exact_input_size = True
        self.config.classes = self._class_map()

        if bool(self.model_config.use_grouppose_keypoints):
            self.config.output_format = ObjectDetectionFormat.poses
            keypoint_counts = list(
                self.model_config.num_keypoints_per_class or []
            )
            maximum_keypoints = max(keypoint_counts, default=0)
            if maximum_keypoints <= 0:
                raise ValueError(
                    "RF-DETR pose checkpoint does not contain an active keypoint schema."
                )
            self.config.keypoint_format = TRex.KeypointFormat(
                maximum_keypoints,
                2,
            )
        elif bool(self.model_config.segmentation_head):
            self.config.output_format = ObjectDetectionFormat.masks
            self.config.keypoint_format = TRex.KeypointFormat(0, 0)
        else:
            self.config.output_format = ObjectDetectionFormat.boxes
            self.config.keypoint_format = TRex.KeypointFormat(0, 0)

    def _apply_inference_optimization(self, wrapper: Any) -> None:
        selected = "torchscript" if self.config.try_optimize else "memory"

        def optimize(mode: str) -> None:
            wrapper.optimize_for_inference(
                compile=mode == "torchscript",
                batch_size=1,
                dtype=torch.float32,
                inplace=mode == "memory",
            )

        try:
            optimize(selected)
        except Exception as error:
            if selected != "torchscript":
                raise
            TRex.warn(
                "RF-DETR TorchScript optimization failed; retrying with the "
                f"in-place memory path: {error}"
            )
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            elif self.device.type == "mps":
                torch.mps.empty_cache()
            optimize("memory")
            selected = "memory"

        optimized = getattr(self.context, "inference_model", None)
        if optimized is None:
            raise RuntimeError(
                "RF-DETR optimization completed without an inference model."
            )
        self.network = optimized.to(self.device)
        self.network.eval()
        self.context.device = self.device
        self.optimization_mode = selected
        self._torchscript_batch_size = 1 if selected == "torchscript" else None
        TRex.log(f"RF-DETR inference optimization selected {selected}.")

    def _normalize_network_output(self, output: Any) -> dict[str, torch.Tensor]:
        """
        Restore RF-DETR's exported tuple to its ordinary inference dictionary.

        RF-DETR export returns boxes before logits and may append either
        keypoints or masks. Its public prediction API performs the same mapping
        before invoking the task postprocessor.
        """
        if isinstance(output, dict):
            return output
        if not isinstance(output, tuple):
            raise TypeError(
                "Expected RF-DETR model output as a dict or exported tuple, "
                f"got {type(output).__name__}."
            )
        if len(output) not in (2, 3):
            raise ValueError(
                "Expected RF-DETR exported output as "
                "(boxes, logits[, keypoints_or_masks]), "
                f"got {len(output)} values."
            )

        result = {
            "pred_boxes": output[0],
            "pred_logits": output[1],
        }
        if len(output) == 3:
            optional_name = (
                "pred_keypoints"
                if bool(self.model_config.use_grouppose_keypoints)
                else "pred_masks"
            )
            result[optional_name] = output[2]
        return result

    def load(self):
        import inspect

        try:
            from rfdetr import RFDETR
        except ImportError as error:
            raise ImportError(
                "RF-DETR checkpoint selected, but RF-DETR could not be imported. "
                'Install or repair the pinned dependency with `pip install "rfdetr==1.8.3"`.'
            ) from error

        loader_parameters = inspect.signature(RFDETR.from_checkpoint).parameters
        loader_kwargs = (
            {"trust_checkpoint": True}
            if "trust_checkpoint" in loader_parameters
            else {}
        )
        accepts_model_overrides = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in loader_parameters.values()
        )
        model_overrides = _checkpoint_model_overrides(self.config.model_path)
        unsupported_overrides = [
            name
            for name in model_overrides
            if name not in loader_parameters and not accepts_model_overrides
        ]
        if unsupported_overrides:
            raise RuntimeError(
                "The installed RF-DETR loader cannot accept checkpoint-derived "
                f"model parameters {unsupported_overrides}. "
                'Install the pinned dependency with `pip install "rfdetr==1.8.3"`.'
            )
        loader_kwargs.update(model_overrides)
        wrapper = RFDETR.from_checkpoint(
            self.config.model_path,
            **loader_kwargs,
        )
        self._finish_load(wrapper)
        super().load()

    def _prepare_batch(
        self,
        images: List[np.ndarray],
    ) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        if self.mean is None or self.std is None:
            raise RuntimeError("RF-DETR model has not been loaded.")

        tensors: list[torch.Tensor] = []
        sizes: list[tuple[int, int]] = []
        for index, image_value in enumerate(images):
            image = np.asarray(image_value)
            if image.ndim != 3:
                raise ValueError(
                    f"RF-DETR input {index} must be RGB HWC, got shape {image.shape}."
                )
            height, width, channels = image.shape
            if (
                height != self.input_height
                or width != self.input_width
            ):
                raise ValueError(
                    "RF-DETR expects the C++ detector image size to match the loaded "
                    f"checkpoint ({self.input_width}x{self.input_height}), but input "
                    f"{index} is {width}x{height}. Check detect_resolution preparation."
                )
            if channels != self.input_channels:
                raise ValueError(
                    f"RF-DETR input {index} has {channels} channels; "
                    f"the checkpoint expects {self.input_channels}."
                )

            tensor = torch.from_numpy(np.ascontiguousarray(image))
            tensor = tensor.permute(2, 0, 1).contiguous()
            if tensor.dtype == torch.uint8:
                tensor = tensor.to(dtype=torch.float32).div_(255.0)
            elif tensor.is_floating_point():
                tensor = tensor.to(dtype=torch.float32)
                if bool(torch.any(tensor < 0)) or bool(torch.any(tensor > 1)):
                    raise ValueError(
                        "Floating RF-DETR input values must be in [0,1]."
                    )
            else:
                raise ValueError(
                    f"Unsupported RF-DETR input dtype {tensor.dtype}."
                )

            tensors.append(tensor)
            sizes.append((height, width))

        batch = torch.stack(tensors).to(
            device=self.device,
            non_blocking=True,
        )
        batch = (batch - self.mean) / self.std
        return batch, sizes

    def _postprocess_segmentation(
        self,
        raw: dict[str, torch.Tensor],
        target_sizes: torch.Tensor,
        threshold: float,
        max_det: int,
        classes: Optional[List[int]],
        valid_class_ids: list[int],
        iou_threshold: Optional[float],
    ) -> list[dict[str, torch.Tensor]]:
        logits = raw["pred_logits"]
        raw_masks = raw["pred_masks"]

        probabilities = logits.sigmoid().flatten(1)
        num_select = int(getattr(self.postprocess, "num_select", 300))
        selected_count = min(num_select, probabilities.shape[1])
        _, flat_indices = torch.topk(
            probabilities,
            selected_count,
            dim=1,
        )
        query_indices = flat_indices // logits.shape[2]
        box_outputs = {
            key: value
            for key, value in raw.items()
            if key != "pred_masks"
        }
        box_results = self.postprocess(
            box_outputs,
            target_sizes=target_sizes,
        )

        results: list[dict[str, torch.Tensor]] = []
        for index, box_result in enumerate(box_results):
            height, width = target_sizes[index]
            result_with_queries = dict(box_result)
            result_with_queries["queries"] = query_indices[
                index,
                :box_result["scores"].shape[0],
            ]
            result = _filter_result(
                result_with_queries,
                threshold,
                max_det,
                classes,
                valid_class_ids,
                iou_threshold,
            )
            retained_queries = result.pop("queries")
            selected_masks = raw_masks[index, retained_queries]
            chunks = [
                F.interpolate(
                    selected_masks[start:start + _MASK_CHUNK_SIZE, None],
                    size=(int(height), int(width)),
                    mode="bilinear",
                    align_corners=False,
                ) > 0.0
                for start in range(0, selected_masks.shape[0], _MASK_CHUNK_SIZE)
            ]
            result["masks"] = (
                torch.cat(chunks, dim=0)
                if chunks
                else raw_masks.new_zeros(
                    (0, 1, int(height), int(width)),
                    dtype=torch.bool,
                )
            )
            results.append(result)
        return results

    def _predict_tile_results(
        self,
        images: List[np.ndarray],
        **kwargs: Any,
    ) -> list[dict[str, torch.Tensor]]:
        if len(images) == 0:
            return []
        if (
            self._torchscript_batch_size is not None
            and len(images) != self._torchscript_batch_size
        ):
            results: list[dict[str, torch.Tensor]] = []
            for image in images:
                results.extend(self._predict_tile_results([image], **kwargs))
            return results

        threshold = float(kwargs.get("conf", 0.1))
        max_det = int(kwargs.get("max_det", 1000))
        classes = kwargs.get("classes")
        iou_threshold = kwargs.get("iou")
        if iou_threshold is not None:
            iou_threshold = float(iou_threshold)
        keypoint_threshold = float(TRex.setting("detect_keypoint_threshold"))
        valid_class_ids = self._valid_class_ids()

        batch, sizes = self._prepare_batch(images)
        with torch.inference_mode():
            raw = self._normalize_network_output(self.network(batch))

        target_sizes = torch.tensor(
            sizes,
            device=self.device,
            dtype=torch.int64,
        )
        if "pred_masks" in raw:
            return self._postprocess_segmentation(
                raw,
                target_sizes,
                threshold,
                max_det,
                classes,
                valid_class_ids,
                iou_threshold,
            )

        results = self.postprocess(
            raw,
            target_sizes=target_sizes,
        )
        return [
            self._zero_low_confidence_keypoints(
                _filter_result(
                    _with_bounded_keypoint_scores(result),
                    threshold,
                    max_det,
                    classes,
                    valid_class_ids,
                    iou_threshold,
                ),
                keypoint_threshold,
            )
            for result in results
        ]

    def predict_boxes(
        self,
        images: List[np.ndarray],
        **kwargs: Any,
    ) -> List[BBox]:
        results = self._predict_tile_results(images, **kwargs)
        return [
            np.ascontiguousarray(
                result["boxes"].detach().cpu().numpy(),
                dtype=np.float32,
            )
            for result in results
        ]

    def predict(
        self,
        images: List[np.ndarray],
        scales: List[Any],
        offsets: List[Any],
        **kwargs: Any,
    ) -> List[StrippedResults]:
        if len(images) != len(scales) or len(images) != len(offsets):
            raise ValueError(
                "RF-DETR predict expects matching images, scales, and offsets, "
                f"got {len(images)}, {len(scales)}, and {len(offsets)}."
            )

        results = self._predict_tile_results(images, **kwargs)
        return [
            StrippedRFDETRResults(
                result,
                scale=np.asarray(scale, dtype=np.float32),
                offset=np.asarray(offset, dtype=np.float32),
                image_shape=image.shape[:2],
            )
            for result, image, scale, offset in zip(
                results,
                images,
                scales,
                offsets,
            )
        ]
