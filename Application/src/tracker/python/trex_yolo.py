from ultralytics import YOLO
import ultralytics.utils

from torch.nn import functional as F
import numpy as np
import torch

import TRex
from TRex import ModelTaskType
from TRex import DetectResolution
from TRex import ObjectDetectionFormat

from trex_detection_model import DetectionModel, StrippedResults, TRexDetection
from trex_detection_model import BBox
from typing import List, Optional, Any

##############
##### since this isnt patched in the ultralytics package yet,
##### we need to patch the following functions in the ultralytics.utils.ops module
##### see https://github.com/ultralytics/ultralytics/issues/8555
##############

import ultralytics
from ultralytics.utils.ops import clip_boxes, crop_mask
from ultralytics.utils import ops
import torch.nn.functional as F

def dilate(mask, k=5):
    # mask: [N, H, W] bool/uint8
    x = mask.float().unsqueeze(1)  # [N,1,H,W]
    y = F.max_pool2d(x, kernel_size=k, stride=1, padding=k // 2)
    return y[:, 0] > 0

def erode(mask, k=5):
    x = mask.float().unsqueeze(1)
    y = -F.max_pool2d(
        -x,
        kernel_size=k,
        stride=1,
        padding=k // 2,
    )
    return y[:, 0] > 0

def close_mask(mask, k=9):
    """Morphologically close a binary mask stack with an odd square kernel."""
    return erode(dilate(mask, k), k)

def expand_boxes_full_mask(masks, bboxes):
    """Expand each XYXY box to enclose its mask's positive pixels.

    ``masks`` has shape ``[N, H, W]`` and may reside on CPU, CUDA, or MPS.
    This function changes boxes in place and never changes mask pixels.
    """

    n, h, w = masks.shape

    for i in range(n):
        ys, xs = torch.nonzero(masks[i], as_tuple=True)
        if xs.numel() == 0:
            continue

        # Convert mask coordinates to same dtype/device as bboxes.
        cx1 = (xs.min() - 1).to(dtype=bboxes.dtype, device=bboxes.device)
        cy1 = (ys.min() - 1).to(dtype=bboxes.dtype, device=bboxes.device)
        cx2 = (xs.max() + 2).to(dtype=bboxes.dtype, device=bboxes.device)
        cy2 = (ys.max() + 2).to(dtype=bboxes.dtype, device=bboxes.device)

        # Clone so subsequent in-place bbox updates don't alter these.
        x1, y1, x2, y2 = bboxes[i].clone()

        nx1 = torch.minimum(x1, cx1).clamp(0, w)
        ny1 = torch.minimum(y1, cy1).clamp(0, h)
        nx2 = torch.maximum(x2, cx2).clamp(0, w)
        ny2 = torch.maximum(y2, cy2).clamp(0, h)

        bboxes[i] = torch.stack((nx1, ny1, nx2, ny2))
        
    # Only four floats per detection need to go back to the
    # inference device.
    return bboxes

def postprocess_masks(masks, bboxes):
    """Apply configured YOLO mask closing and bounding-box expansion."""
    closing_radius = int(TRex.setting("yolo_instance_mask_closing"))
    if closing_radius > 0:
        masks = close_mask(masks, k=closing_radius * 2 + 1).to(masks.dtype)

    if TRex.setting("yolo_instance_mask_expand"):
        bboxes = expand_boxes_full_mask(masks, bboxes)

    return masks, bboxes

def process_mask_native(protos, masks_in, bboxes, shape):
    """Apply masks to bounding boxes using mask head output with native upsampling.

    Args:
        protos (torch.Tensor): Mask prototypes with shape (mask_dim, mask_h, mask_w).
        masks_in (torch.Tensor): Mask coefficients with shape (N, mask_dim) where N is number of masks after NMS.
        bboxes (torch.Tensor): Bounding boxes with shape (N, 4) where N is number of masks after NMS.
        shape (tuple): Input image size as (height, width).

    Returns:
        (torch.Tensor): Binary mask tensor with shape (N, H, W).
    """
    c, mh, mw = protos.shape  # CHW
    h, w = shape
    if masks_in.shape[0] == 0:  # no detections: return a well-formed empty mask stack
        return torch.zeros((0, h, w), dtype=torch.uint8, device=masks_in.device)
    coeffs = masks_in @ protos.float().view(c, -1)  # (N, mh*mw) prototype-resolution mask logits
    # Upsampling all N masks at once allocates an N*H*W float intermediate (~9 GB on a large image with many
    # detections), which OOMs the worker. Upsample in chunks bounded by a pixel budget, thresholding each chunk to
    # uint8 immediately so the float intermediate stays small, then crop the assembled uint8 stack.
    step = max(1, 32_000_000 // (h * w))
    masks = [
        ops.scale_masks(coeffs[i : i + step].view(-1, mh, mw)[None], shape)[0].gt_(0.0).byte()
        for i in range(0, coeffs.shape[0], step)
    ]

    masks = torch.cat(masks, dim=0)

    masks, bboxes = postprocess_masks(masks, bboxes)

    return ops.crop_mask(masks, bboxes)

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None, padding=True, xywh=False):
    """
    Rescales bounding boxes (in the format of xyxy by default) from the shape of the image they were predicted
    specified in (img1_shape) to the shape of a different image (img0_shape).

    Args:
        img1_shape (tuple): The shape of the image that the bounding boxes are for, in the format of (height, width).
        boxes (torch.Tensor): the bounding boxes of the objects in the image, in the format of (x1, y1, x2, y2)
        img0_shape (tuple): the shape of the target image, in the format of (height, width).
        ratio_pad (tuple): a tuple of (ratio, pad) for scaling the boxes. If not provided, the ratio and pad will be
            calculated based on the size difference between the two images.
        padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
            rescaling.
        xywh (bool): The box format is xywh or not, default=False.

    Returns:
        boxes (torch.Tensor): The scaled bounding boxes, in the format of (x1, y1, x2, y2)
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (
            round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1),
            round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1),
        )  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    # Apply padding (if padding is needed) and adjust scaling uniformly
    if padding:
        if not xywh:
            # Adjust all x, y, x2, and y2 by padding, then scale
            pads = torch.tensor([pad[0], pad[1], pad[0], pad[1]], device=boxes.device)
            boxes -= pads
        else:
            # Adjust only x and y for bounding boxes in xywh format
            pads = torch.tensor([pad[0], pad[1], 0, 0], device=boxes.device)
            boxes[:, :2] -= pads[:2]

    # Scale the boxes down
    boxes /= gain

    return clip_boxes(boxes, img0_shape)

def process_mask(protos, masks_in, bboxes, shape, upsample: bool = False):
    """Apply masks to bounding boxes using mask head output.

    Args:
        protos (torch.Tensor): Mask prototypes with shape (mask_dim, mask_h, mask_w).
        masks_in (torch.Tensor): Mask coefficients with shape (N, mask_dim) where N is number of masks after NMS.
        bboxes (torch.Tensor): Bounding boxes with shape (N, 4) where N is number of masks after NMS.
        shape (tuple): Input image size as (height, width).
        upsample (bool): Whether to upsample masks to original image size.

    Returns:
        (torch.Tensor): A binary mask tensor of shape [n, h, w], where n is the number of masks after NMS. When
            upsample=True h and w match the input image size; otherwise they are the prototype mask resolution.
    """
    c, mh, mw = protos.shape  # CHW
    if masks_in.shape[0] == 0:  # no detections: F.interpolate below rejects an empty (N=0) batch
        return torch.zeros((0, *(shape if upsample else (mh, mw))), dtype=torch.uint8, device=masks_in.device)
    masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)  # NHW

    if upsample:
        # Upsample then crop at image resolution; cropping first smears the bilinear edge outside the bbox (#24272)
        masks = F.interpolate(
            masks[None],
            shape,
            mode="bilinear",
        )[0]

        masks = masks.gt_(0.0).byte()

        # masks and bboxes are both input-image coordinates
        masks, bboxes = postprocess_masks(masks, bboxes)

        return ops.crop_mask(masks, bboxes)
    else:
        width_ratio = mw / shape[1]
        height_ratio = mh / shape[0]

        ratios = bboxes.new_tensor([
            width_ratio,
            height_ratio,
            width_ratio,
            height_ratio,
        ])

        bboxes = bboxes * ratios

        masks = masks.gt_(0.0).byte()

        # now both are prototype-resolution coordinates
        masks, bboxes = postprocess_masks(masks, bboxes)

        return ops.crop_mask(masks, bboxes)
    # Binarize before cropping so crop_mask runs on uint8 instead of float32, as in process_mask_native
    #return ops.crop_mask(masks.gt_(0.0).byte(), bboxes)

ultralytics.utils.ops.process_mask = process_mask
ultralytics.utils.ops.scale_boxes = scale_boxes
ultralytics.utils.ops.process_mask_native = process_mask_native

TRex.log("*** patched functions ***")
##############


RUNTIME_PROFILE_LEGACY = "legacy_head"
RUNTIME_PROFILE_MODERN_END2END = "modern_end2end"
RUNTIME_PROFILE_MODERN_END2END_FORCED_NMS = "modern_end2end_forced_nms"

def unscale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """
    Rescale segment coordinates (xyxy) from normalized to source image scale

    Args:
      img1_shape (tuple): The shape of the image that the coords are from.
      coords (torch.Tensor): the coords to be scaled
      img0_shape (tuple): the shape of the image that the segmentation is being applied to
      ratio_pad (tuple): the ratio of the image size to the padded image size.

    Returns:
      coords (torch.Tensor): the segmented image.
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    # Scale up the coordinates from normalized to image scale
    #coords[..., 0] *= img0_shape[1]  # width
    #coords[..., 1] *= img0_shape[0]  # height

    # Remove the effect of padding and scaling
    coords[..., 0] *= gain
    coords[..., 1] *= gain
    coords[..., 0] += pad[0]  # x padding
    coords[..., 1] += pad[1]  # y padding

    # No need to clip coordinates, as we're scaling up rather than down

    return coords

class StrippedYoloResults(StrippedResults):
    def __init__(
        self,
        results: Any,
        scale: np.ndarray,
        offset: np.ndarray,
        box: List[int] = [0, 0, 0, 0]
    ) -> None:
        """
        Initialize StrippedYoloResults from YOLO model output.

        Args:
            results (Any): Raw output from a YOLO model inference. Expected to contain:
                - boxes: tensor of shape [n, 6] with values [clid, conf, x, y, w, h].
                - keypoints (optional): tensor of shape [n, num_keypoints, 2], with (x, y) coordinates.
                - obb (optional): tensor of shape [n, 5] as [x_center, y_center, width, height, angle] before class/conf insertion.
                - masks (optional): tensor of shape [n, mask_h, mask_w] for segmentation masks.
            scale (np.ndarray): A 2-element array [scale_x, scale_y] to map tile coordinates back to source image coordinates.
            offset (np.ndarray): A 2-element array [offset_x, offset_y] to apply in tile coordinates before scaling.
            box (List[int]): A 4-element list [x_offset, y_offset, x_offset, y_offset] for additional cropping offset.
        """
        super().__init__(scale, offset)

        # Extract raw bounding boxes from the model output
        boxes_attr = getattr(results, 'boxes', None)
        if boxes_attr is not None:
            self.boxes = boxes_attr.data.cpu().numpy()
            # Store source-space boxes array for later scaling
        self.orig_shape = getattr(results, 'orig_shape', None)

        semantic_attr = getattr(results, 'semantic_mask', None)
        if semantic_attr is not None:
            semantic = np.asarray(semantic_attr.data.detach().cpu().numpy())
            if semantic.ndim != 2:
                raise ValueError(
                    f"YOLO semantic output must have shape (H, W), got {semantic.shape}."
                )
            if semantic.size > 0:
                if not np.isfinite(semantic).all() or np.any(semantic != np.floor(semantic)):
                    raise ValueError("YOLO semantic output contains non-integral class IDs.")
                minimum = int(semantic.min())
                maximum = int(semantic.max())
                if minimum < 0 or maximum > np.iinfo(np.uint8).max:
                    raise ValueError(
                        "YOLO semantic class IDs must fit in TRex's uint8 class range "
                        f"[0, 255], got [{minimum}, {maximum}]."
                    )
            self.semantic_mask = np.ascontiguousarray(semantic, dtype=np.uint8)

        box_array = np.asarray(box, dtype=np.float32)
        if box_array.shape[0] < 4:
            box_array = np.pad(box_array, (0, 4 - box_array.shape[0]), constant_values=0.0)
        box_offset = box_array[:2]
        crop_offset = box_offset  # legacy naming compatibility

        # Process keypoints: scale valid keypoint coordinates
        keypoints_attr = getattr(results, 'keypoints', None)
        if keypoints_attr is not None:
            # keypoints: list of arrays of shape [num_objects, num_keypoints, 2]
            self.keypoints: List[np.ndarray] = []
            #print(f"keypoints={keypoints_attr}")

            keypoint_tensor = keypoints_attr.cpu().data
            keys = np.ascontiguousarray(
                keypoint_tensor[..., :2].numpy(),
                dtype=np.float32,
            )
            #print("keys=",keys.shape, keypoints_attr.cpu())
            if len(keys) > 0 and len(keys[0]):
                valid_elements = np.isfinite(keys).all(axis=-1)
                valid_elements &= np.logical_or(keys[..., 0] != 0, keys[..., 1] != 0)
                if keypoint_tensor.shape[-1] >= 3:
                    threshold = float(TRex.setting("detect_keypoint_threshold"))
                    confidence = keypoint_tensor[..., 2].numpy()
                    valid_elements &= np.isfinite(confidence)
                    valid_elements &= confidence >= threshold

                keys[..., 0] = (keys[..., 0] + offset[0] + box_offset[0]) * scale[0]
                keys[..., 1] = (keys[..., 1] + offset[1] + box_offset[1]) * scale[1]

                keys[..., 0] = np.where(valid_elements, keys[..., 0], 0)
                keys[..., 1] = np.where(valid_elements, keys[..., 1], 0)

                # Append scaled keypoints if any valid points exist
                self.keypoints.append(keys) # bones * 3 elements

        # Process oriented bounding boxes (OBB): 
        # extract, scale, and annotate with class and confidence
        obb_attr = getattr(results, 'obb', None)
        if obb_attr is not None:
            # obb: array of shape [num_obb, 7] after adding class and confidence
            self.obb: np.ndarray = obb_attr.data[:, :5].cpu().numpy()
            
            # Scale and offset the center coordinates of each OBB
            offset_x, offset_y = offset
            box_dx, box_dy = box_offset
            self.obb[:, 0] = (self.obb[:, 0] + offset_x + box_dx) * scale[0]
            self.obb[:, 1] = (self.obb[:, 1] + offset_y + box_dy) * scale[1]
            self.obb[:, 2] = self.obb[:, 2] * scale[0]
            self.obb[:, 3] = self.obb[:, 3] * scale[1]

            # insert column for confidence in the front
            confs = obb_attr.conf.cpu().numpy()
            self.obb = np.insert(self.obb, 0, confs, axis=1)

            # insert column for class id in the front
            ids = obb_attr.cls.cpu().numpy()
            self.obb = np.insert(self.obb, 0, ids, axis=1)

            # OBBs are now [class, confidence, x_center, y_center, width, height, angle]
            #TRex.log(f"OBB after scaling: {self.obb.shape} {self.obb.dtype} {self.obb}")

        # Process segmentation masks: crop, validate, resize, and store for each box
        masks_attr = getattr(results, 'masks', None)
        mask_source_bounds = None
        if masks_attr is not None:
            # masks: list of 2D numpy arrays corresponding to each box
            self.masks: List[np.ndarray] = []

            # Duplicate boxes and save for scaling/unscaling
            coords : np.ndarray = np.copy(self.boxes)
            unscaled : np.ndarray = np.copy(coords)

            # Scale boxes to resized image coordinates
            offset_x, offset_y = offset
            box_dx, box_dy = crop_offset
            coords[:, 0] = (coords[:, 0] + offset_x + box_dx) * scale[0]
            coords[:, 1] = (coords[:, 1] + offset_y + box_dy) * scale[1]
            coords[:, 2] = (coords[:, 2] + offset_x + box_dx) * scale[0]
            coords[:, 3] = (coords[:, 3] + offset_y + box_dy) * scale[1]

            unscaled[:, 0] *= scale[0]
            unscaled[:, 1] *= scale[1]
            unscaled[:, 2] *= scale[0]
            unscaled[:, 3] *= scale[1]

            # Unscale coordinates back to source image resolution
            # scale xy first and then wh:
            orig_shape_local = self.orig_shape if self.orig_shape is not None else np.array(masks_attr.data.shape[1:])
            new_size: np.ndarray = orig_shape_local * scale
            unscaled[..., :2] = unscale_coords(masks_attr.data.shape[1:],
                                               unscaled[..., :2],
                                               new_size)
            unscaled[..., 2:4] = unscale_coords(masks_attr.data.shape[1:],
                                                unscaled[..., 2:4],
                                                new_size)

            # Ensure each box has a corresponding mask
            assert len(coords) == len(masks_attr.data)
            source_bounds = np.empty((len(coords), 4), dtype=np.int64)
            source_bounds[:, :2] = np.floor(coords[:, :2]).astype(np.int64)
            source_bounds[:, 2:4] = np.ceil(coords[:, 2:4]).astype(np.int64)
            valid_indices = []

            # For each box-mask pair: crop mask, validate, resize, and store
            # convert coords to int for indexing pixels properly (no float needed)
            # convert the returned masks data to uint8 as well, since its image data
            for row_index, (orig, unscale, k) in enumerate(zip(
                    source_bounds,
                    unscaled,
                    (masks_attr.data * 255).byte())):
                # Crop mask within its bounding box
                ux0 = max(0, int(np.floor(unscale[0])))
                uy0 = max(0, int(np.floor(unscale[1])))
                ux1 = max(ux0, int(np.ceil(unscale[2])))
                uy1 = max(uy0, int(np.ceil(unscale[3])))
                sub = k[uy0:uy1, ux0:ux1]

                # If mask is invalid or empty, remove its box and skip
                if orig[3] - orig[1] <= 0 or orig[2] - orig[0] <= 0 or sub.shape[0] <= 0 or sub.shape[1] <= 0:
                    print(f"WARNING: invalid mask size: orig={orig[3] - orig[1]}x{orig[2] - orig[0]} \n\
                          => sub={sub.shape[0]}x{sub.shape[1]} \n\
                          => unscale={unscale} \n\
                          => k={k.shape}\n\
                          => orig={orig}")
                    continue

                valid_indices.append(row_index)

                #print("sub",sub.shape, sub.dtype, "masks_attr: ",masks_attr.data.shape, masks_attr.data.dtype)

                # Resize valid mask to box's size
                ssub = F.interpolate(sub.unsqueeze(0).unsqueeze(0), size=(int(orig[3] - orig[1]), int(orig[2] - orig[0]))).squeeze(0).squeeze(0)
                # Store processed mask
                self.masks.append(ssub.cpu().numpy())

                #print("=>", self.masks[-1].shape, self.masks[-1].dtype, self.masks[-1].flags)
                
                assert self.masks[-1].flags['C_CONTIGUOUS']

            if len(valid_indices) != len(self.boxes):
                self.boxes = self.boxes[valid_indices]
            mask_source_bounds = source_bounds[valid_indices]

        # If we're dealing with a POLO model here we have point predictions
        locs = getattr(results, 'locations', None)
        if locs is not None:
            self.points = locs.data.cpu().numpy()
            TRex.log(f"Got data for locations: {self.points.shape} {self.points}")

            # Scale and offset the center coordinates of each prediction
            # predictions are [N, 4] with values [x_center, y_center, conf, id]
            # need to transform to [id, conf, x_center, y_center, radius]
            TRex.log(f"Scaling points with offset {offset} and box_offset {box_offset} and scale {scale}")
            self.points[:, 0:1] = (self.points[:, 0:1] + offset[0] + box_offset[0]) * scale[0]
            self.points[:, 1:2] = (self.points[:, 1:2] + offset[1] + box_offset[1]) * scale[1]

            # insert column for class id in the front
            #ids = locs.cls.cpu().numpy()
            #self.points = np.insert(self.points, 0, ids, axis=1)

            # move conf column from the end to the front
            confs = self.points[:, -2].copy()
            self.points = np.delete(self.points, -2, axis=1)
            self.points = np.insert(self.points, 0, confs, axis=1)

            # move id column from the end to the front
            ids = self.points[:, -1].copy()
            self.points = np.delete(self.points, -1, axis=1)
            self.points = np.insert(self.points, 0, ids, axis=1)

            # Now the points are in the format [id, conf, x_center, y_center]
            # and we can add a radius column with a default value
            # (e.g., 20 pixels) for each point
            detect_point_radii = TRex.setting("detect_point_radii")
            radii = None
            if isinstance(detect_point_radii, str):
                try:
                    detect_point_radii = eval(detect_point_radii)
                    if isinstance(detect_point_radii, dict):
                        # If radii is a dict, convert it to a list of radii
                        radii = [detect_point_radii.get(int(self.points[i, 0]), float(20)) for i in range(self.points.shape[0])]
                    else:
                        raise ValueError("Radii should be a dict.")
                except ValueError:
                    TRex.warn(f"Invalid radius value: {detect_point_radii}, using default 20")

            if radii is None:
                radii = [float(20)] * len(self.points)
            self.points = np.insert(self.points, 4, radii, axis=1)
            
            # Locations are now [class, confidence, x_center, y_center, radius]
            TRex.log(f"Locations: {self.points}")

        # Finally, scale any remaining bounding boxes to the target image space
        if self.boxes is not None:
            offset_x, offset_y = offset
            box_dx, box_dy = box_offset
            self.boxes[:, 0] = (self.boxes[:, 0] + offset_x + box_dx) * scale[0]
            self.boxes[:, 1] = (self.boxes[:, 1] + offset_y + box_dy) * scale[1]
            self.boxes[:, 2] = (self.boxes[:, 2] + offset_x + box_dx) * scale[0]
            self.boxes[:, 3] = (self.boxes[:, 3] + offset_y + box_dy) * scale[1]

            if mask_source_bounds is not None:
                self.boxes[:, :4] = mask_source_bounds.astype(np.float32, copy=False)

            # If coords has more than 6 columns, it contains tracking information
            # We remove that tracking information by deleting the id-column (at index 4)
            if self.boxes.shape[1] > 6:
                self.boxes = np.delete(self.boxes, 4, axis=1)

class YOLOModel(DetectionModel):
    def __init__(self, config: TRex.ModelConfig):
        """
        Initializes a Model object.

        Args:
            config (ModelConfig): An instance of the ModelConfig C++ class.
        """
        assert isinstance(config, TRex.ModelConfig)
        super().__init__(config)
        self.runtime_profile = RUNTIME_PROFILE_LEGACY
        self.explicit_iou_override: Optional[float] = None
        self.detect_head = None

    def __str__(self) -> str:
        return f"YOLOModel<{str(self.config)}>"

    def _iter_detect_heads(self):
        if not hasattr(self.ptr, "model") or self.ptr.model is None:
            return []
        return [module for module in self.ptr.model.modules() if hasattr(module, "end2end")]

    def _head_has_one2one_branch(self, head: Any) -> bool:
        try:
            one2one = getattr(head, "one2one", None)
            if not isinstance(one2one, dict):
                return False
            return one2one.get("box_head") is not None and one2one.get("cls_head") is not None
        except Exception:
            return False

    def _resolve_runtime_profile(self) -> str:
        self.explicit_iou_override = TRex.setting("detect_iou_threshold")
        self.detect_head = None

        for head in self._iter_detect_heads():
            if getattr(head, "end2end", False) and self._head_has_one2one_branch(head):
                self.detect_head = head
                if self.explicit_iou_override is not None:
                    return RUNTIME_PROFILE_MODERN_END2END_FORCED_NMS
                return RUNTIME_PROFILE_MODERN_END2END

        return RUNTIME_PROFILE_LEGACY

    def _set_detect_head_end2end(self, enabled: bool) -> None:
        for head in self._iter_detect_heads():
            try:
                head.end2end = enabled
            except Exception as exc:
                TRex.warn(f"Could not set end2end={enabled} for {type(head)}: {exc}")

    def _prediction_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        sanitized = dict(kwargs)

        if self.runtime_profile == RUNTIME_PROFILE_MODERN_END2END:
            # Respect upstream NMS-free end2end inference even if callers still pass legacy knobs.
            sanitized.pop("iou", None)
            sanitized.pop("agnostic_nms", None)
        elif sanitized.get("iou", None) is None:
            sanitized.pop("iou", None)
            sanitized.pop("agnostic_nms", None)
        #sanitized["retina_masks"] = True

        return sanitized

    def load(self):
        """
        Load the model from the specified configuration.
        This method should handle the loading of the model parameters and any necessary setup.
        """
        # Load the model from the specified path
        try:
            self.ptr = YOLO(self.config.model_path).to("cpu")
            print(f"Loaded model {self} onto the CPU first.")
        except Exception as e:
            if "LocalizationModel" in str(e):
                # user is trying to load a POLO model in a version of ultralytics that does not support it
                # i.e. the user didnt install POLO, but ultralytics default version
                raise Exception(f"Failed to load model {self}. This model may be a POLO model which requires a version of ultralytics that supports it. Please install ultralytics with POLO support or use a non-POLO model. Original error: {e}")
            else:
                raise Exception(f"Failed to load model {self}. Original error: {e}")

        # initialize the torch device in case this has been broken
        # or the device has changed
        self.reinit_device()

        if self.ptr.task == "segment" and self.device.type == "mps" and TRex.setting("gpu_torch_no_fixes") == "false":
            TRex.log(f"Model {self} cannot be run on MPS due to a bug in PyTorch or Ultralytics. Automatically switching to CPU for this model only. Use -gpu_torch_no_fixes parameter to disable this.")
            self.device = torch.device("cpu")

        if self.ptr.task in {"segment", "semantic"}:
            self.config.output_format = ObjectDetectionFormat.masks
        elif self.ptr.task == "detect":
            self.config.output_format = ObjectDetectionFormat.boxes
        elif self.ptr.task == "pose":
            self.config.output_format = ObjectDetectionFormat.poses
        elif self.ptr.task == "obb":
            self.config.output_format = ObjectDetectionFormat.obb
        elif self.ptr.task == "locate":
            self.config.output_format = ObjectDetectionFormat.points
        else:
            raise Exception(f"Unknown task {self.ptr.task}")
        
        try:
            train_args = self.ptr.ckpt["train_args"]
            imgsz = train_args["imgsz"]
            if isinstance(imgsz, int):
                imgsz = [imgsz, imgsz]
            self.config.trained_resolution = DetectResolution(imgsz[1], imgsz[0])
            self.config.classes = self.ptr.names
            TRex.log(f"set trained_resolution = {self.config.trained_resolution}")

            if(self.config.output_format == ObjectDetectionFormat.poses):
                TRex.log(f"Task is pose, setting keypoint_format to ")
                TRex.log(f"keypoint_format = {self.ptr.kpt_shape}")
                self.config.keypoint_format = TRex.KeypointFormat(self.ptr.kpt_shape[0], self.ptr.kpt_shape[1])
            else:
                TRex.log(f"Task is not pose, not setting keypoint_format")
                self.config.keypoint_format = TRex.KeypointFormat(0, 0)

        except Exception as e:
            TRex.warn("Could not determine trained resolution from model, using " + str(self.config.trained_resolution)+ " ("+ str(e) + ")")
            pass

        self.runtime_profile = self._resolve_runtime_profile()
        TRex.log(
            f"Resolved runtime profile {self.runtime_profile} for model {self.config.model_path} "
            f"(detect_iou_threshold={self.explicit_iou_override})."
        )

        if self.runtime_profile == RUNTIME_PROFILE_MODERN_END2END_FORCED_NMS:
            try:
                self._set_detect_head_end2end(False)
                TRex.log(
                    f"Disabled end2end before fuse() for model {self.config.model_path} because "
                    "detect_iou_threshold is explicitly set."
                )
            except Exception as e:
                TRex.warn(f"Could not disable end2end before fuse: {e}")

        try:
            self.ptr.fuse()
        except Exception as e:
            TRex.warn(f"Model fuse() failed, continuing unfused: {e}")

        # half() is generally only beneficial/valid on CUDA; keep it disabled by default.
        if self.device.type == "cuda":
            try:
                self.ptr.half()
            except Exception as e:
                TRex.warn(f"Model half() failed: {e}")

        self.ptr.to(self.device)
        TRex.log(f"Moved model {self} to device {self.device}.")

        super().load()

    def predict_boxes(self, images : List[np.ndarray], **kwargs) -> List[np.ndarray]:
        if len(images) == 0:
            return []

        kwargs = self._prediction_kwargs(kwargs)
        
        if self.config.use_tracking:
            results = []
            for image in images:
                results.append(self.ptr.track(image, tracker="bytetrack.yaml", persist=True, device=self.device, **kwargs)[0])
            return [bb.boxes.xyxy.cpu().numpy() for bb in results]
        else:
            return [bb.boxes.xyxy.cpu().numpy() for bb in self.ptr.predict(images, device=self.device, stream=True, **kwargs)]
        
    def predict(self, images: List[np.ndarray], scales : List[Any], offsets : List[Any], **kwargs) -> List[StrippedResults]:
        """
        Predict the objects in the image.

        Args:
            images (List[np.ndarray]): A list of images to predict on.
            scales (List[Any]): A list of scales for each image.
            offsets (List[Any]): A list of offsets for each image.
            **kwargs: Additional arguments to be passed to the model.

        Returns:
            List[TRex.Result]: A list of results for each image.
        """
        if len(images) == 0:
            return []

        results = []
        kwargs = self._prediction_kwargs(kwargs)

        # If radii is not provided in kwargs and the model has class names, generate default radii
        if self.config.output_format == ObjectDetectionFormat.points:
            radii = kwargs.get("radii", None)
            if radii is None and hasattr(self.ptr, "names"):
                # Assign a default radius (e.g., 20) for each class
                radii = {i: 20 for i in range(len(self.ptr.names))}
                kwargs["radii"] = radii
        
        if self.config.use_tracking and self.ptr.task != "semantic":
            for image, scale, offset in zip(images, scales, offsets):
                results.append((self.ptr.track(image, tracker="bytetrack.yaml", persist=True, device=self.device, **kwargs)[0], scale, offset))
        else:
            results = self.ptr.predict(images, device=self.device, stream=True, **kwargs)
            results = [(r, scale, offset) for r, scale, offset in zip(results, scales, offsets)]

        return [StrippedYoloResults(r, scale, offset) for r, scale, offset in results]


class TRexYOLO(TRexDetection):
    def __init__(self, models: List[DetectionModel]):
        """
        Initialize the TRexYOLO class with a list of models.

        Args:
            models (List[Model]): A list of models used for region proposal, detection and segmentation.

        Raises:
            AssertionError: If no models are specified.
        """
        super().__init__(models)

    def __str__(self) -> str:
        """
        String representation of the TRexYOLO instance.

        Returns:
            str: A string that represents the TRexYOLO instance.
        """
        return "TRexYOLO<models={}>".format(self.models)
