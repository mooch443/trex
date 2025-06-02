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
from typing import List, Any

##############
##### since this isnt patched in the ultralytics package yet,
##### we need to patch the following functions in the ultralytics.utils.ops module
##### see https://github.com/ultralytics/ultralytics/issues/8555
##############

import ultralytics
from ultralytics.utils.ops import clip_boxes, crop_mask

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None, padding=True, xywh=False):
    """
    Rescales bounding boxes (in the format of xyxy by default) from the shape of the image they were originally
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

def process_mask(protos, masks_in, bboxes, shape, upsample=False):
    """
    Apply masks to bounding boxes using the output of the mask head.

    Args:
        protos (torch.Tensor): A tensor of shape [mask_dim, mask_h, mask_w].
        masks_in (torch.Tensor): A tensor of shape [n, mask_dim], where n is the number of masks after NMS.
        bboxes (torch.Tensor): A tensor of shape [n, 4], where n is the number of masks after NMS.
        shape (tuple): A tuple of integers representing the size of the input image in the format (h, w).
        upsample (bool): A flag to indicate whether to upsample the mask to the original image size. Default is False.

    Returns:
        (torch.Tensor): A binary mask tensor of shape [n, h, w], where n is the number of masks after NMS, and h and w
            are the height and width of the input image. The mask is applied to the bounding boxes.
    """

    c, mh, mw = protos.shape  # CHW
    ih, iw = shape
    masks = (masks_in @ protos.float().view(c, -1)).sigmoid().view(-1, mh, mw)  # CHW
    width_ratio = mw / iw
    height_ratio = mh / ih

    downsampled_bboxes = bboxes.clone()
    scale_factors = torch.tensor([width_ratio, height_ratio, width_ratio, height_ratio], device=bboxes.device)
    downsampled_bboxes = downsampled_bboxes * scale_factors

    masks = crop_mask(masks, downsampled_bboxes)  # CHW
    if upsample:
        masks = F.interpolate(masks[None], shape, mode="bilinear", align_corners=False)[0]  # CHW
    return masks.gt_(0.5)

ultralytics.utils.ops.process_mask = process_mask
ultralytics.utils.ops.scale_boxes = scale_boxes

TRex.log("*** patched functions ***")
##############


def unscale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """
    Rescale segment coordinates (xyxy) from normalized to original image scale

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
    def __init__(self, results, scale, offset, box = [0, 0, 0, 0]):
        super().__init__(scale, offset)

        if results.boxes is not None:
            self.boxes = results.boxes.data.cpu().numpy()
        self.orig_shape = results.orig_shape

        box = np.array([box[0],box[1]])
        box_offset = np.array([box[0], box[1]])
        
        if results.keypoints is not None:
            self.keypoints = []#results.keypoints.cpu().numpy()
            #print(f"keypoints={results.keypoints}")

            keys = results.keypoints.cpu().data[..., :2].numpy()
            #print("keys=",keys.shape, results.keypoints.cpu())
            if len(keys) > 0 and len(keys[0]):
                #print(result.keypoints.cpu().xy, scale)

                # Scale and offset the keypoints, but leave out
                # the ones where both X and Y are zero (invalid)
                zero_elements = np.logical_and(keys[..., 0] == 0, keys[..., 1] == 0)

                keys[..., 0] = (keys[..., 0] + offset[0] + box_offset[0]) * scale[0]
                keys[..., 1] = (keys[..., 1] + offset[1] + box_offset[1]) * scale[1]

                if zero_elements.any():
                    keys[..., 0] = np.where(zero_elements, 0, keys[..., 0])
                    keys[..., 1] = np.where(zero_elements, 0, keys[..., 1])

                #keys[..., 0] = (keys[..., 0] + offset[0]) * scale[0] #+ coords[..., 0]).T
                #keys[..., 1] = (keys[..., 1] + offset[1]) * scale[1] #+ coords[..., 1]).T
                self.keypoints.append(keys) # bones * 3 elements

        if results.obb is not None:
            self.obb = results.obb.data[:, :5].cpu().numpy()
            #TRex.log(f"OBB: {self.obb.shape} {self.obb.dtype} {self.obb}")
            #TRex.log(f"offset={offset}, box_offset={box_offset}, scale={scale}")
            # Scale and offset the OBB coordinates
            self.obb[:, :2] = (self.obb[:, :2] + offset + box_offset) * scale[0]
            self.obb[:, 2:4] = (self.obb[:, 2:4] + offset + box_offset) * scale[1]

            # insert column for confidence in the front
            confs = results.obb.conf.cpu().numpy()
            self.obb = np.insert(self.obb, 0, confs, axis=1)

            # insert column for class id in the front
            ids = results.obb.cls.cpu().numpy()
            self.obb = np.insert(self.obb, 0, ids, axis=1)

            #TRex.log(f"OBB after scaling: {self.obb.shape} {self.obb.dtype} {self.obb}")

        if results.masks is not None:
            self.masks = []
            
            coords = np.copy(self.boxes)
            unscaled = np.copy(coords)

            coords[:, :2] = (coords[:, :2] + offset + box) * scale[0]
            coords[:, 2:4] = (coords[:, 2:4] + offset + box) * scale[1]

            unscaled[:, :2] *= scale[0]
            unscaled[:, 2:4] *= scale[1]
            
            new_size = results.orig_shape * scale
            unscaled[..., :2] = unscale_coords(results.masks.data.shape[1:], unscaled[..., :2], new_size)
            unscaled[..., 2:4] = unscale_coords(results.masks.data.shape[1:], unscaled[..., 2:4], new_size)
            
            assert len(coords) == len(results.masks.data)
            index = 0
            for orig, unscale, k in zip(coords.round().astype(int), unscaled, (results.masks.data * 255).byte()):
                sub = k[max(0, int(unscale[1])):max(0, int(unscale[3])), max(0,int(unscale[0])):max(0, int(unscale[2]))]
                if orig[3] - orig[1] <= 0 or orig[2] - orig[0] <= 0 or sub.shape[0] <= 0 or sub.shape[1] <= 0:
                    print(f"WARNING: invalid mask size: orig={orig[3] - orig[1]}x{orig[2] - orig[0]} \n\
                          => sub={sub.shape[0]}x{sub.shape[1]} \n\
                          => unscale={unscale} \n\
                          => k={k.shape}\n\
                          => orig={orig}")
                    #raise Exception("Invalid mask size")
                    
                    # remove invalid mask from self.boxes at the same index
                    self.boxes = np.delete(self.boxes, index, axis=0)
                    continue

                index += 1

                ssub = F.interpolate(sub.unsqueeze(0).unsqueeze(0), size=(int(orig[3] - orig[1]), int(orig[2] - orig[0]))).squeeze(0).squeeze(0)
                self.masks.append(ssub.cpu().numpy())
                assert self.masks[-1].flags['C_CONTIGUOUS']

        if self.boxes is not None:
            # Scale and offset the bounding boxes
            self.boxes[:, :2] = (self.boxes[:, :2] + offset + box_offset) * scale[0]
            self.boxes[:, 2:4] = (self.boxes[:, 2:4] + offset + box_offset) * scale[1]

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

    def __str__(self) -> str:
        return f"YOLOModel<{str(self.config)}>"

    def load(self):
        """
        Load the model from the specified configuration.
        This method should handle the loading of the model parameters and any necessary setup.
        """
        # Load the model from the specified path
        self.ptr = YOLO(self.config.model_path).to('cpu')
        print(f"Loading model {self} on device {self.device}")

        # initialize the torch device in case this has been broken
        # or the device has changed
        self.reinit_device()

        if self.ptr.task == "segment" and self.device.type == "mps" and TRex.setting("gpu_torch_no_fixes") == "false":
            TRex.log(f"Model {self} cannot be run on MPS due to a bug in PyTorch or Ultralytics. Automatically switching to CPU for this model only. Use -gpu_torch_no_fixes parameter to disable this.")
            self.device = torch.device("cpu")

        if self.ptr.task == "segment":
            self.config.output_format = ObjectDetectionFormat.masks
        elif self.ptr.task == "detect":
            self.config.output_format = ObjectDetectionFormat.boxes
        elif self.ptr.task == "pose":
            self.config.output_format = ObjectDetectionFormat.poses
        elif self.ptr.task == "obb":
            self.config.output_format = ObjectDetectionFormat.obb
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

        self.ptr.fuse()
        self.ptr.half()
        self.ptr.to(self.device)

        super().load()

    def predict_boxes(self, images : List[np.ndarray], **kwargs) -> List[np.ndarray]:
        if len(images) == 0:
            return []
        
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

        if self.config.use_tracking:
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
