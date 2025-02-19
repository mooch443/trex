import torch
import numpy as np
import torchvision
from torchvision import transforms
import torchvision.transforms as T
from torch.nn import functional as F

import time
import cv2
import os
import sys
import argparse
import platform
from enum import Enum

from pathlib import Path
import logging
from functools import lru_cache

from typing import List, Tuple

import ultralytics.utils

import TRex
from TRex import ModelTaskType
from TRex import DetectResolution
from TRex import ObjectDetectionFormat

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

# Set up logging
logging.basicConfig(level=logging.INFO)

class LoggerWriter:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.message = ''  # Buffer for messages
        self.encoding = 'utf-8'

    def write(self, message):
        # Buffer the message
        self.message += message
        # If the message ends with a newline character
        if message.endswith('\n'):
            # Fetch the current frame
            import traceback
            frame = traceback.extract_stack()
            frame = frame[-2]
            # Log the message with additional info, without the trailing newline character
            TRex.log(frame.filename, frame.lineno, f"{self.message[:-1]}")
            # Reset the message buffer
            self.message = ''

    def flush(self):
        pass

# Replace stdout with an instance of LoggerWriter
sys.stdout = LoggerWriter(logging.getLogger(), logging.INFO)

'''if torch.backends.mps.is_available():
    device = torch.device("mps")
    #x = torch.ones(1, device=mps_device)
    #print (x)
else:
    print ("MPS device not found.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")'''


# import the necessary packages
import numpy as np

model = None
image_size = [640,640]
model_path = None
segmentation_path = None
segmentation_resolution = 128
region_path = None
image = None
oimages = None
model_type = None
q_model = None
imgsz = None
device = None
offsets = None
iou_threshold = 0.7
conf_threshold = 0.1

seen, windows, dt = 0, [], None

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

class Model:
    def __init__(self, config):
        """
        Initializes a Model object.

        Args:
            config (ModelConfig): An instance of the ModelConfig C++ class.
        """
        assert isinstance(config, TRex.ModelConfig)
        self.config = config
        self.ptr = None
        self.device : torch.device = None

        # if no device is specified, use cuda if available, otherwise use mps/cpu
        device_from_settings = TRex.setting("gpu_torch_device")
        if device_from_settings != "":
            if device_from_settings == "automatic":
                device_from_settings = ""
            else:
                device_index = eval(TRex.setting("gpu_torch_device_index"))
                if device_index >= 0:
                    device_from_settings = f"{device_from_settings}:{device_index}"
                print(f"Using device {device_from_settings} from settings.")
                self.device = torch.device(device_from_settings)

        if self.device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            elif torch.backends.mps.is_available() and platform.processor() == "arm":
                self.device = torch.device("mps") #mps
            else:
                self.device = torch.device("cpu")

    def __str__(self) -> str:
        return f"Model<{str(self.config)}>"

    def __repr__(self) -> str:
        return self.__str__()
    
    @property
    @lru_cache(maxsize=1)
    def task(self) -> TRex.ModelTaskType:
        #print("task == ", self.config.task, type(self.config.task))
        return self.config.task

    def load(self):
        from ultralytics import YOLO
        self.ptr = YOLO(self.config.model_path).to('cpu')
        #self.device = None

        device_from_settings = TRex.setting("gpu_torch_device")
        if device_from_settings != "":
            if device_from_settings == "automatic":
                device_from_settings = ""
            else:
                device_index = eval(TRex.setting("gpu_torch_device_index"))
                if device_index >= 0:
                    device_from_settings = f"{device_from_settings}:{device_index}"
                print(f"Using device {device_from_settings} from settings.")
                self.device = torch.device(device_from_settings)

        if self.device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            elif torch.backends.mps.is_available() and platform.processor() == "arm":
                self.device = torch.device("mps") #mps
            else:
                self.device = torch.device("cpu")

        if self.ptr.task == "segment" and self.device.type == "mps" and TRex.setting("gpu_torch_no_fixes") == "false":
            TRex.log(f"Model {self} cannot be run on MPS due to a bug in PyTorch or Ultralytics. Automatically switching to CPU for this model only. Use -gpu_torch_no_fixes parameter to disable this.")
            self.device = torch.device("cpu")

        if self.ptr.task == "segment":
            self.config.output_format = ObjectDetectionFormat.masks
        elif self.ptr.task == "detect":
            self.config.output_format = ObjectDetectionFormat.boxes
        elif self.ptr.task == "pose":
            self.config.output_format = ObjectDetectionFormat.poses
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
        TRex.log("Loaded model: {}".format(self))

    def predict(self, images : List[np.ndarray], **kwargs):
        if len(images) == 0:
            return []
        
        if self.config.use_tracking:
            results = []
            for image in images:
                results.append(self.ptr.track(image, tracker="bytetrack.yaml", persist=True, device=self.device, **kwargs)[0])
            return results
        else:
            return self.ptr.predict(images, device=self.device, stream=True, **kwargs)

class StrippedYoloResults:
    def __init__(self, results, scale, offset, box = [0, 0, 0, 0]):
        self.boxes = results.boxes.data.cpu().numpy()
        self.keypoints = None
        self.orig_shape = results.orig_shape
        self.scale = scale
        self.offset = offset

        box = np.array([box[0],box[1]])
        
        if results.keypoints is not None:
            self.keypoints = results.keypoints.cpu().numpy()#[..., :2]
        self.masks = None

        if results.masks is not None:
            self.masks = []
            
            coords = np.copy(self.boxes)#results.boxes.data.clone().to(device)#.cpu()
            #unscaled = coords.clone()
            #coords = coords.numpy()
            unscaled = np.copy(coords)

            #print("devices: ", coords.device, unscaled.device, offset.device, scale.device, 
            #      "box = ", box, "coords = ", coords.shape, "scale = ", scale.shape, "offset = ", offset.shape, "orig_shape = ", results.orig_shape, "scale = ", scale, "offset = ", offset)
            #print("adding masks ", self.offset, self.scale, box,coords.shape)
            coords[:, :2] = (coords[:, :2] + offset + box) * scale[0]
            coords[:, 2:4] = (coords[:, 2:4] + offset + box) * scale[1]

            unscaled[:, :2] *= scale[0]
            unscaled[:, 2:4] *= scale[1]
            
            #print("unscaled = ", unscaled.shape)
            #print("results.orig_shape = ", results.orig_shape, " scale = ", scale)
            #print("results.orig_shape * scale = ", torch.Tensor(results.orig_shape).to(device) * scale)

            #new_size = torch.Tensor(results.orig_shape).to(device) * scale
            new_size = results.orig_shape * scale
            unscaled[..., :2] = unscale_coords(results.masks.data.shape[1:], unscaled[..., :2], new_size)#.round().astype(int)
            unscaled[..., 2:4] = unscale_coords(results.masks.data.shape[1:], unscaled[..., 2:4], new_size)#.round().astype(int)
            
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
                # resize sub to scaled size using pytorch
                #sub = cv2.rectangle((sub * 255).byte().cpu().numpy(), (5,5), (sub.shape[1]-5, sub.shape[0]-5), (0,0,0), 1)
                #print("sub.shape=",sub.shape, " orig=",orig, (orig[3] - orig[1], orig[2] - orig[0]))
                #sub = cv2.resize(sub, (orig[2] - orig[0], orig[3] - orig[1]), interpolation=cv2.INTER_LINEAR)
                #self.masks.append(sub)
                ssub = F.interpolate(sub.unsqueeze(0).unsqueeze(0), size=(int(orig[3] - orig[1]), int(orig[2] - orig[0]))).squeeze(0).squeeze(0)
                self.masks.append(ssub.cpu().numpy())
                assert self.masks[-1].flags['C_CONTIGUOUS']
                #TRex.imshow("mask"+str(len(self.masks)),self.masks[-1])

BBox = np.ndarray#[int]
Image = np.ndarray#[np.uint8]
printed_warning = False

from typing import List, Tuple

def calculate_iou(box1, box2):
    x1, y1, x1_br, y1_br = box1
    x2, y2, x2_br, y2_br = box2

    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1_br, x2_br)
    y_bottom = min(y1_br, y2_br)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (x1_br - x1) * (y1_br - y1)
    box2_area = (x2_br - x2) * (y2_br - y2)
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area

def merge_boxes(boxes, iou_threshold):
    merged_boxes = []
    skipped_boxes = set()

    for i, box1 in enumerate(boxes):
        if i in skipped_boxes:
            continue
        merged_box = box1
        for j, box2 in enumerate(boxes):
            if i != j and j not in skipped_boxes:
                if calculate_iou(merged_box, box2) > iou_threshold:
                    x0 = min(merged_box[0], box2[0])
                    y0 = min(merged_box[1], box2[1])
                    x1 = max(merged_box[2], box2[2])
                    y1 = max(merged_box[3], box2[3])
                    merged_box = [x0, y0, x1, y1]
                    skipped_boxes.add(j)
        merged_boxes.append(merged_box)

    return merged_boxes

class TRexYOLO:
    def __init__(self, models: List[Model]):
        """
        Initialize the TRexYOLO class with a list of models.

        Args:
            models (List[Model]): A list of models used for region proposal, detection and segmentation.

        Raises:
            AssertionError: If no models are specified.
        """
        assert len(models) > 0, "No models specified for TRexYOLO {}".format(models)

        self.models = models
        self.load_models()

        self.boxes_time = 0
        self.boxes_samples = 0
        self.results_time = 0
        self.results_samples = 0

        # log configuration and loaded models
        TRex.log("TRexYOLO configuration: models={}".format(self.models))

    def __str__(self) -> str:
        """
        String representation of the TRexYOLO instance.

        Returns:
            str: A string that represents the TRexYOLO instance.
        """
        return "TRexYOLO<models={}>".format(self.models)

    def has_region_model(self) -> bool:
        """
        Check if the list of models contains a region model.

        Returns:
            bool: True if a region model is found, False otherwise.
        """
        return any(model.task == ModelTaskType.region for model in self.models)

    def has_detect_model(self) -> bool:
        """
        Check if the list of models contains a detection model.

        Returns:
            bool: True if a detection model is found, False otherwise.
        """
        return any(model.task == ModelTaskType.detect for model in self.models)

    def region_proposal(self, images: List[Image], **kwargs) -> List[List[Tuple[BBox, Image]]]:
        """
        Performs region proposals on a list of input images using the region model.
        These regions are focus points for where segmentation and detection models will be applied.
        This can reduce the number of pixels that the segmentation and detection models need to process,
        and thus improve performance.

        Args:
            images (list[Image]): A list of input images to perform region proposals on.
            **kwargs: Additional keyword arguments to pass to the model's `predict` method.

        Returns:
            list[list[tuple[BBox, Image]]]: A list per input image, with a list of bounding boxes (and their image regions) per image.
        """
        assert self.has_region_model(), "No region model found"
        
        #from ultralytics.yolo.utils.ops import scale_coords
        # predict bounding boxes of areas of interest

        #start = time.time()
        # take timing
        model = next((model for model in self.models if model.task == ModelTaskType.region), None)
        assert model is not None

        scaled_size: int = [model.config.trained_resolution.width, model.config.trained_resolution.height]
        #scaled_down_scales: np.ndarray = np.array([(scaled_size / im.shape[1], int(im.shape[0] / im.shape[1] * scaled_size) / im.shape[0]) for im in images])
        #scaled_down: List[Image] = [cv2.resize(im,  (scaled_size, int(im.shape[0] / im.shape[1] * scaled_size))) for im in images]

        scaled_down_scales: np.ndarray = np.array([(1,1) for im in images])
        scaled_down: List[Image] = [im for im in images] #
        
        #import torchvision.transforms as transforms
        #from torchvision.transforms.functional import to_tensor, resize

        #tensors = []
        #for im in images:
        #    tensor = to_tensor(im).to(device)
        #    resized_tensor = resize(tensor,  (scaled_size, scaled_size))
        #    tensors.append(resized_tensor.unsqueeze(0))
        #resized_images = torch.cat(tensors, dim=0)
        #print(resized_images.shape)
        #scaled_down_scales: np.ndarray = np.array([(scaled_size / im.shape[1],scaled_size / im.shape[0]) for im in images])
        #scaled_down: List[Image] = resized_images 

        #print(f"performing region proposals at {scaled_size}x{scaled_size} on {len(images)} images on {model.device} with scaled sizes of {[im.shape for im in scaled_down]}")
        import ultralytics
        bboxes: List[ultralytics.engine.results.Results] = \
                       [r for r in model.predict(images = scaled_down, 
                                     imgsz=scaled_size, 
                                     conf=0.1, 
                                     iou=0.7, 
                                     verbose=False,
                                     **kwargs)]
        padding: int = 7
        results: List[List[Tuple[BBox, Image]]] = []
        
        #print(f"received {bboxes[0].boxes.data.shape} bounding boxes")
        downloaded = [bb.boxes.xyxy.cpu().numpy() for bb in bboxes]
        
        #for i, bb in enumerate(bboxes):
        #    p = bb.cpu().plot(line_width=1)
        #    p = cv2.resize(p, (1280, 900))
        #    TRex.imshow("region proposals"+str(i), p)

        # return a list per input image, with a list of bounding boxes (and their image regions) per image
        for i, bb in enumerate(downloaded):
            boxes = []
            h, w = images[i].shape[:2]

            for box in bb:
                #print("before:",box)
                box[:2] /= scaled_down_scales[i]
                box[2:4] /= scaled_down_scales[i]
                #print("after:",box)

                if padding > 0:
                    box[:2] -= np.array([padding,padding])
                    box[2:4] += np.array([padding,padding])

                box = box.astype(int)

                x0 = int(max(0, box[0]))
                y0 = int(max(0, box[1]))
                x1 = min(w, max(x0, box[2]))
                y1 = min(h, max(y0, box[3]))

                box = np.array([x0,y0,x1,y1])
                #print(box)

                boxes.append(box)

            mois = []
            #print(boxes)
            conservative_boxes = merge_boxes(boxes, iou_threshold=0.0)
            #print(conservative_boxes)

            for box in conservative_boxes:
                x0, y0, x1, y1 = box
                x0 = max(0, x0)
                y0 = max(0, y0)
                x1 = min(w, x1)
                y1 = min(h, y1)

                oi = images[i][y0:y1, x0:x1]
                if oi.shape[0] == 0 or oi.shape[1] == 0:
                    print("region is zero ",i, oi.shape, box)
                    continue

                mois.append((box, oi))

            results.append(mois)

        # take timing
        #end = time.time()
        #print(f"region proposals took {(end - start) / len(images)} seconds ", len(images) )

        return results
    
    def detect_resolution(self) -> int:
        """
        Retrieves the trained resolution of the detection or segmentation model.

        Returns:
            int: The trained resolution of the detection or segmentation model.

        Raises:
            Exception: If no detect or segment model is found.
        """
        detect_model = next((model for model in self.models if model.task == ModelTaskType.detect), None)
        if detect_model is not None:
            return [detect_model.config.trained_resolution.height, 
                    detect_model.config.trained_resolution.width]
        raise Exception("No detect or segment model found")

    def detection_model(self) -> Model:
        """
        Retrieves the detection or segmentation model.

        Returns:
            Model: The detection or segmentation model.

        Raises:
            Exception: If no detect or segment model is found.
        """
        detect_model = next((model for model in self.models if model.task == ModelTaskType.detect), None)
        if detect_model is not None:
            return detect_model
        raise Exception("No detect or segment model found")

    def detect_or_segment(self, **kwargs) -> np.ndarray:
        """
        Perform detection or segmentation on given input.

        Args:
            **kwargs: Additional keyword arguments to pass to the model's `predict` method.

        Returns:
            np.ndarray: The result of the detection or segmentation operation.

        Raises:
            Exception: If no detect or segment model is found.
        """
        detect_model = next((model for model in self.models if model.task == ModelTaskType.detect), None)
        if detect_model is not None:
            return [r for r in detect_model.predict(**kwargs)]
        raise Exception("No detect or segment model found")

    def load_models(self):
        from ultralytics import YOLO
        import torch

        TRex.log("Loading models: models={}".format(self.models))
        for model in self.models:
            model.load()

    def preprocess(self, images : List[Image]):
        """
        Preprocesses an input list of images by converting each image from BGR color space to RGB.

        Parameters:
        images (list): Input list of images

        Returns:
        list: A list of preprocessed images
        """
        return [cv2.cvtColor(np.array(i, copy=False), cv2.COLOR_BGR2RGB) for i in images]
    
    def perform_region_proposal(self, tensor, offsets, scales, ious, confs) -> List[TRex.Result]:
        """
        This function applies the region proposal to a given tensor, performing object detection and segmentation 
        on the proposed regions. It then collects and returns the results.

        Parameters:
        - tensor (np.ndarray): The images to be processed
        - offsets (list): List of offsets for each proposed region.
        - scales (list): List of scales for each proposed region.
        - ious (list): List of Intersection-over-Union scores for each proposed region.
        - confs (list): List of confidence values for each proposed region.

        Returns:
        - list of TRex.Result: Each TRex.Result object contains the result of the object detection and 
        segmentation (optional) for a proposed region.
        """
        # Apply the region proposal method on the given tensor
        # start profiler
        #from pyinstrument import Profiler
        #profiler = Profiler()
        #profiler.start()

        proposal = self.region_proposal(tensor)

        rexsults = []
        all_images = []
        # For each region in the proposal, add the region to all_images
        for regions in proposal:
            all_images.extend([region for _, region in regions])

        # Perform detection or segmentation on the aggregated regions
        # time info:
        start = time.time()
        results = self.detect_or_segment(
            images = all_images, 
            conf = confs, 
            iou = ious, 
            imgsz = self.detect_resolution(),
            #imgsz=160,
            classes=None, 
            agnostic_nms=True,
            verbose = False)

        # Check if the number of results is equal to the number of images
        assert len(results) == len(all_images),f"length of results {len(results)} is not equal to length of all images {len(all_images)}"
        #print("results = ",len(results), " all_images = ",len(all_images), " time = ", (time.time() - start) / len(all_images))
        #print(ious)

        # Reorder the results list to match the order of the proposal list
        cursor = 0
        all_results = []
        for regions in proposal:
            end = cursor + len(regions)
            assert len(results) >= end
            all_results.append(results[cursor:end])
            cursor = end

        # Perform segmentation predictions for each identified region
        for i, (regions, results) in enumerate(zip(proposal, all_results)):
            scale = scales[i]
            offset = offsets[i]

            collected_boxes = []
            collected_masks = []
            collected_keypoints = []

            # For each result, perform postprocessing and gather the boxes and masks
            for t, (result, (box, region)) in enumerate(zip(results, regions)):
                #r = result.cpu().plot(img=region, line_width=1)
                #TRex.imshow("segmentation"+str(i)+","+str(t), r)

                result = StrippedYoloResults(result, scale=scale, offset=offset)
                
                coords, masks, keypoints = self.postprocess_result(i, result, offset, scale, box)
                collected_boxes.append(coords)
                collected_masks.extend(masks)
                if len(keypoints) > 0:
                    collected_keypoints.extend(keypoints)

            # Concatenate all collected boxes
            if len(collected_boxes) > 0:
                collected_boxes = np.concatenate(collected_boxes, axis=0)

            if len(collected_keypoints) > 0:
                collected_keypoints = np.concatenate(collected_keypoints, axis=0)

            # Check if the number of boxes is equal to the number of masks
            assert len(collected_masks) == 0 or len(collected_boxes) == len(collected_masks)
            assert len(collected_keypoints) == 0 or len(collected_boxes) == len(collected_keypoints)

            # perform nms on collected_boxes, use indexes to filter keypoints, boxes and masks in the same way
            if len(collected_boxes) > 0:
                from torchvision.ops import nms
                indexes = nms(torch.Tensor(collected_boxes[..., :4]), torch.Tensor(collected_boxes[..., 4]), ious)
                if(sorted(indexes.numpy()) != list(range(len(indexes)))):
                    #print("filtering boxes: ", collected_boxes,"with",ious)
                    #print("using indexes: ", sorted(indexes.numpy()))
                    collected_boxes = [collected_boxes[i] for i in indexes]
                    if len(collected_masks) > 0:
                        collected_masks = [collected_masks[i] for i in indexes]
                    if len(collected_keypoints) > 0:
                        collected_keypoints = [collected_keypoints[i] for i in indexes]

            # Append the result to the list of results
            rexsults.append(TRex.Result(i, TRex.Boxes(collected_boxes), collected_masks, TRex.KeypointData(collected_keypoints)))

        # stop profiler
        #profiler.stop()
        #print(profiler.output_text(unicode=True, color=False, show_all=True))
        # Return the final list of results
        return rexsults


    def postprocess_result(self, i, result, offset, scale, box = [0, 0, 0, 0]) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """
        This function postprocesses the result of the object detection and segmentation for a proposed region.

        Parameters:
        - i (int): The index of the result to be postprocessed.
        - result (TRex.Result): The result to be postprocessed.
        - offset (list): The offset for the proposed region.
        - scale (list): The scale for the proposed region.
        - box (list): The bounding box of the proposed region.

        Returns:
        - tuple[np.ndarray, list[np.ndarray]]: The postprocessed result as a tuple of the coordinates and masks.
        """
        # Extract bounding boxes from result
        coords = result.boxes
        unscaled = np.copy(coords)

        # Scale and offset the bounding boxes
        box_offset = np.array([box[0], box[1]])
        coords[:, :2] = (coords[:, :2] + offset + box_offset) * scale[0]
        coords[:, 2:4] = (coords[:, 2:4] + offset + box_offset) * scale[1]

        # If coords has more than 6 columns, it contains tracking information
        # We remove that tracking information by deleting the id-column (at index 4)
        if coords.shape[1] > 6:
            coords = np.delete(coords, 4, axis=1)

        # Initialize the list of masks
        masks = []
        keypoints = []

        # If result contains keypoints, process them
        if result.keypoints is not None:
            #print(i, ": x1 y1 x1 y1 conf clid =",coords.shape, coords)
            #print(result.keypoints.shape)
            #print("orig_shape...=",result.orig_shape)

            keys = result.keypoints.cpu().data[..., :2]
            #print("keys=",keys.shape, result.keypoints.cpu())
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
                keypoints.append(keys) # bones * 3 elements

        #print("collected ", len(keypoints), keypoints)

        # If result contains masks, add them to the output
        if result.masks is not None:
            return coords, result.masks, keypoints

        return coords, masks, keypoints

    @staticmethod
    def calculate_memory(num_images, h, w, c, dtype):
        """
        Calculates the memory usage for a batch of images.

        Parameters:
        num_images (int): Number of images
        h (int): Height of each image
        w (int): Width of each image
        c (int): Number of color channels in each image
        dtype (torch.dtype): PyTorch datatype of the images

        Returns:
        int: The memory usage of the images in bytes
        """
        # Get the number of elements in each image tensor
        num_elements = h * w * c

        # Get the size of the datatype
        if dtype == torch.float32 or dtype == torch.int32:
            datatype_size = 4
        elif dtype == torch.float64 or dtype == torch.int64:
            datatype_size = 8
        elif dtype == torch.float16 or dtype == torch.int16:
            datatype_size = 2
        elif dtype == torch.uint8:
            datatype_size = 1
        else:
            raise ValueError("Unsupported datatype")

        # Calculate the total memory size
        total_memory = num_images * num_elements * datatype_size

        return int(total_memory)

    @staticmethod
    def get_free_memory(device):
        """
        Gets the amount of free memory on a device.

        Parameters:
        device (torch.device): a PyTorch device

        Returns:
        int: The free memory on the device in bytes
        """
        if "cuda" in device.type:
            torch.cuda.synchronize(device)
            total_memory = torch.cuda.get_device_properties(device).total_memory
            reserved_memory = torch.cuda.memory_reserved(device)
            free_memory = total_memory - reserved_memory
            return int(free_memory)
        else:
            global printed_warning
            if not printed_warning:
                print("Memory profiling is supported only for cuda devices, assuming 640x640x3x32x8 bytes")
                printed_warning = True
            return 640 * 640 * 3 * 32 * 8


    def inference(self, input, max_det = 1000, conf_threshold=0.1, iou_threshold=0.7) -> list[TRex.Result]:
        """
        Performs inference on the input images.

        Parameters:
        input (torch.Tensor): Input tensor of images
        max_det (int): Maximum number of detections
        conf_threshold (float): Confidence threshold for detection
        iou_threshold (float): Intersection over Union threshold for detection

        Returns:
        list[TRex.Result]: List of detection results
        """
        global receive

        offsets = input.offsets()
        offsets = np.array([(o.x, o.y) for o in offsets], dtype=np.float32)

        scales = input.scales()
        scales = np.array([(o.x, o.y) for o in scales], dtype=np.float32)

        offsets = np.reshape(offsets, (-1, 2))
        scales = np.reshape(scales, (-1, 2)).astype(np.float32)

        orig_id = np.array(input.orig_id(), copy=False, dtype=np.uint64)
        #print(f"Received orig_id = {orig_id}")

        im = input.images()
        tensor = self.preprocess(im)

        # if we have a box detection model, we can focus on parts of the image
        # based on the detection boxes by that model
        if self.has_region_model():
            return self.perform_region_proposal(tensor, offsets = offsets, scales = scales, ious = iou_threshold, confs = conf_threshold)

        rexsults = []

        # otherwise, we need to segment the entire image
        if len(tensor) == 0:
            return rexsults

        w, h, c = tensor[0].shape[1], tensor[0].shape[0], tensor[0].shape[2]
        #print("tensor[0].shape = ",tensor[0].shape)
        #print("resolution = ",self.detect_resolution())

        # get total memory of the gpu:
        total_memory = TRexYOLO.get_free_memory(self.detection_model().device) * 0.75

        # if the total memory is not enough, we need to send the images in packages
        normal_res = int(w * h)
        memory_per_image = TRexYOLO.calculate_memory(1, h, w, c, torch.float64)
        max_len = int(max(1, total_memory // memory_per_image // 100))
        #print(f"Calculated max_len = {max_len} based on total_memory = {total_memory} and normal_res = {normal_res} and {w}x{h} pixels / image, memory_per_image = {memory_per_image}")

        if len(tensor) > max_len:
            # send data in packages of X images
            #print(f"Sending images in packages of {X} images, total {len(tensor)} images, {w}x{h} pixels, based on normal_res = {normal_res}")
            #print(f"Sending images in packages of {max_len} images, total {len(tensor)} images, {w}x{h} pixels / image")

            results = []
            for i in range(0, len(tensor), max_len):
                #print(f"Sending images {i} to {i+max_len}")
                rs = self.detect_or_segment(images = tensor[i:i+max_len], 
                                            conf = conf_threshold, 
                                            iou = iou_threshold, 
                                            #offsets = offsets, 
                                            imgsz = self.detect_resolution(),
                                            classes=None, 
                                            agnostic_nms=True,
                                            verbose = False,
                                            max_det = max_det)
                for r, scale, offset in zip(rs, scales[i:i+max_len], offsets[i:i+max_len]):
                    results.append(StrippedYoloResults(r, scale=scale, offset=offset))
                #results.extend(rs)
                torch.cuda.empty_cache()

        else:
            #print(f"Sending all {len(tensor)} images at once, given {w}x{h} pixels, amounting to {len(tensor) * w * h} pixels")
            rs = self.detect_or_segment(images = tensor, 
                                        conf = conf_threshold, 
                                        iou = iou_threshold, 
                                        #offsets = offsets, 
                                        imgsz = self.detect_resolution(),
                                        classes=None, 
                                        agnostic_nms=True,
                                        verbose = False,
                                        max_det = max_det)
            results = []
            for r, scale, offset in zip(rs, scales, offsets):
                    results.append(StrippedYoloResults(r, scale=scale, offset=offset))
            #torch.cuda.empty_cache()

        from itertools import groupby

        # use groupby to group the list elements by id
        results = [[x[1] for x in group] for _, group in groupby(list(zip(orig_id, results)), lambda x: x[0])]
        offsets = [[x[1] for x in group] for _, group in groupby(list(zip(orig_id, offsets)), lambda x: x[0])]
        scales = [[x[1] for x in group] for _, group in groupby(list(zip(orig_id, scales)), lambda x: x[0])]
        #print(f"len(results) = {len(results)} len(offsets) = {len(offsets)} len(scales) = {len(scales)}")

        index = 0
        for i, (tiles, scale, offset) in enumerate(zip(results, scales, offsets)):
            coords = []
            masks = []
            keypoints = []
            for j, (tile, s, o) in enumerate(zip(tiles, scale, offset)):
                try:
                    c, m, k = self.postprocess_result(index, tile, offset = o, scale = s)
                    #print("c.shape= ",c.shape, " len(m)=", len(m))
                    coords.append(c)
                    masks.extend(m)
                    keypoints.extend(k)

                    #print("appended keypoints: ", len(keypoints), " at ", index, "with", tile)

                    #r = result.cpu().plot(img=im[i], line_width=1)
                    #TRex.imshow("result"+str(i), r)
                except Exception as e:
                    print("Exception when postprocessing result", e," at ",index, "with", tile)
                    #print("result.boxes.data.cpu().numpy() = ", result.boxes.data.cpu().numpy())
                    #r = tile.cpu().plot(img=im[i], line_width=1)
                    #TRex.imshow("result"+str(i), r)
                    #    raise e
                finally:
                    index += 1
            
            coords = np.concatenate(coords, axis=0)
            if len(keypoints) > 0:
                keypoints = np.concatenate(keypoints, axis=0, dtype=np.float32)
            
            rexsults.append(TRex.Result(index, TRex.Boxes(coords), masks, TRex.KeypointData(keypoints)))
        
        return rexsults

def load_yolo(configs : List[TRex.ModelConfig]):
    import torch
    TRex.log("Clearing caches...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    #torch.set_grad_enabled(False)

    global model
    models = []
    for config in configs:
        models.append(Model(config))
    
    print("Configs: ", models)
    model = TRexYOLO(models)
    TRex.log("Loaded YOLO models: "+str([model.config for model in model.models]))
    return [model.config for model in model.models]

def predict(input : TRex.YoloInput) -> List[TRex.Result]:
    global model
    conf_threshold = float(TRex.setting("detect_conf_threshold"))
    iou_threshold = float(TRex.setting("detect_iou_threshold"))

    return model.inference(input, 
                           conf_threshold = conf_threshold, 
                           iou_threshold = iou_threshold)

def apply():
    #from pyinstrument import Profiler

    #profiler = Profiler()
    #profiler.start()

    import time
    start = time.time()

    try:
        global model, image_size, segmentation_resolution, image, oimages, model_type, offsets, device
        if model_type == "yolo" or model_type == "yolo8" or model_type == "yolo8seg":
            #im = np.array(image, copy=False)[..., :3]
            model.inference(image, offsets=offsets, conf_threshold=conf_threshold, iou_threshold=iou_threshold)

        else:
            raise Exception("model_type was not set before running inference:")

    finally:
        #e = time.time()
        #profiler.stop()
        #profiler.print(show_all=True)
        pass
        #print("Took ", (e - start)*1000, "ms")
