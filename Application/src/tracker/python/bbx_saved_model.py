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

import TRex
from TRex import ModelTaskType

# Set up logging
logging.basicConfig(level=logging.INFO)

class LoggerWriter:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.message = ''  # Buffer for messages

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

def t_predict(t_model, device, image_size, offsets, im, conf_threshold = 0.25, iou_threshold = 0.1, mask_res = 56, max_det = 1000):
    def crop(masks, boxes):
        """
        "Crop" predicted masks by zeroing out everything not in the predicted bbox.
        Vectorized by Chong (thanks Chong).

        Args:
            - masks should be a size [h, w, n] tensor of masks
            - boxes should be a size [n, 4] tensor of bbox coords in relative point form
        """

        n, h, w = masks.shape
        #print(n,h,w)
        x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(1,1,n)
        #print(x1,y1,x2,y2)
        r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # rows shape(1,w,1)
        #print(r)
        c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # cols shape(h,1,1)
        #print(c)
        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

    if len(im.shape) < 4:
        im = im[np.newaxis, ...]

    print(image_size)
    if type(image_size) == np.ndarray or type(image_size) == list:
        image_size = int(image_size[0])

    if im.shape[1:] != (image_size,image_size,3):
        print("Image shape unexpected, got ", im.shape, " expected (:",",",image_size,",",image_size,",3)")
    assert im.shape[1:] == (image_size,image_size,3)
    im0 = im.shape[1:]
    im = im.transpose((0, 3, 1, 2))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)

    assert len(im.shape) == 4

    #dt = (Profile(), Profile(), Profile())
    global dt
    from utils.general import Profile
    if type(dt) == type(None):
        dt = (Profile(), Profile(), Profile())

    with dt[0]:
        im = torch.from_numpy(im).to(device)
        im = im.half() if t_model.fp16 else im.float()  # uint8 to fp16/32
        print(im.dtype)
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

    def clip_coords(boxes, shape):
        # Clip bounding xyxy bounding boxes to image shape (height, width)
        if isinstance(boxes, torch.Tensor):  # faster individually
            boxes[:, 0].clamp_(0, shape[1])  # x1
            boxes[:, 1].clamp_(0, shape[0])  # y1
            boxes[:, 2].clamp_(0, shape[1])  # x2
            boxes[:, 3].clamp_(0, shape[0])  # y2
        else:  # np.array (faster grouped)
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

    def scale_coords(img1_shape, coords, img0_shape):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding

        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, :4] /= gain
        clip_coords(coords, img0_shape)
        return coords

    def apply(pred, index, im, prot):
        meta = []
        indexes = []
        shapes = []
        ih, iw = im.shape[2:]

        c, mh, mw = prot.shape  # CHW

        downsampled_bboxes = pred[:, :4].clone()
        downsampled_bboxes[:, 0] *= mw / iw
        downsampled_bboxes[:, 2] *= mw / iw
        downsampled_bboxes[:, 3] *= mh / ih
        downsampled_bboxes[:, 1] *= mh / ih

        masks = (pred[:, 6:] @ prot.float().view(c, -1)).sigmoid().view(-1, mh, mw)  # CHW
        masks = crop(masks, downsampled_bboxes)

        pred[:, :4] = scale_coords(im.shape[2:], pred[:, :4], im0).round()

        confs = pred[:, 4]
        clids = pred[:, 5]

        #for i, (det, proto) in enumerate(zip(pred, prot)):
            

        for j in range(len(masks)):
            x = masks[j]
            
            box = downsampled_bboxes[j]
            x0,y0,x1,y1 = box.cpu().numpy().astype(int)

            if x1-x0 > 0 and y1-y0 > 0 and x1+1 <= x.shape[1] and y1+1 <= x.shape[0] and x0 >= 0 and y0 >= 0:
                x = x[y0:y1+1, x0:x1+1]
                x = (T.Resize((mask_res,mask_res))(x[None])[0])# * 255).to(torch.uint8)
                shapes.append(x.cpu().numpy())
                indexes.append(index)
                #if not x.shape[0] == 0 and not x.shape[1] == 0:
                #    shapes.append(cv2.resize(x, (56,56)))
                meta.append(pred[j, :6].to(torch.float32).cpu().numpy())
            else:
                print("image empty :(")

            #meta.append(det[:, :6].to(torch.float32).cpu().numpy())
        assert len(meta) == len(indexes)
        return meta, indexes, shapes

    results = []
    shapes = []
    meta = []
    indexes = []

    #print("processing im.shape", im.shape)
    #print(offsets)
    offsets = np.reshape(offsets, (-1, 2)).astype(int)
    #print(len(pred))
    #assert len(offsets) == len(pred)
    pred = None
    prediction = None
    proto = None

    from utils.general import non_max_suppression

    with dt[1]:
        preds, outs = t_model(im, augment=None, visualize=None)
        print(preds.shape, outs[1].shape, len(im))
    with dt[2]:
        preds = non_max_suppression(preds, conf_threshold, iou_threshold, None, True, max_det=max_det, nm=32)

    for i, pred, proto in zip(range(len(im)), preds, outs[1]):
        # Inference
        #local = (im[i:i+1].cpu().numpy() * 255).astype(np.uint8)[0, 0, ...]
        #print("local = ",local.shape, " ", local.dtype)
        #TRex.imshow("im"+str(len(im) - 1 - i), local)

        #with dt[1]:
            #pred, out = t_model(im[i][None], augment=None, visualize=None)
            #proto = out[1]

            #print(proto.shape)

            # NMS
            #with dt[2]:
            #conf_thres = 0.1
            #iou_thres = 0.0

        #with dt[2]:
        #    pred = [a.to(device) for a in non_max_suppression(pred[None], conf_threshold, iou_threshold, None, True, max_det=max_det, nm=32)]
            #print("nonmaxsupp proc", (pred[0]).int().cpu().numpy())
        _meta, _index, _shapes = apply(pred, index=i, im=im[i:i+1], prot = proto)
            #print("RESULT FOR ",i,"=",_meta)
        meta += _meta
        shapes += _shapes
        indexes += np.repeat(len(im) - 1 - i, len(_meta)).tolist()

    #print("meta: ", meta)
    #print("meta: ", np.concatenate(meta, axis=0))
    if len(meta) > 0:
        meta = np.concatenate(meta, axis=0, dtype=np.float32)
    else:
        meta = np.array([], dtype=np.float32)

    return np.array(shapes, dtype=np.float32).flatten(), meta.flatten(), np.array(indexes, dtype=int)

'''if torch.backends.mps.is_available():
    device = torch.device("mps")
    #x = torch.ones(1, device=mps_device)
    #print (x)
else:
    print ("MPS device not found.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")'''

t_model = None

hyp = {
    "mask_resolution": 56,
    "attn_resolution": 14,
    "num_base": 5
}

# import the necessary packages
import numpy as np
# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return [],[]
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    #if boxes.dtype.kind == "i":
    #    boxes = boxes.astype("float")
    # initialize the list of picked indexes 
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return pick, boxes[pick].astype(int)


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
    def __init__(self, config, use_tracking = False):
        """
        Initializes a Model object.

        Args:
            config (ModelConfig): An instance of the ModelConfig C++ class.
        """
        assert isinstance(config, TRex.ModelConfig)
        self.config = config
        self.ptr = None
        self.use_tracking = use_tracking

        # if no device is specified, use cuda if available, otherwise use mps/cpu
        self.device = eval(TRex.setting("gpu_torch_device"))
        if self.device == "":
            self.device = None
        else:
            print(f"Using device {self.device} from settings.")
            self.device = torch.device(self.device)

        if self.device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            elif torch.backends.mps.is_available():
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
        self.ptr = YOLO(self.config.model_path)

        device_from_settings = eval(TRex.setting("gpu_torch_device"))
        if device_from_settings != "":
            print(f"Using device {device_from_settings} from settings.")
            self.device = torch.device(device_from_settings)

        if self.config.task == ModelTaskType.detect or self.config.task == ModelTaskType.segment:
            if self.ptr.task == "segment":
                self.config.task = ModelTaskType.segment
                if self.device == torch.device("mps") and device_from_settings == "":
                    self.device = torch.device("cpu")
            elif self.ptr.task == "detect":
                self.config.task = ModelTaskType.detect

        self.ptr.to(self.device)
        self.ptr.fuse()
        TRex.log("Loaded model: {}".format(self))

    def predict(self, images : List[np.ndarray], **kwargs):
        if len(images) == 0:
            return []
        
        if self.use_tracking:
            results = []
            for image in images:
                results.append(self.ptr.track(image, persist=True, device=self.device, **kwargs)[0])
            return results
        else:
            return self.ptr.predict(images, device=self.device, **kwargs)

class StrippedYolo8Results:
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
            for orig, unscale, k in zip(coords.round().astype(int), unscaled, (results.masks.data * 255).byte()):
                sub = k[max(0, int(unscale[1])):max(0, int(unscale[3])), max(0,int(unscale[0])):max(0, int(unscale[2]))]
                if orig[3] - orig[1] <= 0 or orig[2] - orig[0] <= 0 or sub.shape[0] <= 0 or sub.shape[1] <= 0:
                    print(f"WARNING: invalid mask size: orig={orig[3] - orig[1]}x{orig[2] - orig[0]} \n\
                          => sub={sub.shape[0]}x{sub.shape[1]} \n\
                          => unscale={unscale} \n\
                          => k={k.shape}\n\
                          => orig={orig}")
                    raise Exception("Invalid mask size")
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

class TRexYOLO8:
    def __init__(self, models: List[Model]):
        """
        Initialize the TRexYOLO8 class with a list of models.

        Args:
            models (List[Model]): A list of models used for region proposal, detection and segmentation.

        Raises:
            AssertionError: If no models are specified.
        """
        assert len(models) > 0, "No models specified for TRexYOLO8 {}".format(models)

        self.models = models
        self.load_models()

        self.boxes_time = 0
        self.boxes_samples = 0
        self.results_time = 0
        self.results_samples = 0

        # log configuration and loaded models
        TRex.log("TRexYOLO8 configuration: models={}".format(self.models))

    def __str__(self) -> str:
        """
        String representation of the TRexYOLO8 instance.

        Returns:
            str: A string that represents the TRexYOLO8 instance.
        """
        return "TRexYOLO8<models={}>".format(self.models)

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

    def has_segment_model(self) -> bool:
        """
        Check if the list of models contains a segmentation model.

        Returns:
            bool: True if a segmentation model is found, False otherwise.
        """
        return any(model.task == ModelTaskType.segment for model in self.models)

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
        
        from ultralytics.yolo.utils.ops import scale_coords
        # predict bounding boxes of areas of interest
        model = next((model for model in self.models if model.task == ModelTaskType.region), None)
        assert model is not None
        
        scaled_size: int = model.config.trained_resolution
        scaled_down_scales: np.ndarray = np.array([(scaled_size / im.shape[1], int(im.shape[0] / im.shape[1] * scaled_size) / im.shape[0]) for im in images])
        scaled_down: List[Image] = [cv2.resize(im, (scaled_size, int(im.shape[0] / im.shape[1] * scaled_size))) for im in images]
        #print(f"performing region proposals at {scaled_size}x{scaled_size} on {len(images)} images")
        bboxes: list = model.predict(images = scaled_down, imgsz=scaled_size, conf=0.25, iou=0.3, verbose=False, **kwargs)
        padding: int = 7
        results: List[List[Tuple[BBox, Image]]] = []
        
        #for i, bb in enumerate(bboxes):
        #    p = bb.cpu().plot(line_width=1)
        #    TRex.imshow("region proposals"+str(i), p)

        # return a list per input image, with a list of bounding boxes (and their image regions) per image
        for i, bb in enumerate([bb.boxes.xyxy.cpu().numpy() for bb in bboxes]):
            mois = []
            for box in bb:
                box[:2] /= scaled_down_scales[i]
                box[2:4] /= scaled_down_scales[i]

                if padding > 0:
                    box[:2] -= np.array([padding,padding])
                    box[2:4] += np.array([padding,padding])

                box = box.astype(int)
                h, w = images[i].shape[:2]

                x0 = int(max(0, box[0]))
                y0 = int(max(0, box[1]))
                x1 = min(w, max(x0, box[2]))
                y1 = min(h, max(y0, box[3]))

                box = np.array([x0,y0,x1,y1])

                oi = images[i][y0:y1, x0:x1]
                if oi.shape[0] == 0 or oi.shape[1] == 0:
                    print("region is zero ",i, oi.shape, box)
                    continue

                mois.append((box, oi))

            results.append(mois)

        return results
    
    def detection_resolution(self) -> int:
        """
        Retrieves the trained resolution of the detection or segmentation model.

        Returns:
            int: The trained resolution of the detection or segmentation model.

        Raises:
            Exception: If no detect or segment model is found.
        """
        detect_model = next((model for model in self.models if model.task == ModelTaskType.detect or model.task == ModelTaskType.segment), None)
        if detect_model is not None:
            return detect_model.config.trained_resolution
        raise Exception("No detect or segment model found")

    def detection_model(self) -> Model:
        """
        Retrieves the detection or segmentation model.

        Returns:
            Model: The detection or segmentation model.

        Raises:
            Exception: If no detect or segment model is found.
        """
        detect_model = next((model for model in self.models if model.task == ModelTaskType.detect or model.task == ModelTaskType.segment), None)
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
        detect_model = next((model for model in self.models if model.task == ModelTaskType.detect or model.task == ModelTaskType.segment), None)
        if detect_model is not None:
            return detect_model.predict(**kwargs)
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
        proposal = self.region_proposal(tensor)

        rexsults = []
        all_images = []
        # For each region in the proposal, add the region to all_images
        for regions in proposal:
            all_images.extend([region for _, region in regions])

        # Perform detection or segmentation on the aggregated regions
        results = self.detect_or_segment(
            images = all_images, 
            conf = confs, 
            iou = ious, 
            imgsz = self.detection_resolution(),
            #imgsz=160,
            classes=None, 
            agnostic_nms=True,
            verbose = False)

        # Check if the number of results is equal to the number of images
        assert len(results) == len(all_images),f"length of results {len(results)} is not equal to length of all images {len(all_images)}"

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
                result = StrippedYolo8Results(result, scale=scale, offset=offset)
                #r = result.cpu().plot(img=region, line_width=1)
                #TRex.imshow("segmentation"+str(i)+","+str(t), r)
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

            # Append the result to the list of results
            rexsults.append(TRex.Result(i, TRex.Boxes(collected_boxes), collected_masks, TRex.KeypointData(collected_keypoints)))

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
                #keys[..., 0] = (keys[..., 0] + offset[0] + box_offset[0]) * scale[0]
                #keys[..., 1] = (keys[..., 1] + offset[1] + box_offset[1]) * scale[1]
                keys[..., 0] = (keys[..., 0] * scale[0]) #+ coords[..., 0]).T
                keys[..., 1] = (keys[..., 1] * scale[1]) #+ coords[..., 1]).T
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
        #print("resolution = ",self.detection_resolution())

        # get total memory of the gpu:
        total_memory = TRexYOLO8.get_free_memory(self.detection_model().device) * 0.75

        # if the total memory is not enough, we need to send the images in packages
        normal_res = int(w * h)
        memory_per_image = TRexYOLO8.calculate_memory(1, h, w, c, torch.float64)
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
                                            imgsz = self.detection_resolution(),
                                            classes=None, 
                                            agnostic_nms=True,
                                            verbose = False,
                                            max_det = max_det)
                for r, scale, offset in zip(rs, scales[i:i+max_len], offsets[i:i+max_len]):
                    results.append(StrippedYolo8Results(r, scale=scale, offset=offset))
                #results.extend(rs)
                torch.cuda.empty_cache()

        else:
            #print(f"Sending all {len(tensor)} images at once, given {w}x{h} pixels, amounting to {len(tensor) * w * h} pixels")
            rs = self.detect_or_segment(images = tensor, 
                                        conf = conf_threshold, 
                                        iou = iou_threshold, 
                                        #offsets = offsets, 
                                        imgsz = self.detection_resolution(),
                                        classes=None, 
                                        agnostic_nms=True,
                                        verbose = False,
                                        max_det = max_det)
            results = []
            for r, scale, offset in zip(rs, scales, offsets):
                    results.append(StrippedYolo8Results(r, scale=scale, offset=offset))
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

def load_yolo8(configs : List[TRex.ModelConfig]):
    global model
    models = []
    for config in configs:
        models.append(Model(config))
    
    print("Loaded models ", models)
    model = TRexYOLO8(models)

def load_model():
    global model, model_path, segmentation_path, region_path, image_size, t_model, imgsz, WEIGHTS_PATH, device, model_type, t_predict, q_model
    print("loading model type", model_type)
    if model_type == "yolo8seg" or model_type == "yolo8":
        raise Exception("Please use the load_yolo8 method.")
    elif model_type == "yolo7seg":
        from models.common import DetectMultiBackend
        if torch.backends.mps.is_available():
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        t_model = DetectMultiBackend(model_path, device=device, dnn=False, fp16=False)
        imgsz = tuple(np.array(image_size).astype(int))
        stride, names, pt = t_model.stride, t_model.names, t_model.pt

        #from utils.general import check_img_size
        #imgsz = check_img_size(imgsz, s=stride)
        t_model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))
        print("Loaded and warmed up")

    elif model_type == "customseg":
        if torch.backends.mps.is_available():
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        import pickle
        import cloudpickle
        with open(model_path, "rb") as f:
            content = pickle.load(f)

        # loading a dictionary with both "model" and a "predict" function in it
        # the predict function is supposed to have the following definition:
        #    predict(t_model, device, image_size, offsets, im)
        assert "predict" in content
        assert "model" in content

        t_model = content["model"].to(device)
        t_predict = content["predict"]

    else:
        image_size = np.array(image_size).astype(int)
        print("loading tensorflow model")
        import tensorflow as tf
        model = tf.saved_model.load(model_path)
        full_model = tf.function(lambda x: model(images=x))
        full_model = full_model.get_concrete_function(
            tf.TensorSpec((None, 3, image_size[1], image_size[0]), tf.float32))
        # Get frozen ConcreteFunction

        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
        frozen_func = convert_variables_to_constants_v2(full_model)
        frozen_func.graph.as_graph_def()
        model = frozen_func

        from tensorflow.keras import backend
        backend.clear_session()
        print("loaded ", model_path)

        ## load additional segmentation model
        ## to generate outlines from masked image portions (64x64px)
        if torch.backends.mps.is_available():
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        '''import pickle
        with open("Z:/work/shark/models/shark-cropped-0.712-loops-128-seg-windows.pth", "rb") as f:
            content = pickle.load(f)

        # loading a dictionary with both "model" and a "predict" function in it
        # the predict function is supposed to have the following definition:
        #    predict(t_model, device, image_size, offsets, im)
        assert "predict" in content
        assert "model" in content

        t_model = content["model"].to(device)
        t_predict = content["predict"]'''
        if not type(segmentation_path) == type(None) and os.path.exists(segmentation_path):
            from models.common import DetectMultiBackend
            t_model = DetectMultiBackend(segmentation_path, device=device, dnn=True, fp16=False)

            #t_model.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            #torch.quantization.prepare(t_model.model, inplace=True)
            #torch.quantization.convert(t_model.model, inplace=True)
            #q_model = torch.quantization.quantize_dynamic(t_model.model, {torch.nn.Conv2d, torch.nn.Linear}, dtype=torch.qint8)

            #t_predict = predict_custom_yolo7_seg
        else:
            t_model = None

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes -= [pad[0], pad[1], pad[0], pad[1]]
    #boxes[..., [0, 2]] -= pad[0]  # x padding
    #boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes /= gain
    #boxes = boxes.numpy()
    #boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes#.astype(int)
def clip_boxes(boxes, shape):
    #tf.clip_by_value(boxes, [[0], [shape[1]]], [[0], [shape[0]]])
    #boxes.clip(0, shape)
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2
    #boxes[..., 0] = tf.clip_by_value(boxes[..., 0], 0, shape[1])
    #boxes[..., 1] = tf.clip_by_value(boxes[..., 1], 0, shape[0])
    #boxes[..., 2] = tf.clip_by_value(boxes[..., 2], 0, shape[1])
    #boxes[..., 3] = tf.clip_by_value(boxes[..., 3], 0, shape[0])

def inference(model, im, size=(640,640)):
    im0 = im.shape
    import tensorflow as tf
    im = tf.image.resize(im, size)

    def _xywh2xyxy(xywh):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        x, y, w, h = tf.split(xywh, num_or_size_splits=4, axis=-1)
        return tf.concat([x - w / 2, y - h / 2, x + w / 2, y + h / 2], axis=-1)

    topk_per_class=100
    topk_all=100
    iou_thres=0.45
    conf_thres=0.45

    b, h, w, ch = im.shape  # batch, channel, height, width
    #y = model(np.zeros((1, 640, 640, 3), dtype=float))
    y = model(im / 255.0)[0]
    #return y
    #y = [x if isinstance(x, np.ndarray) else x.numpy() for x in y]
    #  # xywh normalized to pixels
    
    results = []
    boxes = _xywh2xyxy(y[..., :4])
    boxes *= [w, h, w, h]

    probs = y[:, :, 4:5]
    classes = y[:, :, 5:]
    scores = probs * classes

    #boxes = tf.expand_dims(boxes, 2)
    #nms = tf.image.combined_non_max_suppression(boxes,
    #                                            scores,
    #                                            topk_per_class,
    #                                            topk_all,
    #                                            iou_thres,
    #                                            conf_thres,
    #                                            clip_boxes=False)

    for i, det in enumerate(boxes):
        #print(det.shape)
        print("in loop", i, " - ",det.shape)
        nms = np.array(non_max_suppression_fast(det, iou_thres))

        #if len(nms.shape) > 1:
        if len(nms) > 0:
            #det = det.numpy()
            b = tf.round(scale_boxes(im.shape[1:3], nms[:, :4], im.shape[1:3]))
            for *xyxy, conf, cls in reversed(nms):
                for i in range(len(xyxy)):
                    xy = xyxy[i]
                    if not type(xy) is np.ndarray:
                        continue
                    ratio = (im0[2] / im.shape[2], im0[1] / im.shape[1])
                    pt0 = np.array((xy[0] * ratio[0], xy[1] * ratio[1])).astype(int)
                    pt1 = np.array((np.array((xy[2] * ratio[0], xy[3] * ratio[1])))).astype(int)
                    if xy[2] > 0:
                        results.append((pt0, pt1))

    return np.array(results, dtype=int)

def predict_yolov7(offsets, img, image_shape=(640,640)):
    #from utils.augmentations import augment_hsv, copy_paste, letterbox
    import tensorflow as tf

    def perform_filtering(im0, im, y):
        global iou_threshold, conf_threshold

        def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
            # Rescale boxes (xyxy) from img1_shape to img0_shape
            if ratio_pad is None:  # calculate from img0_shape
                gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
                pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
            else:
                gain = ratio_pad[0][0]
                pad = ratio_pad[1]

            boxes -= [pad[0], pad[1], pad[0], pad[1]]
            #boxes[..., [0, 2]] -= pad[0]  # x padding
            #boxes[..., [1, 3]] -= pad[1]  # y padding
            boxes /= gain
            #boxes = boxes.numpy()
            #boxes[..., :4] /= gain
            clip_boxes(boxes, img0_shape)
            return boxes#.astype(int)
        def clip_boxes(boxes, shape):
            #tf.clip_by_value(boxes, [[0], [shape[1]]], [[0], [shape[0]]])
            #boxes.clip(0, shape)
            boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
            boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2
            #boxes[..., 0] = tf.clip_by_value(boxes[..., 0], 0, shape[1])
            #boxes[..., 1] = tf.clip_by_value(boxes[..., 1], 0, shape[0])
            #boxes[..., 2] = tf.clip_by_value(boxes[..., 2], 0, shape[1])
            #boxes[..., 3] = tf.clip_by_value(boxes[..., 3], 0, shape[0])
        def _xywh2xyxy(xywh):
            # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
            x, y, w, h = tf.split(xywh, num_or_size_splits=4, axis=-1)
            return tf.concat([x - w / 2, y - h / 2, x + w / 2, y + h / 2], axis=-1)

        topk_per_class=200
        topk_all=200

        b, ch, h, w = im.shape
        #y = np.concatenate((y, y), axis=0)

        if False:
            all_results = []
            #y = y.numpy()
            boxes = _xywh2xyxy(y[..., :4])
            ratio = (im0[2] / im.shape[-1], im0[1] / im.shape[-2]) * 2

            #nmsed_boxes = tf.math.multiply(nms.nmsed_boxes, ratio)
            #nboxes = nmsed_boxes.numpy()
            #nscores = nms.nmsed_scores.numpy()
            #nclasses = nms.nmsed_classes.numpy()

            for i in range(len(y)):        
                classes = y[i, ..., 4:5] * y[i, ..., 5:]
                max_probs = tf.reduce_max(classes, axis=-1)
                mask = max_probs >= conf_threshold

                Y = y[i][mask].numpy()
                nms_boxes = boxes[i][mask].numpy()
                nms, nms_boxes = non_max_suppression_fast(nms_boxes, iou_threshold)
                if len(nms) == 0:
                    all_results += [np.array([], np.float32)]
                    continue;

                Y = Y[nms]

                nms_boxes = np.multiply(nms_boxes, ratio)
                nms_scores = np.max(Y[..., 5:], axis=-1)
                nms_classes = np.argmax(Y[..., 5:], axis=-1)

                #nms_boxes = nboxes[i, :nms_valid[i]]
                #nms_scores = nscores[i, :nms_valid[i]]
                #nms_classes = nclasses[i, :nms_valid[i]]

                #print("\tclasses:", nms_classes)
                #print("\nms_scores:", nms_scores.shape)
                #print(len(nms_boxes), len(nms_scores), len(nms_classes))

                results = []
                for xy, score, clid in zip(nms_boxes, nms_scores, nms_classes):
                    #pt0 = np.array((xy[0] * ratio[0], xy[1] * ratio[1])).astype(int)
                    #pt1 = np.array((np.array((xy[2] * ratio[0], xy[3] * ratio[1])))).astype(int)
                    pt0 = np.array((xy[0] , xy[1] )).astype(int)
                    pt1 = np.array((np.array((xy[2], xy[3] )))).astype(int)
                    #print(score, clid, pt0, pt1)
                    results.append((score, clid, pt0[0], pt0[1], pt1[0], pt1[1]))

                all_results += [np.array(results, dtype=np.float32)]

            return [len(i) for i in all_results], all_results

        #y = y.numpy()
        boxes = _xywh2xyxy(y[..., :4])
        #boxes = tf.expand_dims(boxes, 2)

        probs = y[..., 4:5]
        classes = y[..., 5:]
        scores = probs * classes
        #print("y=",y.shape)
        
        #nms = tf.image.combined_non_max_suppression(boxes,
        #                                            scores,
        #                                            topk_per_class,
        #                                            topk_all,
        #                                            iou_threshold,
        #                                            conf_threshold,
        #                                            clip_boxes=False)

        #nms_valid = nms.valid_detections.numpy()
        ratio = (im0[2] / im.shape[-1], im0[1] / im.shape[-2]) * 2
        ratio = tf.constant(ratio, dtype=tf.float32)
        #print(ratio, nms.nmsed_boxes.shape)

        all_results = []

        nmsed_boxes = tf.math.multiply(boxes, ratio)
        nboxes = nmsed_boxes#.numpy()
        nscores = scores#.numpy()
        nclasses = classes#.numpy()
        #nscores = nms.nmsed_scores.numpy()
        #nclasses = nms.nmsed_classes.numpy()

        for i in range(len(y)):        
            #classes = y[i, ..., 4:5] * y[i, ..., 5:]
            #max_probs = tf.reduce_max(classes, axis=-1)
            #mask = max_probs >= conf_threshold

            #Y = y[i][mask].numpy()
            #nms_boxes = boxes[i][mask].numpy()

            #nms, nms_boxes = non_max_suppression_fast(nms_boxes, iou_threshold)
            #if len(nms) == 0:
            #    all_results += [np.array([], np.float32)]
            #    continue;

            #Y = Y[nms]

            #nms_boxes = np.multiply(nms_boxes, ratio) #, :nms_valid[i]]
            #nms_scores = np.max(Y[..., 5:], axis=-1)#, :nms_valid[i]]
            #nms_classes = np.argmax(Y[..., 5:], axis=-1)#nclasses[i]#, :nms_valid[i]]

            #nms_boxes = nboxes[i, :nms_valid[i]]
            #nms_scores = nscores[i, :nms_valid[i]]
            #nms_classes = nclasses[i, :nms_valid[i]]

            #print(boxes[i:i+1].shape, scores[i:i+1].shape, probs[i].shape, scores[i].shape, np.max(scores[i], axis=-1).shape)
            s = tf.reduce_max(scores[i], axis=-1)
            nms = tf.image.non_max_suppression(
                boxes[i],
                s,
                topk_all,
                iou_threshold=iou_threshold,
                score_threshold=conf_threshold)
            #print("nms",nms.shape, nms.dtype)
            #print("boxes",boxes[i].shape)
            #print("nboxes",nboxes[i, nms].shape)
            #print("classes",nclasses[i, nms].shape)

            nms_scores = tf.gather(s,nms).numpy()
            nms_boxes = tf.gather(boxes[i],nms).numpy()
            nms_classes = np.argmax(tf.gather(scores[i], nms).numpy(), axis=-1)
            #nms_scores = np.max(nms_scores, axis=-1)
            #nms_classes = nclasses[i, nms]
            #nms_scores = probs[i, ..., 0][nms]

            #print("\tclasses:", nms_classes.shape, nms_classes)
            #print("\nms_scores:", nms_scores.shape, nms_scores)
            #print(len(nms_boxes), len(nms_scores), len(nms_classes))

            results = []
            for xy, score, clid in zip(nms_boxes, nms_scores, nms_classes):
                #pt0 = np.array((xy[0] * ratio[0], xy[1] * ratio[1])).astype(int)
                #pt1 = np.array((np.array((xy[2] * ratio[0], xy[3] * ratio[1])))).astype(int)
                pt0 = np.array((xy[0] , xy[1] )).astype(int)
                pt1 = np.array((np.array((xy[2], xy[3] )))).astype(int)
                #print(score, clid, pt0, pt1)
                results.append((score, clid, pt0[0], pt0[1], pt1[0], pt1[1]))

            all_results += [np.array(results, dtype=np.float32)]

        #print("all_results:", np.shape(all_results))
        return [len(i) for i in all_results], all_results

    #img = (images[0].copy() * 255).astype(np.uint8)
    def transform_image(img, image_shape):
        shape = img.shape[1:3]
        stride = 32
        #img, ratio, dwdh = letterbox(img, new_shape=image_shape, auto=False)
        ratio = min(image_shape[0] / shape[0], image_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * ratio)), int(round(shape[0] * ratio))
        
        dw, dh = image_shape[1] - new_unpad[0], image_shape[0] - new_unpad[1]  # wh padding
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        img = img.transpose(0, 3, 1, 2)#[np.newaxis, ...]
        #img = img.astype(np.float32) / 255.0
        
        return tf.constant(img.astype(np.float32) / 255.0, dtype=tf.float32), ratio, (dw, dh)
    
    #print("prelim shape", img.shape)
    if len(img.shape) < 4:
        img = img[np.newaxis, ...]
    if np.min(img.shape) == 0:
        print("WARNING: Shape of image is zero: ", img.shape)
        return [], np.array([], dtype=np.float32)

    im, ratio, dwdh = transform_image(img, image_shape=image_shape)
    #print("final shape", im.shape)
    output_data = model(im)[0]
    offsets = np.reshape(offsets, (-1, 2))
    #print(offsets)

    Ns, R = perform_filtering(img.shape, im, output_data)
    _Ns = []
    rs = []
    for i in range(len(output_data)):
        r = R[i]
        N = 0
        if len(np.shape(r)) == 2:
            #print(r)
            r[:, 2:4] += offsets[i]
            r[:, 4:6] += offsets[i]
            #print("->",r)
            N += len(r)
            rs.append(r)
        _Ns.append(N)
        #else:
            #print("empty:",r)

    _Ns = np.array(_Ns, dtype=np.uint64)
    #print("Returning ", _Ns)
    if len(rs) > 0:
        #print("RS:",np.concatenate(rs, axis=0).shape)
        return _Ns, np.concatenate(rs, axis=0)
    return [], np.array([], dtype=np.float32)

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
        global model, image_size, segmentation_resolution, receive, image, oimages, model_type, offsets, t_predict, t_model, device
        if model_type == "yolo5":
            import tensorflow as tf
            image = tf.constant(np.array(image, copy=False)[..., :3], dtype=tf.uint8)
            #print(image)
            results = inference(model, image, size=image_size)
            #print(np.array(results, dtype=int).flatten())
            receive(np.array(results, dtype=int).flatten())

        elif model_type == "customseg":
            im = np.array(image, copy=False)[..., :3]
            print(im.shape, hyp)
            results = t_predict(
                t_model = t_model, 
                device = device, 
                image_size = int(image_size[0]), 
                offsets = offsets, 
                im = im, 
                conf_threshold = conf_threshold, 
                iou_threshold = iou_threshold,
                mask_res = hyp['mask_resolution'])
            print("sending: ", results[0].shape, results[1])
            receive(results[0], results[1], results[2])

        elif model_type == "yolo7seg":
            im = np.array(image, copy=False)[..., :3]
            results = predict_custom_yolo7_seg(offsets, im)
            print("sending: ", results[0].shape, results[1])
            receive(results[0], results[1], results[2])

        elif model_type == "yolo8" or model_type == "yolo8seg":
            #im = np.array(image, copy=False)[..., :3]
            model.inference(image, offsets=offsets, conf_threshold=conf_threshold, iou_threshold=iou_threshold)

        elif model_type == "yolo7":
            #profiler = Profiler()
            #profiler.start()
            try:
                #im = tf.convert_to_tensor(np.array(image, copy=False)[..., :3], dtype=tf.float32)
                im = np.array(image, copy=False)[..., :3]
                oim = np.array(oimages, copy=False)
                assert len(im) == len(oim)
                #print("shape: ", im.shape, " image_size=",image_size)
                #print(np.shape(offsets))
                #results = predict_custom_yolo7_seg(im)
                #print("sending: ", results[0].shape, results[1])

                #receive_seg(results[0], results[1])
                #s0 = time.time()
                Ns, results = predict_yolov7(offsets, im, image_shape=image_size)
                #e0 = time.time()

                ## apply additional segmentation
                if type(t_model) != type(None):
                    #print("applying segmentation to", Ns.shape, results.shape, im.shape)
                    index = 0
                    offset_percentage = 0.1

                    # per box, so has to be segmented wrt original images
                    image_indexes = []
                    scales = []
                    distorted_boxes = []
                    subs = []

                    for image_index, (img, resized, N) in enumerate(zip(oim, im, Ns)):
                        # rw, rh
                        ratio = np.array((img.shape[1] / resized.shape[1], img.shape[0] / resized.shape[0]) * 2)

                        s = results[int(index):int(index+N)]#.copy()
                        index += N

                        assert N == len(s)
                        boxes = s[..., 2:] * ratio
                        clid = s[..., 1]
                        print(img.shape, s.shape, N, boxes.shape, clid)

                        for (x0,y0,x1,y1), c in zip(boxes, clid):
                            # filter for class ID to only generate relevant outlines
                            #if c != 1:
                            #    continue

                            w = max(0, x1 - x0) + 1
                            h = max(0, y1 - y0) + 1

                            woff = w * offset_percentage
                            hoff = h * offset_percentage

                            dx0 = int(max(0, x0 - woff))
                            dy0 = int(max(0, y0 - hoff))

                            dx1 = int(x1 + np.abs(x0 - dx0) * 2)
                            dy1 = int(y1 + np.abs(y0 - dy0) * 2)

                            sub = img[dy0:dy1, dx0:dx1, :]

                            #print(sub.shape, x0, y0, x1, y1, " -> ",  dx1, dy1, "increased by", (dx1-dx0) - w, ",",(dy1-dy0) - h)

                            if np.min(sub.shape[:2]) < 10:
                                continue

                            distorted_boxes.append([dx0, dy0, dx0 + sub.shape[1], dy0 + sub.shape[0]])
                            #import TRex
                            #TRex.imshow("mask1",np.ascontiguousarray(sub.astype(np.uint8)))
                            subs.append(cv2.resize(sub, (segmentation_resolution,segmentation_resolution)))
                            
                            image_indexes.append(image_index)
                            scales.append((sub.shape[1] / subs[-1].shape[1], sub.shape[0] / subs[-1].shape[0]) * 2)

                    rs = None
                    if len(scales) > 0:
                        scales = np.array(scales)
                        distorted_boxes = np.array(distorted_boxes)
                        image_indexes = np.array(image_indexes)

                        rs = t_predict(t_model = t_model, 
                                      device = device, 
                                      image_size = segmentation_resolution, 
                                      offsets = offsets, 
                                      im = np.array(subs), 
                                      conf_threshold = conf_threshold,
                                      iou_threshold = iou_threshold,
                                      mask_res = hyp['mask_resolution'],
                                      max_det = 1)

                        shapes = rs[0].reshape((-1, 56, 56))
                        meta = rs[1].reshape((-1, 6))
                        indexes = rs[2]

                        meta[..., :4] = meta[..., :4] * scales[rs[2]]
                        meta[..., :2] += distorted_boxes[rs[2]][..., :2]
                        meta[..., 2:4] += distorted_boxes[rs[2]][..., :2]

                        sizes = meta.astype(int)[..., [2,3]] - meta.astype(int)[..., [0,1]]
                        deformed =  (shapes * 255).astype(np.uint8)
                        masks = []
                        last_index = None

                        ons = {}
                        oindexes = image_indexes[indexes]
                        for index in range(0, len(oim)):
                            ons[index] = 0

                        for index in oindexes:
                            ons[index] += 1
                        segNs = []
                        for index in range(0, len(oim)):
                            if index in ons:
                                segNs.append(ons[index])
                            else:
                                segNs.append(0)
                        segNs = np.array(segNs).astype(int)

                        rs = [
                            shapes.copy().flatten(), 
                            meta.copy().flatten(), 
                            oindexes.astype(int).copy(), 
                            segNs # indexed like the original images
                        ]

                        if False:
                            print(shapes.shape, " shapes and ", meta.shape, " meta are transformed into ", rs[0].shape, " and ", rs[1].shape, " and ", rs[-1].shape)

                            for i, shape, m, s, d in zip(indexes, shapes, meta, sizes, deformed):
                                image_index = image_indexes[i]
                                img = oim[image_index]
                                print("image",image_indexes[i]," with shape ", shape.shape," and meta ", m, s, " img ", img.shape)

                                crop = img[int(m[..., 1]):int(m[..., 3]), int(m[..., 0]):int(m[..., 2]), :]
                                undistorted = (cv2.resize(d, tuple(s.T))).astype(np.uint8)

                                crop[..., 1] = np.where(
                                    undistorted[:crop.shape[0], :crop.shape[1]] > 153, 
                                    undistorted[:crop.shape[0], :crop.shape[1]], 
                                    crop[..., 0])

                                undistorted[undistorted < 100] = 0
                                contours, _ = cv2.findContours(undistorted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                cv2.drawContours(crop, contours, -1, (0, 255, 0), 1)  # Green color for contours

                                #print("masks = ", len(masks))
                                if image_index != last_index:
                                    if len(masks) > 0:
                                        pass
                                        #import TRex
                                        #for i in range(0, min(len(masks), 10)):
                                        #    if masks[i].shape[0] < 128:
                                        #       TRex.imshow("mask"+str(i),cv2.resize(np.ascontiguousarray(masks[i].astype(np.uint8)), (128,128)))
                                        #    else:
                                        #        TRex.imshow("mask"+str(i),np.ascontiguousarray(masks[i].astype(np.uint8)))
                                        #TRex.imshow("whole",np.ascontiguousarray(oim[last_index].astype(np.uint8)))
                                    last_index = image_index
                                    masks = []


                                if crop.shape[0] > 0 and crop.shape[1] > 0:
                                    masks.append(crop)

                    if type(rs) != type(None):
                        receive_with_seg(Ns, np.array(results, dtype=np.float32).flatten(), rs[0], rs[1], rs[2], rs[3])
                    else:
                        receive(Ns, np.array(results, dtype=np.float32).flatten())
                else:
                    receive(Ns, np.array(results, dtype=np.float32).flatten())
                #multi = 10
                #d0, d1 = np.concatenate((offsets, )*multi, axis=0), np.concatenate((im, )*multi, axis=0)
                #s1 = time.time()

                #predict_yolov7(d0, d1, image_shape=(image_size,image_size))

                #e1 = time.time()

                #print("******",(e0-s0)*1000,"ms im.shape=", im.shape)

            finally:
                #profiler.stop()
                #profiler.print(show_all=True)
                pass

        else:
            raise Exception("model_type was not set before running inference:")

    finally:
        #e = time.time()
        #profiler.stop()
        #profiler.print(show_all=True)
        pass
        #print("Took ", (e - start)*1000, "ms")


def predict_custom_yolo7_seg(offsets, im, mask_res=56):
    global t_model, device, conf_threshold, iou_threshold

    from detectron2.modeling.poolers import ROIPooler
    from detectron2.structures import Boxes

    def crop(masks, boxes):
        """
        "Crop" predicted masks by zeroing out everything not in the predicted bbox.
        Vectorized by Chong (thanks Chong).

        Args:
            - masks should be a size [h, w, n] tensor of masks
            - boxes should be a size [n, 4] tensor of bbox coords in relative point form
        """

        n, h, w = masks.shape
        print(n,h,w)
        x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(1,1,n)
        #print(x1,y1,x2,y2)
        r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # rows shape(1,w,1)
        #print(r)
        c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # cols shape(h,1,1)
        #print(c)
        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

    if len(im.shape) < 4:
        im = im[np.newaxis, ...]

    assert im.shape[1:] == (image_size[1],image_size[0],3)
    im0 = im.shape[1:]
    im = im.transpose((0, 3, 1, 2))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)

    assert len(im.shape) == 4

    #dt = (Profile(), Profile(), Profile())

    #with dt[0]:
    im = torch.from_numpy(im).to(device)
    im = im.half() if t_model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim



    def clip_coords(boxes, shape):
        # Clip bounding xyxy bounding boxes to image shape (height, width)
        if isinstance(boxes, torch.Tensor):  # faster individually
            boxes[:, 0].clamp_(0, shape[1])  # x1
            boxes[:, 1].clamp_(0, shape[0])  # y1
            boxes[:, 2].clamp_(0, shape[1])  # x2
            boxes[:, 3].clamp_(0, shape[0])  # y2
        else:  # np.array (faster grouped)
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

    def scale_coords(img1_shape, coords, img0_shape):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding

        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, :4] /= gain
        clip_coords(coords, img0_shape)
        return coords

    def apply(pred, index, im, prot):
        meta = []
        indexes = []
        shapes = []
        ih, iw = im.shape[2:]

        print(prot.shape)
        c, mh, mw = prot.shape  # CHW

        downsampled_bboxes = pred[:, :4].clone()
        downsampled_bboxes[:, 0] *= mw / iw
        downsampled_bboxes[:, 2] *= mw / iw
        downsampled_bboxes[:, 3] *= mh / ih
        downsampled_bboxes[:, 1] *= mh / ih

        masks = (pred[:, 6:] @ prot.float().view(c, -1)).sigmoid().view(-1, mh, mw)  # CHW
        masks = crop(masks, downsampled_bboxes)

        #print("offsets=",offsets[i])
        #print("det.shape=",det.cpu().numpy().shape)
        #print("det[:,2:4]=",det[:, 2:4].cpu().numpy())
        #print("sclaing to ", im.shape[2:])

        pred[:, :4] = scale_coords(im.shape[2:], pred[:, :4], im0).round()

        #print(i,"processing shape", im.shape, pred.cpu().numpy().shape, pred[i].cpu().numpy().astype(int))

        #print("det[:,2:4]*scale=",det[:, 2:4].cpu().numpy())
        #det[:, 0] += offsets[index, 0]
        #det[:, 1] += offsets[index, 1]
        #det[:, 2] += offsets[index, 0]
        #det[:, 3] += offsets[index, 1]
        #print("det[:,2:4]+offset=",det[:, 2:4].cpu().numpy())

        confs = pred[:, 4]
        clids = pred[:, 5]

        #for i, (det, proto) in enumerate(zip(pred, prot)):
            

        for j in range(len(masks)):
            x = masks[j]
            
            box = downsampled_bboxes[j]
            print(box.cpu().numpy().astype(int))
            x0,y0,x1,y1 = box.cpu().numpy().astype(int)

            if x1-x0 > 0 and y1-y0 > 0 and x1+1 <= x.shape[1] and y1+1 <= x.shape[0] and x0 >= 0 and y0 >= 0:
                x = x[y0:y1+1, x0:x1+1]
                x = (T.Resize((mask_res,mask_res))(x[None])[0])# * 255).to(torch.uint8)
                shapes.append(x.cpu().numpy())
                indexes.append(index)
                #if not x.shape[0] == 0 and not x.shape[1] == 0:
                #    shapes.append(cv2.resize(x, (56,56)))
                meta.append(pred[j, :6].to(torch.float32).cpu().numpy())
            else:
                print("image empty :(")

            #meta.append(det[:, :6].to(torch.float32).cpu().numpy())
        assert len(meta) == len(indexes)
        return meta, indexes, shapes

    results = []
    shapes = []
    meta = []
    indexes = []

    from utils.general import non_max_suppression

    #print("processing im.shape", im.shape)
    #print(offsets)
    offsets = np.reshape(offsets, (-1, 2)).astype(int)
    #print(len(pred))
    #assert len(offsets) == len(pred)
    pred = None
    prediction = None
    proto = None

    for i in range(len(im)):
        # Inference
        #local = (im[i:i+1].cpu().numpy() * 255).astype(np.uint8)[0, 0, ...]
        #print("local = ",local.shape, " ", local.dtype)
        #TRex.imshow("im"+str(len(im) - 1 - i), local)

        #with dt[1]:
        with torch.no_grad():
            pred, out = t_model(im[i][None], augment=None, visualize=None)
            proto = out[1]

            #print(proto.shape)

            # NMS
            #with dt[2]:
            conf_thres = 0.25
            iou_thres = 0.45
            
            pred = [a.to(device) for a in non_max_suppression(pred.cpu(), conf_thres, iou_thres, None, False, max_det=1000, nm=32)]
            #print("nonmaxsupp proc", (pred[0]).int().cpu().numpy())
            _meta, _index, _shapes = apply(pred[0], index=i, im=im[i:i+1], prot = proto[0])
            #print("RESULT FOR ",i,"=",_meta)
        meta += _meta
        shapes += _shapes
        indexes += np.repeat(len(im) - 1 - i, len(_meta)).tolist()

    #print("meta: ", meta)
    #print("meta: ", np.concatenate(meta, axis=0))
    if len(meta) > 0:
        meta = np.concatenate(meta, axis=0, dtype=np.float32)
    else:
        meta = np.array([], dtype=np.float32)

    return np.array(shapes, dtype=np.float32).flatten(), meta.flatten(), np.array(indexes, dtype=int)
    #return np.array(shapes, dtype=np.uint8).flatten(), torch.flatten(det[:, :6].to(torch.float32)).cpu().numpy()

def predict_yolo7_seg(image):
    global t_model

    from utils.general import non_max_suppression# (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                               #increment_path, non_max_suppression,scale_segments, print_args, strip_optimizer, xyxy2xywh)

    def xywh2xyxy(x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def merge_bases(rois, coeffs, attn_r, num_b, location_to_inds=None):
        # merge predictions
        # N = coeffs.size(0)
        if location_to_inds is not None:
            rois = rois[location_to_inds]
        N, B, H, W = rois.size()
        if coeffs.dim() != 4:
            coeffs = coeffs.view(N, num_b, attn_r, attn_r)
        # NA = coeffs.shape[1] //  B
        coeffs = F.interpolate(coeffs, (H, W),
                               mode="bilinear").softmax(dim=1)
        # coeffs = coeffs.view(N, -1, B, H, W)
        # rois = rois[:, None, ...].repeat(1, NA, 1, 1, 1)
        # masks_preds, _ = (rois * coeffs).sum(dim=2) # c.max(dim=1)
        masks_preds = (rois * coeffs).sum(dim=1)
        return masks_preds

    def non_max_suppression_mask_conf(prediction, attn, bases, pooler, hyp, conf_thres=0.1, iou_thres=0.6, merge=False, classes=None, agnostic=False, mask_iou=None, vote=False):
        from detectron2.structures import Boxes

        if prediction.dtype is torch.float16:
            prediction = prediction.float()  # to FP32
        nc = prediction[0].shape[1] - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates
        # Settings
        min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
        max_det = 300  # maximum number of detections per image
        time_limit = 10.0  # seconds to quit after
        redundant = True  # require redundant detections
        multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

        t = time.time()
        output = [None] * prediction.shape[0]
        output_mask = [None] * prediction.shape[0]
        output_mask_score = [None] * prediction.shape[0]
        output_ac = [None] * prediction.shape[0]
        output_ab = [None] * prediction.shape[0]

        def RMS_contrast(masks):
            mu = torch.mean(masks, dim=-1, keepdim=True)
            return torch.sqrt(torch.mean((masks - mu)**2, dim=-1, keepdim=True))


        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence
            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = xywh2xyxy(x[:, :4])

            # If none remain process next image
            if not x.shape[0]:
                continue

            a = attn[xi][xc[xi]]
            base = bases[xi]

            bboxes = Boxes(box)
            pooled_bases = pooler([base[None]], [bboxes])

            pred_masks = merge_bases(pooled_bases, a, hyp["attn_resolution"], hyp["num_base"]).view(a.shape[0], -1).sigmoid()

            if mask_iou is not None:
                mask_score = mask_iou[xi][xc[xi]][..., None]
            else:
                temp = pred_masks.clone()
                temp[temp < 0.5] = 1 - temp[temp < 0.5]
                mask_score = torch.exp(torch.log(temp).mean(dim=-1, keepdims=True))#torch.mean(temp, dim=-1, keepdims=True)

            x[:, 5:] *= x[:, 4:5] * mask_score # x[:, 4:5] *   * mask_conf * non_mask_conf  # conf = obj_conf * cls_conf

            if multi_label:
                i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
                mask_score = mask_score[i]
                if attn is not None:    
                    pred_masks = pred_masks[i]
            else:  # best class only
                conf, j = x[:, 5:].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

            # Filter by class
            if classes:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]


            # If none remain process next image
            n = x.shape[0]  # number of boxes
            if not n:
                continue

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            # scores *= mask_score
            i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]


            all_candidates = []
            all_boxes = []
            if vote:
                ious = box_iou(boxes[i], boxes) > iou_thres
                for iou in ious: 
                    selected_masks = pred_masks[iou]
                    k = min(10, selected_masks.shape[0])
                    _, tfive = torch.topk(scores[iou], k)
                    all_candidates.append(pred_masks[iou][tfive])
                    all_boxes.append(x[iou, :4][tfive])
            #exit()

            if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                    iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                    weights = iou * scores[None]  # box weights
                    x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                    if redundant:
                        i = i[iou.sum(1) > 1]  # require redundancy
                except Exception as e:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                    print(x, i, x.shape, i.shape)
                    print(e)
                    pass

            output[xi] = x[i]
            output_mask_score[xi] = mask_score[i]
            output_ac[xi] = all_candidates
            output_ab[xi] = all_boxes
            if attn is not None:
                output_mask[xi] = pred_masks[i]
            if (time.time() - t) > time_limit:
                break  # time limit exceeded

        return output, output_mask, output_mask_score, output_ac, output_ab
    
    #image = letterbox(image, 640, stride=64, auto=True)[0]
    #print(image.shape, image.dtype)
    if len(image.shape) > 3:
        image = image[0]
    #image = cv2.resize(image, (640, 640))#.astype(np.float32) / 255.0
    #assert image.shape == (480, 480, 3)
    #print(image.shape)
    #print(image.shape)
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))
    image = image.to(device)
    image = image.half()
    #print(image.shape, image.dtype)

    output = t_model(image)
    #print(output)
    '''print([key for key in output])
    for key in output:
        if output[key] is None:
            continue
        print(key)
        if type(output[key]) is list:
            print(len(output[key]), [k[0].shape for k in output[key]])
        else:
            print(output[key].shape)'''
    inf_out, train_out, attn, mask_iou, bases, sem_output = output['test'], output['bbox_and_cls'], output['attn'], output['mask_iou'], output['bases'], output['sem']
    #print(bases.shape, sem_output.shape, sem_output.cpu().numpy().shape)
    #plt.imshow(sem_output.cpu().numpy())
    #plt.show()
    bases = torch.cat([bases, sem_output], dim=1)
    nb, _, height, width = image.shape
    names = t_model.names
    pooler_scale = t_model.pooler_scale

    from detectron2.structures import Boxes
    from detectron2.modeling.poolers import ROIPooler
    pooler = ROIPooler(output_size=hyp['mask_resolution'], scales=(pooler_scale,), sampling_ratio=1, pooler_type='ROIAlignV2', canonical_level=2)

    output, output_mask, output_mask_score, output_ac, output_ab = non_max_suppression_mask_conf(inf_out, attn, bases, pooler, hyp, conf_thres=0.15, iou_thres=1.0, merge=False, vote=False, mask_iou=None)
    pred, pred_masks = output[0], output_mask[0]
    if pred is None:
        return (np.array([], dtype=np.uint8).flatten(), np.array([], dtype=np.float32))

    base = bases[0]
    bboxes = Boxes(pred[:, :4])
    original_pred_masks = pred_masks.view(-1, hyp['mask_resolution'], hyp['mask_resolution'])
    #print(original_pred_masks.cpu().numpy().shape)
    #print(original_pred_masks.cpu().numpy())
    #print(bboxes.tensor.detach().cpu().numpy().astype(int))
    pred_cls = pred[:, 5].detach().cpu().numpy()
    pred_conf = pred[:, 4].detach().cpu().numpy()

    results = []
    shapes = []
    bboxes = bboxes.tensor.detach().cpu().numpy().astype(int)
    #print(image.shape)
    bboxes[:, [0,2]] = np.clip(bboxes[:, [0,2]], 0, image.shape[3]-1);
    bboxes[:, [1,3]] = np.clip(bboxes[:, [1,3]], 0, image.shape[2]-1);
    print(bboxes)

    #bboxes = np.clip(bboxes, 0, np.max(image.shape)) #[[0, 0, 0, 0]], [[image.shape[1], image.shape[0], image.shape[1], image.shape[0]]])

    for i, (conf, clid, (x0,y0,x1,y1), mask) in enumerate(zip(pred_conf, pred_cls, bboxes, original_pred_masks.cpu().numpy())):
        #mask = (original_pred_masks[i].cpu().numpy() * 255).astype(np.uint8)
        #if i == 0:
        #    print(i, mask.shape, " -> ", (x1 - x0, y1 - y0))

        results.append((conf, clid, x0, y0, x1, y1))
        mask = (mask * 255).astype(np.uint8)
        #mask[mask < 150] = 0
        #mimg[mimg > 0] = 255
        shapes.append(mask)

        '''if i == 0:
            print(i, " -> ", (x1 - x0, y1 - y0), mask)
            mimg = cv2.resize(mask, (x1 - x0, y1 - y0))
            mimg[mimg < 200] = 0
            mimg[mimg > 0] = 255
            cv2.imshow("blob"+str(i), mimg)
            cv2.waitKey(1)'''
        


    return np.ascontiguousarray(np.array(shapes, dtype=np.uint8).flatten()), np.array(results, dtype=np.float32)

    from detectron2.utils.memory import retry_if_cuda_oom
    from detectron2.layers import paste_masks_in_image

    pred_masks = retry_if_cuda_oom(paste_masks_in_image)( original_pred_masks, bboxes, (height, width), threshold=0.5)
    pred_masks_np = pred_masks.detach().cpu().numpy()
    pred_cls = pred[:, 5].detach().cpu().numpy()
    pred_conf = pred[:, 4].detach().cpu().numpy()
    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    #nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
    nbboxes = bboxes.tensor.detach().cpu().numpy().astype(int)
    pnimg = nimg.copy()
    for one_mask, bbox, cls, conf in zip(pred_masks_np, nbboxes, pred_cls, pred_conf):
        if conf < 0.25:
            continue
        color = [np.random.randint(255), np.random.randint(255), np.random.randint(255)]

        #print(one_mask.shape)
        #print(one_mask)
        ##pnimg[one_mask] = pnimg[one_mask] * 0.5 + np.array(color, dtype=np.uint8) * 0.5
        ##pnimg = cv2.rectangle(pnimg, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        #label = '%s %.3f' % (names[int(cls)], conf)
        #t_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=1)[0]
        #c2 = bbox[0] + t_size[0], bbox[1] - t_size[1] - 3
        #pnimg = cv2.rectangle(pnimg, (bbox[0], bbox[1]), c2, color, -1, cv2.LINE_AA)  # filled
        #pnimg = cv2.putText(pnimg, label, (bbox[0], bbox[1] - 2), 0, 0.5, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)  
    
    #print(pnimg.shape)
    #cv2.imshow("webcam", pnimg)
    #cv2.waitKey(1)
    #plt.figure(figsize=(8,8), dpi=300)
    #plt.axis('off')
    #plt.imshow(pnimg)
    #plt.show()
