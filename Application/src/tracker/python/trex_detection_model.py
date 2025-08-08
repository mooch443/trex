import TRex
from TRex import ModelTaskType
from TRex import DetectResolution
from TRex import ObjectDetectionFormat
from TRex import Result

import torch
from functools import lru_cache
import platform
from typing import Optional, List, Any, Tuple
from itertools import groupby
import numpy as np
import cv2

BBox = np.ndarray
Image = np.ndarray

printed_warning = False

class StrippedResults:
    """
    Base class for stripped detection results, storing bounding boxes, keypoints, masks, and oriented bounding boxes along with scale and offset for coordinate transformations.
    """
    def __init__(self, scale: np.ndarray, offset: np.ndarray) -> None:
        """
        Initialize StrippedResults with scale and offset for coordinate transformations.

        Args:
            scale (np.ndarray): A 2-element array [scale_x, scale_y] representing scaling factors applied to model output coordinates to map them back to the original image.
            offset (np.ndarray): A 2-element array [offset_x, offset_y] representing pixel offsets added to model output coordinates before scaling.

        Attributes:
            boxes: Array of clid, conf, and bounding boxes in format [clid, conf, x, y, w, h] in original image coordinates.

            keypoints: List of arrays (same length as the boxes array), each of shape [num_keypoints, 2], containing (x, y) coordinates of keypoints in original image space.

            masks: List of 2D numpy arrays (uint8) representing segmentation masks aligned to the original image dimensions. Same length as boxes.

            obb: Array of oriented bounding boxes, with each row formatted as [class_id, confidence, x_center, y_center, width, height, angle] in original image coordinates. Note: if obb is set, boxes is not required and has to be empty.

            points: Array of point detections, with each row formatted as [class_id, confidence, x, y, radius]

            orig_shape: Original image shape tuple (height, width) as reported by the model output.
        """
        self.boxes: Optional[np.ndarray] = None
        self.keypoints: Optional[List[np.ndarray]] = None
        self.masks: Optional[List[np.ndarray]] = None
        self.orig_shape: Optional[Any] = None
        self.scale: np.ndarray = scale
        self.offset: np.ndarray = offset
        self.obb: Optional[np.ndarray] = None
        self.points: Optional[np.ndarray] = None
        self.locations: Optional[np.ndarray] = None

    def __str__(self) -> str:
        return f"StrippedResults<boxes={self.boxes}, keypoints={self.keypoints}, orig_shape={self.orig_shape}, scale={self.scale}, offset={self.offset}, obb={self.obb}, points={self.points}>"

    def __repr__(self) -> str:
        return self.__str__()

class DetectionModel:
    """
    Abstract base class for detection models. Provides interface for loading models and performing inference.
    """
    def __init__(self, config: TRex.ModelConfig) -> None:
        """Initialize DetectionModel with a given ModelConfig and set up the computation device."""
        self.config = config
        self.ptr = None
        self.device : Optional[torch.device] = None

        self.reinit_device()

    
    def __str__(self) -> str:
        return f"DetectionModel<{str(self.config)}>"

    def __repr__(self) -> str:
        return self.__str__()
    
    def reinit_device(self):
        self.device = None

        # if no device is specified, use cuda if available, otherwise use mps/cpu
        device_from_settings = TRex.setting("gpu_torch_device")
        if device_from_settings != "":
            if device_from_settings == "automatic":
                device_from_settings = ""
            else:
                device_index = eval(TRex.setting("gpu_torch_device_index"))
                if device_index >= 0:
                    device_from_settings = f"{device_from_settings}:{device_index}"
                print(f"Using device {device_from_settings} from settings for {self}.")
                self.device = torch.device(device_from_settings)
        
        # if no device is specified, use cuda if available, otherwise use mps/cpu
        if self.device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            elif torch.backends.mps.is_available() and platform.processor() == "arm":
                self.device = torch.device("mps") #mps
            else:
                self.device = torch.device("cpu")

    @property
    @lru_cache(maxsize=1)
    def task(self) -> TRex.ModelTaskType:
        #print("task == ", self.config.task, type(self.config.task))
        return self.config.task
    
    def load(self):
        """
        Load the model from the specified configuration.
        This method should handle the loading of the model parameters and any necessary setup.

        It should initialise the following attributes:
        - self.ptr: A pointer to the model object.
        - self.config.task: The task type of the model.
        - self.config.trained_resolution(TRex.DetectResolution): The resolution of the model.
        - self.config.classes: The classes of the model.
        - self.config.output_format(TRex.ObjectDetectionFormat): The output format of the model.
        - self.config.keypoint_format(TRex.KeypointFormat): The keypoint format of the model.
        """

        TRex.log("Loaded model: {}".format(self))

    def predict_boxes(self, images: List[np.ndarray], **kwargs) -> List[BBox]:
        """
        Perform inference on the input images.

        Args:
            images (List[np.ndarray]): A list of images to perform inference on.

        Returns:
            List[BBox]: A list of bounding boxes containing the detection results for each image.
        """
        raise NotImplementedError("predict_boxes method not implemented")

    def predict(self, images: List[np.ndarray], scales : List[Any], offsets : List[Any], **kwargs) -> List[StrippedResults]:
        """
        Perform inference on the input images.

        Args:
            images (List[np.ndarray]): A list of images to perform inference on.

        Returns:
            List[StrippedResults]: A list of results containing the detection results for each image.
        """
        raise NotImplementedError("predict method not implemented")


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

# This class is the abstract parent of for example TRexYOLO
# and will encapsulate all functionality shared by all model types
class TRexDetection:
    def __init__(self, models: List[DetectionModel]):
        """
        Initialize the TRexDetection class with a list of models.

        Args:
            models (List[Model]): A list of models used for region proposal, detection and segmentation.

        Raises:
            AssertionError: If no models are specified.
        """
        assert len(models) > 0, "No models specified for TRexDetection {}".format(models)

        self.models: List[DetectionModel] = models
        self.load_models()

        # log configuration and loaded models
        TRex.log("TRexDetection configuration: models={}".format(self.models))

    def __str__(self):
        """
        String representation of the TRexDetection instance.

        Returns:
            str: A string that represents the TRexDetection instance.
        """
        return "TRexDetection<models={}>".format(self.models)
    
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
        
        model = next((model for model in self.models if model.task == ModelTaskType.region), None)
        assert model is not None
        
        scaled_size: List[int] = [
            model.config.trained_resolution.width, 
            model.config.trained_resolution.height
        ]

        scaled_down_scales: np.ndarray = np.array([(1,1) for im in images])
        scaled_down: List[Image] = [im for im in images] #
        
        bboxes: List[BBox] = \
                       model.predict_boxes(images = scaled_down, 
                                     imgsz=scaled_size, 
                                     conf=0.1, 
                                     iou=0.7, 
                                     verbose=False,
                                     **kwargs)
        padding: int = 7
        results: List[List[Tuple[BBox, Image]]] = []
        
        # return a list per input image, with a list of bounding boxes (and their image regions) per image
        for i, bb in enumerate(bboxes):
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

    def detection_model(self) -> DetectionModel:
        """
        Retrieves the detection or segmentation model.

        Returns:
            DetectionModel: The detection or segmentation model.

        Raises:
            Exception: If no detect or segment model is found.
        """
        detect_model = next((model for model in self.models if model.task == ModelTaskType.detect), None)
        if detect_model is not None:
            return detect_model
        raise Exception("No detect or segment model found")

    def detect_or_segment(self, images :List[np.ndarray], 
                          scales : List[Any], 
                          offsets : List[Any],
                            **kwargs) -> List[StrippedResults]:
        """
        Perform detection or segmentation on given input.

        Args:
            images (List[np.ndarray]): A list of images to predict on.
            scales (List[Any]): A list of scales for each proposed region.
            offsets (List[any]): A list of offsets for each proposed region.
            **kwargs: Additional keyword arguments to pass to the model's `predict` method.

        Returns:
            List[TRex.Result]: The result of the detection or segmentation operation.

        Raises:
            Exception: If no detect or segment model is found.
        """
        return self.detection_model().predict(images = images, scales = scales, offsets = offsets, **kwargs)

    def load_models(self):
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

        rexsults : List[TRex.Result] = []

        # otherwise, we need to segment the entire image
        if len(tensor) == 0:
            return rexsults

        w, h, c = tensor[0].shape[1], tensor[0].shape[0], tensor[0].shape[2]
        #print("tensor[0].shape = ",tensor[0].shape)
        #print("resolution = ",self.detect_resolution())

        # get total memory of the gpu:
        total_memory = TRexDetection.get_free_memory(self.detection_model().device) * 0.75

        # if the total memory is not enough, we need to send the images in packages
        normal_res = int(w * h)
        memory_per_image = TRexDetection.calculate_memory(1, h, w, c, torch.float64)
        max_len = int(max(1, total_memory // memory_per_image // 100))
        #print(f"Calculated max_len = {max_len} based on total_memory = {total_memory} and normal_res = {normal_res} and {w}x{h} pixels / image, memory_per_image = {memory_per_image}")

        results : List[StrippedResults] = []

        if len(tensor) > max_len:
            # send data in packages of X images
            #print(f"Sending images in packages of {X} images, total {len(tensor)} images, {w}x{h} pixels, based on normal_res = {normal_res}")
            #print(f"Sending images in packages of {max_len} images, total {len(tensor)} images, {w}x{h} pixels / image")

            for i in range(0, len(tensor), max_len):
                #print(f"Sending images {i} to {i+max_len}")
                results.extend(self.detect_or_segment(images = tensor[i:i+max_len], 
                                            scales = scales[i:i+max_len],
                                            offsets = offsets[i:i+max_len],
                                            conf = conf_threshold, 
                                            iou = iou_threshold, 
                                            #offsets = offsets, 
                                            imgsz = self.detect_resolution(),
                                            classes=None, 
                                            agnostic_nms=True,
                                            verbose = False,
                                            max_det = max_det))
                '''for r, scale, offset in zip(rs, scales[i:i+max_len], offsets[i:i+max_len]):
                    results.append(StrippedYoloResults(r, scale=scale, offset=offset))
                #results.extend(rs)
                torch.cuda.empty_cache()'''

        else:
            #print(f"Sending all {len(tensor)} images at once, given {w}x{h} pixels, amounting to {len(tensor) * w * h} pixels")
            results.extend(self.detect_or_segment(images = tensor, 
                                        scales = scales,
                                        offsets = offsets,
                                        conf = conf_threshold, 
                                        iou = iou_threshold, 
                                        #offsets = offsets, 
                                        imgsz = self.detect_resolution(),
                                        classes=None, 
                                        agnostic_nms=True,
                                        verbose = False,
                                        max_det = max_det))
            '''results = []
            for r, scale, offset in zip(rs, scales, offsets):
                    results.append(StrippedYoloResults(r, scale=scale, offset=offset))
            #torch.cuda.empty_cache()'''

        # use groupby to group the list elements by id
        results = [[x[1] for x in group] for _, group in groupby(list(zip(orig_id, results)), lambda x: x[0])]
        offsets = [[x[1] for x in group] for _, group in groupby(list(zip(orig_id, offsets)), lambda x: x[0])]
        scales = [[x[1] for x in group] for _, group in groupby(list(zip(orig_id, scales)), lambda x: x[0])]
        #print(f"len(results) = {len(results)} len(offsets) = {len(offsets)} len(scales) = {len(scales)}")

        index = 0
        for i, tiles in enumerate(results):
            coords = []
            masks = []
            keypoints = []
            obbs = []
            points = []
            for j, tile in enumerate(tiles):
                if True:
                #try:
                    #c, m, k = self.postprocess_result(index, tile)
                    #print("c.shape= ",c.shape, " len(m)=", len(m))
                    coords.append(tile.boxes)
                    if tile.masks is not None and len(tile.masks) > 0:
                        masks.extend(tile.masks)
                    if tile.keypoints is not None and len(tile.keypoints) > 0:
                        keypoints.extend(tile.keypoints)
                    if tile.obb is not None and len(tile.obb) > 0:
                        obbs.extend(tile.obb)
                    if tile.points is not None and len(tile.points) > 0:
                        points.extend(tile.points)

                    #print("appended keypoints: ", len(keypoints), " at ", index, "with", tile)

                    #r = result.cpu().plot(img=im[i], line_width=1)
                    #TRex.imshow("result"+str(i), r)
                    '''except Exception as e:
                        print("Exception when postprocessing result", e," at ",index, "with", tile)
                        #print("result.boxes.data.cpu().numpy() = ", result.boxes.data.cpu().numpy())
                        #r = tile.cpu().plot(img=im[i], line_width=1)
                        #TRex.imshow("result"+str(i), r)
                        raise e
                    finally:'''
                    index += 1
            
            if len(keypoints) > 0:
                keypoints = np.concatenate(keypoints, axis=0, dtype=np.float32)
            if len(obbs) > 0:
                obbs = np.concatenate(obbs, axis=0, dtype=np.float32)
                coords = np.array([], dtype=np.float32)
            elif len(points) > 0:
                points = np.concatenate(points, axis=0, dtype=np.float32)
                coords = np.array([], dtype=np.float32)
            else:
                coords = np.concatenate(coords, axis=0)

            rexsults.append(TRex.Result(index, TRex.Boxes(coords), masks, TRex.KeypointData(keypoints), TRex.ObbData(obbs), TRex.PointData(points)))

        return rexsults

    def perform_region_proposal(self, tensor: np.ndarray, offsets: List[float], scales: List[float], ious: List[float], confs: List[float]) -> List[TRex.Result]:
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
        results : List[StrippedResults] = self.detect_or_segment(
            images = all_images, 
            scales = scales,
            offsets=offsets,

            conf = confs, 
            iou = ious, 
            imgsz = self.detect_resolution(),
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
            #scale = scales[i]
            #offset = offsets[i]

            collected_boxes = []
            collected_masks = []
            collected_keypoints = []

            # For each result, perform postprocessing and gather the boxes and masks
            for t, (result, (box, region)) in enumerate(zip(results, regions)):
                #r = result.cpu().plot(img=region, line_width=1)
                #TRex.imshow("segmentation"+str(i)+","+str(t), r)

                #result = StrippedYoloResults(result, scale=scale, offset=offset)
                
                #coords, masks, keypoints = self.postprocess_result(i, result, offset, scale, box)
                collected_boxes.append(result.boxes)
                collected_masks.extend(result.masks)
                if len(result.keypoints) > 0:
                    collected_keypoints.extend(result.keypoints)

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

        # Return the final list of results
        return rexsults

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
