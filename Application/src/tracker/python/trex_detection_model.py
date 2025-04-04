import TRex
from TRex import ModelTaskType
from TRex import DetectResolution
from TRex import ObjectDetectionFormat
from TRex import Result

import torch
from functools import lru_cache
import platform
from typing import Optional, List, Any
import numpy as np

BBox = np.ndarray

class StrippedResults:
    def __init__(self, scale, offset):
        self.boxes : List[np.ndarray] = None
        self.keypoints : List[Any] = None
        self.masks : List[Any] = None
        self.orig_shape = None
        self.scale = scale
        self.offset = offset

    def __str__(self) -> str:
        return f"StrippedResults<boxes={self.boxes}, keypoints={self.keypoints}, orig_shape={self.orig_shape}, scale={self.scale}, offset={self.offset}>"

    def __repr__(self) -> str:
        return self.__str__()

class DetectionModel:
    def __init__(self, config : TRex.ModelConfig):
        self.config = config
        self.ptr = None
        self.device : Optional[torch.device] = None
        self.task   : Optional[TRex.ModelTaskType] = None

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
