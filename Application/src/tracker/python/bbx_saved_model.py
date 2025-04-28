# -*- coding: utf-8 -*-
from typing import List, Tuple, Any, Optional

from trex_detection_model import DetectionModel, TRexDetection
import TRex

from trex_yolo import YOLOModel, TRexYOLO

model: Optional[TRexDetection] = None
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
        models.append(YOLOModel(config))
    
    print("Configs: ", models)
    model = TRexYOLO(models)
    TRex.log("Loaded YOLO models: "+str([model.config for model in model.models]))
    return [model.config for model in model.models]

def predict(input : TRex.YoloInput) -> List[TRex.Result]:
    global model
    conf_threshold = float(TRex.setting("detect_conf_threshold"))
    iou_threshold = float(TRex.setting("detect_iou_threshold"))
    if not model:
        raise ValueError("Model not loaded. Please load the model before predicting.")

    return model.inference(input, 
                           conf_threshold = conf_threshold, 
                           iou_threshold = iou_threshold)