# -*- coding: utf-8 -*-
"""Production entry points used by the C++ detector bridge."""

from typing import List, Optional

from trex_detection_model import TRexDetection
import TRex

from trex_yolo import YOLOModel
from trex_rfdetr import RFDETRModel, is_rfdetr_checkpoint

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
        model_class = (
            RFDETRModel
            if is_rfdetr_checkpoint(config.model_path)
            else YOLOModel
        )
        models.append(model_class(config))

    if any(isinstance(candidate, RFDETRModel) for candidate in models):
        if any(config.task == TRex.ModelTaskType.region for config in configs):
            raise ValueError(
                "RF-DETR cannot currently be combined with a region_model. "
                "Region crops do not yet pass through the C++ exact-size preparation path."
            )
    
    print("Configs: ", models)
    model = TRexDetection(models)
    TRex.log("Loaded detection models: "+str([model.config for model in model.models]))
    return [model.config for model in model.models]

def predict(input : TRex.YoloInput) -> List[TRex.Result]:
    import os

    global model
    conf_threshold = float(TRex.setting("detect_conf_threshold"))
    iou_threshold : Optional[float] = TRex.setting("detect_iou_threshold")
    if not model:
        raise ValueError("Model not loaded. Please load the model before predicting.")

    destination = os.environ.get("TREX_RFDETR_E2E_APP_DUMP")
    if destination:
        # TEST-ONLY: the helper lives in Application/Tests and is available
        # only to the opt-in full application parity test.
        from trex_detector_e2e_capture import (
            capture_input,
            write_result_dump,
        )
        input_path = capture_input(input, destination)
    results = model.inference(
        input,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
    )
    if destination:
        write_result_dump(
            results,
            destination,
            input_path,
            conf_threshold,
            iou_threshold,
        )
    return results
