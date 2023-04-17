import torch
import pickle
import cloudpickle
from torchvision import transforms
from torch.nn import functional as F
import torchvision.transforms as T
import torchvision
import torch.backends.cudnn as cudnn

from utils.general import non_max_suppression
import numpy as np
import detectron2
from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.layers import paste_masks_in_image

from models.common import DetectMultiBackend
from utils.general import non_max_suppression# (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, 

#model_path = "Z:/work/octopus/yolov7-seg.pt"
#model_path = "/Volumes/Public/work/yolov7-seg.pt"
model_path= "/Users/tristan/Downloads/tortoise-640-32-seg.pt"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

t_model = DetectMultiBackend(model_path, device=device, dnn=False, fp16=False)

def predict_custom_yolo7_seg(t_model, device, image_size, offsets, im, conf_threshold = 0.25, iou_threshold = 0.1, mask_res = 56):
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

    if im.shape[1:] != (image_size,image_size,3):
    	print("Image shape unexpected, got ", im.shape, " expected (:",",",image_size,",",image_size,",3)")
    assert im.shape[1:] == (image_size,image_size,3)
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
            #conf_thres = 0.1
            #iou_thres = 0.0
            
            pred = [a.to(device) for a in non_max_suppression(pred.cpu(), conf_threshold, iou_threshold, None, True, max_det=1000, nm=32)]
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

import models
import utils
cloudpickle.register_pickle_by_value(models)
cloudpickle.register_pickle_by_value(utils)
cloudpickle.register_pickle_by_value(detectron2)

output_path = "/Volumes/Public/work/tali/models/640p/tortoise-640-32-seg-macosx.pth"
print("Output to",output_path)

with open(output_path, "wb") as f:
	cloudpickle.dump({"model":t_model, "predict":predict_custom_yolo7_seg}, f)
