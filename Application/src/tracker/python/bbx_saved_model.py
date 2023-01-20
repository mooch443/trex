import torch
from torchvision import transforms
from torch.nn import functional as F
import torchvision

from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.layers import paste_masks_in_image

import tensorflow as tf
import TRex
import numpy as np
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import time
import cv2


print("UPDATED SCRIPT ---------------------------------------------")

WEIGHTS_PATH = "C:/Users/tristan/Downloads/yolov7-mask.pt"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

weigths = torch.load(WEIGHTS_PATH)
t_model = weigths['model']
t_model = t_model.half().to(device)
_ = t_model.eval()

hyp = {
    "mask_resolution": 56,
    "attn_resolution": 14,
    "num_base": 5
}


model = None
image_size = 640
model_path = None
image = None
model_type = None

def load_model():
    global model, model_path, image_size
    '''model = tf.saved_model.load(model_path)
    full_model = tf.function(lambda x: model(images=x))
    full_model = full_model.get_concrete_function(
        tf.TensorSpec((1, 3, image_size, image_size), tf.float32))
    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()
    model = frozen_func'''
    pass

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
    
    boxes = y[..., :4]
    boxes = _xywh2xyxy(y[..., :4])

    boxes *= [w, h, w, h]
    #print(w, h)

    probs = y[:, :, 4:5]
    classes = y[:, :, 5:]
    scores = probs * classes

    boxes = tf.expand_dims(boxes, 2)
    nms = tf.image.combined_non_max_suppression(boxes,
                                                scores,
                                                topk_per_class,
                                                topk_all,
                                                iou_thres,
                                                conf_thres,
                                                clip_boxes=False)

    #y = y[0]  # [x(1,6300,85), ...] to x(6300,85)
    #xywh = y[..., :4]  # x(6300,4) boxes
    #conf = y[..., 4:5]  # x(6300,1) confidences
    #cls = tf.reshape(tf.cast(tf.argmax(y[..., 5:], axis=1), tf.float32), (-1, 1))  # x(6300,1)  classes
    #y = tf.concat([conf, cls, xywh], 1)

    #cx, cy, w, h, conf = y[0][0, :, 0:5].T#.shape
    #clid = tf.reshape(tf.cast(tf.argmax(y[0][..., 5:], axis=1), tf.float32), (-1, 1)) 

    for i, det in enumerate(nms):
        #print(det.shape)
        if len(det.shape) > 1:
            det = det.numpy()

            boxes = tf.round(scale_boxes(im.shape[1:3], det[:, :4], im.shape[1:3]))
            for *xyxy, conf, cls in reversed(det):
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



def predict_yolov7(img, image_shape=(640,640)):
    def perform_filtering(im0, im, y):
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

        topk_per_class=100
        topk_all=200
        iou_thres=0.45
        conf_thres=0.25
        b, ch, h, w = im.shape
        results = []

        boxes = y[..., :4]
        boxes = _xywh2xyxy(y[..., :4])

        #boxes *= [w, h, w, h]
        #print(w, h)

        probs = y[:, :, 4:5]
        classes = y[:, :, 5:]
        scores = probs * classes

        boxes = tf.expand_dims(boxes, 2)
        nms = tf.image.combined_non_max_suppression(boxes,
                                                    scores,
                                                    topk_per_class,
                                                    topk_all,
                                                    iou_thres,
                                                    conf_thres,
                                                    clip_boxes=False)

        #print(nms)
        #print("nmsed_boxes: ",nms.nmsed_boxes)

        nms_valid = nms.valid_detections[0]
        nms_boxes = nms.nmsed_boxes.numpy()[0, :nms_valid]
        nms_scores = nms.nmsed_scores.numpy()[0, :nms_valid]
        nms_classes = nms.nmsed_classes.numpy()[0, :nms_valid]

        if len(nms_boxes) > 0:
            print("nms_boxes: ", nms_boxes)
            print("nms_scores: ", nms_scores)
            print("nms_classes: ", nms_classes)
            print("nms_valid: ", nms_valid)

        ratio = (im0[1] / im.shape[-1], im0[0] / im.shape[-2])

        for xy, score, clid in zip(nms_boxes, nms_scores, nms_classes):
            pt0 = np.array((xy[0] * ratio[0], xy[1] * ratio[1])).astype(int)
            pt1 = np.array((np.array((xy[2] * ratio[0], xy[3] * ratio[1])))).astype(int)
            print(score, clid, pt0, pt1)
            results.append(np.array((score, clid, pt0[0], pt0[1], pt1[0], pt1[1])))


        '''for i, det in enumerate(nms):
            if len(det.shape) > 1:
                det = det.numpy()
                boxes = np.round(scale_boxes(im.shape[2:4], det[:, :4], im.shape[2:4]))

                for *xyxy, conf, cls in reversed(det):
                    line = (i, cls, *xyxy, np.array([conf]))
                    #print(line)
                    print(i, "=>", len(xyxy), xyxy[i], np.shape(xyxy), np.shape(conf), conf)

                for *xyxy, conf, cls in reversed(det):
                    for i in range(len(xyxy)):
                        xy = xyxy[i]
                        if not type(xy) is np.ndarray:
                            continue
                        ratio = (im0[1] / im.shape[-1], im0[0] / im.shape[-2])
                        pt0 = np.array((xy[0] * ratio[0], xy[1] * ratio[1])).astype(int)
                        pt1 = np.array((np.array((xy[2] * ratio[0], xy[3] * ratio[1])))).astype(int)
                        if xy[2] > 0:
                            #print(classes[i], probs[i], scores[i], conf, cls, clid, pt0[0], pt0[1], pt1[0], pt1[1])
                            print(conf[i], cls[i])
                            results.append(np.array((conf[i], cls[i], pt0[0], pt0[1], pt1[0], pt1[1])))'''
        return np.array(results, dtype=np.float32)
    
    def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv.resize(im, new_unpad, interpolation=cv.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv.copyMakeBorder(im, top, bottom, left, right, cv.BORDER_CONSTANT, value=color)  # add border
        return im, r, (dw, dh)

    #img = (images[0].copy() * 255).astype(np.uint8)
    def transform_image(img, image_shape):
        
        image = img
        #print(image.shape)
        shape = img.shape[:2]
        stride = 32
        #image, ratio, dwdh = letterbox(image, new_shape=image_shape, auto=False)
        ratio = min(image_shape[0] / shape[0], image_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * ratio)), int(round(shape[0] * ratio))
        
        dw, dh = image_shape[1] - new_unpad[0], image_shape[0] - new_unpad[1]  # wh padding
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2
        im = image

        #im = np.swapaxes(im, 2, 0)[np.newaxis, ...]

        im = im.transpose(2, 0, 1)[np.newaxis, ...]
        #print("BEFORE:",np.histogram(im), im.shape)
        im = im.astype(np.float32) / 255.0
        
        return tf.convert_to_tensor(im), ratio, (dw, dh)
    
    if len(img.shape) > 3:
        img = img[0, ...]
    im, ratio, dwdh = transform_image(img, image_shape=image_shape)
    output_data = model(im)[0]
    return perform_filtering(img.shape, im, output_data)

def apply():
    global model, image_size, receive, image, model_type
    if model_type == "yolo5":
        image = tf.constant(np.array(image, copy=False)[..., :3], dtype=tf.uint8)
        #print(image)
        results = inference(model, image, size=(image_size, image_size))
        #print(np.array(results, dtype=int).flatten())
        receive(np.array(results, dtype=int).flatten())
    elif model_type == "yolo7":
        #im = tf.convert_to_tensor(np.array(image, copy=False)[..., :3], dtype=tf.float32)
        im = np.array(image, copy=False)[..., :3]
        results = predict_yolo7_seg(im)
        print("sending: ", results[1])

        receive_seg(results[0], results[1].flatten())
        #results = predict_yolov7(im, image_shape=(image_size,image_size))
        #receive(np.array(results[1], dtype=np.float32).flatten())
    else:
        raise Exception("model_type was not set before running inference")

def predict_yolo7_seg(image):
    global t_model
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
                except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                    print(x, i, x.shape, i.shape)
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
    pooler = ROIPooler(output_size=hyp['mask_resolution'], scales=(pooler_scale,), sampling_ratio=1, pooler_type='ROIAlignV2', canonical_level=2)

    output, output_mask, output_mask_score, output_ac, output_ab = non_max_suppression_mask_conf(inf_out, attn, bases, pooler, hyp, conf_thres=0.35, iou_thres=0.65, merge=False, mask_iou=None)
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
    for i, (conf, clid, (x0,y0,x1,y1), mask) in enumerate(zip(pred_conf, pred_cls, bboxes.tensor.detach().cpu().numpy().astype(int), original_pred_masks.cpu().numpy())):
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
