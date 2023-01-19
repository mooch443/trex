import tensorflow as tf
import TRex
import numpy as np
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

model = None
image_size = 640
model_path = None
image = None
model_type = None

def load_model():
	global model, model_path, image_size
	model = tf.saved_model.load(model_path)
	full_model = tf.function(lambda x: model(images=x))
	full_model = full_model.get_concrete_function(
	    tf.TensorSpec((1, 3, image_size, image_size), tf.float32))
	# Get frozen ConcreteFunction
	frozen_func = convert_variables_to_constants_v2(full_model)
	frozen_func.graph.as_graph_def()
	model = frozen_func

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
        topk_all=100
        iou_thres=0.45
        conf_thres=0.45
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

        for i, det in enumerate(nms):
            if len(det.shape) > 1:
                det = det.numpy()
                boxes = np.round(scale_boxes(im.shape[2:4], det[:, :4], im.shape[2:4]))
                for *xyxy, conf, cls in reversed(det):
                    for i in range(len(xyxy)):
                        xy = xyxy[i]
                        if not type(xy) is np.ndarray:
                            continue
                        ratio = (im0[1] / im.shape[-1], im0[0] / im.shape[-2])
                        pt0 = np.array((xy[0] * ratio[0], xy[1] * ratio[1])).astype(int)
                        pt1 = np.array((np.array((xy[2] * ratio[0], xy[3] * ratio[1])))).astype(int)
                        if xy[2] > 0:
                            results.append(np.array((pt0, pt1)))
        return np.array(results, dtype=int)
    
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
		results = predict_yolov7(im, image_shape=(image_size,image_size))
		receive(np.array(results, dtype=int).flatten())
	else:
		raise Exception("model_type was not set before running inference")
