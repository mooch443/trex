import tensorflow as tf
import TRex
import numpy as np

model = None
image_size = 640
model_path = None
image = None

def load_model():
	global model, model_path
	model = tf.saved_model.load(model_path)

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

def apply():
	global model, image_size, receive, image
	image = tf.constant(np.array(image, copy=False)[..., :3], dtype=tf.uint8)
	#print(image)
	results = inference(model, image, size=(image_size, image_size))
	#print(np.array(results, dtype=int).flatten())
	receive(np.array(results, dtype=int).flatten())
