# -*- coding: utf-8 -*-
import TRex
import time
import numpy as np
import cv2

quit_app = False
video_size = None
skeleton = None

def update_thread(status):
    global quit_app, video_size, skeleton
    try:
        if status is None:
            TRex.log("Empty status received from C++")
            time.sleep(1)
            return False
        
        if video_size is None:
            video_size = eval(TRex.setting("meta_video_size"))
            skeleton = eval(TRex.setting("detect_skeleton"))[-1]

        # draw a picture of objects
        image = np.zeros((video_size[1], video_size[0], 3), np.uint8)
        cv2.rectangle(image, (0, 0), (video_size[0], video_size[1]), (255, 255, 255), 3)

        for obj in status["objects"]:
            clr = tuple(obj['color'])
            box = np.array(obj["box"])
            center = box[:2] + box[2:] / 2

            # Log object information
            #TRex.log(f"Object {obj['id']} is at {center} and of size {box[2:]} with color {clr} and pose {obj['pose']}")

            # Draw circles and rectangles on the image
            cv2.circle(image, (int(center[0]), int(center[1])), 15, (125,125,125), 1)
            cv2.circle(image, (int(center[0]), int(center[1])), 10, clr, 1)
            cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[0] + box[2]), int(box[1] + box[3])), (125,125,125), 3)

            if obj["selected"]:
                cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[0] + box[2]), int(box[1] + box[3])), clr, 3)

            # Draw pose lines and circles
            if obj['pose'] is not None and len(obj['pose']) > 0:
                for A,B in skeleton:
                    if A >= len(obj['pose']) or B >= len(obj['pose']):
                        continue
                    x1,y1 = obj['pose'][A]
                    x2,y2 = obj['pose'][B]

                    if (x1 != 0 or y1 != 0) and (x2 != 0 or y2 != 0):
                        cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), clr, 3)
                
                for x,y in obj['pose']:
                    if x == 0 and y == 0:
                        continue
                    cv2.circle(image, (int(x), int(y)), 10, clr, 1)        

        if image.shape[1] >= 800:
            ratio = image.shape[1] / image.shape[0]
            new_width = 800
            new_height = int(new_width / ratio)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            #TRex.log(f"Resized image to {image.shape[1]}x{image.shape[0]} {image.dtype} {image.shape}")
        
        TRex.imshow("Objects", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        #TRex.log(f"Skeleton = {TRex.setting('detect_skeleton')}")
    except Exception as e:
        TRex.log(f"Error polling C++: {e}")
    start = time.time()
    time.sleep(0.01)  # Sleep for 100 milliseconds
    return True

def init():
    TRex.log("hi message")

def deinit():
    global video_size, skeleton
    TRex.log("deinit called")
    TRex.destroyAllWindows()
    video_size = None
    skeleton = None

def update(status):
    #TRex.log(f"Updating with status: {status}")
    update_thread(status)