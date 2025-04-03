import json
import TRex
import threading
import time
import numpy as np
import cv2

quit_app = False
polling_thread = None

def poll_cpp():
    global quit_app, frame_info
    start = time.time()
    video_size = None
    skeleton = None

    while not quit_app:
        try:
            TRex.log("Polling C++ for status...")
            TRex.log(f"frame_info = {frame_info}")
            status = None
            status = frame_info()
            #TRex.log(f"status = {status} {type(status)}")

            if status is None or status == "None":
                TRex.log("No status received from C++")
                time.sleep(1)
                continue
            
            #TRex.log("Converting status to JSON...")
            status = json.loads(status)
            #TRex.log(f"status.json = {status}")

            if video_size is None:
                video_size = eval(TRex.setting("meta_video_size"))
                skeleton = eval(TRex.setting("detect_skeleton"))[-1]

            # draw a picture of objects
            image = np.zeros((video_size[1], video_size[0], 3), np.uint8)

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

            TRex.imshow("Objects", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            #TRex.log(f"Skeleton = {TRex.setting('detect_skeleton')}")
            #TRex.log(f"Polled C++ message and got: {status} (took {time.time() - start} seconds)")
        except Exception as e:
            TRex.log(f"Error polling C++: {e}")
        start = time.time()
        time.sleep(0.01)  # Sleep for 100 milliseconds
        
    TRex.log("Ending poll_cpp thread.")

TRex.log("import message")

def init():
    TRex.log("hi message")

    global polling_thread, quit_app
    if polling_thread is not None:
        TRex.log("ending thread for reinit...")
        quit_app = True
        polling_thread.join()
        polling_thread = None
    
    TRex.log("starting thread...")
    quit_app = False
    polling_thread = threading.Thread(target=poll_cpp, name="poll_cpp_thread", daemon=True)
    polling_thread.start()

def deinit():
    global quit_app, polling_thread
    if polling_thread is not None:
        quit_app = True
        polling_thread.join()
        polling_thread = None
        quit_app = False
        TRex.log("deinit message")

def update():
    global quit_app
    if quit_app:
        TRex.log("we are quitting, so not updating")
        return
    #TRex.log("update message")
    #time.sleep(0.1)