import numpy as np
import TRex
import subprocess
from queue import Queue, Empty
import sys
from threading  import Thread
import platform

ON_POSIX = 'posix' in sys.builtin_module_names

def enqueue_output(out, queue):
    for line in iter(out.readline, b''):
        queue.put(line)
    out.close()

q = Queue()
t = None

dimensions = None
process = None
scale_factor = 0.5

def request_features():
    return "position" #"position,visual_field,midline"

def update_tracking():
    global process, t, q, dimensions
    
    if type(None) == type(process):
        print("initializing subprocess --")
        process = subprocess.Popen(['python', '-i'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, bufsize=1, close_fds=ON_POSIX)
        t = Thread(target=enqueue_output, args=(process.stdout, q))
        t.daemon = True # thread dies with the program
        t.start()
        
        dimensions = TRex.video_size()
    
    if frame % 100 == 0:
        s = (str(frame)+"*2\n").encode('ascii')
        print(frame, "sending message", s)
        process.stdin.write(s)
        process.stdin.flush()
        
    try:
        while True:
            line = q.get_nowait() # or q.get(timeout=.1)
            line = line.decode('utf-8')
            print("output:",line)
            # do work with 'line'
    except Empty:
        pass

    image = np.zeros((int(dimensions["height"] * scale_factor), int(dimensions["width"] * scale_factor), 3), dtype=np.uint8)
    #print(positions.shape, ids.shape)
    #print(visual_field.shape, ids, colors)
    #print(midlines)
    for i, key in zip(range(len(ids)), ids):
        color = (int(colors[i * 3]), int(colors[i * 3 + 1]), int(colors[i * 3 + 2]))
        pos = positions[i] * scale_factor

        if len(midlines) > i:
            midline = (midlines[i] * scale_factor + pos ).astype(np.int)
        pos = tuple((pos + centers[i] * scale_factor).astype(np.int))

        if len(midlines) > i and not np.isinf(midline[0]).any():
            print(midlines, i, midline[0].min(), midline[0].max())
            #cv.circle(image, tuple(midline[0]), 5, (255, 0, 0), -1)
            #for j in range(1, len(midline)):
            #    cv.line(image, tuple(midline[j-1, :]), tuple(midline[j]), (255, 255, 255))

        #cv.circle(image, pos, 5, color, -1)

        if key != 1 or i >= len(visual_field):
           continue 
        for id in np.unique(np.concatenate((visual_field[i]))):
            if id == -1 or id == key or not id in ids:
                continue
            j = np.where(ids == id)[0]
            other = tuple(((positions[j] + centers[j]) * scale_factor)[0].astype(np.int))
            #cv.line(image, pos, other, (255, 255, 255))
        #print("tracking", frame, key, positions[key][3])
        #cv.imwrite("image.png", image)
    #cv.putText(image, str(frame), (10, 10), cv.FONT_HERSHEY_PLAIN, 0.5, (255, 255, 255))
    #image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    #TRex.imshow("image", image)
