import matplotlib
#matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import io
from PIL import Image
import sys
import glob

if __name__ == "__main__":
    if len(sys.argv) < 2 or ".npz" in sys.argv[1] or "_posture_" in sys.argv[1]:
        raise Exception("Please provide the path to a fishdata file and omit the '*posture*fish[ID]*.npz' part. Basically just put the fishdata_folder/video_name.")
    
    data = {}
    files = sorted(glob.glob(sys.argv[1]+"*_posture*.npz"))
    min_frame = np.inf
    max_frame = -np.inf
    
    screen = [np.inf, np.inf, -np.inf, -np.inf]
    
    # open all files that have been found and preprocess data
    for f in files:
        print("loading",f)
        data[f] = {}
        
        with np.load(f) as npz:
            array = {}
            midline = {}
            offset = npz["offset"]
            frames = npz["frames"]
            
            if min_frame > frames.min():
                min_frame = frames.min()
            if max_frame < frames.max():
                max_frame = frames.max()
                
            if offset.T[0].min() < screen[0]:
                screen[0] = offset.T[0].min()
            if offset.T[1].min() < screen[1]:
                screen[1] = offset.T[1].min()
                
            if offset.T[0].max() > screen[2]:
                screen[2] = offset.T[0].max()
            if offset.T[1].max() > screen[3]:
                screen[3] = offset.T[1].max()
            
            # if all midlines have the same length (e.g. 12), the midline_points
            # array has a shape of (frames, midline_length, 2). otherwise the
            # shape is (overall_number_points_for_all_frames, 2) and where a midline ends
            # or begins is determined by the "midline_lengths" array of shape (frames, 1)
            # containing the number of points in each midline per frame
            midline = {}
            if len(npz["midline_points"].shape) == 2:
                i = 0
                indices = []
                for l in npz["midline_lengths"][:-1]:
                    i += l
                    indices.append(int(i))
                points = np.split(npz["midline_points"], indices, axis=0)
                for frame, point, off in zip(frames, points, offset):
                    midline[frame] = point + off
            else:
                for mpt, off, frame in zip(npz["midline_points"], offset, frames):
                    midline[frame] = mpt + off
            
            # outlines always have varying lengths, so they are always stored as
            # "outline_points" shape: (overall_number_outline_points, 2)
            # and
            # "outline_lengths" shape: (frames, 1)
            # as described above for midline_lengths/midline_points.
            i = 0
            indices = []
            for l in npz["outline_lengths"][:-1]:
                i += l
                indices.append(int(i))
            points = np.split(npz["outline_points"], indices, axis=0)
            outline = {}
            for frame, point, off in zip(frames, points, offset):
                outline[frame] = point + off
            
            data[f]["midline"] = midline
            data[f]["outline"] = outline
    
    # the dimensions of the tank
    screen[0] -= 10
    screen[1] -= 10
    screen[2] *= 1.1
    screen[3] *= 1.1
    #screen = [screen[0], screen[1], screen[2] - screen[0], screen[3] - screen[1]]
    input_shape = (screen[2] - screen[0], screen[3] - screen[1])#screen[2], screen[3])
    output_shape = (640, 640)
    fps = 40.0
    frame_range = []#[100, 500]
    
    if len(sys.argv) == 4:
        print("reading frame range from command line (", sys.argv[2:4],")")
        frame_range = [int(sys.argv[2]), int(sys.argv[3])]

    fourcc = cv.VideoWriter_fourcc(*'MJPG')
    filename = "output.avi"
    out = cv.VideoWriter(filename, fourcc, fps, output_shape, True)

    print("Writing video '"+filename+"' with frames",frames.min(),"-",frames.max())

    # this determines speed of the calculation and font sizes
    dpi = 192 / 4
    cv.destroyAllWindows()
    
    frames = np.arange(min_frame, max_frame+1)
    if len(frame_range) == 2:
        frames = frames[np.logical_and(frames >= frame_range[0], frames <= frame_range[1])]
        
    for chosen_frame in frames:
        fig = plt.figure(figsize=(output_shape[0] / dpi, output_shape[1] / dpi), dpi=dpi)
        fig.set_tight_layout(True)
        
        for key in data:
            if not chosen_frame in data[key]["outline"]:
                continue
            
            outline = np.array(data[key]["outline"][chosen_frame])
            plt.scatter(outline.T[0], outline.T[1], label="outline", s = 1)
    
            midline = data[key]["midline"]
            m = midline[chosen_frame]
            plt.scatter(m.T[0], m.T[1], label="midline", s = 1)
        
        plt.xlim(screen[0],screen[2])
        plt.ylim(screen[1],screen[3])
    
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        im = Image.open(buf)
        im = np.array(im).astype(np.uint8)
        buf.close()
        plt.close(fig)
    
        if int(chosen_frame) % int((frames.max() - frames.min())*0.1) == 0:
            print(chosen_frame,"/",frames.max())
    
        if (im.shape[1], im.shape[0]) != output_shape:
            print("different shape", output_shape, im.shape)
    
        im = im[:,:,0:3]
        out.write(im)
        cv.imshow("movie", im)
        cv.waitKey(1)
    out.release()
