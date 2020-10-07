def axes_for_array(length):
    import numpy as np
    import matplotlib.pyplot as plt

    rest = 0
    cols = min(16, length)
    if length % cols > 0:
        rest = 1

    figsize = (cols * 5, 5 * (length // cols + rest))
    fig, axes = plt.subplots(length // cols + rest, cols, figsize=figsize)
    print(figsize)
    
    axes = np.array(axes).flatten()
    for ax in axes:
        ax.axis('off')
    axes = axes[:length]

    return fig, axes

def figure_as_image():
    import io
    from PIL import Image
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2 as cv
    import os
    
    plt.gcf().set_tight_layout(True)
    #plt.gcf().patch.set_facecolor('black')
    plt.gcf().patch.set_alpha(0.75)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', transparent=True, facecolor=plt.gcf().get_facecolor(), edgecolor='none')
    buf.seek(0)
    im = Image.open(buf)
    im = np.array(im).astype(np.uint8)
    buf.close()
    plt.close(plt.gcf())
    
    return im

def show_figure(title="plot", output_path="", im = None):
    import TRex
    from PIL import Image
    import numpy as np
    import cv2 as cv
    import os

    if type(im) == type(None):
        im = Image.fromarray(figure_as_image())
    else:
        im = Image.fromarray(im)
    try:
        path = "/var/www/example.com/html/"+title.replace(" ", "_").replace("/", "-")+".png"
        im.save(path, "PNG")
        TRex.log("saved as"+str(path))
    except Exception as e:
        TRex.warn(str(e))

    if len(output_path) > 0:
        try:
            if not output_path.endswith(os.sep):
                output_path = output_path + os.sep
            path = output_path+title.replace(" ", "_").replace("/", "-")+".png"
            im.save(path, "PNG")
            TRex.log("saved as"+str(path))
        except Exception as e:
            TRex.warn(str(e))
    
    im = np.array(im).astype(np.uint8)

    #TRex.imshow(title, im)
    
