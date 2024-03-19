.. include:: names.rst

.. toctree::
   :maxdepth: 2

TRexA: An Overview
==================

TRexA is a beta version of the TRex software suite that aims to consolidate the functionality of TRex and TGrabs into a single, streamlined tool. TRexA is capable of performing segmentation, tracking, and additional machine learning tasks. However, as the GUI is still under development, both trexa and trex are currently in use. Currently, trexa essentially serves as a replacement for tgrabs, but it will eventually replace both TGrabs and TRex.

TRexA Beta Installation Guide
=============================

You can use one of the installers provided here, which installs a minimal portable version of conda and TRexA. The installers are available for Windows, Linux, and MacOS.

`Google Drive - TRexA Beta Installers <https://drive.google.com/drive/folders/1TNGnDQP4gqPwCBvAgBRDlJ_U4CCdPSAx?usp=drive_link>`_

On MacOS, if you installed to ~/trex-beta, you can then open a new terminal and type:

.. code-block:: bash

    eval "$(~/trex-beta/condabin/conda shell.bash hook)"

On other operating-systems, you'll have an Icon on your Desktop/in your start-menu that you can use to launch TRexA instead.

Installing via Conda
====================

Alternatively, to install the program using Conda, create an environment named 'beta':

Linux, Windows:
---------------
.. code-block:: bash

    conda create -n beta --override-channels -c trex-beta -c pytorch -c nvidia -c defaults trex pytorch-cuda=11.7

I personally had to add `=h67b0de4_1` to the end of that line on Windows, but try if it works without (could be a temporary problem with the anaconda servers) and only append if required.
Also on Windows you need to do this:

.. code-block:: bash

    conda activate beta
    python -m pip install opencv-python ultralytics "numpy>=1.23,<1.24" 'tensorflow-gpu>=2,<3'

In order to properly install all required packages.

MacOS:
------
.. code-block:: bash

    conda create -n beta --override-channels -c trex-beta -c pytorch-nightly -c conda-forge trex

Once installed, you can access TRexA by typing `trexa` in the terminal. You can pass parameters to the program using the syntax `-parameter <value>`, a standard functionality of Unix-like terminals.

Basic TRexA Parameters
======================

Here is a table of parameters for TRexA:

- `-i path/to/video.mp4`: Defines the input video.
- `-d path/to/output/folder`: Specifies the directory for the output .pv file and other resulting files.
- `-m path/to/model.pt`: Specifies a PyTorch model for segmentation/bbx detection/keypoint detection of video frames or a region cropped by a region model.
- `-dim/-detect_resolution`: Sets the resolution for the `-m` model.
- `-region_resolution`: Sets the resolution for the region proposal network.
- `-bm path/to/region_model.pt` (optional): Specifies a region proposal network (bounding box detection) that crops out certain regions from a larger image.
- `-tile_image N`: Tiles the image into a grid of `N` crops, each of which will be processed by the `-m` model.
- `-detect_iou_threshold t`: Sets the Intersection over Union (IoU) threshold `0<t<1`. IoU measures the overlap between two boundaries. Increasing `t` allows more overlap between objects. Decrease `t` if you encounter duplicate objects on top of each other.
- `-detect_conf_threshold t`: Sets a confidence threshold `0<=t<1`. This threshold controls how confident the model should be in its predictions to consider them valid. Lowering this threshold may result in more detections, but also potentially more false positives.

In addition to the parameters listed above, you can also use other parameters from TRex and TGrabs, provided they are relevant to your situation. Please refer to the TRex documentation at trex.run/docs for more details.

Example Use Case: Using TRexA to Convert Tortoise Videos
=========================================================

I use both trex and trexa for the conversions with the following parameters:

.. code-block:: bash

    trexa -d output/folder/path -i GH014629.MP4 -m /path/to/160-yolov8n-seg-cropped-2023-06-07-22_dtcrop_-1-mAP5095_0.74629-mAP50_0.995.pt -bm /path/to/480-yolov8n-2023-06-07-23_joint_dataset_dts1_dttr2-1-mAP5095_0.65648-mAP50_0.96707.pt -detect_resolution 160 -region_resolution 480 -detect_iou_threshold 0.5

This launches a user interface where you can experiment with the impact of various parameters on tracking and other tasks. Please note that the `cm_per_pixel` parameter may be incorrectly set, and tracking may not work optimally at this time. If the program closes without reporting any issues, you can open the resulting file for tracking:

.. code-block:: bash

    trex -d output/folder/path -i GH014629 -track_max_individuals 2 -meta_real_width 20 -track_size_filter '[0.05,1]' -track_threshold 0 -track_posture_threshold 0 -midline_invert -track_do_history_split false -track_label_confidence_threshold 0.1 -output_format csv -track_max_speed 100

The blob size ranges and max speed are dependent on the value set for `meta_real_width` or `cm_per_pixel`, so adjust accordingly. Adding `-auto_quit` to the command-line will automatically quit the application once tracking is complete. You can also use `-auto_train` in addition to that to automatically train a model on the resulting data and retrack with that information. However for e.g. GH019044 I got a perfect result + no tracking mistakes without any additional visual identification, so I didn't need to do that.

Otherwise, please refer to this section in the docs: `Visual identification <https://trex.run/docs/identification.html>`_. Be aware, however, that all results produced by a machine should in theory be checked again by a human. At least quickly. I have not done this for all videos I have ever tracked of course, because theory != reality, but it is certainly a good idea to do so. Mostly what I did was to check the results from the .csv files and see if there are many straight lines indicating potential jumps/errors. If there are, I would go back and check the video again. If not, I would just use the results as is.

Setting the Width of the Arena in the Program
=============================================

To set the width of the arena within the program, Ctrl+click (or CMD+click on Mac) on your reference points in the image background. This is preferably done in Debug mode (press the D key), as there are fewer clickable objects that could potentially interfere. Then, click on "use known length to calibrate" and enter the known length in centimeters. Once done, you can copy the value from `cm_per_pixel` and add it instead of `meta_real_width` to your other files, provided the conversion factor remains the same.

For more information, visit the `changing the cm to px conversion factor <https://trex.run/docs/gui.html#changing-the-cm-px-conversion-factor>`_ section in the TRex documentation.
