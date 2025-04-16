.. include:: names.rst

.. toctree::
   :maxdepth: 2


Workflow & Quick Start
======================

First, install the software as described in :doc:`install`.

The general workflow of using |trex| then is quite straight-forward. Usually, you'd have your videos already recorded and will simply

1. Open |trex|
2. Select the input video file
3. Select the detection mode (e.g. ``background_subtraction``, or ``yolo``)
4. Depending on the mode, select the parameters for the detection algorithm (e.g. the YOLO model)
5. Click **Convert** to start the conversion process
6. Wait a bit until you're dropped into **Tracking View**
7. Check for mistakes (e.g. using keys M and N) and, if OK, export the data by pressing ``S`` twice.

.. NOTE::

   To improve tracking performance, the software will produce a *cached* version of your video file (``.pv``) that contains all the information needed, but not more. This includes all objects of interest (i.e. not the background per frame) as well as a single averaged background image.

Command-Line
------------

The workflow for the terminal is similar, but can be condensed down into a single command::

   trex -i <videofile> -task convert -detect_type yolo -m <yolo_model_path.pt>
   trex -i <videofile> -task convert -detect_type background_subtraction -detect_threshold 50

On Video Files and File Sizes
-----------------------------

Standard encoded video files, such as `.mp4`, can often be surprisingly difficult to scrub through - you may have noticed this, for example, as `delays` when trying to rewind or fast-forward a movie you're watching. |trex| *preprocessed* video files are designed to make scrubbing faster by avoiding *delta encoding* (i.e. storing only the changes between frames) and not having to perform detection over and over again every time you change some setting: 

All objects of interest in every frame are stored, but all background pixels are removed. This enables seamless jumps (e.g. during `4x` playback or going backwards) and fast random data access during tracking. On the downside, this approach can (occasionally) result in slightly larger file sizes compared to the original `.mp4` — though this depends on your specific situation and is not always the case. Generally speaking, if you have small*ish* objects you're going to be fine.

The file size of a |trex| video also depends on your settings. For instance, the :param:`meta_encoding` parameter determines whether all RGB channels are stored, only greyscale, or none at all (resulting in much smaller files). Refer to the documentation for more details on these options.

If you're running out of storage space, you can delete the .pv file and reconvert the video later using the settings you previously saved. R3G3B2 encoding was added specifically for that purpose:

Let's say you have attached color tags to your individuals, so you really want to keep some kind of color information. This makes visual identification an easy to solve problem, ... but those are also pretty long videos, so optimally you'd want to minimize filesize. R3G3B2 encoding was added specifically for this purpose. It (lossy) compresses the video to a size equivalent to the greyscale encoding, but sorts everything into 256 different colors across the spectrum. The result is a video that is much smaller than the original, but still retains enough color information for visual identification.
