.. include:: names.rst

.. toctree::
   :maxdepth: 2


Workflow & Quick Start
======================

First, install the software as described in :doc:`install`.

The general workflow of using |trex| then is quite straight-forward. Usually, you'd have your videos already recorded and will simply

1. Open |trex|
2. Open the video file, change a few settings and click **Convert**
3. Wait a bit until you're dropped into **Tracking View**
4. Quickly check for mistakes and, if OK, export the data by pressing ``S``

To improve tracking performance, the software will produce a *cached* version of your video file (``.pv``) that contains all the information needed, but not more. This includes all objects of interest (i.e. not the background per frame) as well as a single averaged background image.

On Video Files and File Sizes
-----------------------------

Standard encoded video files, such as `.mp4`, can often be surprisingly difficult to scrub through - you may have noticed this, for example, as `delays` when trying to rewind or fast-forward a movie you're watching. |trex| *preprocessed* video files are designed to make scrubbing faster by avoiding *delta encoding* (i.e. storing only the changes between frames). Instead, all objects of interest in every frame are stored in full - omitting all background pixels. This enables seamless jumps (e.g. during `4x` playback) and fast random data access during tracking. On the downside, this approach can sometimes result in slightly larger file sizes compared to the original `.mp4` — though this depends on your specific situation and is not always the case.

The file size of a |trex| video also depends on your settings. For instance, the :param:`meta_encoding` parameter determines whether all RGB channels are stored, only greyscale, or none at all (resulting in much smaller files). Refer to the documentation for more details on these options.

If you're running out of storage space, you can delete the .pv file and reconvert the video later using the settings you previously saved.

