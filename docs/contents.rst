.. include:: names.rst

Welcome to TRex's documentation!
################################

.. NOTE::
	This page (and the github repository at https://github.com/mooch443/trex) will be updated frequently at the moment, since |trex| is still in active development. If you find any issues, please report them using the github issue tracker!

|trex| is a tracking software designed to track and identify individuals and other moving entities using computer vision and machine learning. 

Some of the key features are:

1. Record or convert existing videos
2. Track multiple individuals in these videos
3. Automatic visual identification of individuals to correct tracking errors
4. Visual data exploration
5. Export task-specific data, such as trajectories, heatmaps, extract individual videos, and more
6. Adjust tracking parameters with visual feedback

See :doc:`install` and :doc:`run` for instructions on how to install and use our software.

.. raw:: html

	<video autoplay muted loop playsinline id="myVideo" style="position: relative; padding-bottom: 15px; height: 0; overflow: hidden; max-width: 100%; height: auto;">
		<source src="composite_tracking_video.webm" type='video/webm; codecs="vp9"' />
		<source src="composite_tracking_video_.mp4" type='video/mp4; codecs="avc1"'>
		<source src="composite_tracking_video.mov" type='video/mp4; codecs="h264"' />
	</video>

Use-cases
=========

|trex| is a versatile tool, designed with animal behavior research in mind, integrating tracking capabilities with support for both background subtraction in simple scenarios and advanced YOLOv8 detect/segmentation/pose analysis for more complex scenarios. It features a graphical interface for preprocessing trajectory data, with options for exporting to Python/R (npz or csv files) for further study. It also offers advanced machine learning tools to automatically identify specific individuals/objects in a video, or categorize them based on their appearance with few manual clicks.

There is lots of functionality in |trex|, but here are some common use-cases that we have in mind when developing the software:

* **Tracking large groups**: |trex| can track many individuals in a group with a changing number of individuals, e.g. when individuals are entering or leaving the video frame / hide in shelters. In the simplest case identities can not be guaranteed, but the software provides additional information about reliable trajectory pieces (consecutive segments) per individual.

* **Tracking < 100 individuals while maintaining identities**: |trex| can use visual identification to recognize individuals in a fixed group, 'getting to know them personally', and maintaining their identities throughout the video. This process takes longer than just tracking, but may be required for your research (see :doc:`identification` for more information). It is possible to identify individuals in the same group across videos, too.

* **Categorize individuals based on appearance**: |trex| can be used to categorize individuals based on their appearance, e.g. to distinguish different species or phenotypes (if sufficiently visually distinct). This also works for large quantities of individuals, such as termite workers vs. soldiers.

* **Adjusting parameters with visual feedback**: This is useful, e.g. when trying to figure out parameters for a batch or adapting parameters for specific videos.

* **Extracting data for further analysis**: |trex| can export data in a variety of formats, including NPZ files, CSV files, and videos. This allows you to use the data in your favorite analysis software, or to share it with others.
 
* **Exploring the results and generating videos for presentations**: |trex| provides a rich set of functionalities for generating heatmaps and other useful visual information, as well as offering support to record anything that is on-screen to a AVI/MP4 video file. For example, one can follow a subset of individuals and record every frame (lag-free and making sure that no frames are skipped). Any changes to the interface will be visible in the video as well.

Reference
=========

If you use this software in your work, please cite our `open-access paper <https://elifesciences.org/articles/64000>`_:

.. code:: raw

   @article {walter2020trex,
      article_type = {journal},
      title = {TRex, a fast multi-animal tracking system with markerless identification, and 2D estimation of posture and visual fields},
      author = {Walter, Tristan and Couzin, Iain D},
      editor = {Lentink, David},
      volume = 10,
      year = 2021,
      month = {feb},
      pub_date = {2021-02-26},
      pages = {e64000},
      citation = {eLife 2021;10:e64000},
      doi = {10.7554/eLife.64000},
      url = {https://doi.org/10.7554/eLife.64000},
      journal = {eLife},
      issn = {2050-084X},
      publisher = {eLife Sciences Publications, Ltd},
   }

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:
   
   install
   run
   update
   examples
   gui
   identification
   batch
   formats
   parameters_trex
   faq