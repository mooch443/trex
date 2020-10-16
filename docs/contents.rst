.. include:: names.rst

Welcome to TRex's documentation!
================================

.. NOTE::
	This page (and the github repository at https://github.com/mooch443/trex) will be updated frequently at the moment, since |trex| is still in active development. If you find any issues, please report them using the github issue tracker!

|trex| is a tracking software designed to track and identify individuals and other moving entities using computer vision and machine learning. The work-load is split into two (not entirely separate) tools:

* **TGrabs**: Record or convert existing videos, perform live-tracking and closed-loop experiments
* **TRex**: Track converted videos (in PV format), use the automatic visual recognition, explore the data with visual helpers, export task-specific data, and adapt tracking parameters to specific use-cases

.. raw:: html

	<video autoplay muted loop playsinline id="myVideo" style="position: relative; padding-bottom: 15px; height: 0; overflow: hidden; max-width: 100%; height: auto;">
		<source src="composite_tracking_video.webm" type='video/webm; codecs="vp9"' />
		<source src="composite_tracking_video_.mp4" type='video/mp4; codecs="avc1"'>
		<source src="composite_tracking_video.mov" type='video/mp4; codecs="h264"' />
	</video>

Workflow
--------

|grabs| always has to be used first. |trex| is optional in some cases. Use-cases where |trex| is not required include:

* *Just give me tracks*: The user has a video and wants positional, or posture-related data for the individuals seen in the video. Maintaining identities is not required.
* *Closed-loop*: React to the behavior of individuals during a trial, e.g. lighting an LED when individuals get close to it, or run a python script every time individual 2 sees individual 3.

Whereas other use-cases are:

* *Maintaining identities*: Individuals are required to be assigned consistent identities throughout the entire video. Any results involving automatic identity correction will have to use |trex|.
* *Adjusting parameters with visual feedback*: While |grabs| includes a lot of the functionality of |trex|, it currently has no interface to directly test out parameters. Tracking parameters, specifically, have to be tested in |trex|. This is useful, e.g. when trying to figure out parameters for a batch process or adapting parameters for specific purposes.
* *Exploring and generating videos for presentations*: |trex| provides a rich set of functionalities for generating heatmaps and other useful visual information, as well as offering support to record anything that is on-screen to a AVI video file. For example, one can follow a subset of individuals and record every frame (lag-free and making sure that no frames are skipped). Any changes to the interface will be visible in the video as well.


.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   install
   run
   examples
   formats
   parameters_trex
   parameters_tgrabs
   faq