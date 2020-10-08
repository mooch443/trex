.. include:: names.rst

.. toctree::
   :maxdepth: 2


Basic usage
===========

|trex| can be launched simply by double-clicking it, or launching it from the command-line without parameters, which will show a file opening dialog. Its younger twin, |grabs|, can only be started from the terminal at the moment (we will be working on changing that). The following sections address the issue of directly providing parameters using the command-line for both softwares (e.g. in a batch processing, or generally a more command-line affine use-case). If you're launching |trex| by double-clicking it (or using voice commands), most parameters can be adjusted after loading a video -- others can be modified before that (e.g. the number of expected individuals).

Basic principles
----------------

Both |grabs| and |trex| offer many parameters that can be set by users. However, that does not mean that all parameters have to be considered in all cases. In fact, most of them will already be set to reasonable values that work in many situations:

	If there is no problem, do not change parameters.
	
Otherwise, if your problem cannot be solved by repeatedly mashing the same sequence of buttons (but with an increasingly stern expression on your face), please consider the list of probable solutions in (`Frequently asked questions and solutions to weird problems`_).

.. NOTE::

	Explicitly discouraging this specific kind of problem-solving is motivated by the results of informal (but extensive) user-studies conducted in the last three years.

Parameters are either changed directly as part of the command-line (using ``-PARAMETER VALUE``), in settings files or from within the graphical user interface (or magically). Settings files are called [VIDEONAME].settings, and located either in the same folder as the video, or in the output folder (set using ``-d /folder/path``, or ``~/Videos`` by default). These settings files are automatically loaded along with the video. Command-line parameters always override settings files. They can be generated automatically by opening |trex|, changing parameters and pressing ``menu -> save config``.


Running TGrabs
--------------

Running |grabs| usually involves the following parameters::

	./tgrabs -i [SOURCE] -o [DESTINATION] [ADDITIONAL]

**Source** can be any of the following:

* **basler**: A Basler camera. This can be followed up by :code:`-cam_serial_number [SERIAL]` if multiple cameras are attached.
* **webcam**: A generic webcam, e.g. attached via USB or integrated into the computer.
* **files**: Path to a video file, or an array of video files like :code:`-i [FILE1,FILE2,...,FILEN]` which are going to be concatenated in the order they appear.
* **pattern**: A patterned path to where the video file is located. This could be, e.g. :code:`-i /path/to/video/%6d.mp4` to indicate that the video-files are named :code:`000000.mp4` to any number :code:`XXXXXX.mp4`. If the video starts at a number higher than zero, one can attach :code:`-i /path/to/video/%6.S.E.mp4` to indicate start (:code:`S`) and end (:code:`E`) indexes.
* **interactive**: Is a test environment, generating a number of individuals based on the tracking parameter :code:`track_max_individuals` and a set :code:`frame_rate` (:code:`25` by default).

**Destination** is expected to be either a full path, or the name of the video (without extension). This will save a :code:`DESTINATION.pv` video file in the :code:`~/Videos/` folder.

**Additional** can be any number of parameters -- be it for tracking or image processing. A full reference of available parameters for |grabs| can be found at :doc:`parameters_tgrabs`.

Running TRex
------------

The tracker only expects an input file::

	./trex -i [VIDEONAME]

``VIDEONAME`` is either a full path to the video file, or the name of a video file in the default output folder (``~/Videos`` by default). This will open |trex| with all settings set to default, except if there is a ``[VIDEONAME].settings`` file present next to the video file or in the default output folder.

Just like with |grabs|, you can attach any number of additional parameters to the command-line, simply using ``-PARAMETER VALUE`` (see :doc:`parameters_trex`).

Frequently asked questions and solutions to weird problems
==========================================================

Segmentation/objects barely visible or too large!
	Is the background image good enough (if you're using background subtraction), e.g. are there artifacts from individuals visible in the background, or does the background change during the video?
		If the background is dynamic, you might have to disable subtraction by setting :func:`enable_difference` to false, and adjusting :func:`threshold` to a cut-off greyscale value. Otherwise, consider using a different :func:`averaging_method`. For example, max/mode are good for white backgrounds and short videos, or barely moving individuals.
	Is there a lot of magenta-colored noise outside of objects?
		Increase :func:`threshold` during recording/conversion.
	Objects are too small?
		Decrease :func:`threshold` during recording/conversion.
		
Trajectories jump around a lot for no particular reason!
	Changing :func:`track_max_speed` might help to mitigate this problem. Generally this can be the symptom of many different problems that lead to individuals being lost: size, speed, visibility issues, etc.: Sometimes individuals are lost because they are moving too fast (faster than the maximally allowed speed), or because they are expected to move much faster. Try lowering or increasing that limit. To get a hint at which speed to set, open up |trex|, track a few frames and select an individual - if there are consecutive frames for that individual, it will display a cm/s speed in the top-left overlay.

I set :func:`track_max_individuals` to zero, but it still does not track all individuals!
	Probably what's happening is that you have not created a ``.settings`` file for the given video yet (or not in the right folder). Try that, and then attach the command-line option again.

|trex| is really **laggy** and frequently complains about **too many combinations**!
	Pause the analysis (``,`` key, this may take a few seconds). The matching algorithm has diffculty separating individuals into distinct cliques of neighboring individuals, or there are simply too many of them. This could be because your video contains too many trackable objects and no limit on the number of individuals has been set (:func:`track_max_individuals`), or there are significant time-jumps in the video. If the number of individuals should be much lower than detected, check your :func:`track_threshold`/:func:`blob_size_ranges` settings. It is advisable to start |trex| with likely parameters, or pausing analysis to change parameters. Otherwise, if that does not fix anything, check your settings for :func:`track_max_speed`, which controls the size of the neighbourhood to be considered during matching, and reduce it until there are no further warnings.
	
|grabs| does not quit and only shows "[...] not properly closed [...]"!
	You may have to forcibly quit the application, either using a task manager, or by finding and manually ending its process::
		
		ps aux | grep tgrabs

I have attached my Basler camera, and now |grabs| is stuck initializing the camera! 
	Most likely the camera driver crashed. Try restarting your computer to fix it.