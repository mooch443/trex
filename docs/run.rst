.. include:: names.rst

.. toctree::
   :maxdepth: 2


Basic usage
===========

|trex| can be launched simply by double-clicking it, or launching it from the command-line without parameters, which will show a file opening dialog. Its younger sibling, |grabs|, also offers a graphical user interface, but can only be started from the terminal at the moment (we will be working on changing that, and also potentially integrating it completely with |trex|). The following sections address the issue of directly providing parameters using the command-line for both softwares (e.g. in a batch processing, or generally a more command-line affine use-case). If you're launching |trex| by double-clicking it (or using voice commands), most parameters (except system-variables) can be adjusted after loading a video.

Basic principles
----------------

Both |grabs| and |trex| offer many parameters that can be set by users. However, that does not mean that all parameters have to be considered in all cases. In fact, most of them will already be set to reasonable values that work in many situations:

	If there is no problem, do not change parameters.
	
Otherwise, if your problem cannot be solved by repeatedly mashing the same sequence of buttons (but with an increasingly stern expression on your face), please consider the list of probable solutions in (`Frequently asked questions and solutions to weird problems`_).

.. NOTE::

	Explicitly discouraging this specific kind of problem-solving is motivated by the results of informal (but extensive) user-studies conducted in the last four years.

Parameters are either changed directly as part of the command-line (using ``-PARAMETER VALUE``), in settings files or from within the graphical user interface (or magically). Settings files are called [VIDEONAME].settings, and located either in the same folder as the video, or in the output folder (``~/Videos`` by default, or set using the command-line option ``-d /folder/path``). These settings files are automatically loaded along with the video. Settings that were changed by command-line parameters can be saved by pressing ``menu -> save config``.

Command-line parameters always override settings files.

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

If multiple files match the **pattern**, then they will be concatenated into one long video. This can be useful for videos that have been split into many small parts, or just a convenient way of e.g. training visual identification on multiple videos of the same individuals.

If there are ``[XXXXX].npz`` files (named exactly like the video files but with a different extension) in the video folder, then |grabs| will attempt to use them for frame-timestamps. The format of these files is expected to be::

	- 'frame_time': an array of N doubles for all N frames in the video segment
	- 'imgshape': a tuple of integers (width, height) of the video
	
Other useful options are::

	- 'meta_real_width': The width of what is seen in the video in cms. This is used to convert px -> cm internally, and is saved as meta information inside the .pv file.
	- 'meta_species': Species (meta-information, entirely optional)

Running TRex
------------

The tracker only expects an input file::

	./trex -i [VIDEONAME]

``VIDEONAME`` is either a full path to the video file, or the name of a video file in the default output folder (``~/Videos`` by default). This will open |trex| with all settings set to default, except if there is a ``[VIDEONAME].settings`` file present next to the video file or in the default output folder.

Just like with |grabs|, you can attach any number of additional parameters to the command-line, simply using ``-PARAMETER VALUE`` (see :doc:`parameters_trex`).

Other command-line tools
------------------------

Another tool is included, called ``pvinfo``, which can be used programmatically (e.g. in jupyter notebooks) to retrieve information about converted videos and the settings used to track them::

	$ pvinfo -i <VIDEO> -print_parameters [analysis_range,video_size] -quiet
	analysis_range = [1000,50000]
	video_size = [3712,3712]
	
If no settings (apart from ``-i <pv-videopath>``) are provided, it displays some information on the PV file::

	$ pvinfo -i group_1
	[15:24:00] pv::File<V6, 182.97MB, '/Users/tristan/Videos/group_1', [3712 x 3712], 19317 frames, no mask>
	
	crop_offsets: [0,0,0,0]
	Time of recording: 'Tue Apr 28 06:42:28 2020'
	Length of recording: '00d:00h:10m:03s'
	Framerate: 32fps (31.25ms)
	
	Metadata: {"meta_species": "Zebrafish", "meta_age_days": 25, "meta_conditions": "", "meta_misc": "", "cam_limit_exposure": 5500, "meta_real_width": 30, "meta_source_path": "/media/nvme/group_1.avi", "meta_cmd": " ./tgrabs -i /media/nvme/group_1.avi -s /media/nvme/convert_group_1.settings -enable_live_tracking", "meta_build": "<hash>", "meta_conversion_time": "28-04-2020 06:42:09", "frame_rate": 32}

It can be useful for retrieving values of tracking parameters from external scripts, like in python (also things like ``video_size``)::

	def get_parameter(video, parameter, root = "", prefix = "", info_path = "/path/to/pvinfo"):
		"""Uses the pvinfo utility to retrieve information about tracked videos. 
		This information is usually saved in the .pv files / .settings / .results files. 
		The pvinfo utility will emulate a call to TRex + loading existing .results files 
		and settings files in the same order.

		Parameters
		----------
		video : str
			Name of the video, or absolute path to the video
		parameter : str
			Name of the parameter to retrieve (see https://trex.run/docs/parameters_trex.html)
		root : str, optional
			Either empty (default), or path to the folder the video is in
		prefix : str, optional
			Either empty (default), or name of the output_prefix to be used
		info_path : str
			Absolute path to the pvinfo utility executable

		Returns
		-------
		object
			The return type is determined by the type of the parameter requested
		"""
	
		from subprocess import Popen, PIPE
		import ast
	
		if type(info_path) == type(None):
			process = Popen(["which", "tgrabs"], stdout=PIPE)
			(output, err) = process.communicate()
			exit_code = process.wait()
			if exit_code != 0:
				raise Exception("Cannot retrieve info path.")
			info_path = output.decode("utf-8").split("=")[-1][1:-1]
			info_path = info_path.split("/tgrabs")[0] + "/info"
			print(info_path)
	
		parameter_list = [info_path]
		if root != "":
			parameter_list += ["-d",root]
		if prefix != "":
			parameter_list += ["-p", prefix]
	
		parameter_list += ["-i", video,"-quiet","-print_parameters","["+parameter+"]"]
	
		process = Popen(parameter_list, stdout=PIPE)
		(output, err) = process.communicate()
		exit_code = process.wait()
		if exit_code == 0:
			return ast.literal_eval(output.decode("utf-8").split("=")[-1][1:-1])
		raise Exception("Cannot retrieve "+parameter+" from "+video+": "+output.decode("utf-8")+" ("+str(exit_code)+") "+" ".join(parameter_list))

But it has many other uses, too! For example, it can be used to save heatmap information that can be visualized in |trex| (but can not currently be saved directly from |trex| -> will be soon)::

	pvinfo -i video -heatmap -heatmap_source SPEED -heatmap_resolution 128 -heatmap_frames 100 -heatmap_dynamic
	
Or to display information about objects inside the saved frames::

	$ pvinfo -i <VIDEO> -blob_detail -quiet
	[15:24:07] 190315246 bytes (190.32MB) of blob data
	[15:24:07] Images average at 512.654519 px / blob and the range is [2-2154] with a median of 616.
	[15:24:07] There are 10 blobs in each frame (median).



Batch processing support
========================

|trex| and |grabs| both offer full batch processing support. All parameters that can be setup via the settings box (and even some that are read-only when the program is already started), can be appended to the command-line -- as mentioned above. For batch processing, special parameters are available::

	auto_quit			  # automatically saves all requested data to the output folder and quits the app
	auto_train			 # automatically attempts to train the visual identification if successfully tracked
	auto_apply			 # automatically attempts to load weights from a previous training and auto correct the video
	auto_no_results		# do not save a .results file
	auto_no_tracking_data  # do not save the data/file_fishX.npz files
	auto_no_memory_stats   # (enabled by default) do not save memory statistics


Frequently asked questions and solutions to weird problems
==========================================================

I am using Windows and Python cannot be initialized successfully!
	If you compiled or installed |trex| in a conda environment and it is having trouble finding the necessary files, you can try to help it by editing the default.settings file inside the application folder (be careful if you're compiling the program on your own, the one in the build folder gets replaced by the one in the folder ``[root]/Application/default.settings`` after every build that altered anything). Just put a ``python_path = "C:\Users\[USERNAME]\Anaconda3\envs\[ENVNAME]"`` inside it and try starting again! Otherwise, you can be more savage and set the ``PYTHONHOME`` variable to the same folder and add it to ``PATH``, too (this breaks other anaconda stuff most likely).

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