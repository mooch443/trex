.. include:: names.rst

.. toctree::
   :maxdepth: 2


Basic usage
===========

|trex| can be launched simply by double-clicking it, or launching it from the command-line without parameters, which will show a file opening dialog. Its younger sibling, |grabs|, also offers a graphical user interface, but can only be started from the terminal at the moment (we will be working on changing that, and also potentially integrating it completely with |trex|). The following sections address the issue of directly providing parameters using the command-line for both softwares (e.g. in a batch processing, or generally a more command-line affine use-case). If you're launching |trex| by double-clicking it (or using voice commands), most parameters (except system-variables) can be adjusted after loading a video.

This page is a reference for some commonly used parameters of our software. Some common real-life usage examples can be found at :doc:`examples`.

Basic principles & Good practices
---------------------------------

Both |grabs| and |trex| offer many parameters that can be set by users. However, that does not mean that all parameters have to be considered in all cases. In fact, most of them will already be set to reasonable values that work in many situations:

	If there is no problem, do not change parameters.
	
Otherwise, if your problem cannot be solved by repeatedly mashing the same sequence of buttons (but with an increasingly stern expression on your face), please consider the list of things that happen frequently in (:doc:`faq`).

Parameters are either changed directly as part of the command-line (using ``-PARAMETER VALUE``), in settings files or from within the graphical user interface (or magically). Settings files are called [VIDEONAME].settings, and located either in the same folder as the video, or in the output folder (``~/Videos`` by default, or set using the command-line option ``-d /folder/path``). These settings files are automatically loaded along with the video. Settings that were changed by command-line parameters can be saved by pressing ``menu -> save config``.

Command-line parameters always override settings files.

If you know the number of individuals, specify before you do the tracking (using the parameter ``track_max_individuals``).

If you have more than 200 individuals and they are always in very close proximity to each other (or you get a lot of warnings), the tree-based matching method might be in trouble (combinatorically speaking). Consider changing your matching algorithm (``match_mode``) to ``approximate`` or ``hungarian``. These algorithms have down-sides to them, but they do scale better for many individuals. If you need something trustworthy: ``hungarian`` is the well-known Hungarian algorithm (https://en.wikipedia.org/wiki/Hungarian_algorithm)!

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

	def get_parameter(video, parameter, root = "", prefix = "", conda_env = "", info_path = "pvinfo"):
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
	    conda_env : path, optional
	            Either empty (default), or an absolute path to the conda environment in which pvinfo is installed 
	            (e.g. C:\\Users\\Tristan\\Anaconda3\\envs\\tracking)
	    info_path : str
	            Absolute path to the pvinfo utility executable

	    Returns
	    -------
	    object
	            The return type is determined by the type of the parameter requested
	    """

	    from subprocess import Popen, PIPE
	    import os
	    import ast
    
	    if len(conda_env) > 0:
	        if info_path == "pvinfo":
	            info_path = conda_env+os.sep+"bin"+os.sep+"pvinfo"
	    parameter_list = [info_path]
	    if root != "":
	            parameter_list += ["-d",root]
	    if prefix != "":
	            parameter_list += ["-p", prefix]

	    parameter_list += ["-i", video,"-quiet", "-print_parameters","["+parameter+"]"]

	    my_env = os.environ.copy()
	    if len(conda_env) > 0:
	        my_env["CONDA_PREFIX"] = conda_env
	        my_env["PATH"] = conda_env+";"+conda_env+os.sep+"bin;"+my_env["PATH"]

	    process = Popen(parameter_list, stdout=PIPE, env=my_env)
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

