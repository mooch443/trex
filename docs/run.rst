.. include:: names.rst

.. toctree::
   :maxdepth: 2


Basic Command-line Usage
========================

|trex| is usually launched from the command-line (with or without parameters), which will show a file opening dialog by default.

You probably *should* get familiar with the graphical user-interface of |trex| as well - for example, it makes it easy to detect trouble with parameter values immediately and gives you feedback after changing tracking parameters that you wouldn't get on the command-line. To see how that works, and for some additional detail on the general principles, have a look at :doc:`tutorials`.

The following sections, however, address directly providing parameters using the command-line (e.g. in a batch processing, or generally a more command-line affine use-case). Any additional number of parameters can be passed to |trex| as::

	trex [...] -PARAMETER VALUE

For example, in order to set the number of individuals to 5 and prefix all individuals names/files with "termite" instead of "fish", just change the values of ``track_max_individuals`` and ``individual_prefix`` when launching the application like so (we are also opening a file called "example.mp4" and use a specific settings file)::

	trex -i example.mp4 -s tmp.settings -track_max_individuals 5 -individual_prefix "termite"

If you've never run the program before on that specific video file, it'll first convert the mp4 file to a pv file and save an ``example.settings`` file. Conversion might take a bit - especially if its a particularly long video. Next time, however, it will open any existing pv files in tracking mode directly - if you want to generate the pv file again, you'd have to add force the task to ``convert`` like this::

	trex -i example.mp4 [...] -task convert

This will overwrite an existing ``example.pv`` (and ``example.settings``) file in that same folder.

Most parameters (except system-variables) can be adjusted after loading a video. A full reference of them is at :doc:`parameters_trex`.

.. NOTE::

	There is the possibility of starting it directly using a desktop short-cut (or double-clicking the executable itself), but this requires either a manual compile or changes to your environment variables (see :ref:`run-by-clicking`) which are not officially supported at the moment. 

Basic principles & Good practices
---------------------------------

|trex| offers many parameters that can be set by users. However, that does not mean that all parameters have to be considered in all cases. In fact, most of them will already be set to reasonable values that work in many situations:

	*If there is no problem, do not change parameters.*
	
Otherwise, if your problem cannot be solved by repeatedly mashing the same sequence of buttons (but with an increasingly stern expression on your face), please consider the list of things that happen frequently in (:doc:`faq`).

Parameters are either changed directly as part of the command-line (using ``-PARAMETER VALUE``), in settings files or from within the graphical user interface (or magically). Settings files are called [VIDEONAME].settings, and located either in the same folder as the video, or in the output folder (``~/Videos`` by default, or set using the command-line option ``-d /folder/path``). These settings files are automatically loaded along with the video. Settings that were changed by command-line parameters can be saved by pressing ``menu -> save config``.

Command-line parameters always override settings files.

If you know the number of individuals, specify before you do the tracking (using the parameter :param:`track_max_individuals`).

When converting videos, :param:`cm_per_pixel` should always be set to provide a valid conversion factor between pixels and real-world coordinates. By default, it's set to ``1`` - meaning there's no conversion and all units (even if it says they are ``cm``) are in pixels. You can set ``cm_per_pixel`` within the |trex| GUI, either in the initial settings dialog (under "tracking" > "calibrate") or in the tracking view. There, simply CTRL/⌘ + click into an empty spot of the arena. Hold CTRL/⌘ and click somewhere else: a button will pop up to define the selected length as a specific real-world length. See here: `changing the cm to px conversion factor <https://trex.run/docs/gui.html#changing-the-cm-px-conversion-factor>`_.

Using settings files
--------------------

The format is::

	parameter_name = value
	parameter2 = value2

If you save this as ``videoname.settings`` in your output directory (e.g. by default the same folder as your input video), |trex| will load these settings before loading any command-line parameters. Command-line, however, always overwrites anything loaded from ``.settings`` files.

.. _parameter-order:

Setting parameters in the correct order
---------------------------------------

Preferably set parameters in this order (with the goal to only match those objects that are your objects of interest, and exclude the ones that you do not want to track):

	- :param:`cm_per_pixel` (in .settings files, command-line or in |trex|)
	- :param:`track_threshold`
	- :param:`track_size_filter`

Now all objects of interest should have a cyan number next to them in RAW view (pressing ``D`` in tracking view switches to RAW and vice-versa). More "optional" parameters like can now be set in order to maximize the length of tracklets:

	- :param:`track_max_speed`
	- :param:`track_max_reassign_time`
	- :param:`track_posture_threshold`
	- :param:`outline_resample`
	- :param:`outline_curvature_range_ratio`

Other command-line tools
------------------------

Another tool is included, called ``pvinfo``, which can be used programmatically (e.g. in jupyter notebooks) to retrieve information about converted videos and the settings used to track them::

	$ pvinfo -i <VIDEO> -print_parameters [analysis_range,meta_video_size] -quiet
	analysis_range = [1000,50000]
	meta_video_size = [3712,3712]
	
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

.. _run-by-clicking:

Run software directly using shortcuts
-------------------------------------

Interaction with software on Unix-systems often takes place within a terminal. Under Windows, a lot of the typical interactions take place within a graphical user interface -- however, especially when installed within a conda environment, some additional environment variables need to be set. I am unsure whether this may influence other software or applications, so starting directly from the Anaconda3 PowerShell terminal should be the preferred approach.

If you still want to go the dangerous path, e.g. under Windows 10, you can adjust your environment variables by right-clicking "This Computer" -> "Properties" -> "Advanced System Settings" -> "Environment variables". Inside the "System variables" box, if it does not contain the variable "CONDA_PREFIX" yet, click on "New" to add it. Example::

	Name of the variable: CONDA_PREFIX
	Value of the variable: C:\Users\tristan\Anaconda3\envs\tracking

If it already exists, change the value accordingly. Now you should be able to create a Desktop shortcut for TRex and start it with a double-click.
