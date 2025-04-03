.. include:: names.rst

.. toctree::
   :maxdepth: 2

Usage examples
##############

This section contains an assortment of common usage examples, taken from the real world, as well as some terminal basics. We will explain each example shortly and move on quickly. If certain details are unclear, :doc:`parameters_trex` might help!

Command-line
============

.. |movie| replace:: path/to/VIDEO.mp4

Often, the command-line interface is the most powerful way to use |trex|, as it allows you to automate tasks and run them in batch mode. The following examples will show you how to use the command-line interface to achieve common tasks. Just to note: you can reach most or all of these functions through the graphical user interface, using the same parameter names. Instructions from here should be fairly easy to transfer over!

As you may know, simply starting the program without any arguments will open the graphical user interface. If you want to use the command-line interface, you have to specify the input using the ``-i`` option. For example, to open a video file, you would use::

	trex -i webcam

.. NOTE::
	For multiple webcams, add the :param:`webcam_index` parameter. For example, to open the second webcam, use ``-i webcam -webcam_index 1``. You can also specify a video file using the ``-i`` option, e.g. ``-i /path/to/VIDEO.mp4``.

.. raw:: html

   <p>This will open the webcam, if you have one installed and allow the program to use it, and use <code class="docutils literal notranslate"><span class="pre">yolov8n-pose</span></code> (see <a href="https://docs.ultralytics.com/models/yolov11/#supported-tasks-and-modes" target="_blank">YOLOv11 models</a>) to find you in the picture.</p>
   
Just for fun, we also set a different :param:`detect_iou_threshold` which will change the IOU threshold for YOLO object detection - the higher the percentage, the more overlap between bounding boxes is allowed. The default is 70%, but we set it to 35%::

	trex -i webcam -detect_iou_threshold 0.35

You may have already noticed that, by default, |trex| will see if a PV file already exists for the video you're trying to open. If it does, it will open it and you will end up in the tracking view immediately. However, we want to start over from scratch here - which can be enforced by adding the ``-task convert`` option in the same way::

	trex -i webcam -task convert -detect_iou_threshold 0.35

The ``detect_iou_threshold`` here is simply the parameter :param:`detect_iou_threshold`, as described in the documentation. You may add any parameter found in there to the command-line, and it will be evaluated when the program starts - if there are any errors, an ``ERROR`` will be displayed somewhere in the command-line output. Those errors might also be interesting in case its not a user error, but a software bug (which you are welcome to `report here <https://github.com/mooch443/trex/issues/new?assignees=mooch443&labels=bug&template=bug_report.md&title=>`_ on GitHub!).

For example, we can also limit the number of individuals to track::

	trex -i webcam -task convert -detect_iou_threshold 0.35 -track_max_individuals 1

This will force |trex| to (re-)convert the video to PV format, overwriting an existing ``VIDEO.pv`` file in the current folder.

If you want the program to quit after it's done, you can use the ``-auto_quit`` option, which also exports trajectory data (if not disabled by ``-auto_no_tracking_data``). Other options omitted, this would look like this::

	trex -i webcam [...] -auto_quit

By default, |trex| will save the resulting .pv file in the same folder as the source video (as well as any exported trajectory data, which will land inside a ``data`` folder). If you want to save it somewhere else, you can use the ``-d`` option::

	trex -i webcam [...] -d /path/to/output/to

Parameters, often also called settings, can be stored in settings files. Almost all parameters can be passed to the program via such a settings file using the ``-s`` option::
	
	trex -i webcam [...] -s /path/to/default.settings

TRex: general usage examples
****************************

.. NOTE::
	Keep in mind that all parameters specified here in the command-line can also be accessed if you're already within the graphical user interface. Just type into the textfield on the bottom left of the screen and it will auto-complete parameter names for you. See also :doc:`gui`.

Open a video::

	trex -i <VIDEO>

*Hey, but I know how many individuals I have!* If you do, you should always specify (although |trex| tries to find it out by itself)::

	trex -i <VIDEO> -track_max_individuals 8

Open a video, track it using the parameters set in command-line and the settings file at path <SETTINGS>, and save all selected output options to ``/data/<VIDEO_NAME>_fish*.npz``, etc.::

	trex -i <VIDEO> -s <SETTINGS> -auto_quit

Same as above, but don't save tracking data (only posture and ``.results``)::

	trex -i <VIDEO> -s <SETTINGS> -auto_quit -auto_no_tracking_data

Launch the tracking software, track it and automatically correct it using the visual identification. For this, you always have to specify the number of individuals, either via the command-line or in ``<VIDEO_NAME>.settings``::

	trex -i <VIDEO> -track_max_individuals 10 -auto_train

Same as above, but also save stuff and quit (e.g. for batch processing)::

	trex -i <VIDEO> -track_max_individuals 10 -auto_train -auto_quit

Don't show the graphical user-interface (really only useful when combined with ``auto_`` options or some serious terminal-based hacking). It can be quit using CTRL+C (or whatever is the equivalent in your system/terminal)::

	trex -i <VIDEO> -nowindow

Create tracklet images to use with other posture estimation software
********************************************************************

If your desired output are images of each individual, you can combine either of the options above and set ``output_image_per_tracklet`` to ``true`` and ``tracklet_max_images`` to ``0`` (which means 'no limit'). We will output only tracklet images, no tracking data/results files::

	trex -i <VIDEO> -output_image_per_tracklet -tracklet_max_images 0 \
		 -tracklet_normalize_orientation true -auto_quit -auto_no_results \
		 -auto_no_tracking_data

This will save a couple of files named ``<VIDEO>_tracklet_images_*.npz`` in your output/data directory. This format is described in more detail, and with an example of how it can then be used in Python, is `here <formats.html#tracklet-images>`_.

Create a short clip of objects with or w/o background after converting to PV
****************************************************************************

The tool ``pvconvert``, included in the standard installation of |trex|, can be used to achieve this. It reads the PV file format and exports sequences of images. For example::

	pvconvert -i /Volumes/Public/videos/group_1  \
		-disable_background true             \
		-start_frame 0 -end_frame 20         \
		-o /Volumes/Public/frames            \
		-as_gif true                         \
		-scale 0.75

produces this gif, which is cropped, scaled, short, and has lost its background:

.. image:: animated_frames.gif

Closed-loop
===========

To enable closed-loop, open |trex| using::

	trex -i webcam -closed_loop_enable -track_max_individuals 1

.. NOTE::
	Now you also have to attach ``track_`` parameters and set everything up properly for tracking (see next section)!

Every frame that has been tracked will be forwarded to your python script. Be aware that if your script takes too long, frames might be dropped and the tracking might become less reliable. In cases like that, or with many individuals, it might be beneficial to change ``match_mode`` to ``approximate`` (if you don't need extremely good identity consistency, just general position information).

You may, of course, edit the closed_loop file according to your needs. The exact path is displayed in the temrinal, which you maybe shouldn't modify but *copy* to a different location. You can then use :param:`closed_loop_path` to point |trex| to the correct script with custom code. The path used is displayed in the terminal (by default its in ``$CONDA_PREFIX/usr/share/trex/closed_loop_beta.py``).