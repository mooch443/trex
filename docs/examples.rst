.. include:: names.rst

.. toctree::
   :maxdepth: 2

Usage Examples
##############

This section contains an assortment of common usage examples, taken from the real world. We will explain each example shortly and move on quickly. If certain details are unclear, :doc:`parameters_trex` or :doc:`parameters_tgrabs` might help!

TGrabs examples
***************

Some things that are good to know:

	- Converting/recording has to be done before (or at the same time) as tracking!
	- Everything that appears pink in |grabs| is considered to be noise. If |grabs| is too choosy in your opinion, consider lowering ``threshold``, change ``blob_size_range`` to include the objects that are considered noise, or enabling ``use_closing``!
	- You should not delete your AVI after converting it to PV. Objects that are not considered noise, are saved losslessly in PV, but the rest is removed (that's the compression here).

Converting videos
=================

Just open a movie file and convert it to the PV format (it will be saved to the default output location, and named after the input file). Just for fun, we also set a different (higher) threshold::

	tgrabs -i <MOVIE> -threshold 35

We can switch to a different method for generating the background that is used for background-subtraction (which is how the foreground objects are detected), by using :func:`averaging_method`::

	tgrabs -i <MOVIE> -threshold 35 -averaging_method mode -reset_average 

The background will be saved to a png file in the output folder. You can edit it manually, too (until you use use ``reset_average``).

Record using a Basler camera
============================

Same options as above, but the input is different (note that you'll have to compile the software yourself in order to use this - with the Basler SDK enabled/installed on your system)::

	tgrabs -i basler

Closed-loop
===========

To enable closed-loop, edit the ``closed_loop.py`` file (it contains a few examples) and open tgrabs using::

	tgrabs -i basler -enable_closed_loop -threshold 35 -track_threshold 35

.. NOTE::
	Now you also have to attach ``track_`` parameters and set everything up properly for tracking (see next section)!

Every frame that has been tracked will be forwarded to your python script. Be aware that if your script takes too long, frames might be dropped and the tracking might become less reliable. In cases like that, or with many individuals, it might be beneficial to change ``match_mode`` to ``approximate`` (if you don't need extremely good identity consistency, just general position information).

TRex: general usage
*******************

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

The tool ``pvconvert``, included in the standard install of |trex|, can be used to achieve this. It reads the PV file format and exports sequences of images. For example::

	pvconvert -i /Volumes/Public/videos/group_1  \
		-disable_background true             \
		-start_frame 0 -end_frame 20         \
		-o /Volumes/Public/frames            \
		-as_gif true                         \
		-scale 0.75

produces this gif, which is cropped, scaled, short, and has lost its background:

.. image:: animated_frames.gif

