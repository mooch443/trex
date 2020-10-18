.. include:: names.rst

.. toctree::
   :maxdepth: 2

Examples
########

This section contains an assortment of common usage examples, taken from the real world. We will explain each example shortly and move on quickly. If certain details are unclear, :doc:`parameters_trex` or :doc:`parameters_tgrabs` might help!

TGrabs
******

Ground-roules:

	- Converting/recording has to be done before (or at the same time) as tracking!
	- Everything that appears pink in |grabs| is considered to be noise. If |grabs| is too choosy in your opinion, consider lowering ``threshold``, change ``blob_size_range`` to include the objects that are considered noise, or enabling ``use_closing``!
	- You should not delete your AVI after converting it to PV. Objects that are not considered noise, are saved losslessly in PV, but the rest is removed (that's the compression here).

Converting videos
=================

Just open a movie file and convert it to the PV format (it will be saved to the default output location, and named after the input file). Just for fun, we also set a different (higher) threshold::

	tgrabs -i <MOVIE> -threshold 35

We can switch to a different background subtraction method, by using::

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

TRex
****

.. NOTE::
	Keep in mind that all parameters specified here in the the command-line can also be accessed if you're already within the graphical user interface.

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

If your desired output are images of each individual, you can combine either of the options above and set ``output_image_per_tracklet`` to ``true`` and ``tracklet_max_images`` to ``0`` (which means 'no limit'). We will output only tracklet images, no tracking data/results files::

	trex -i <VIDEO> -output_image_per_tracklet -tracklet_max_images 0 \
		 -tracklet_normalize_orientation false -auto_quit -auto_no_results \
		 -auto_no_tracking_data

