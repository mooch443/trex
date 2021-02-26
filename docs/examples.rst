.. include:: names.rst

.. toctree::
   :maxdepth: 2

Usage Examples
##############

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

This will save a couple of files named ``<VIDEO>_tracklet_images_*.npz`` in your output/data directory. ``<VIDEO>_tracklet_images.npz`` contains summaries of all consecutive segments for each individual in the form of ``['images', 'meta']``, where ``meta`` is matrix of ``Nx3`` (where ``N`` is the number of segments). The three columns are ID, segment start and segment end (frame numbers). Images is a matrix of ``NxWxH`` depending on the image dimensions set in :func:`recognition_image_size`. Here is an example:

How to use this data in Python
==============================

.. code:: ipython3

    import numpy as np
    import matplotlib.pyplot as plt

Open all images of a certain individual ID
------------------------------------------

.. code:: ipython3

    with np.load("Videos/data/video_tracklet_images_single_part0.npz") as npz:
        print(npz.files)
        
        images = npz["images"]  # load images
        ids    = npz["ids"]     # ids have the same length as images
        frames = npz["frames"]  # the frame for each row
        
        # so we can use them as a mask for the images array:
        print(images[ids == 0].shape)
        
        # now draw a median image of this fish. since it is normalized (orientation),
        # it will be a nice, clean picture of what it looks like most of the time.
        # if it does not, then your posture settings are probably off.
        # this only works after successful visual identification + correction of course.
        plt.figure(figsize=(5,5))
        plt.imshow(np.median(images[ids == 0], axis=0), cmap="Greys")
        plt.show()


.. parsed-literal::

    ['images', 'frames', 'ids']
    (19247, 80, 80)
    


.. image:: output_2_1.png


Now we want to see that for all individuals
-------------------------------------------

But we are using the meta tracklet pack for this. It contains only one
image per consecutive segment.

.. code:: ipython3

    with np.load("Videos/data/video_tracklet_images.npz") as npz:
        meta = npz["meta"]
        N = len(np.unique(meta[:, 0])) # how many fish do we have here?
        
        # plot all individuals in a row. this will probably be real tiny for many more individuals.
        f, axes = plt.subplots(1, N, figsize=(5*N, 5))
        for ax, i in zip(axes, ids):
            ax.axis('off')
            ax.imshow(np.median(npz["images"][npz["meta"][:, 0] == i], axis=0), cmap="Greys")
            ax.set_title(str(i))
        plt.show()



.. image:: output_4_0.png


We can now map from segments (meta) to tracklet images from the big file
------------------------------------------------------------------------

.. code:: ipython3

    for ID, start, end in meta:
        mask = np.logical_and(ids == ID, np.logical_and(frames >= start, frames <= end))
        print(ID, start,"-",end, images[mask].shape)


.. parsed-literal::

    0 0 - 40 (41, 80, 80)
    0 42 - 50 (9, 80, 80)

    [...]
    
    9 19235 - 19242 (8, 80, 80)
    9 19245 - 19251 (7, 80, 80)
    9 19252 - 19305 (54, 80, 80)
    9 19306 - 19316 (11, 80, 80)
    

