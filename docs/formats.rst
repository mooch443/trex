.. include:: names.rst

.. toctree::
   :maxdepth: 2

File formats
************

Positional Data
===============

Upon hitting the ``S`` key/clicking the export tracking data button in the menu, or when using the ``auto_quit`` option, |trex| will save one file per individual. They are, by default, saved to ``data/[VIDEONAME]_fish[NUMBER].npz`` and contain all of the data fields selected from the export options. These include by default::

	X#wcentroid         (x-position of the centroid)
	Y#wcentroid         (y-   *)
	X                   (x-position of the head)
	Y
	SPEED#wcentroid     (magnitude of the second derivative of position, based on centroid)
	SPEED
	AX                  (x-component of acceleration in cm/s)
	AY                  (y-component *)
	time                (time in seconds from start of the video, consistent across individuals)
	frame               (monotonically increasing integer number, consistent across individuals)
	num_pixels          (number of pixels in the object)
	...

Each of these metrics is saved per frame, meaning that each metric mentioned here has the same number of values per individual. However, not all individuals have the same number of frames and do not necessarily start at the same time (even though the time variable is consistent across individuals). If an individual is first detected later in the video (not in the first frame), then that frame is the first one to appear in the exported data. Individuals may also not appear towards the end of the video, in which case there also won't be any data exported for that part.

.. NOTE::
	There are a couple hashtags in there -- these simply mean that the data-source for that metric is different. For example, ``wcentroid`` means that this metric is centered on the centroid (weighted by pixel values) of each individual. If no hashtag is provided, the metric centers on the head of each individual. These are usually closely related, but are different from each other e.g. when the individual moves its head independently from other parts of the body. There the head-based metric would show much more wiggling around than the centroid-based metric.

If one metric (such as anything posture-related) is not available in a frame -- either because the individual was not found, or because no valid posture was found -- then all affected metrics (sometimes all) will be set to ``infinity``. A typical way of opening such a file and plotting a trajectory would be::

    import numpy as np
    import matplotlib.pyplot as plt
    
    with np.load("video_fish0.npz") as npz:
        # sample output: ['threshold_reached', 'num_pixels', 'time',     \
        #     'midline_length', 'frame', 'Y#wcentroid', 'Y', 'missing',  \
        #     'X', 'SPEED', 'SPEED#pcentroid', 'MIDLINE_OFFSET',         \
        #     'X#wcentroid', 'SPEED#wcentroid']
        print(npz.files)
        
        X = npz["X#wcentroid"]
        Y = npz["Y#wcentroid"]
        
        # sample output: (30269,)
        # just a stream of X positions for fish0
        print(X.shape)
        
        # using the mask gets rid of np.inf values (otherwise the plot
        # might get weird). "missing" is 1 whenever the individual is 
        # not tracked in that frame:
        mask = ~npz["missing"].astype(np.bool)
        
        # plot a connected trajectory for fish0:
        plt.figure(figsize=(5,5))
        plt.plot(X[mask], Y[mask])
        plt.show()

Posture-data
============

For each individual (as long as :func:`output_posture_data` is set to ``true``), |trex| writes a file ``[output_path]/[FILENAME]_posture_*.npz`` containing the following fields:

.. raw:: html

	<style>
	.wy-table-responsive table td, .wy-table-responsive table th {
		white-space: normal !important;
	}
	.rst-content table.docutils th {
		border-color: #939799;
	}
	</style>

.. csv-table::
	:header: Name,Description
	:widths: 15, 35
	:stub-columns: 1
	
	frames, "Frame index for all data contained in all other arrays."
	offset, "Top-left corner (px) of the object that this posture was generated from."
	midline_lengths, "Number of points in the midline."
	midline_centimeters, "Length of the midline in cms (based on the :func:`cm_per_pixel` parameter)"
 	midline_offsets, "A bit obscure: Angle (rad) from start to end of the midline (line through first and last point)."
	midline_angle, "Angle (rad) of a line from head-position (first midline segment) through the midline segment at a fraction of :func:`midline_stiff_percentage` of the midline - approximating the heading of the individual."
	posture_area, "Area of the polygon spanned by the outline points (calculated using the Shoe lace formula)."
	midline_points, "2D Points (px) of the midline in real-world coordinates."
	midline_points_raw, "2D Points (px) of the midline in normalized coordinates."
	outline_lengths, "Number of points in the outline."
	outline_points, "Each outline point consists of X and Y, but each outline can be of a different length. To iterate through these points, one must keep a current index that increases by ``outline_lengths[frame]`` per frame."

For an example of how to use this data, have a look at `this <https://github.com/mooch443/trex/blob/master/docs/scripts/plot_posture_output.py>`_ example from the repository. It demonstrates loading all posture files from one video, drawing the posture output in matplotlib and saving it to a new movie file. This is what the output could look like:

.. image:: matplotlib_posture.gif

Tracklet images
===============

Tracklet images can be quite useful when a different software has to perform operations on individual images (or summary-images per segment), like, for example, using DeepLabCut or DeepPoseKit to estimate poses in more detail -- after tracking and visual identification in |trex| have ensured that we know who is whom. Interoperability between the tools can have other advantages, too. Images can be normalized, for example, and become easier to annotate manually -- these altered video sequences can be saved for each individual.

.. image:: collection.gif

The container ``<VIDEO>_tracklet_images.npz`` contains summaries of all consecutive segments for each individual in the form of ``['images', 'meta']``, where ``meta`` is matrix of ``Nx3`` (where ``N`` is the number of segments). The three columns are ID, segment start and segment end (frame numbers). Images is a matrix of ``NxWxH`` depending on the image dimensions set in :func:`recognition_image_size`.

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
        for ax, i in zip(axes, np.unique(meta[:, 0])):
            ax.axis('off')
            ax.imshow(np.median(npz["images"][meta[:, 0] == i], axis=0), cmap="Greys")
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
    

Visual fields
=============

(todo)

PreprocessedVideo (pv)
======================

Videos in the PV format are structured as follows::

	**[HEADER SECTION]**
	(string) "PV" + (version_nr)
	(byte)   channels
	(uint16) width
	(uint16) height
	(Rect2i) four ushorts with the mask-offsets left,top,right,bottom
	(uchar)  sizeof(HorizontalLine)
	(uint32) number of frames
	(uint64) pointer to index at the end of file
	(uint64) timestamp (time since 1970 in microseconds)
	(string) project name
	(byte*)  average img ([width x height] x channels)
	(size_t) mask present / mask size in bytes (if 0 no mask)
	[byte*]  mask, but only present if mask_size != NULL
	
	**[DATA SECTION]**
	for each frame:
	  (uchar) compression flag (if 1, the whole frame is compressed)
	  if compressed:
	     (uint32) original size
	     (uint32) compressed size
	     (byte*) lzo1x compressed data (see below for uncompressed)
	  else:
	     [UNCOMPRESSED DATA PER FRAME] {
	         (uint64) timestamp (in microseconds) since start of movie
	         (uint16) number of individual cropped images

	         for each object/blob:
	             (uint16) y of first HorizontalLine
	             (uint16) (n)umber of HorizontalLine structs
	             (byte*)  n * sizeof(HorizontalLine)
	             (byte*)  original image pixels ordered exactly as in HorizontalLines (BGR, CV_8UC(n))
	     }

	**[INDEX TABLE]**
	for each frame
	  (uint64) frame start position in file

	**[METADATA]**
	  (string) JSONized metadata array

Where the HorizontalLine struct is made up of::

	(uint16) x0
	(uint15) y0
	(1 bit ) eol

The last bit (EOL, or "end of line") in this case suggests to the reader that this line is the last on the current y-coordinate, meaning that the y-counter has to be incremented by one. This, together with the "y of first HorizontalLine", is enough to reconstruct the entire object.