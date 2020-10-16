.. include:: names.rst

.. toctree::
   :maxdepth: 2

File formats
============

Export
******

Data
----

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

If one metric (such as anything posture-related) is not available in a frame -- either because the individual was not found, or because no valid posture was found -- then all affected metrics (sometimes all) will be set to infinity.

Posture
-------

(todo)

Tracklet images
---------------

(todo)

Visual fields
-------------

(todo)

PreprocessedVideo (pv)
**********************

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