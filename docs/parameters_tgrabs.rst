.. toctree::
   :maxdepth: 2

TGrabs parameters
#################
.. include:: names.rst

.. NOTE::
	|grabs| has a live-tracking feature, allowing users to extract positions and postures of individuals while recording/converting. For this process, all parameters relevant for tracking are available in |grabs| as well -- for a reference of those, please refer to :doc:`parameters_trex`.
.. function:: adaptive_threshold_scale(float)

	**default value:** 2


	Threshold value to be used for adaptive thresholding, if enabled.



.. function:: approximate_length_minutes(uint)

	**default value:** 0


	If available, please provide the approximate length of the video in minutes here, so that the encoding strategy can be chosen intelligently. If set to 0, infinity is assumed. This setting is overwritten by ``stop_after_minutes``.

	.. seealso:: :func:`stop_after_minutes`, 


.. function:: average_samples(uint)

	**default value:** 100


	Number of samples taken to generate an average image. Usually fewer are necessary for ``averaging_method``s max, and min.

	.. seealso:: :func:`averaging_method`, 


.. function:: averaging_method(averaging_method_t)

	**default value:** mode

	**possible values:**
		- `mean`: Sum all samples and divide by N.
		- `mode`: Calculate a per-pixel median of the samples to avoid noise. More computationally involved than mean, but often better results.
		- `max`: Use a per-pixel minimum across samples. Usually a good choice for short videos with black backgrounds and individuals that do not move much.
		- `min`: Use a per-pixel maximum across samples. Usually a good choice for short videos with white backgrounds and individuals that do not move much.

	Determines the way in which the background samples are combined. The background generated in the process will be used to subtract background from foreground objects during conversion.




.. function:: blob_size_range(range<float>)

	**default value:** [0.01,500000]


	Minimum or maximum size of the individuals on screen after thresholding. Anything smaller or bigger than these values will be disregarded as noise.



.. function:: cam_framerate(int)

	**default value:** -1


	If set to anything else than 0, this will limit the basler camera framerate to the given fps value.



.. function:: cam_limit_exposure(int)

	**default value:** 5500


	Sets the cameras exposure time in micro seconds.



.. function:: cam_resolution(size<int>)

	**default value:** [-1,-1]


	Defines the dimensions of the camera image.



.. function:: closing_size(int)

	**default value:** 3


	Size of the dilation/erosion filters for if ``use_closing`` is enabled.

	.. seealso:: :func:`use_closing`, 


.. function:: color_channel(uchar)

	**default value:** 1


	Index (0-2) of the color channel to be used during video conversion, if more than one channel is present in the video file.



.. function:: correct_luminance(bool)

	**default value:** false


	Attempts to correct for badly lit backgrounds by evening out luminance across the background.



.. function:: crop_offsets(offsets)

	**default value:** [0,0,0,0]


	Percentage offsets [left, top, right, bottom] that will be cut off the input images (e.g. [0.1,0.1,0.5,0.5] will remove 10%% from the left and top and 50%% from the right and bottom and the video will be 60%% smaller in X and Y).



.. function:: crop_window(bool)

	**default value:** false


	If set to true, the grabber will open a window before the analysis starts where the user can drag+drop points defining the crop_offsets.



.. function:: dilation_size(int)

	**default value:** 0


	If set to a value greater than zero, detected shapes will be inflated (and potentially merged). When set to a value smaller than zero, detected shapes will be shrunk (and potentially split).



.. function:: enable_closed_loop(bool)

	**default value:** false


	When enabled, live tracking will be executed for every frame received. Frames will be sent to the 'closed_loop.py' script - see this script for more information. Sets ``enable_live_tracking`` to true. Allows the tracker to skip frames by default, in order to catch up to the video.

	.. seealso:: :func:`enable_live_tracking`, 


.. function:: enable_difference(bool)

	**default value:** true


	Enables background subtraction. If disabled, ``threshold`` will be applied to the raw greyscale values instead of difference values.

	.. seealso:: :func:`threshold`, 


.. function:: enable_live_tracking(bool)

	**default value:** false


	When enabled, the program will save a .results file for the recorded video plus export the data (see ``output_graphs`` in the tracker documentation).



.. function:: equalize_histogram(bool)

	**default value:** false


	Equalizes the histogram of the image before thresholding and background subtraction.



.. function:: ffmpeg_crf(uint)

	**default value:** 20


	Quality for crf (see ffmpeg documentation) used when encoding as libx264.



.. function:: grabber_force_settings(bool)

	**default value:** false


	If set to true, live tracking will always overwrite a settings file with ``filename``.settings in the output folder.

	.. seealso:: :func:`filename`, 


.. function:: image_adjust(bool)

	**default value:** false


	Converts the image to floating-point (temporarily) and performs f(x,y) * ``image_contrast_increase`` + ``image_brightness_increase`` plus, if enabled, squares the image (``image_square_brightness``).

	.. seealso:: :func:`image_contrast_increase`, :func:`image_brightness_increase`, :func:`image_square_brightness`, 


.. function:: image_brightness_increase(float)

	**default value:** 0


	Value that is added to the preprocessed image before applying the threshold (see ``image_adjust``). The neutral value is 0 here.

	.. seealso:: :func:`image_adjust`, 


.. function:: image_contrast_increase(float)

	**default value:** 3


	Value that is multiplied to the preprocessed image before applying the threshold (see ``image_adjust``). The neutral value is 1 here.

	.. seealso:: :func:`image_adjust`, 


.. function:: image_square_brightness(bool)

	**default value:** false


	Squares the floating point input image after background subtraction. This brightens brighter parts of the image, and darkens darker regions.



.. function:: mask_path(path)

	**default value:** ""


	Path to a video file containing a mask to be applied to the video while recording. Only works for conversions.



.. function:: meta_age_days(int)

	**default value:** -1


	Age of the individuals used in days.



.. function:: meta_build(string)

	**default value:** ""


	The current commit hash. The video is branded with this information for later inspection of errors that might have occured.



.. function:: meta_cmd(string)

	**default value:** ""


	Command-line of the framegrabber when conversion was started.



.. function:: meta_conditions(string)

	**default value:** ""


	Treatment name.



.. function:: meta_conversion_time(string)

	**default value:** ""


	This contains the time of when this video was converted / recorded as a string.



.. function:: meta_misc(string)

	**default value:** ""


	Other information.



.. function:: meta_species(string)

	**default value:** ""


	Name of the species used.



.. function:: meta_write_these(array<string>)

	**default value:** ["meta_species","meta_age_days","meta_conditions","meta_misc","cam_limit_exposure","meta_real_width","meta_source_path","meta_cmd","meta_build","meta_conversion_time","frame_rate","cam_undistort_vector","cam_matrix"]


	The given settings values will be written to the video file.



.. function:: nowindow(bool)

	**default value:** false


	Start without a window enabled (for terminal-only use).



.. function:: quit_after_average(bool)

	**default value:** false


	If set to true, this will terminate the program directly after generating (or loading) a background average image.



.. function:: recording(bool)

	**default value:** true


	If set to true, the program will record frames whenever individuals are found.



.. function:: reset_average(bool)

	**default value:** false


	If set to true, the average will be regenerated using the live stream of images (video or camera).



.. function:: save_raw_movie(bool)

	**default value:** false


	Saves a RAW movie (.mov) with a similar name in the same folder, while also recording to a PV file. This might reduce the maximum framerate slightly, but it gives you the best of both worlds.



.. function:: solid_background_color(uchar)

	**default value:** 255


	A greyscale value in case ``enable_difference`` is set to false - TGrabs will automatically generate a background image with the given color.

	.. seealso:: :func:`enable_difference`, 


.. function:: stop_after_minutes(uint)

	**default value:** 0


	If set to a value above 0, the video will stop recording after X minutes of recording time.



.. function:: system_memory_limit(uint64)

	**default value:** 0


	Custom override of how many bytes of system RAM the program is allowed to fill. If ``approximate_length_minutes`` or ``stop_after_minutes`` are set, this might help to increase the resulting RAW video footage frame_rate.

	.. seealso:: :func:`approximate_length_minutes`, :func:`stop_after_minutes`, 


.. function:: tags_approximation(float)

	**default value:** 0.025


	Higher values (up to 1.0) will lead to coarser approximation of the rectangle/tag shapes.



.. function:: tags_debug(bool)

	**default value:** false


	(beta) enable debugging for tags.



.. function:: tags_enable(bool)

	**default value:** false


	(beta) live tracking of tags.



.. function:: tags_equalize_hist(bool)

	**default value:** true


	



.. function:: tags_model_path(path)

	**default value:** "tag_recognition_network.h5"


	The pretrained model used to recognize QRcodes/tags according to `<https://github.com/jgraving/pinpoint/blob/2d7f6803b38f52acb28facd12bd106754cad89bd/barcodes/old_barcodes_py2/4x4_4bit/master_list.pdf>`_. Path to a pretrained network .h5 file that takes 32x32px images of tags and returns a (N, 122) shaped tensor with 1-hot encoding.



.. function:: tags_num_sides(range<int>)

	**default value:** [3,7]


	The number of sides of the tag (e.g. should be 4 if it is a rectangle).



.. function:: tags_recognize(bool)

	**default value:** false


	(beta) apply an existing machine learning network to get tag ids.



.. function:: tags_save_predictions(bool)

	**default value:** false


	Save images of tags, sorted into folders labelled according to network predictions (i.e. 'tag 22') to '``output_dir``/tags_``filename``/<individual>.<frame>/*'. 

	.. seealso:: :func:`output_dir`, :func:`filename`, 


.. function:: tags_saved_only(bool)

	**default value:** false


	(beta) if set to true, all objects other than the detected blobs are removed and not written to the output video file.



.. function:: tags_size_range(range<double>)

	**default value:** [0,10]


	



.. function:: tags_threshold(int)

	**default value:** -5


	Threshold passed on to cv::adaptiveThreshold, lower numbers (below zero) are equivalent to higher thresholds / removing more of the pixels of objects and shrinking them.



.. function:: terminate_error(bool)

	**default value:** false


	Internal variable.



.. function:: test_image(string)

	**default value:** "checkerboard"


	Defines, which test image will be used if ``video_source`` is set to 'test_image'.

	.. seealso:: :func:`video_source`, 


.. function:: tgrabs_use_threads(bool)

	**default value:** true


	Use threads to process images (specifically the blob detection).



.. function:: threshold(int)

	**default value:** 9


	Threshold to be applied to the input image to find blobs.



.. function:: threshold_maximum(int)

	**default value:** 255


	



.. function:: use_adaptive_threshold(bool)

	**default value:** false


	Enables or disables adaptive thresholding (slower than normal threshold). Deals better with weird backgrounds.



.. function:: use_closing(bool)

	**default value:** false


	Toggles the attempt to close weird blobs using dilation/erosion with ``closing_size`` sized filters.

	.. seealso:: :func:`closing_size`, 


.. function:: video_conversion_range(pair<int,int>)

	**default value:** [-1,-1]


	If set to a valid value (!= -1), start and end values determine the range converted.



.. function:: video_reading_use_threads(bool)

	**default value:** true


	Use threads to read images from a video file.



.. function:: video_source(string)

	**default value:** "webcam"


	Where the video is recorded from. Can be the name of a file, or one of the keywords ['basler', 'webcam', 'test_image'].



