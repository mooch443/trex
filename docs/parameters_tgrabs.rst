.. toctree::
   :maxdepth: 2

TGrabs parameters
#################
.. include:: names.rst

.. NOTE::
	|grabs| has a live-tracking feature, allowing users to extract positions and postures of individuals while recording/converting. For this process, all parameters relevant for tracking are available in |grabs| as well -- for a reference of those, please refer to :doc:`parameters_trex`.
.. function:: enable_closed_loop(bool)
	:noindex:

	**default value:** false


	When enabled, live tracking will be executed for every frame received. Frames will be sent to the 'closed_loop.py' script - see this script for more information. Sets `enable_live_tracking` to true. Allows the tracker to skip frames by default, in order to catch up to the video.

	.. seealso:: :func:`enable_live_tracking`, 


.. function:: mask_path(path)
	:noindex:

	**default value:** ""


	Path to a video file containing a mask to be applied to the video while recording. Only works for conversions.



.. function:: meta_build(string)
	:noindex:

	**default value:** ""


	The current commit hash. The video is branded with this information for later inspection of errors that might have occured.



.. function:: meta_misc(string)
	:noindex:

	**default value:** ""


	Other information.



.. function:: meta_species(string)
	:noindex:

	**default value:** ""


	Name of the species used.



.. function:: cam_framerate(int)
	:noindex:

	**default value:** 30


	[BASLER] If set to anything else than 0, this will limit the basler camera framerate to the given fps value.



.. function:: cam_resolution(size<int>)
	:noindex:

	**default value:** [2048,2048]


	[BASLER] Defines the dimensions of the camera image.



.. function:: averaging_method(string)
	:noindex:

	**default value:** "mean"


	This can be either 'mean', 'mode', 'min' or 'max'. All accumulated background images (to be used for generating an average background) will be combined using the max or mean function.



.. function:: correct_luminance(bool)
	:noindex:

	**default value:** false


	Attempts to correct for badly lit backgrounds by evening out luminance across the background.



.. function:: enable_difference(bool)
	:noindex:

	**default value:** true


	Enables background subtraction. If disabled, `threshold` will be applied to the raw greyscale values instead of difference values.

	.. seealso:: :func:`threshold`, 


.. function:: image_adjust(bool)
	:noindex:

	**default value:** false


	Converts the image to floating-point (temporarily) and performs f(x,y) * `image_contrast_increase` + `image_brightness_increase` plus, if enabled, squares the image (`image_square_brightness`).

	.. seealso:: :func:`image_contrast_increase`, :func:`image_brightness_increase`, :func:`image_square_brightness`, 


.. function:: grabber_force_settings(bool)
	:noindex:

	**default value:** false


	If set to true, live tracking will always overwrite a settings file with `filename`.settings in the output folder.

	.. seealso:: :func:`filename`, 


.. function:: closing_size(int)
	:noindex:

	**default value:** 3


	Size of the dilation/erosion filters for if `use_closing` is enabled.

	.. seealso:: :func:`use_closing`, 


.. function:: use_adaptive_threshold(bool)
	:noindex:

	**default value:** false


	Enables or disables adaptive thresholding (slower than normal threshold). Deals better with weird backgrounds.



.. function:: grabber_use_threads(bool)
	:noindex:

	**default value:** true


	Use threads to process images (specifically the blob detection).



.. function:: terminate_error(bool)
	:noindex:

	**default value:** false


	Internal variable.



.. function:: recording(bool)
	:noindex:

	**default value:** true


	If set to true, the program will record frames whenever individuals are found.



.. function:: video_conversion_range(pair<int,int>)
	:noindex:

	**default value:** [-1,-1]


	If set to a valid value (!= -1), start and end values determine the range converted.



.. function:: save_raw_movie(bool)
	:noindex:

	**default value:** false


	Saves a RAW movie (.mov) with a similar name in the same folder, while also recording to a PV file. This might reduce the maximum framerate slightly, but it gives you the best of both worlds.



.. function:: stop_after_minutes(uint)
	:noindex:

	**default value:** 0


	If set to a value above 0, the video will stop recording after X minutes of recording time.



.. function:: blob_size_range(rangef)
	:noindex:

	**default value:** [0.01,500000]


	Minimum or maximum size of the individuals on screen after thresholding. Anything smaller or bigger than these values will be disregarded as noise.



.. function:: image_brightness_increase(float)
	:noindex:

	**default value:** 0


	Value that is added to the preprocessed image before applying the threshold (see `image_adjust`). The neutral value is 0 here.

	.. seealso:: :func:`image_adjust`, 


.. function:: enable_live_tracking(bool)
	:noindex:

	**default value:** false


	When enabled, the program will save a .results file for the recorded video plus export the data (see `output_graphs` in the tracker documentation).

	.. seealso:: :func:`output_graphs`, 


.. function:: dilation_size(int)
	:noindex:

	**default value:** 0


	If set to a value greater than zero, detected shapes will be inflated (and potentially merged). When set to a value smaller than zero, detected shapes will be shrunk (and potentially split).



.. function:: meta_write_these(array<string>)
	:noindex:

	**default value:** ["meta_species","meta_age_days","meta_conditions","meta_misc","cam_limit_exposure","meta_real_width","meta_source_path","meta_cmd","meta_build","meta_conversion_time","frame_rate","cam_undistort_vector","cam_matrix"]


	The given settings values will be written to the video file.



.. function:: video_source(string)
	:noindex:

	**default value:** "basler"


	Where the video is recorded from. Can be the name of a file, or one of the keywords ['basler', 'webcam', 'test_image'].



.. function:: image_contrast_increase(float)
	:noindex:

	**default value:** 3


	Value that is multiplied to the preprocessed image before applying the threshold (see `image_adjust`). The neutral value is 1 here.

	.. seealso:: :func:`image_adjust`, 


.. function:: meta_conversion_time(string)
	:noindex:

	**default value:** ""


	This contains the time of when this video was converted / recorded as a string.



.. function:: color_channel(ulong)
	:noindex:

	**default value:** 1


	Index (0-2) of the color channel to be used during video conversion, if more than one channel is present in the video file.



.. function:: image_square_brightness(bool)
	:noindex:

	**default value:** false


	Squares the floating point input image after background subtraction. This brightens brighter parts of the image, and darkens darker regions.



.. function:: meta_cmd(string)
	:noindex:

	**default value:** ""


	Command-line of the framegrabber when conversion was started.



.. function:: test_image(string)
	:noindex:

	**default value:** "checkerboard"


	Defines, which test image will be used if `video_source` is set to 'test_image'.

	.. seealso:: :func:`video_source`, 


.. function:: average_samples(int)
	:noindex:

	**default value:** 100


	Number of samples taken to generate an average image. Usually has to be less if `average_method` is set to max.

	.. seealso:: :func:`average_method`, 


.. function:: reset_average(bool)
	:noindex:

	**default value:** false


	If set to true, the average will be regenerated using the live stream of images (video or camera).



.. function:: meta_age_days(int)
	:noindex:

	**default value:** -1


	Age of the individuals used in days.



.. function:: threshold_maximum(int)
	:noindex:

	**default value:** 255


	



.. function:: meta_conditions(string)
	:noindex:

	**default value:** ""


	Treatment name.



.. function:: threshold(int)
	:noindex:

	**default value:** 9


	Threshold to be applied to the input image to find blobs.



.. function:: system_memory_limit(uint64)
	:noindex:

	**default value:** 0


	Custom override of how many bytes of system RAM the program is allowed to fill. If `approximate_length_minutes` or `stop_after_minutes` are set, this might help to increase the resulting RAW video footage frame_rate.

	.. seealso:: :func:`approximate_length_minutes`, :func:`stop_after_minutes`, 


.. function:: approximate_length_minutes(uint)
	:noindex:

	**default value:** 0


	If available, please provide the approximate length of the video in minutes here, so that the encoding strategy can be chosen intelligently. If set to 0, infinity is assumed. This setting is overwritten by `stop_after_minutes`.

	.. seealso:: :func:`stop_after_minutes`, 


.. function:: quit_after_average(bool)
	:noindex:

	**default value:** false


	If set to true, this will terminate the program directly after generating (or loading) a background average image.



.. function:: crop_offsets(offsets)
	:noindex:

	**default value:** [0,0,0,0]


	Percentage offsets [left, top, right, bottom] that will be cut off the input images (e.g. [0.1,0.1,0.5,0.5] will remove 10%% from the left and top and 50%% from the right and bottom and the video will be 60%% smaller in X and Y).



.. function:: equalize_histogram(bool)
	:noindex:

	**default value:** false


	Equalizes the histogram of the image before thresholding and background subtraction.



.. function:: crop_window(bool)
	:noindex:

	**default value:** false


	If set to true, the grabber will open a window before the analysis starts where the user can drag+drop points defining the crop_offsets.



.. function:: adaptive_threshold_scale(float)
	:noindex:

	**default value:** 2


	Threshold value to be used for adaptive thresholding, if enabled.



.. function:: use_closing(bool)
	:noindex:

	**default value:** false


	Toggles the attempt to close weird blobs using dilation/erosion with `closing_size` sized filters.

	.. seealso:: :func:`closing_size`, 


.. function:: cam_limit_exposure(int)
	:noindex:

	**default value:** 5500


	[BASLER] Sets the cameras exposure time in micro seconds.



