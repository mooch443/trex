.. toctree::
   :maxdepth: 2

TRex parameters
###############
.. function:: accumulation_enable(bool)

	**default value:** true


	Enables or disables the idtrackerai-esque accumulation protocol cascade. It is usually a good thing to enable this (especially in more complicated videos), but can be disabled as a fallback (e.g. if computation time is a major constraint).



.. function:: accumulation_enable_final_step(bool)

	**default value:** true


	If enabled, the network will be trained on all the validation + training data accumulated, as a last step of the accumulation protocol cascade. This is intentional overfitting.



.. function:: accumulation_max_tracklets(uint)

	**default value:** 15


	If there are more than  global tracklets to be trained on, they will be filtered according to their quality until said limit is reached.



.. function:: accumulation_sufficient_uniqueness(float)

	**default value:** 0


	If changed (from 0), the ratio given here will be the acceptable uniqueness for the video - which will stop accumulation if reached.



.. function:: accumulation_tracklet_add_factor(float)

	**default value:** 1.5


	This factor will be multiplied with the probability that would be pure chance, during the decision whether a tracklet is to be added or not. The default value of 1.5 suggests that the minimum probability for each identity has to be 1.5 times chance (e.g. 0.5 in the case of two individuals).



.. function:: adaptive_threshold_scale(float)

	**default value:** 2


	Threshold value to be used for adaptive thresholding, if enabled.



.. function:: analysis_range(range<int>)

	**default value:** [-1,-1]


	Sets start and end of the analysed frames.



.. function:: app_check_for_updates(app_update_check_t)

	**default value:** none

	**possible values:**
		- `none`: No status has been set yet and the program will ask the user.
		- `manually`: Manually check for updates, do not automatically check for them online.
		- `automatically`: Automatically check for updates periodically (once per week).

	If enabled, the application will regularly check for updates online (`https://api.github.com/re[...]43/trex/releases <https://api.github.com/repos/mooch443/trex/releases>`_).




.. function:: app_last_update_version(string)

	**default value:** ""


	



.. function:: approximate_length_minutes(uint)

	**default value:** 0


	If available, please provide the approximate length of the video in minutes here, so that the encoding strategy can be chosen intelligently. If set to 0, infinity is assumed. This setting is overwritten by ``stop_after_minutes``.

	.. seealso:: :param:`stop_after_minutes`


.. function:: auto_apply(bool)

	**default value:** false


	If set to true, the application will automatically apply the network with existing weights once the analysis is done. It will then automatically correct and reanalyse the video.



.. function:: auto_categorize(bool)

	**default value:** false


	If set to true, the program will try to load <video>_categories.npz from the ``output_dir``. If successful, then categories will be computed according to the current categories_ settings. Combine this with the ``auto_quit`` parameter to automatically save and quit afterwards. If weights cannot be loaded, the app crashes.

	.. seealso:: :param:`output_dir`, :param:`auto_quit`


.. function:: auto_minmax_size(bool)

	**default value:** false


	Program will try to find minimum / maximum size of the individuals automatically for the current ``cm_per_pixel`` setting. Can only be passed as an argument upon startup. The calculation is based on the median blob size in the video and assumes a relatively low level of noise.

	.. seealso:: :param:`cm_per_pixel`


.. function:: auto_no_memory_stats(bool)

	**default value:** true


	If set to true, no memory statistics will be saved on auto_quit.



.. function:: auto_no_outputs(bool)

	**default value:** false


	If set to true, no data will be exported upon ``auto_quit``. Not even a .settings file will be saved.

	.. seealso:: :param:`auto_quit`


.. function:: auto_no_results(bool)

	**default value:** false


	If set to true, the auto_quit option will NOT save a .results file along with the NPZ (or CSV) files. This saves time and space, but also means that the tracked portion cannot be loaded via -load afterwards. Useful, if you only want to analyse the resulting data and never look at the tracked video again.



.. function:: auto_no_tracking_data(bool)

	**default value:** false


	If set to true, the auto_quit option will NOT save any ``output_fields`` tracking data - just the posture data (if enabled) and the results file (if not disabled). This saves time and space if that is a need.

	.. seealso:: :param:`output_fields`


.. function:: auto_number_individuals(bool)

	**default value:** false


	Program will automatically try to find the number of individuals (with sizes given in ``track_size_filter``) and set ``track_max_individuals`` to that value.

	.. seealso:: :param:`track_size_filter`, :param:`track_max_individuals`


.. function:: auto_quit(bool)

	**default value:** false


	If set to true, the application will automatically save all results and export CSV files and quit, after the analysis is complete.



.. function:: auto_tags(bool)

	**default value:** false


	If set to true, the application will automatically apply available tag information once the results file has been loaded. It will then automatically correct potential tracking mistakes based on this information.



.. function:: auto_train(bool)

	**default value:** false


	If set to true, the application will automatically train the recognition network with the best track tracklet and apply it to the video.



.. function:: auto_train_dont_apply(bool)

	**default value:** false


	If set to true, setting ``auto_train`` will only train and not apply the trained network.

	.. seealso:: :param:`auto_train`


.. function:: average_samples(uint)

	**default value:** 25


	Number of samples taken to generate an average image. Usually fewer are necessary for ``averaging_method``'s max, and min.

	.. seealso:: :param:`averaging_method`


.. function:: averaging_method(averaging_method_t)

	**default value:** mean

	**possible values:**
		- `mean`: Sum all samples and divide by N.
		- `mode`: Calculate a per-pixel median of the samples to avoid noise. More computationally involved than mean, but often better results.
		- `max`: Use a per-pixel minimum across samples. Usually a good choice for short videos with black backgrounds and individuals that do not move much.
		- `min`: Use a per-pixel maximum across samples. Usually a good choice for short videos with white backgrounds and individuals that do not move much.

	Determines the way in which the background samples are combined. The background generated in the process will be used to subtract background from foreground objects during conversion.




.. function:: blob_size_range(range<float>)

	**default value:** [0.01,500000]


	Minimum or maximum size of the individuals on screen after thresholding. Anything smaller or bigger than these values will be disregarded as noise.



.. function:: blob_split_algorithm(blob_split_algorithm_t)

	**default value:** threshold

	**possible values:**
		- `threshold`: Adaptively increase the threshold of closeby objects, until separation.
		- `threshold_approximate`: Same as threshold, but use heuristics to produce results faster. These results might not be as deterministic as with threshold, but usually only differ by 1 or 2 in found threshold value. It is guaranteed, however, that a solution is found if one exists.
		- `fill`: Use the previously known positions of objects to place a seed within the overlapped objects and perform a watershed run.
		- `none`: Do not actually attempt to split blobs. Just ignore blobs until they split by themselves.

	The default splitting algorithm used to split objects that are too close together.




.. function:: blob_split_global_shrink_limit(float)

	**default value:** 0.2


	The minimum percentage of the minimum in ``track_size_filter``, that a blob is allowed to be reduced to during splitting. If this value is set too low, the program might start recognizing parts of individual as other individual too quickly.

	.. seealso:: :param:`track_size_filter`


.. function:: blob_split_max_shrink(float)

	**default value:** 0.2


	The minimum percentage of the starting blob size (after thresholding), that a blob is allowed to be reduced to during splitting. If this value is set too low, the program might start recognizing parts of individual as other individual too quickly.



.. function:: blobs_per_thread(float)

	**default value:** 150


	Number of blobs for which properties will be calculated per thread.



.. function:: blur_difference(bool)

	**default value:** false


	Enables a special mode that will 1. truncate all difference values below threshold, 2. blur the remaining difference, 3. threshold again.



.. function:: calculate_posture(bool)

	**default value:** true


	Enables or disables posture calculation. Can only be set before the video is analysed (e.g. in a settings file or as a startup parameter).



.. function:: cam_circle_mask(bool)

	**default value:** false


	If set to true, a circle with a diameter of the width of the video image will mask the video. Anything outside that circle will be disregarded as background.



.. function:: cam_framerate(int)

	**default value:** -1


	If set to anything else than 0, this will limit the basler camera framerate to the given fps value.



.. function:: cam_limit_exposure(int)

	**default value:** 5500


	Sets the cameras exposure time in micro seconds.



.. function:: cam_matrix(array<double>)

	**default value:** []


	



.. function:: cam_resolution(size)

	**default value:** [-1,-1]


	Defines the dimensions of the camera image.



.. function:: cam_scale(float)

	**default value:** 1


	Scales the image down or up by the given factor.



.. function:: cam_undistort(bool)

	**default value:** false


	If set to true, the recorded video image will be undistorted using ``cam_undistort_vector`` (1x5) and ``cam_matrix`` (3x3).

	.. seealso:: :param:`cam_undistort_vector`, :param:`cam_matrix`


.. function:: cam_undistort_vector(array<double>)

	**default value:** []


	



.. function:: categories_apply_min_tracklet_length(uint)

	**default value:** 0


	Minimum number of images for a sample to be considered relevant when applying the categorization. This defaults to 0, meaning all samples are valid. If set to anything higher, only tracklets with more than N frames will be processed.



.. function:: categories_ordered(array<string>)

	**default value:** []


	Ordered list of names of categories that are used in categorization (classification of types of individuals).



.. function:: categories_train_min_tracklet_length(uint)

	**default value:** 50


	Minimum number of images for a sample to be considered relevant for training categorization. Will default to 50, meaning all tracklets longer than that will be presented for training.



.. function:: closed_loop_enable(bool)

	**default value:** false


	When enabled, live tracking will be executed for every frame received. Frames will be sent to the 'closed_loop.py' script - see this script for more information. Allows the tracker to skip frames by default, in order to catch up to the video.



.. function:: closed_loop_path(path)

	**default value:** "closed_loop_beta.py"


	Set the path to a Python file to be used in closed_loop. Please also enable closed loop processing by setting ``closed_loop_enable`` to true.

	.. seealso:: :param:`closed_loop_enable`


.. function:: closing_size(int)

	**default value:** 3


	Size of the dilation/erosion filters for if ``use_closing`` is enabled.

	.. seealso:: :param:`use_closing`


.. function:: cm_per_pixel(float)

	**default value:** 0


	The ratio of ``meta_real_width / video_width`` that is used to convert pixels to centimeters. Will be automatically calculated based on a meta-parameter saved inside the video file (``meta_real_width``) and does not need to be set manually.

	.. seealso:: :param:`meta_real_width`


.. function:: color_channel(optional<uchar>)

	**default value:** null


	Index (0-2) of the color channel to be used during video conversion, if more than one channel is present in the video file. If set to null it will use the default conversion that OpenCV uses for cv::COLOR_BGRA2GRAY.



.. function:: correct_illegal_lines(bool)

	**default value:** false


	In older versions of the software, blobs can be constructed in 'illegal' ways, meaning the lines might be overlapping. If the software is printing warnings about it, this should probably be enabled (makes it slower).



.. function:: correct_luminance(bool)

	**default value:** false


	Attempts to correct for badly lit backgrounds by evening out luminance across the background.



.. function:: crop_offsets(offsets)

	**default value:** [0,0,0,0]


	Percentage offsets [left, top, right, bottom] that will be cut off the input images (e.g. [0.1,0.1,0.5,0.5] will remove 10%% from the left and top and 50%% from the right and bottom and the video will be 60%% smaller in X and Y).



.. function:: crop_window(bool)

	**default value:** false


	If set to true, the grabber will open a window before the analysis starts where the user can drag+drop points defining the crop_offsets.



.. function:: data_prefix(path)

	**default value:** "data"


	Subfolder (below ``output_dir``) where the exported NPZ or CSV files will be saved (see ``output_fields``).

	.. seealso:: :param:`output_dir`, :param:`output_fields`


.. function:: debug_recognition_output_all_methods(bool)

	**default value:** false


	If set to true, a complete training will attempt to output all images for each identity with all available normalization methods.



.. function:: detect_batch_size(uchar)

	**default value:** 1


	The batching size for object detection.



.. function:: detect_classes(optional<map<uint16,string>>)

	**default value:** null


	Class names for object classification in video during conversion.



.. function:: detect_conf_threshold(float)

	**default value:** 0.1


	Confidence threshold (``0<=value<1``) for object detection / segmentation networks. Confidence is higher if the network is more *sure* about the object. Anything with a confidence level below  will not be considered an object and not saved to the PV file during conversion.



.. function:: detect_format(ObjectDetectionFormat)

	**default value:** none


	The type of data returned by the ``detect_model``, which can be an instance segmentation

	.. seealso:: :param:`detect_model`


.. function:: detect_iou_threshold(float)

	**default value:** 0.5


	Higher (==1) indicates that all overlaps are allowed, while lower values (>0) will filter out more of the overlaps. This depends strongly on the situation, but values between 0.25 and 0.7 are common.



.. function:: detect_keypoint_names(KeypointNames)

	**default value:** null


	An array of names in the correct keypoint index order for the given model.



.. function:: detect_model(path)

	**default value:** ""


	The path to a .pt file that contains a valid PyTorch object detection model (currently only YOLO networks are supported).



.. function:: detect_only_classes(array<uchar>)

	**default value:** []


	An array of class ids that you would like to detect (as returned from the model). If left empty, no class will be filtered out.



.. function:: detect_precomputed_file(PathArray)

	**default value:** ""


	If ``detect_type`` is set to ``precomputed``, this should point to a csv file (or npz files) containing the necessary tracking data for the given ``source`` video.

	.. seealso:: :param:`detect_type`, :param:`source`


.. function:: detect_size_filter(SizeFilters)

	**default value:** []


	During the detection phase, objects outside this size range will be filtered out. If empty, no objects will be filtered out.



.. function:: detect_skeleton(optional<Skeletons>)

	**default value:** null


	Skeleton to be used when displaying pose data.



.. function:: detect_threshold(int)

	**default value:** 15


	Threshold to be applied to the input image to find blobs.



.. function:: detect_threshold_is_absolute(bool)

	**default value:** true


	If enabled, uses absolute difference values and disregards any pixel |p| < ``threshold`` during conversion. Otherwise the equation is p < ``threshold``, meaning that e.g. bright spots may not be considered trackable when dark spots would. Same as ``track_threshold_is_absolute``, but during conversion instead of tracking.

	.. seealso:: :param:`track_threshold_is_absolute`


.. function:: detect_tile_image(uchar)

	**default value:** 0


	If > 1, this will tile the input image for Object detection (SAHI method) before passing it to the network. These tiles will be ``detect_resolution`` pixels high and wide (with zero padding).

	.. seealso:: :param:`detect_resolution`


.. function:: detect_type(ObjectDetectionType)

	**default value:** none


	The method used to separate background from foreground when converting videos.



.. function:: dilation_size(int)

	**default value:** 0


	If set to a value greater than zero, detected shapes will be inflated (and potentially merged). When set to a value smaller than zero, detected shapes will be shrunk (and potentially split).



.. function:: enable_difference(bool)

	**default value:** true


	Enables background subtraction. If disabled, ``threshold`` will be applied to the raw greyscale values instead of difference values.



.. function:: equalize_histogram(bool)

	**default value:** false


	Equalizes the histogram of the image before thresholding and background subtraction.



.. function:: evaluate_thresholds(bool)

	**default value:** false


	This option, if enabled, previews the effects of all possible thresholds when applied to the given video. These are shown as a graph in a separate window. Can be used to debug parameters instead of try-and-error. Might take a few minutes to finish calculating.



.. function:: event_min_peak_offset(float)

	**default value:** 0.15


	



.. function:: exec(path)

	**default value:** ""


	This can be set to the path of an additional settings file that is executed after the normal settings file.



.. function:: ffmpeg_crf(uint)

	**default value:** 20


	Quality for crf (see ffmpeg documentation) used when encoding as libx264.



.. function:: ffmpeg_path(path)

	**default value:** ""


	Path to an ffmpeg executable file. This is used for converting videos after recording them (from the GUI). It is not a critical component of the software, but mostly for convenience.



.. function:: filename(path)

	**default value:** ""


	The converted video file (.pv file) or target for video conversion. Typically it would have the same basename as the video source (i.e. an MP4 file), but a different extension: pv.



.. function:: frame_rate(uint)

	**default value:** 0


	Specifies the frame rate of the video. It is used e.g. for playback speed and certain parts of the matching algorithm. Will be set by the metadata of the video. If you want to set a custom frame rate, different from the video metadata, you should set it during conversion. This guarantees that the timestamps generated will match up with your custom framerate during tracking.



.. function:: gpu_learning_rate(float)

	**default value:** 0.0001


	Learning rate for training a recognition network.



.. function:: gpu_max_cache(float)

	**default value:** 2


	Size of the image cache (transferring to GPU) in GigaBytes when applying the network.



.. function:: gpu_max_epochs(uchar)

	**default value:** 150


	Maximum number of epochs for training a recognition network (0 means infinite).



.. function:: gpu_max_sample_gb(float)

	**default value:** 2


	Maximum size of per-individual sample images in GigaBytes. If the collected images are too many, they will be sub-sampled in regular intervals.



.. function:: gpu_min_elements(uint)

	**default value:** 25000


	Minimum number of images being collected, before sending them to the GPU.



.. function:: gpu_min_iterations(uchar)

	**default value:** 100


	Minimum number of iterations per epoch for training a recognition network.



.. function:: gpu_torch_device(gpu_torch_device_t)

	**default value:** automatic

	**possible values:**
		- `automatic`: The device is automatically chosen by PyTorch.
		- `cuda`: Use a CUDA device (requires an NVIDIA graphics card).
		- `mps`: Use a METAL device (requires an Apple Silicone Mac).
		- `cpu`: Use the CPU (everybody should have this).

	If specified, indicate something like 'cuda:0' to use the first cuda device when doing machine learning using pytorch (e.g. TRexA). Other options can be looked up at `https://pytorch.org/docs/[...]orch.cuda.device <https://pytorch.org/docs/stable/generated/torch.cuda.device.html#torch.cuda.device>`_.




.. function:: gpu_torch_device_index(int)

	**default value:** -1


	Index of the GPU used by torch (or -1 for automatic selection).



.. function:: gpu_torch_no_fixes(bool)

	**default value:** true


	Disable the fix for PyTorch on MPS devices that will automatically switch to CPU specifically for Ultralytics segmentation models.



.. function:: gpu_verbosity(gpu_verbosity_t)

	**default value:** full

	**possible values:**
		- `silent`: No output during training.
		- `full`: An animated bar with detailed information about the training progress.
		- `oneline`: One line per epoch.

	Determines the nature of the output on the command-line during training. This does not change any behaviour in the graphical interface.




.. function:: grabber_force_settings(bool)

	**default value:** false


	If set to true, live tracking will always overwrite a settings file with ``filename``.settings in the output folder.

	.. seealso:: :param:`filename`


.. function:: grid_points(array<vec>)

	**default value:** []


	Whenever there is an identification network loaded and this array contains more than one point ``[[x0,y0],[x1,y1],...]``, then the network will only be applied to blobs within circles around these points. The size of these circles is half of the average distance between the points.



.. function:: grid_points_scaling(float)

	**default value:** 0.8


	Scaling applied to the average distance between the points in order to shrink or increase the size of the circles for recognition (see ``grid_points``).

	.. seealso:: :param:`grid_points`


.. function:: gui_auto_scale(bool)

	**default value:** false


	If set to true, the tracker will always try to zoom in on the whole group. This is useful for some individuals in a huge video (because if they are too tiny, you cant see them and their posture anymore).



.. function:: gui_auto_scale_focus_one(bool)

	**default value:** true


	If set to true (and ``gui_auto_scale`` set to true, too), the tracker will zoom in on the selected individual, if one is selected.

	.. seealso:: :param:`gui_auto_scale`


.. function:: gui_background_color(color)

	**default value:** [0,0,0,255]


	Values < 255 will make the background (or video background) more transparent in standard view. This might be useful with very bright backgrounds.



.. function:: gui_blob_label(string)

	**default value:** "{if:{dock}:{name} :''}{if:{active}:<a>:''}{real_size}{if:{split}: <gray>split</gray>:''}{if:{tried_to_split}: <orange>split tried</orange>:''}{if:{prediction}: {prediction}:''}{if:{instance}: <gray>instance</gray>:''}{if:{dock}:{if:{filter_reason}: [<gray>{filter_reason}</gray>]:''}:''}{if:{active}:</a>:''}{if:{category}: {category}:''}"


	This is what the graphical user interfaces displays as a label for each blob in raw view. Replace this with {help} to see available variables.



.. function:: gui_connectivity_matrix(map<int,array<float>>)

	**default value:** {}


	Internally used to store the connectivity matrix.



.. function:: gui_connectivity_matrix_file(path)

	**default value:** ""


	Path to connectivity table. Expected structure is a csv table with columns [frame | #(track_max_individuals^2) values] and frames in y-direction.



.. function:: gui_displayed_frame(frame)

	**default value:** 0


	The currently visible frame.



.. function:: gui_draw_blobs_separately(bool)

	**default value:** false


	Draw blobs separately. If false, blobs will be drawn on a single full-screen texture and displayed. The second option may be better on some computers (not supported if ``gui_macos_blur`` is set to true).

	.. seealso:: :param:`gui_macos_blur`


.. function:: gui_draw_only_filtered_out(bool)

	**default value:** false


	Only show filtered out blob texts.



.. function:: gui_equalize_blob_histograms(bool)

	**default value:** false


	Equalize histograms of blobs wihtin videos (makes them more visible).



.. function:: gui_faded_brightness(uchar)

	**default value:** 255


	The alpha value of tracking-related elements when timeline is hidden (0-255).



.. function:: gui_fish_color(string)

	**default value:** "identity"


	



.. function:: gui_fish_label(string)

	**default value:** "{if:{not:{has_pred}}:{name}:{if:{equal:{at:0:{max_pred}}:{id}}:<green>{name}</green>:<red>{name}</red> <i>loc</i>[<c><nr>{at:0:{max_pred}}</nr>:<nr>{int:{*:100:{at:1:{max_pred}}}}</nr><i>%</i></c>]}}{if:{tag}:' <a>tag:{tag.id} ({dec:2:{tag.p}})</a>':''}{if:{average_category}:' <nr>{average_category}</nr>':''}{if:{&&:{category}:{not:{equal:{category}:{average_category}}}}:' <b><i>{category}</i></b>':''}"


	This is what the graphical user interface displays as a label for each individual. Replace this with {help} to see the available variables.



.. function:: gui_focus_group(array<Idx_t>)

	**default value:** []


	Focus on this group of individuals.



.. function:: gui_foi_name(string)

	**default value:** "correcting"


	If not empty, the gui will display the given FOI type in the timeline and allow to navigate between them via M/N.



.. function:: gui_frame(frame)

	**default value:** 0


	The currently selected frame. ``gui_displayed_frame`` might differ, if loading from file is currently slow.

	.. seealso:: :param:`gui_displayed_frame`


.. function:: gui_happy_mode(bool)

	**default value:** false


	If ``calculate_posture`` is enabled, enabling this option likely improves your experience with TRex.

	.. seealso:: :param:`calculate_posture`


.. function:: gui_highlight_categories(bool)

	**default value:** false


	If enabled, categories (if applied in the video) will be highlighted in the tracking view.



.. function:: gui_macos_blur(bool)

	**default value:** false


	MacOS supports a blur filter that can be applied to make unselected individuals look more interesting. Purely a visual effect. Does nothing on other operating systems.



.. function:: gui_max_path_time(float)

	**default value:** 3


	Length (in time) of the trails shown in GUI.



.. function:: gui_mode(mode_t)

	**default value:** tracking


	The currently used display mode for the GUI.



.. function:: gui_outline_thickness(uchar)

	**default value:** 1


	The thickness of outline / midlines in the GUI.



.. function:: gui_playback_speed(float)

	**default value:** 1


	Playback speed when pressing SPACE.



.. function:: gui_pose_smoothing(frame)

	**default value:** 0


	Blending between the current and previous / future frames for displaying smoother poses in the graphical user-interface. This does not affect data output.



.. function:: gui_recording_format(gui_recording_format_t)

	**default value:** mp4

	**possible values:**
		- `avi`: AVI / video format (codec MJPG is used)
		- `mp4`: MP4 / video format (codec H264 is used)
		- `jpg`: individual images in JPEG format
		- `png`: individual images in PNG format

	Sets the format for recording mode (when R is pressed in the GUI). Supported formats are 'avi', 'jpg' and 'png'. JPEGs have 75%% compression, AVI is using MJPEG compression.




.. function:: gui_run(bool)

	**default value:** false


	When set to true, the GUI starts playing back the video and stops once it reaches the end, or is set to false.



.. function:: gui_show_autoident_controls(bool)

	**default value:** false


	Showing or hiding controls for removing forced auto-ident in the info card if an individual is selected.



.. function:: gui_show_blobs(bool)

	**default value:** true


	Showing or hiding individual raw blobs in tracking view (are always shown in RAW mode).



.. function:: gui_show_boundary_crossings(bool)

	**default value:** true


	If set to true (and the number of individuals is set to a number > 0), the tracker will show whenever an individual enters the recognition boundary. Indicated by an expanding cyan circle around it.



.. function:: gui_show_cliques(bool)

	**default value:** false


	Show/hide cliques of potentially difficult tracking situations.



.. function:: gui_show_dataset(bool)

	**default value:** false


	Show/hide detailed dataset information on-screen.



.. function:: gui_show_detailed_probabilities(bool)

	**default value:** false


	Show/hide detailed probability stats when an individual is selected.



.. function:: gui_show_export_options(bool)

	**default value:** false


	Show/hide the export options widget.



.. function:: gui_show_fish(tuple<blob,frame,>)

	**default value:** [null,null]


	Show debug output for {blob_id, fish_id}.



.. function:: gui_show_graph(bool)

	**default value:** false


	Show/hide the data time-series graph.



.. function:: gui_show_heatmap(bool)

	**default value:** false


	Showing a heatmap per identity, normalized by maximum samples per grid-cell.



.. function:: gui_show_histograms(bool)

	**default value:** false


	Equivalent to the checkbox visible in GUI on the bottom-left.



.. function:: gui_show_inactive_individuals(bool)

	**default value:** false


	Show/hide individuals that have not been seen for longer than ``track_max_reassign_time``.

	.. seealso:: :param:`track_max_reassign_time`


.. function:: gui_show_individual_preview(bool)

	**default value:** false


	Shows preview images for all selected individuals as they would be processed during network training, based on settings like ``individual_image_size``, ``individual_image_scale`` and ``individual_image_normalization``.

	.. seealso:: :param:`individual_image_size`, :param:`individual_image_scale`, :param:`individual_image_normalization`


.. function:: gui_show_infocard(bool)

	**default value:** true


	Showing / hiding some facts about the currently selected individual on the top left of the window.



.. function:: gui_show_match_modes(bool)

	**default value:** false


	Shows the match mode used for every tracked object. Green is 'approximate', yellow is 'hungarian', and red is 'created/loaded'.



.. function:: gui_show_matching_info(bool)

	**default value:** true


	Showing or hiding probabilities for relevant blobs in the info card if an individual is selected.



.. function:: gui_show_memory_stats(bool)

	**default value:** false


	Showing or hiding memory statistics.



.. function:: gui_show_midline(bool)

	**default value:** true


	Showing or hiding individual midlines in tracking view.



.. function:: gui_show_midline_histogram(bool)

	**default value:** false


	Displays a histogram for midline lengths.



.. function:: gui_show_misc_metrics(bool)

	**default value:** true


	Showing or hiding some metrics for a selected individual in the info card.



.. function:: gui_show_number_individuals(bool)

	**default value:** false


	Show/hide the #individuals time-series graph.



.. function:: gui_show_only_unassigned(bool)

	**default value:** false


	Showing only unassigned objects.



.. function:: gui_show_outline(bool)

	**default value:** true


	Showing or hiding individual outlines in tracking view.



.. function:: gui_show_paths(bool)

	**default value:** true


	Equivalent to the checkbox visible in GUI on the bottom-left.



.. function:: gui_show_pixel_grid(bool)

	**default value:** false


	Shows the proximity grid generated for all blobs, which is used for history splitting.



.. function:: gui_show_posture(bool)

	**default value:** false


	Show/hide the posture window on the top-right.



.. function:: gui_show_probabilities(bool)

	**default value:** false


	Show/hide probability visualisation when an individual is selected.



.. function:: gui_show_processing_time(bool)

	**default value:** false


	Show/hide the ms/frame time-series graph.



.. function:: gui_show_recognition_bounds(bool)

	**default value:** true


	Shows what is contained within tht recognition boundary as a cyan background. (See ``recognition_border`` for details.)

	.. seealso:: :param:`recognition_border`


.. function:: gui_show_recognition_summary(bool)

	**default value:** false


	Show/hide confusion matrix (if network is loaded).



.. function:: gui_show_selections(bool)

	**default value:** true


	Show/hide circles around selected individual.



.. function:: gui_show_shadows(bool)

	**default value:** true


	Showing or hiding individual shadows in tracking view.



.. function:: gui_show_skeletons(bool)

	**default value:** true


	Shows / hides keypoint data being shown in the graphical interface.



.. function:: gui_show_texts(bool)

	**default value:** true


	Showing or hiding individual identity (and related) texts in tracking view.



.. function:: gui_show_timeline(bool)

	**default value:** true


	If enabled, the timeline (top of the screen) will be shown in the tracking view.



.. function:: gui_show_timing_stats(bool)

	**default value:** false


	Showing / hiding rendering information.



.. function:: gui_show_uniqueness(bool)

	**default value:** false


	Show/hide uniqueness overview after training.



.. function:: gui_show_video_background(bool)

	**default value:** true


	If available, show an animated background of the original video.



.. function:: gui_show_visualfield(bool)

	**default value:** false


	Show/hide the visual field rays.



.. function:: gui_show_visualfield_ts(bool)

	**default value:** false


	Show/hide the visual field time series.



.. function:: gui_single_identity_color(color)

	**default value:** [0,0,0,0]


	If set to something else than transparent, all individuals will be displayed with this color.



.. function:: gui_timeline_alpha(uchar)

	**default value:** 200


	Determines the Alpha value for the timeline / tracklets display.



.. function:: gui_wait_for_background(bool)

	**default value:** true


	Sacrifice video playback speed to wait for the background video the load in. This only applies if the background is actually displayed (``gui_show_video_background``).

	.. seealso:: :param:`gui_show_video_background`


.. function:: gui_wait_for_pv(bool)

	**default value:** true


	Sacrifice video playback speed to wait for the pv file the load in.



.. function:: gui_zoom_limit(size)

	**default value:** [300,300]


	



.. function:: gui_zoom_polygon(array<vec>)

	**default value:** []


	If this is non-empty, the view will be zoomed in on the center of the polygon with approximately the dimensions of the polygon.



.. function:: heatmap_dynamic(bool)

	**default value:** false


	If enabled the heatmap will only show frames before the frame currently displayed in the graphical user interface.



.. function:: heatmap_frames(uint)

	**default value:** 0


	If ``heatmap_dynamic`` is enabled, this variable determines the range of frames that are considered. If set to 0, all frames up to the current frame are considered. Otherwise, this number determines the number of frames previous to the current frame that are considered.

	.. seealso:: :param:`heatmap_dynamic`


.. function:: heatmap_ids(array<Idx_t>)

	**default value:** []


	Add ID numbers to this array to exclusively display heatmap values for those individuals.



.. function:: heatmap_normalization(heatmap_normalization_t)

	**default value:** cell

	**possible values:**
		- `none`: No normalization at all. Values will only be averaged per cell.
		- `value`: Normalization based in value-space. The average of each cell will be divided by the maximum value encountered.
		- `cell`: The cell sum will be divided by the maximum cell value encountered.
		- `variance`: Displays the variation within each cell.

	Normalization used for the heatmaps. If ``value`` is selected, then the maximum of all values encountered will be used to normalize the average of each cell. If ``cell`` is selected, the sum of each cell will be divided by the maximum cell value encountered.




.. function:: heatmap_resolution(uint)

	**default value:** 64


	Square resolution of individual heatmaps displayed with ``gui_show_heatmap``. Will generate a square grid, each cell with dimensions (video_width / N, video_height / N), and sort all positions of each identity into it.

	.. seealso:: :param:`gui_show_heatmap`


.. function:: heatmap_smooth(double)

	**default value:** 0.05


	Value between 0 and 1, think of as  times video-width, indicating the maximum upscaled size of the heatmaps shown in the tracker. Makes them prettier, but maybe much slower.



.. function:: heatmap_source(string)

	**default value:** ""


	If empty, the source will simply be an individuals identity. Otherwise, information from export data sources will be used.



.. function:: heatmap_value_range(range<double>)

	**default value:** [-1,-1]


	Give a custom value range that is used to normalize heatmap cell values.



.. function:: history_matching_log(path)

	**default value:** ""


	If this is set to a valid html file path, a detailed matching history log will be written to the given file for each frame.



.. function:: huge_timestamp_seconds(double)

	**default value:** 0.2


	Defaults to 0.5s (500ms), can be set to any value that should be recognized as being huge.



.. function:: image_adjust(bool)

	**default value:** false


	Converts the image to floating-point (temporarily) and performs f(x,y) * ``image_contrast_increase`` + ``image_brightness_increase`` plus, if enabled, squares the image (``image_square_brightness``).

	.. seealso:: :param:`image_contrast_increase`, :param:`image_brightness_increase`, :param:`image_square_brightness`


.. function:: image_brightness_increase(float)

	**default value:** 0


	Value that is added to the preprocessed image before applying the threshold (see ``image_adjust``). The neutral value is 0 here.

	.. seealso:: :param:`image_adjust`


.. function:: image_contrast_increase(float)

	**default value:** 3


	Value that is multiplied to the preprocessed image before applying the threshold (see ``image_adjust``). The neutral value is 1 here.

	.. seealso:: :param:`image_adjust`


.. function:: image_invert(bool)

	**default value:** false


	Inverts the image greyscale values before thresholding.



.. function:: image_square_brightness(bool)

	**default value:** false


	Squares the floating point input image after background subtraction. This brightens brighter parts of the image, and darkens darker regions.



.. function:: individual_image_normalization(individual_image_normalization_t)

	**default value:** posture

	**possible values:**
		- `none`: No normalization. Images will only be cropped out and used as-is.
		- `moments`: Images will be cropped out and aligned as in idtracker.ai using the main axis calculated using `image moments`.
		- `posture`: Images will be cropped out and rotated so that the head will be fixed in one position and only the tail moves.
		- `legacy`: Images will be aligned parallel to the x axis.

	This enables or disable normalizing the images before training. If set to ``none``, the images will be sent to the GPU raw - they will only be cropped out. Otherwise they will be normalized based on head orientation (posture) or the main axis calculated using ``image moments``.




.. function:: individual_image_scale(float)

	**default value:** 1


	Scaling applied to the images before passing them to the network.



.. function:: individual_image_size(size)

	**default value:** [80,80]


	Size of each image generated for network training.



.. function:: individual_names(map<Idx_t,string>)

	**default value:** {}


	A map of ``{individual-id: "individual-name", ...}`` that names individuals in the GUI and exported data.



.. function:: individual_prefix(string)

	**default value:** "fish"


	The prefix that is added to all the files containing certain IDs. So individual 0 will turn into '[prefix]0' for all the npz files and within the program.



.. function:: individuals_per_thread(float)

	**default value:** 1


	Number of individuals for which positions will be estimated per thread.



.. function:: limit(float)

	**default value:** 0.09


	Limit for tailbeat event detection.



.. function:: log_file(path)

	**default value:** ""


	Set this to a path you want to save the log file to.



.. function:: manual_matches(map<frame,map<Idx_t,blob>>)

	**default value:** {}


	A map of manually defined matches (also updated by GUI menu for assigning manual identities). ``{{frame: {fish0: blob2, fish1: blob0}}, ...}``



.. function:: manual_splits(map<frame,set<blob>>)

	**default value:** {}


	This map contains ``{frame: [blobid1,blobid2,...]}`` where frame and blobid are integers. When this is read during tracking for a frame, the tracker will attempt to force-split the given blob ids.



.. function:: manually_approved(map<int,int>)

	**default value:** {}


	A list of ranges of manually approved frames that may be used for generating training datasets, e.g. ``{232:233,5555:5560}`` where each of the numbers is a frame number. Meaning that frames 232-233 and 5555-5560 are manually set to be manually checked for any identity switches, and individual identities can be assumed to be consistent throughout these frames.



.. function:: mask_path(path)

	**default value:** ""


	Path to a video file containing a mask to be applied to the video while recording. Only works for conversions.



.. function:: match_min_probability(float)

	**default value:** 0.1


	The probability below which a possible connection between blob and identity is considered too low. The probability depends largely upon settings like ``track_max_speed``.

	.. seealso:: :param:`track_max_speed`


.. function:: match_mode(matching_mode_t)

	**default value:** automatic

	**possible values:**
		- `tree`: Maximizes the probability sum by assigning (or potentially not assigning) individuals to objects in the frame. This returns the correct solution, but might take long for high quantities of individuals.
		- `approximate`: Simply assigns the highest probability edges (blob to individual) to all individuals - first come, first serve. Parameters have to be set very strictly (especially speed) in order to have as few objects to choose from as possible and limit the error.
		- `hungarian`: The hungarian algorithm (as implemented in O(n^3) by Mattias Andrée `https://github.com/maandree/hungarian-algorithm-n3`).
		- `benchmark`: Runs all algorithms and pits them against each other, outputting statistics every few frames.
		- `automatic`: Uses automatic selection based on density.
		- `none`: No algorithm, direct assignment.

	Changes the default algorithm to be used for matching blobs in one frame with blobs in the next frame. The accurate algorithm performs best, but also scales less well for more individuals than the approximate one. However, if it is too slow (temporarily) in a few frames, the program falls back to using the approximate one that doesnt slow down.




.. function:: meta_age_days(int)

	**default value:** -1


	Age of the individuals used in days.



.. function:: meta_conditions(string)

	**default value:** ""


	Treatment name.



.. function:: meta_conversion_time(string)

	**default value:** ""


	This contains the time of when this video was converted / recorded as a string.



.. function:: meta_encoding(meta_encoding_t)

	**default value:** rgb8

	**possible values:**
		- `gray`: No color information is stored. This makes .pv video files very small, but loses all greyscale or color information.
		- `r3g3b2`: Grayscale video, calculated by simply extracting one channel (default R) from the video.
		- `rgb8`: Encode all colors into a 256-colors unsigned 8-bit integer. The top 2 bits are blue (4 shades), the following 3 bits green (8 shades) and the last 3 bits red (8 shades).
		- `binary`: Encode all colors into a full color 8-bit R8G8B8 array.

	The encoding used for the given .pv video.




.. function:: meta_mass_mg(float)

	**default value:** 200


	Used for exporting event-energy levels.



.. function:: meta_misc(string)

	**default value:** ""


	Other information.



.. function:: meta_real_width(float)

	**default value:** 0


	Used to calculate the ``cm_per_pixel`` conversion factor, relevant for e.g. converting the speed of individuals from px/s to cm/s (to compare to ``track_max_speed`` which is given in cm/s). By default set to 30 if no other values are available (e.g. via command-line). This variable should reflect actual width (in cm) of what is seen in the video image. For example, if the video shows a tank that is 50cm in X-direction and 30cm in Y-direction, and the image is cropped exactly to the size of the tank, then this variable should be set to 50.

	.. seealso:: :param:`cm_per_pixel`, :param:`track_max_speed`


.. function:: meta_source_path(string)

	**default value:** ""


	Path of the original video file for conversions (saved as debug info).



.. function:: meta_species(string)

	**default value:** ""


	Name of the species used.



.. function:: meta_video_scale(float)

	**default value:** 1


	Scale applied to the original video / footage.



.. function:: meta_video_size(size)

	**default value:** [0,0]


	Resolution of the original video.



.. function:: meta_write_these(array<string>)

	**default value:** ["meta_species","meta_age_days","meta_conditions","meta_misc","cam_limit_exposure","meta_real_width","meta_source_path","meta_cmd","meta_build","meta_conversion_time","meta_video_scale","meta_video_size","detect_classes","meta_encoding","detect_skeleton","frame_rate","calculate_posture","cam_undistort_vector","cam_matrix","cm_per_pixel","track_size_filter","track_threshold","track_posture_threshold","track_do_history_split","track_max_individuals","track_background_subtraction","track_max_speed","detect_model","region_model","detect_resolution","region_resolution","detect_batch_size","detect_type","detect_iou_threshold","detect_conf_threshold","video_conversion_range","detect_batch_size","detect_threshold","output_dir","output_prefix","filename"]


	The given settings values will be written to the video file.



.. function:: midline_invert(bool)

	**default value:** false


	If enabled, all midlines will be inverted (tail/head swapped).



.. function:: midline_resolution(uint)

	**default value:** 25


	Number of midline points that are saved. Higher number increases detail.



.. function:: midline_start_with_head(bool)

	**default value:** false


	If enabled, the midline is going to be estimated starting at the head instead of the tail.



.. function:: midline_stiff_percentage(float)

	**default value:** 0.15


	Percentage of the midline that can be assumed to be stiff. If the head position seems poorly approximated (straighened out too much), then decrease this value.



.. function:: midline_walk_offset(float)

	**default value:** 0.025


	This percentage of the number of outline points is the amount of points that the midline-algorithm is allowed to move left and right upon each step. Higher numbers will make midlines more straight, especially when extremities are present (that need to be skipped over), but higher numbers will also potentially decrease accuracy for less detailed objects.



.. function:: nowindow(bool)

	**default value:** false


	If set to true, no GUI will be created on startup (e.g. when starting from SSH).



.. function:: outline_approximate(uchar)

	**default value:** 3


	If this is a number > 0, the outline detected from the image will be passed through an elliptical fourier transform with  number of coefficients. When the given number is sufficiently low, the outline will be smoothed significantly (and more so for lower numbers of coefficients).



.. function:: outline_compression(float)

	**default value:** 0


	Applies a *lossy* compression to the outlines generated by segmentation models. Walking around the outline, it removes line segments that do not introduce any noticable change in direction. The factor specified here controls how much proportional difference in radians/angle is allowed. The value isnt in real radians, as the true downsampling depends on the size of the object (smaller objects = smaller differences allowed).



.. function:: outline_curvature_range_ratio(float)

	**default value:** 0.03


	Determines the ratio between number of outline points and distance used to calculate its curvature. Program will look at index +- ``ratio * size()`` and calculate the distance between these points (see posture window red/green color).



.. function:: outline_resample(float)

	**default value:** 1


	Spacing between outline points in pixels (``0<value<255``), after resampling the outline. A lower value here can drastically increase the number of outline points being generated (and decrease analysis speed), while a higher value is going to do the opposite. By default this value is 1-pixel, meaning that there is no artificial interpolation or down-sampling.



.. function:: outline_smooth_samples(uchar)

	**default value:** 4


	Use N samples for smoothing the outline. More samples will generate a smoother (less detailed) outline.



.. function:: outline_smooth_step(uchar)

	**default value:** 1


	Jump over N outline points when smoothing (reducing accuracy).



.. function:: outline_use_dft(bool)

	**default value:** true


	If enabled, the program tries to reduce outline noise by convolution of the curvature array with a low pass filter.



.. function:: output_annotations(map<string,string>)

	**default value:** {"ACCELERATION":"cm/s2","ACCELERATION_SMOOTH":"cm/s2","BORDER_DISTANCE":"cm","NEIGHBOR_DISTANCE":"cm","ORIENTATION":"rad","SPEED":"cm/s","SPEED_OLD":"cm/s","SPEED_SMOOTH":"cm/s","VX":"cm/s","VY":"cm/s","X":"cm","Y":"cm","global":"px"}


	Units (as a string) of output functions to be annotated in various places like graphs.



.. function:: output_auto_pose(bool)

	**default value:** true


	If this is set to false, then no poseX[n] and poseY[n] fields will automatically be added to the ``output_fields`` based on what the keypoint model reports. You can still manually add them if you like.

	.. seealso:: :param:`output_fields`


.. function:: output_centered(bool)

	**default value:** false


	If set to true, the origin of all X and Y coordinates is going to be set to the center of the video. Using this overrides ``output_origin``.

	.. seealso:: :param:`output_origin`


.. function:: output_csv_decimals(uchar)

	**default value:** 2


	Maximum number of decimal places that is written into CSV files (a text-based format for storing data). A value of 0 results in integer values.



.. function:: output_default_options(map<string,array<string>>)

	**default value:** {"event_acceleration":["/10"],"ACCELERATION":["/15","CENTROID"],"L_V":["/10"],"v_direction":["/10"],"DOT_V":["/10"],"ANGULAR_V":["/10","CENTROID"],"ANGULAR_A":["/1000","CENTROID"],"NEIGHBOR_VECTOR_T":["/1"],"SPEED":["/10"],"NEIGHBOR_DISTANCE":["/10"],"X":["/100"],"Y":["/100"],"tailbeat_threshold":["pm"],"tailbeat_peak":["pm"],"threshold_reached":["POINTS"],"midline_length":["/15"],"amplitude":["/100"],"outline_size":["/100"],"global":["/10"]}


	Default scaling and smoothing options for output functions, which are applied to functions in ``output_fields`` during export.

	.. seealso:: :param:`output_fields`


.. function:: output_dir(path)

	**default value:** ""


	Default output-/input-directory. Change this in order to omit paths in front of filenames for open and save.



.. function:: output_fields(array<pair<string,array<string>>>)

	**default value:** [["X",["RAW","WCENTROID"]],["Y",["RAW","WCENTROID"]],["X",["RAW","HEAD"]],["Y",["RAW","HEAD"]],["VX",["RAW","HEAD"]],["VY",["RAW","HEAD"]],["AX",["RAW","HEAD"]],["AY",["RAW","HEAD"]],["ANGLE",["RAW"]],["ANGULAR_V",["RAW"]],["ANGULAR_A",["RAW"]],["MIDLINE_OFFSET",["RAW"]],["normalized_midline",["RAW"]],["midline_length",["RAW"]],["midline_x",["RAW"]],["midline_y",["RAW"]],["midline_segment_length",["RAW"]],["SPEED",["RAW","WCENTROID"]],["SPEED",["RAW","PCENTROID"]],["SPEED",["RAW","HEAD"]],["BORDER_DISTANCE",["PCENTROID"]],["time",[]],["timestamp",[]],["frame",[]],["missing",[]],["num_pixels",[]],["ACCELERATION",["RAW","PCENTROID"]],["ACCELERATION",["RAW","WCENTROID"]]]


	The functions that will be exported when saving to CSV, or shown in the graph. ``[['X',[option], ...]]``



.. function:: output_format(output_format_t)

	**default value:** npz

	**possible values:**
		- `csv`: A standard data format, comma-separated columns for each data stream. Use `output_csv_decimals` to adjust the maximum precision for exported data.
		- `npz`: NPZ is basically a collection of binary arrays, readable by NumPy and other plugins (there are plugins available for Matlab and R).

	When pressing the S(ave) button or using ``auto_quit``, this setting allows to switch between CSV and NPZ output. NPZ files are recommended and will be used by default - some functionality (such as visual fields, posture data, etc.) will remain in NPZ format due to technical constraints.


	.. seealso:: :param:`auto_quit`


.. function:: output_frame_window(uint)

	**default value:** 100


	If an individual is selected during CSV output, use these number of frames around it (or -1 for all frames).



.. function:: output_heatmaps(bool)

	**default value:** false


	When set to true, heatmaps are going to be saved to a separate file, or set of files '_p*' - with all the settings in heatmap_* applied.



.. function:: output_interpolate_positions(bool)

	**default value:** false


	If turned on this function will linearly interpolate X/Y, and SPEED values, for all frames in which an individual is missing.



.. function:: output_invalid_value(output_invalid_t)

	**default value:** inf

	**possible values:**
		- `inf`: Infinity (e.g. np.inf)
		- `nan`: NaN (e.g. np.nan)

	Determines, what is exported in cases where the individual was not found (or a certain value could not be calculated). For example, if an individual is found but posture could not successfully be generated, then all posture-based values (e.g. ``midline_length``) default to the value specified here. By default (and for historic reasons), any invalid value is marked by 'inf'.




.. function:: output_min_frames(uint16)

	**default value:** 1


	Filters all individual with less than N frames when exporting. Individuals with fewer than N frames will also be hidden in the GUI unless ``gui_show_inactive_individuals`` is enabled (default).

	.. seealso:: :param:`gui_show_inactive_individuals`


.. function:: output_normalize_midline_data(bool)

	**default value:** false


	If enabled: save a normalized version of the midline data saved whenever ``output_posture_data`` is set to true. Normalized means that the position of the midline points is normalized across frames (or the distance between head and point n in the midline array).

	.. seealso:: :param:`output_posture_data`


.. function:: output_origin(vec)

	**default value:** [0,0]


	When exporting the data, positions will be relative to this point - unless ``output_centered`` is set, which takes precedence.

	.. seealso:: :param:`output_centered`


.. function:: output_posture_data(bool)

	**default value:** false


	Save posture data npz file along with the usual NPZ/CSV files containing positions and such. If set to true, a file called '``output_dir``/``fish_data_dir``/``<filename>_posture_fishXXX.npz``' will be created for each individual XXX.

	.. seealso:: :param:`output_dir`


.. function:: output_prefix(string)

	**default value:** ""


	If this is not empty, all output files will go into ``output_dir`` /  / ... instead of just into ``output_dir``. The output directory is usually the folder where the video is, unless set to a different folder by you.

	.. seealso:: :param:`output_dir`, :param:`output_dir`


.. function:: output_recognition_data(bool)

	**default value:** false


	Save recognition / probability data npz file along with the usual NPZ/CSV files containing positions and such. If set to true, a file called '``output_dir``/``fish_data_dir``/``<filename>_recognition_fishXXX.npz``' will be created for each individual XXX.

	.. seealso:: :param:`output_dir`


.. function:: output_statistics(bool)

	**default value:** false


	Save an NPZ file containing an array with shape Nx16 and contents [``adding_seconds``, ``combined_posture_seconds``, ``number_fish``, ``loading_seconds``, ``posture_seconds``, ``match_number_fish``, ``match_number_blob``, ``match_number_edges``, ``match_stack_objects``, ``match_max_edges_per_blob``, ``match_max_edges_per_fish``, ``match_mean_edges_per_blob``, ``match_mean_edges_per_fish``, ``match_improvements_made``, ``match_leafs_visited``, ``method_used``] and an 1D-array containing all frame numbers. If set to true, a file called '``output_dir``/``fish_data_dir``/``<filename>_statistics.npz``' will be created. This will not output anything interesting, if the data was loaded instead of analysed.

	.. seealso:: :param:`output_dir`


.. function:: output_tracklet_images(bool)

	**default value:** false


	If set to true, the program will output one median image per tracklet (time-series segment) and save it alongside the npz/csv files (inside ``<filename>_tracklet_images.npz``). It will also output (if ``tracklet_max_images`` is 0) all images of each tracklet in a separate npz files named ``<filename>_tracklet_images_single_*.npz``.

	.. seealso:: :param:`tracklet_max_images`


.. function:: output_visual_fields(bool)

	**default value:** false


	Export visual fields for all individuals upon saving.



.. function:: panic_button(int)

	**default value:** 0


	42



.. function:: peak_mode(peak_mode_t)

	**default value:** pointy

	**possible values:**
		- `pointy`: The head is broader than the tail.
		- `broad`: The tail is broader than the head.

	This determines whether the tail of an individual should be expected to be pointy or broad.




.. function:: pose_midline_indexes(PoseMidlineIndexes)

	**default value:** []


	This is an array of joint indexes (in the order as predicted by a YOLO-pose model), which are used to determine the joints making up the midline of an object. The first index is the head, the last the tail. This is used to generate a posture when using YOLO-pose models with ``calculate_posture`` enabled.

	.. seealso:: :param:`calculate_posture`


.. function:: posture_closing_size(uchar)

	**default value:** 2


	The kernel size for erosion / dilation of the posture algorithm. Only has an effect with  ``posture_closing_steps`` > 0.

	.. seealso:: :param:`posture_closing_steps`


.. function:: posture_closing_steps(uchar)

	**default value:** 0


	When enabled (> 0), posture will be processed using a combination of erode / dilate in order to close holes in the shape and get rid of extremities. An increased number of steps will shrink the shape, but will also be more time intensive.



.. function:: posture_direction_smoothing(uint16)

	**default value:** 0


	Enables or disables smoothing of the posture orientation based on previous frames (not good for fast turns).



.. function:: posture_head_percentage(float)

	**default value:** 0.1


	The percentage of the midline-length that the head is moved away from the front of the body.



.. function:: postures_per_thread(float)

	**default value:** 1


	Number of individuals for which postures will be estimated per thread.



.. function:: python_path(path)

	**default value:** "/Users/tristan/miniforge3/envs/trex/bin/python3.11"


	Path to the python home folder. If left empty, the user is required to make sure that all necessary libraries are in-scope the PATH environment variable.



.. function:: quit_after_average(bool)

	**default value:** false


	If set to true, this will terminate the program directly after generating (or loading) a background average image.



.. function:: recognition_border(recognition_border_t)

	**default value:** none

	**possible values:**
		- `none`: No border at all. All points are inside the recognition boundary. (default)
		- `heatmap`: Looks at a subset of frames from the video, trying to find out where individuals go and masking all the places they do not.
		- `outline`: Similar to heatmap, but tries to build a convex border around the around (without holes in it).
		- `shapes`: Any array of convex shapes. Set coordinates by changing `recognition_shapes`.
		- `grid`: The points defined in `grid_points` are turned into N different circles inside the arena (with points in `grid_points` being the circle centers), which define in/out if inside/outside any of the circles.
		- `circle`: The video-file provides a binary mask (e.g. when `cam_circle_mask` was set to true during recording), which is then used to determine in/out.

	This defines the type of border that is used in all automatic recognition routines. Depending on the type set here, you might need to set other parameters as well (e.g. ``recognition_shapes``). In general, this defines whether an image of an individual is usable for automatic recognition. If it is inside the defined border, then it will be passed on to the recognition network - if not, then it wont.


	.. seealso:: :param:`recognition_shapes`


.. function:: recognition_border_shrink_percent(float)

	**default value:** 0.3


	The amount by which the recognition border is shrunk after generating it (roughly and depends on the method).



.. function:: recognition_border_size_rescale(float)

	**default value:** 0.5


	The amount that blob sizes for calculating the heatmap are allowed to go below or above values specified in ``track_size_filter`` (e.g. 0.5 means that the sizes can range between ``track_size_filter.min * (1 - 0.5)`` and ``track_size_filter.max * (1 + 0.5)``).

	.. seealso:: :param:`track_size_filter`


.. function:: recognition_coeff(uint16)

	**default value:** 50


	If ``recognition_border`` is 'outline', this is the number of coefficients to use when smoothing the ``recognition_border``.

	.. seealso:: :param:`recognition_border`, :param:`recognition_border`


.. function:: recognition_save_progress_images(bool)

	**default value:** false


	If set to true, an image will be saved for all training epochs, documenting the uniqueness in each step.



.. function:: recognition_shapes(array<array<vec>>)

	**default value:** []


	If ``recognition_border`` is set to 'shapes', then the identification network will only be applied to blobs within the convex shapes specified here.

	.. seealso:: :param:`recognition_border`


.. function:: recognition_smooth_amount(uint16)

	**default value:** 200


	If ``recognition_border`` is 'outline', this is the amount that the ``recognition_border`` is smoothed (similar to ``outline_smooth_samples``), where larger numbers will smooth more.

	.. seealso:: :param:`recognition_border`, :param:`recognition_border`, :param:`outline_smooth_samples`


.. function:: recording(bool)

	**default value:** true


	If set to true, the program will record frames whenever individuals are found.



.. function:: region_model(path)

	**default value:** ""


	The path to a .pt file that contains a valid PyTorch object detection model used for region proposal (currently only YOLO networks are supported).



.. function:: reset_average(bool)

	**default value:** false


	If set to true, the average will be regenerated using the live stream of images (video or camera).



.. function:: save_raw_movie(bool)

	**default value:** false


	Saves a RAW movie (.mov) with a similar name in the same folder, while also recording to a PV file. This might reduce the maximum framerate slightly, but it gives you the best of both worlds.



.. function:: save_raw_movie_path(path)

	**default value:** ""


	The path to the raw movie file. If empty, the same path as the PV file will be used (but as a .mov).



.. function:: settings_file(path)

	**default value:** ""


	Name of the settings file. By default, this will be set to ``filename``.settings in the same folder as ``filename``.

	.. seealso:: :param:`filename`, :param:`filename`


.. function:: smooth_window(uint)

	**default value:** 2


	Smoothing window used for exported data with the #smooth tag.



.. function:: solid_background_color(uchar)

	**default value:** 255


	A greyscale value in case ``enable_difference`` is set to false - TGrabs will automatically generate a background image with the given color.

	.. seealso:: :param:`enable_difference`


.. function:: source(PathArray)

	**default value:** ""


	This is the (video) source for the current session. Typically this would point to the original video source of ``filename``.

	.. seealso:: :param:`filename`


.. function:: speed_extrapolation(float)

	**default value:** 3


	Used for matching when estimating the next position of an individual. Smaller values are appropriate for lower frame rates. The higher this value is, the more previous frames will have significant weight in estimating the next position (with an exponential decay).



.. function:: stop_after_minutes(uint)

	**default value:** 0


	If set to a value above 0, the video will stop recording after X minutes of recording time.



.. function:: system_memory_limit(uint64)

	**default value:** 0


	Custom override of how many bytes of system RAM the program is allowed to fill. If ``approximate_length_minutes`` or ``stop_after_minutes`` are set, this might help to increase the resulting RAW video footage frame_rate.

	.. seealso:: :param:`approximate_length_minutes`, :param:`stop_after_minutes`


.. function:: tags_approximation(float)

	**default value:** 0.025


	Higher values (up to 1.0) will lead to coarser approximation of the rectangle/tag shapes.



.. function:: tags_debug(bool)

	**default value:** false


	(beta) Enable debugging for tags.



.. function:: tags_dont_track(bool)

	**default value:** true


	If true, disables the tracking of tags as objects in TRex. This means that tags are not displayed like other objects and are instead only used as additional 'information' to correct tracks. However, if you enabled ``tags_saved_only`` in TGrabs, setting this parameter to true will make your TRex look quite empty.

	.. seealso:: :param:`tags_saved_only`


.. function:: tags_enable(bool)

	**default value:** false


	(beta) If enabled, TGrabs will search for (black) square shapes with white insides (and other stuff inside them) - like QRCodes or similar tags. These can then be recognized using a pre-trained machine learning network (see ``tags_recognize``), and/or exported to PNG files using ``tags_save_predictions``.

	.. seealso:: :param:`tags_recognize`, :param:`tags_save_predictions`


.. function:: tags_equalize_hist(bool)

	**default value:** false


	Apply a histogram equalization before applying a threshold. Mostly this should not be necessary due to using adaptive thresholds anyway.



.. function:: tags_image_size(size)

	**default value:** [32,32]


	The image size that tag images are normalized to.



.. function:: tags_maximum_image_size(size)

	**default value:** [80,80]


	Tags that are bigger than these pixel dimensions may be cropped off. All extracted tags are then pre-aligned to any of their sides, and normalized/scaled down or up to a 32x32 picture (to make life for the machine learning network easier).



.. function:: tags_model_path(path)

	**default value:** "tag_recognition_network.h5"


	The pretrained model used to recognize QRcodes/tags according to `https://github.com/jgravi[...]/master_list.pdf <https://github.com/jgraving/pinpoint/blob/2d7f6803b38f52acb28facd12bd106754cad89bd/barcodes/old_barcodes_py2/4x4_4bit/master_list.pdf>`_. Path to a pretrained network .h5 file that takes 32x32px images of tags and returns a (N, 122) shaped tensor with 1-hot encoding.



.. function:: tags_num_sides(range<int>)

	**default value:** [3,7]


	The number of sides of the tag (e.g. should be 4 if it is a rectangle).



.. function:: tags_path(path)

	**default value:** ""


	If this path is set, the program will try to find tags and save them at the specified location.



.. function:: tags_recognize(bool)

	**default value:** false


	(beta) Apply an existing machine learning network to turn images of tags into tag ids (numbers, e.g. 1-122). Be sure to set ``tags_model_path`` along-side this.

	.. seealso:: :param:`tags_model_path`


.. function:: tags_save_predictions(bool)

	**default value:** false


	Save images of tags, sorted into folders labelled according to network predictions (i.e. 'tag 22') to '``output_dir`` / ``tags_`` ``filename`` / ``<individual>.<frame>`` / ``*``'. 

	.. seealso:: :param:`output_dir`, :param:`filename`


.. function:: tags_saved_only(bool)

	**default value:** false


	(beta) If set to true, all objects other than the detected blobs are removed and will not be written to the output video file.



.. function:: tags_size_range(range<double>)

	**default value:** [0.08,2]


	The minimum and maximum area accepted as a (square) physical tag on the individuals.



.. function:: tags_threshold(int)

	**default value:** -5


	Threshold passed on to cv::adaptiveThreshold, lower numbers (below zero) are equivalent to higher thresholds / removing more of the pixels of objects and shrinking them. Positive numbers may invert the image/mask.



.. function:: task(TRexTask_t)

	**default value:** none

	**possible values:**
		- `none`: No task forced. Auto-select.
		- `track`: Load an existing .pv file and track / edit individuals.
		- `convert`: Convert source material to .pv file.
		- `annotate`: Annotate video or image source material.
		- `rst`: Save .rst parameter documentation files to the output folder.

	The task selected by the user upon startup. This is used to determine which GUI mode to start in.




.. function:: terminate_training(bool)

	**default value:** false


	Setting this to true aborts the training in progress.



.. function:: test_image(string)

	**default value:** "checkerboard"


	Defines, which test image will be used if ``video_source`` is set to 'test_image'.

	.. seealso:: :param:`video_source`


.. function:: threshold_maximum(int)

	**default value:** 255


	



.. function:: threshold_ratio_range(range<float>)

	**default value:** [0.5,1]


	If ``track_threshold_2`` is not equal to zero, this ratio will be multiplied by the number of pixels present before the second threshold. If the resulting size falls within the given range, the blob is deemed okay.

	.. seealso:: :param:`track_threshold_2`


.. function:: track_background_subtraction(bool)

	**default value:** false


	If enabled, objects in .pv videos will first be contrasted against the background before thresholding (background_colors - object_colors). ``track_threshold_is_absolute`` then decides whether this term is evaluated in an absolute or signed manner.

	.. seealso:: :param:`track_threshold_is_absolute`


.. function:: track_conf_threshold(float)

	**default value:** 0.1


	During tracking, detections with confidence levels below the given fraction (0-1) for labels (assigned by an ML network during video conversion) will be discarded. These objects will not be assigned to any individual.



.. function:: track_consistent_categories(bool)

	**default value:** false


	Utilise categories (if present) when tracking. This may break trajectories in places with imperfect categorization, but only applies once categories have been applied.



.. function:: track_do_history_split(bool)

	**default value:** true


	If disabled, blobs will not be split automatically in order to separate overlapping individuals. This usually happens based on their history.



.. function:: track_enforce_frame_rate(bool)

	**default value:** true


	Enforce the ``frame_rate`` and override the frame_rate provided by the video file for calculating kinematic properties and probabilities. If this is not enabled, ``frame_rate`` is only a cosmetic property that influences the GUI and not exported data (for example).

	.. seealso:: :param:`frame_rate`, :param:`frame_rate`


.. function:: track_history_split_threshold(frame)

	**default value:** null


	If this is greater than 0, then individuals with tracklets < this threshold will not be considered for the splitting algorithm. That means that objects have to be detected for at least ``N`` frames in a row to play a role in history splitting.



.. function:: track_ignore(array<array<vec>>)

	**default value:** []


	If this is not empty, objects within the given rectangles or polygons (>= 3 points) ``[[x0,y0],[x1,y1](, ...)], ...]`` will be ignored during tracking.



.. function:: track_ignore_bdx(map<frame,set<blob>>)

	**default value:** {}


	This is a map of frame -> [bdx0, bdx1, ...] of blob ids that are specifically set to be ignored in the given frame. Can be reached using the GUI by clicking on a blob in raw mode.



.. function:: track_include(array<array<vec>>)

	**default value:** []


	If this is not empty, objects within the given rectangles or polygons (>= 3 points) ``[[x0,y0],[x1,y1](, ...)], ...]`` will be the only objects being tracked. (overwrites ``track_ignore``)

	.. seealso:: :param:`track_ignore`


.. function:: track_intensity_range(range<int>)

	**default value:** [-1,-1]


	When set to valid values, objects will be filtered to have an average pixel intensity within the given range.



.. function:: track_max_individuals(uint)

	**default value:** 1024


	The maximal number of individual that are assigned at the same time (infinite if set to zero). If the given number is below the actual number of individual, then only a (random) subset of individual are assigned and a warning is shown.



.. function:: track_max_reassign_time(float)

	**default value:** 0.5


	Distance in time (seconds) where the matcher will stop trying to reassign an individual based on previous position. After this time runs out, depending on the settings, the tracker will try to find it based on other criteria, or generate a new individual.



.. function:: track_max_speed(float)

	**default value:** 0


	The maximum speed an individual can have (=> the maximum distance an individual can travel within one second) in cm/s. Uses and is influenced by ``meta_real_width`` and ``cm_per_pixel`` as follows: ``speed(px/s) * cm_per_pixel(cm/px) -> cm/s``.

	.. seealso:: :param:`meta_real_width`, :param:`cm_per_pixel`


.. function:: track_only_categories(array<string>)

	**default value:** []


	If this is a non-empty list, only objects that have previously been assigned one of the correct categories will be tracked. Note that this also affects anything below ``categories_apply_min_tracklet_length`` length (e.g. noise particles or short tracklets).

	.. seealso:: :param:`categories_apply_min_tracklet_length`


.. function:: track_only_classes(array<string>)

	**default value:** []


	If this is a non-empty list, only objects that have any of the given labels (assigned by a ML network during video conversion) will be tracked.



.. function:: track_only_segmentations(bool)

	**default value:** false


	If this is enabled, only segmentation results will be tracked - this avoids double tracking of bounding boxes and segmentation masks.



.. function:: track_pause(bool)

	**default value:** false


	Halts the analysis.



.. function:: track_posture_threshold(int)

	**default value:** 0


	Same as ``track_threshold``, but for posture estimation.

	.. seealso:: :param:`track_threshold`


.. function:: track_size_filter(SizeFilters)

	**default value:** []


	Blobs below the lower bound are recognized as noise instead of individuals. Blobs bigger than the upper bound are considered to potentially contain more than one individual. You can look these values up by pressing ``D`` in TRex to get to the raw view (see `<https://trex.run/docs/gui.html>`_ for details). The unit is #pixels * (cm/px)^2. ``cm_per_pixel`` is used for this conversion.

	.. seealso:: :param:`cm_per_pixel`


.. function:: track_speed_decay(float)

	**default value:** 1


	The amount the expected speed is reduced over time when an individual is lost. When individuals collide, depending on the expected behavior for the given species, one should choose different values for this variable. If the individuals usually stop when they collide, this should be set to 1. If the individuals are expected to move over one another, the value should be set to ``0.7 > value > 0``.



.. function:: track_threshold(int)

	**default value:** 0


	Constant used in background subtraction. Pixels with grey values above this threshold will be interpreted as potential individuals, while pixels below this threshold will be ignored.



.. function:: track_threshold_2(int)

	**default value:** 0


	If not zero, a second threshold will be applied to all objects after they have been deemed do be theoretically large enough. Then they are compared to #before_pixels * ``threshold_ratio_range`` to see how much they have been shrunk).

	.. seealso:: :param:`threshold_ratio_range`


.. function:: track_threshold_is_absolute(bool)

	**default value:** true


	If enabled, uses absolute difference values and disregards any pixel |p| < ``threshold`` during conversion. Otherwise the equation is p < ``threshold``, meaning that e.g. bright spots may not be considered trackable when dark spots would. Same as ``detect_threshold_is_absolute``, but during tracking instead of converting.

	.. seealso:: :param:`detect_threshold_is_absolute`


.. function:: track_time_probability_enabled(bool)

	**default value:** true


	



.. function:: track_trusted_probability(float)

	**default value:** 0.25


	If the (purely kinematic-based) probability that is used to assign an individual to an object is smaller than this value, the current tracklet ends and a new one starts. Even if the individual may still be assigned to the object, TRex will be *unsure* and no longer assume that it is definitely the same individual.



.. function:: tracklet_force_normal_color(bool)

	**default value:** true


	If set to true (default) then all images are saved as they appear in the original video. Otherwise, all images are exported according to the individual image settings (as seen in the image settings when an individual is selected) - in which case the background may have been subtracted from the original image and a threshold may have been applied (if ``track_threshold`` > 0 and ``track_background_subtraction`` is true).

	.. seealso:: :param:`track_threshold`, :param:`track_background_subtraction`


.. function:: tracklet_max_images(uint16)

	**default value:** 0


	This limits the maximum number of images that are being exported per tracklet given that ``output_tracklet_images`` is true. If the number is 0 (default), then every image will be exported. Otherwise, only a uniformly sampled subset of N images will be exported.

	.. seealso:: :param:`output_tracklet_images`


.. function:: tracklet_max_length(float)

	**default value:** 0


	If set to something bigger than zero, this represents the maximum number of seconds that a tracklet can be.



.. function:: tracklet_normalize(bool)

	**default value:** true


	If enabled, all exported tracklet images are normalized according to the ``individual_image_normalization`` and padded / shrunk to ``individual_image_size`` (they appear as they do in the image preview when selecting an individual in the GUI).

	.. seealso:: :param:`individual_image_normalization`, :param:`individual_image_size`


.. function:: tracklet_punish_speeding(bool)

	**default value:** true


	Sometimes individuals might be assigned to blobs that are far away from the previous position. This could indicate wrong assignments, but not necessarily. If this variable is set to true, tracklets will end whenever high speeds are reached, just to be on the safe side. For scenarios with lots of individuals (and no recognition) this might spam yellow bars in the timeline and may be disabled.



.. function:: tracklet_punish_timedelta(bool)

	**default value:** true


	If enabled, a huge timestamp difference will end the current trajectory tracklet and will be displayed as a reason in the tracklet overview at the top of the selected individual info card.



.. function:: use_adaptive_threshold(bool)

	**default value:** false


	Enables or disables adaptive thresholding (slower than normal threshold). Deals better with weird backgrounds.



.. function:: use_closing(bool)

	**default value:** false


	Toggles the attempt to close weird blobs using dilation/erosion with ``closing_size`` sized filters.

	.. seealso:: :param:`closing_size`


.. function:: use_differences(bool)

	**default value:** false


	This should be set to false unless when using really old files.



.. function:: video_conversion_range(range<int>)

	**default value:** [-1,-1]


	This determines which part of the video will be converted. By default (``[-1,-1]``) the entire video will be converted. If set to a valid value (not -1), start and end values determine the range converted (each one can be valid independently of the other).



.. function:: video_length(uint64)

	**default value:** 0


	The length of the video in frames



.. function:: video_reading_use_threads(bool)

	**default value:** true


	Use threads to read images from a video file.



.. function:: video_size(size)

	**default value:** [-1,-1]


	The dimensions of the currently loaded video.



.. function:: video_source(string)

	**default value:** "webcam"


	Where the video is recorded from. Can be the name of a file, or one of the keywords ['basler', 'webcam', 'test_image'].



.. function:: visual_field_eye_offset(float)

	**default value:** 0.15


	A percentage telling the program how much the eye positions are offset from the start of the midline.



.. function:: visual_field_eye_separation(float)

	**default value:** 60


	Degrees of separation between the eye and looking straight ahead. Results in the eye looking towards head.angle +- .



.. function:: visual_field_history_smoothing(uchar)

	**default value:** 0


	The maximum number of previous values (and look-back in frames) to take into account when smoothing visual field orientations. If greater than 0, visual fields will use smoothed previous eye positions to determine the optimal current eye position. This is usually only necessary when postures are somewhat noisy to a degree that makes visual fields unreliable.



.. function:: visual_field_shapes(array<array<vec>>)

	**default value:** []


	A list of shapes that should be handled as view-blocking in visual field calculations.



.. function:: visual_identification_model_path(optional<path>)

	**default value:** null


	If this is set to a path, visual identification 'load weights' or 'apply' will try to load this path first if it exists. This way you can facilitate transfer learning (taking a model file from one video and applying it to a different video of the same individuals).



.. function:: visual_identification_save_images(bool)

	**default value:** false


	If set to true, the program will save the images used for a successful training of the visual identification to ``output_dir``.

	.. seealso:: :param:`output_dir`


.. function:: visual_identification_version(visual_identification_version_t)

	**default value:** v118_3

	**possible values:**
		- `current`: This always points to the current version.
		- `v200`: The v200 model introduces a deeper architecture with five convolutional layers, compared to the four layers in v119. The convolutional layers in v200 have channel sizes of 64, 128, 256, 512, and 512, whereas v119 has channel sizes of 256, 128, 32, and 128. Additionally, v200 incorporates global average pooling and larger dropout rates, enhancing regularization. Both models use Batch Normalization, ReLU activations, and MaxPooling, but v200 features a more complex fully connected layer structure with an additional dropout layer before the final classification.
		- `v119`: The v119 model introduces an additional convolutional layer, increasing the total to four layers with channel sizes of 256, 128, 32, and 128, compared to the three layers in v118_3 with channel sizes of 16, 64, and 100. Additionally, v119 features larger fully connected layers with 1024 units in the first layer, whereas v118_3 has 100 units. Both models maintain the use of MaxPooling, Dropout, Batch Normalization, and a final Softmax activation for class probabilities.
		- `v118_3`: The order of Max-Pooling layers was changed, along with some other minor changes.
		- `v110`: Changed activation order, added BatchNormalization. No Flattening to maintain spatial context.
		- `v100`: The original layout.
		- `convnext_base`: The ConvNeXtBase architecture is a deep convolutional neural network (CNN) designed for image classification tasks, featuring a modernized ResNet-inspired structure with large kernel sizes, efficient attention mechanisms, and an optimized design for both computational efficiency and performance, aimed at achieving state-of-the-art results on large-scale image datasets.
		- `vgg_16`: The VGG16 architecture is a deep convolutional neural network (CNN) designed for image classification tasks, featuring a simple and straightforward structure with small kernel sizes, a large number of layers, and a focus on simplicity and ease of use, aimed at achieving strong results on small-scale image datasets.
		- `vgg_19`: The VGG19 architecture is a deep convolutional neural network (CNN) designed for image classification tasks, featuring a simple and straightforward structure with small kernel sizes, a large number of layers, and a focus on simplicity and ease of use, aimed at achieving strong results on small-scale image datasets.
		- `mobilenet_v3_small`: The MobileNetV3Small architecture is a deep convolutional neural network (CNN) designed for image classification tasks, featuring a lightweight structure with small kernel sizes, efficient depthwise separable convolutions, and an emphasis on computational efficiency and performance, aimed at achieving strong results on mobile and edge devices with limited computational resources.
		- `mobilenet_v3_large`: The MobileNetV3Large architecture is a deep convolutional neural network (CNN) designed for image classification tasks, featuring a lightweight structure with small kernel sizes, efficient depthwise separable convolutions, and an emphasis on computational efficiency and performance, aimed at achieving strong results on mobile and edge devices with limited computational resources.
		- `inception_v3`: The InceptionV3 architecture is a deep convolutional neural network (CNN) designed for image classification tasks, featuring a complex structure with multiple parallel paths, efficient factorization methods, and an emphasis on computational efficiency and performance, aimed at achieving strong results on large-scale image datasets.
		- `resnet_50_v2`: The ResNet50V2 architecture is a deep convolutional neural network (CNN) designed for image classification tasks, featuring a modernized ResNet-inspired structure with bottleneck blocks, efficient skip connections, and an optimized design for both computational efficiency and performance, aimed at achieving strong results on large-scale image datasets.
		- `efficientnet_b0`: The EfficientNetB0 architecture is a deep convolutional neural network (CNN) designed for image classification tasks, featuring a lightweight structure with small kernel sizes, efficient depthwise separable convolutions, and an emphasis on computational efficiency and performance, aimed at achieving strong results on mobile and edge devices with limited computational resources.
		- `resnet_18`: The ResNet18 architecture is a deep convolutional neural network (CNN) designed for image classification tasks, featuring a modernized ResNet-inspired structure with bottleneck blocks, efficient skip connections, and an optimized design for both computational efficiency and performance, aimed at achieving strong results on large-scale image datasets.

	Newer versions of TRex sometimes change the network layout for (e.g.) visual identification, which will make them incompatible with older trained models. This parameter allows you to change the expected version back, to ensure backwards compatibility. It also features many public network layouts available from the Keras package. In case training results do not match expectations, please first check the quality of your trajectories before trying out different network layouts.




.. function:: web_quality(int)

	**default value:** 75


	JPEG quality of images transferred over the web interface.



.. function:: web_time_threshold(float)

	**default value:** 0.05


	Maximum refresh rate in seconds for the web interface.



.. function:: webcam_index(uchar)

	**default value:** 0


	cv::VideoCapture index of the current webcam. If the program chooses the wrong webcam (``source`` = webcam), increase this index until it finds the correct one.

	.. seealso:: :param:`source`


.. function:: yolo_region_tracking_enabled(bool)

	**default value:** false


	If set to true, the program will try to use yolov8s internal tracking routine to improve results for region tracking. This can be significantly slower and disables batching.



.. function:: yolo_tracking_enabled(bool)

	**default value:** false


	If set to true, the program will try to use yolov8s internal tracking routine to improve results. This can be significantly slower and disables batching.



