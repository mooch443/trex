.. toctree::
   :maxdepth: 2

TRex parameters
###############
.. function:: analysis_paused(bool)

	**default value:** false


	Halts the analysis.



.. function:: analysis_range(pair<int,int>)

	**default value:** [-1,-1]


	Sets start and end of the analysed frames.



.. function:: app_check_for_updates(app_update_check_t)

	**default value:** none

	**possible values:**
		- `none`: No status has been set yet and the program will ask the user.
		- `manually`: Manually check for updates, do not automatically check for them online.
		- `automatically`: Automatically check for updates periodically (once per week).

	If enabled, the application will regularly check for updates online (`<https://api.github.com/repos/mooch443/trex/releases>`_).




.. function:: app_last_update_check(uint64)

	**default value:** 0


	Time-point of when the application has last checked for an update.



.. function:: app_last_update_version(string)

	**default value:** ""


	



.. function:: app_name(string)

	**default value:** "TRex"


	Name of the application.



.. function:: auto_apply(bool)

	**default value:** false


	If set to true, the application will automatically apply the network with existing weights once the analysis is done. It will then automatically correct and reanalyse the video.



.. function:: auto_categorize(bool)

	**default value:** false


	If set to true, the program will try to load <video>_categories.npz from the ``output_dir``. If successful, then categories will be computed according to the current categories_ settings. Combine this with the ``auto_quit`` parameter to automatically save and quit afterwards. If weights cannot be loaded, the app crashes.

	.. seealso:: :func:`output_dir`, :func:`auto_quit`, 


.. function:: auto_minmax_size(bool)

	**default value:** false


	Program will try to find minimum / maximum size of the individuals automatically for the current ``cm_per_pixel`` setting. Can only be passed as an argument upon startup. The calculation is based on the median blob size in the video and assumes a relatively low level of noise.

	.. seealso:: :func:`cm_per_pixel`, 


.. function:: auto_no_memory_stats(bool)

	**default value:** true


	If set to true, no memory statistics will be saved on auto_quit.



.. function:: auto_no_results(bool)

	**default value:** false


	If set to true, the auto_quit option will NOT save a .results file along with the NPZ (or CSV) files. This saves time and space, but also means that the tracked portion cannot be loaded via -load afterwards. Useful, if you only want to analyse the resulting data and never look at the tracked video again.



.. function:: auto_no_tracking_data(bool)

	**default value:** false


	If set to true, the auto_quit option will NOT save any ``output_graphs`` tracking data - just the posture data (if enabled) and the results file (if not disabled). This saves time and space if that is a need.

	.. seealso:: :func:`output_graphs`, 


.. function:: auto_number_individuals(bool)

	**default value:** false


	Program will automatically try to find the number of individuals (with sizes given in ``blob_size_ranges``) and set ``track_max_individuals`` to that value.

	.. seealso:: :func:`blob_size_ranges`, :func:`track_max_individuals`, 


.. function:: auto_quit(bool)

	**default value:** false


	If set to true, the application will automatically save all results and export CSV files and quit, after the analysis is complete.



.. function:: auto_train(bool)

	**default value:** false


	If set to true (and ``recognition_enable`` is also set to true), the application will automatically train the recognition network with the best track segment and apply it to the video.

	.. seealso:: :func:`recognition_enable`, 


.. function:: auto_train_dont_apply(bool)

	**default value:** false


	If set to true, setting ``auto_train`` will only train and not apply the trained network.

	.. seealso:: :func:`auto_train`, 


.. function:: auto_train_on_startup(bool)

	**default value:** false


	This is a parameter that is used by the system to determine whether ``auto_train`` was set on startup, and thus also whether a failure of ``auto_train`` should result in a crash (return code != 0).

	.. seealso:: :func:`auto_train`, :func:`auto_train`, 


.. function:: blob_size_ranges(BlobSizeRange)

	**default value:** [[0.1,3]]


	Blobs below the lower bound are recognized as noise instead of individuals. Blobs bigger than the upper bound are considered to potentially contain more than one individual. You can look these values up by pressing ``D`` in TRex to get to the raw view (see `<https://trex.run/docs/gui.html>`_ for details). The unit is #pixels * (cm/px)^2. ``cm_per_pixel`` is used for this conversion.

	.. seealso:: :func:`cm_per_pixel`, 


.. function:: blob_split_global_shrink_limit(float)

	**default value:** 0.2


	The minimum percentage of the minimum in ``blob_size_ranges``, that a blob is allowed to be reduced to during splitting. If this value is set too low, the program might start recognizing parts of individual as other individual too quickly.

	.. seealso:: :func:`blob_size_ranges`, 


.. function:: blob_split_max_shrink(float)

	**default value:** 0.2


	The minimum percentage of the starting blob size (after thresholding), that a blob is allowed to be reduced to during splitting. If this value is set too low, the program might start recognizing parts of individual as other individual too quickly.



.. function:: blobs_per_thread(float)

	**default value:** 150


	Number of blobs for which properties will be calculated per thread.



.. function:: build(string)

	**default value:** ""


	Current build version



.. function:: build_architecture(string)

	**default value:** "AMD64"


	The architecture this executable was built for.



.. function:: build_cxx_options(string)

	**default value:** " -fvisibility-inlines-hidden -fvisibility=hidden -Wno-c++98-compat-pedantic -O3 -DNDEBUG -O3 -Wno-nullability-extension"


	The mode the application was built in.



.. function:: build_is_debug(string)

	**default value:** "release"


	If built in debug mode, this will show 'debug'.



.. function:: build_type(string)

	**default value:** "$<$<CONFIG:Debug>:Release>$<$<CONFIG:Release>:Debug>"


	The mode the application was built in.



.. function:: calculate_posture(bool)

	**default value:** true


	Enables or disables posture calculation. Can only be set before the video is analysed (e.g. in a settings file or as a startup parameter).



.. function:: cam_circle_mask(bool)

	**default value:** false


	If set to true, a circle with a diameter of the width of the video image will mask the video. Anything outside that circle will be disregarded as background.



.. function:: cam_matrix(array<float>)

	**default value:** [2945.0896,0,617.255432,0,2942.825195,682.473633,0,0,1]


	



.. function:: cam_scale(float)

	**default value:** 1


	Scales the image down or up by the given factor.



.. function:: cam_undistort(bool)

	**default value:** false


	If set to true, the recorded video image will be undistorted using ``cam_undistort_vector`` (1x5) and ``cam_matrix`` (3x3).

	.. seealso:: :func:`cam_undistort_vector`, :func:`cam_matrix`, 


.. function:: cam_undistort_vector(array<float>)

	**default value:** [-0.257663,-0.192336,0.002455,0.003988,0.35924]


	



.. function:: categories_min_sample_images(uint)

	**default value:** 50


	Minimum number of images for a sample to be considered relevant. This will default to 50, or ten percent of ``track_segment_max_length``, if that parameter is set. If ``track_segment_max_length`` is set, the value of this parameter will be ignored. If set to zero or one, then all samples are valid.

	.. seealso:: :func:`track_segment_max_length`, :func:`track_segment_max_length`, 


.. function:: categories_ordered(array<string>)

	**default value:** []


	Ordered list of names of categories that are used in categorization (classification of types of individuals).



.. function:: cm_per_pixel(float)

	**default value:** 0


	The ratio of ``meta_real_width / video_width`` that is used to convert pixels to centimeters. Will be automatically calculated based on a meta-parameter saved inside the video file (``meta_real_width``) and does not need to be set manually.

	.. seealso:: :func:`meta_real_width`, 


.. function:: cmd_line(string)

	**default value:** ""


	An approximation of the command-line arguments passed to the program.



.. function:: correct_illegal_lines(bool)

	**default value:** false


	In older versions of the software, blobs can be constructed in 'illegal' ways, meaning the lines might be overlapping. If the software is printing warnings about it, this should probably be enabled (makes it slower).



.. function:: debug(bool)

	**default value:** false


	Enables some verbose debug print-outs.



.. function:: debug_recognition_output_all_methods(bool)

	**default value:** false


	If set to true, a complete training will attempt to output all images for each identity with all available normalization methods.



.. function:: enable_absolute_difference(bool)

	**default value:** true


	If set to true, the threshold values will be applied to abs(image - background). Otherwise max(0, image - background).



.. function:: error_terminate(bool)

	**default value:** false


	



.. function:: event_min_peak_offset(float)

	**default value:** 0.15


	



.. function:: exec(path)

	**default value:** ""


	This can be set to the path of an additional settings file that is executed after the normal settings file.



.. function:: ffmpeg_path(path)

	**default value:** ""


	Path to an ffmpeg executable file. This is used for converting videos after recording them (from the GUI). It is not a critical component of the software, but mostly for convenience.



.. function:: filename(path)

	**default value:** ""


	Opened filename (without .pv).



.. function:: fishdata_dir(path)

	**default value:** "data"


	Subfolder (below ``output_dir``) where the exported NPZ or CSV files will be saved (see ``output_graphs``).

	.. seealso:: :func:`output_dir`, :func:`output_graphs`, 


.. function:: frame_rate(int)

	**default value:** 0


	Specifies the frame rate of the video. It is used e.g. for playback speed and certain parts of the matching algorithm. Will be set by the .settings of a video (or by the video itself).



.. function:: gpu_accepted_uniqueness(float)

	**default value:** 0


	If changed (from 0), the ratio given here will be the acceptable uniqueness for the video - which will stop accumulation if reached.



.. function:: gpu_accumulation_enable_final_step(bool)

	**default value:** true


	If enabled, the network will be trained on all the validation + training data accumulated, as a last step of the accumulation protocol cascade. This is intentional overfitting.



.. function:: gpu_accumulation_max_segments(uint)

	**default value:** 15


	If there are more than  global segments to be trained on, they will be filtered according to their quality until said limit is reached.



.. function:: gpu_enable_accumulation(bool)

	**default value:** true


	Enables or disables the idtrackerai-esque accumulation protocol cascade. It is usually a good thing to enable this (especially in more complicated videos), but can be disabled as a fallback (e.g. if computation time is a major constraint).



.. function:: gpu_learning_rate(float)

	**default value:** 0.0005


	Learning rate for training a recognition network.



.. function:: gpu_max_cache(float)

	**default value:** 2


	Size of the image cache (transferring to GPU) in GigaBytes when applying the network.



.. function:: gpu_max_epochs(uint64)

	**default value:** 150


	Maximum number of epochs for training a recognition network.



.. function:: gpu_max_sample_gb(float)

	**default value:** 2


	Maximum size of per-individual sample images in GigaBytes. If the collected images are too many, they will be sub-sampled in regular intervals.



.. function:: gpu_min_elements(uint64)

	**default value:** 25000


	Minimum number of images being collected, before sending them to the GPU.



.. function:: gpu_min_iterations(uint64)

	**default value:** 100


	Minimum number of iterations per epoch for training a recognition network.



.. function:: gpu_verbosity(gpu_verbosity_t)

	**default value:** full

	**possible values:**
		- `silent`: No output during training.
		- `full`: An animated bar with detailed information about the training progress.
		- `oneline`: One line per epoch.

	Determines the nature of the output on the command-line during training. This does not change any behaviour in the graphical interface.




.. function:: grid_points(array<vec>)

	**default value:** []


	Whenever there is an identification network loaded and this array contains more than one point ``[[x0,y0],[x1,y1],...]``, then the network will only be applied to blobs within circles around these points. The size of these circles is half of the average distance between the points.



.. function:: grid_points_scaling(float)

	**default value:** 0.8


	Scaling applied to the average distance between the points in order to shrink or increase the size of the circles for recognition (see ``grid_points``).

	.. seealso:: :func:`grid_points`, 


.. function:: gui_auto_scale(bool)

	**default value:** false


	If set to true, the tracker will always try to zoom in on the whole group. This is useful for some individuals in a huge video (because if they are too tiny, you cant see them and their posture anymore).



.. function:: gui_auto_scale_focus_one(bool)

	**default value:** true


	If set to true (and ``gui_auto_scale`` set to true, too), the tracker will zoom in on the selected individual, if one is selected.

	.. seealso:: :func:`gui_auto_scale`, 


.. function:: gui_background_color(color)

	**default value:** [0,0,0,150]


	Values < 255 will make the background more transparent in standard view. This might be useful with very bright backgrounds.



.. function:: gui_connectivity_matrix(map<int,array<float>>)

	**default value:** {}


	Internally used to store the connectivity matrix.



.. function:: gui_connectivity_matrix_file(path)

	**default value:** ""


	Path to connectivity table. Expected structure is a csv table with columns [frame | #(track_max_individuals^2) values] and frames in y-direction.



.. function:: gui_draw_only_filtered_out(bool)

	**default value:** false


	Only show filtered out blob texts.



.. function:: gui_equalize_blob_histograms(bool)

	**default value:** true


	Equalize histograms of blobs wihtin videos (makes them more visible).



.. function:: gui_faded_brightness(uchar)

	**default value:** 255


	The alpha value of tracking-related elements when timeline is hidden (0-255).



.. function:: gui_fish_color(string)

	**default value:** "identity"


	



.. function:: gui_focus_group(array<Idx_t>)

	**default value:** []


	Focus on this group of individuals.



.. function:: gui_foi_name(string)

	**default value:** "correcting"


	If not empty, the gui will display the given FOI type in the timeline and allow to navigate between them via M/N.



.. function:: gui_foi_types(array<string>)

	**default value:** []


	A list of all the foi types registered.



.. function:: gui_frame(int)

	**default value:** 0


	The currently visible frame.



.. function:: gui_happy_mode(bool)

	**default value:** false


	If ``calculate_posture`` is enabled, enabling this option likely improves your experience with TRex.

	.. seealso:: :func:`calculate_posture`, 


.. function:: gui_highlight_categories(bool)

	**default value:** false


	If enabled, categories (if applied in the video) will be highlighted in the tracking view.



.. function:: gui_interface_scale(float)

	**default value:** 1.25


	Scales the whole interface. A value greater than 1 will make it smaller.



.. function:: gui_max_path_time(float)

	**default value:** 3


	Length (in time) of the trails shown in GUI.



.. function:: gui_mode(mode_t)

	**default value:** tracking


	The currently used display mode for the GUI.



.. function:: gui_outline_thickness(uint64)

	**default value:** 1


	The thickness of outline / midlines in the GUI.



.. function:: gui_playback_speed(float)

	**default value:** 1


	Playback speed when pressing SPACE.



.. function:: gui_recording_format(gui_recording_format_t)

	**default value:** avi

	**possible values:**
		- `avi`: AVI / video format (codec FFV1 is used in unix systems)
		- `jpg`: individual images in JPEG format
		- `png`: individual images in PNG format

	Sets the format for recording mode (when R is pressed in the GUI). Supported formats are 'avi', 'jpg' and 'png'. JPEGs have 75%% compression, AVI is using MJPEG compression.




.. function:: gui_run(bool)

	**default value:** false


	When set to true, the GUI starts playing back the video and stops once it reaches the end, or is set to false.



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



.. function:: gui_show_fish(pair<int64,int>)

	**default value:** [-1,-1]


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

	**default value:** true


	Show/hide individuals that have not been seen for longer than ``track_max_reassign_time``.

	.. seealso:: :func:`track_max_reassign_time`, 


.. function:: gui_show_match_modes(bool)

	**default value:** false


	Shows the match mode used for every tracked object. Green is 'approximate', yellow is 'hungarian', and red is 'created/loaded'.



.. function:: gui_show_memory_stats(bool)

	**default value:** false


	Showing or hiding memory statistics.



.. function:: gui_show_midline(bool)

	**default value:** true


	Showing or hiding individual midlines in tracking view.



.. function:: gui_show_midline_histogram(bool)

	**default value:** false


	Displays a histogram for midline lengths.



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

	**default value:** true


	Show/hide the posture window on the top-right.



.. function:: gui_show_probabilities(bool)

	**default value:** false


	Show/hide probability visualisation when an individual is selected.



.. function:: gui_show_recognition_bounds(bool)

	**default value:** true


	Shows what is contained within tht recognition boundary as a cyan background. (See ``recognition_border`` for details.)

	.. seealso:: :func:`recognition_border`, 


.. function:: gui_show_recognition_summary(bool)

	**default value:** false


	Show/hide confusion matrix (if network is loaded).



.. function:: gui_show_selections(bool)

	**default value:** true


	Show/hide circles around selected individual.



.. function:: gui_show_shadows(bool)

	**default value:** true


	Showing or hiding individual shadows in tracking view.



.. function:: gui_show_texts(bool)

	**default value:** true


	Showing or hiding individual identity (and related) texts in tracking view.



.. function:: gui_show_uniqueness(bool)

	**default value:** false


	Show/hide uniqueness overview after training.



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


	Determines the Alpha value for the timeline / consecutive segments display.



.. function:: gui_transparent_background(bool)

	**default value:** false


	If enabled, fonts might look weird but you can record movies (and images) with transparent background (if gui_background_color.alpha is < 255).



.. function:: gui_zoom_limit(size)

	**default value:** [300,300]


	



.. function:: heatmap_dynamic(bool)

	**default value:** false


	If enabled the heatmap will only show frames before the frame currently displayed in the graphical user interface.



.. function:: heatmap_frames(uint)

	**default value:** 0


	If ``heatmap_dynamic`` is enabled, this variable determines the range of frames that are considered. If set to 0, all frames up to the current frame are considered. Otherwise, this number determines the number of frames previous to the current frame that are considered.

	.. seealso:: :func:`heatmap_dynamic`, 


.. function:: heatmap_ids(array<uint>)

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

	.. seealso:: :func:`gui_show_heatmap`, 


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



.. function:: httpd_accepted_ip(string)

	**default value:** ""


	Set this to an IP address that you want to accept exclusively.



.. function:: httpd_port(int)

	**default value:** 8080


	This is where the webserver tries to establish a socket. If it fails, this will be set to the port that was chosen.



.. function:: huge_timestamp_ends_segment(bool)

	**default value:** true


	



.. function:: huge_timestamp_seconds(double)

	**default value:** 0.2


	Defaults to 0.5s (500ms), can be set to any value that should be recognized as being huge.



.. function:: image_invert(bool)

	**default value:** false


	Inverts the image greyscale values before thresholding.



.. function:: individual_names(map<uint,string>)

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



.. function:: manual_identities(set<Idx_t>)

	**default value:** []


	



.. function:: manual_matches(map<int,map<Idx_t,int64>>)

	**default value:** {}


	A map of manually defined matches (also updated by GUI menu for assigning manual identities). ``{{frame: {fish0: blob2, fish1: blob0}}, ...}``



.. function:: manual_splits(map<int,set<int64>>)

	**default value:** {}


	This map contains ``{frame: [blobid1,blobid2,...]}`` where frame and blobid are integers. When this is read during tracking for a frame, the tracker will attempt to force-split the given blob ids.



.. function:: manually_approved(map<int,int>)

	**default value:** {}


	A list of ranges of manually approved frames that may be used for generating training datasets, e.g. ``{232:233,5555:5560}`` where each of the numbers is a frame number. Meaning that frames 232-233 and 5555-5560 are manually set to be manually checked for any identity switches, and individual identities can be assumed to be consistent throughout these frames.



.. function:: match_mode(matching_mode_t)

	**default value:** automatic

	**possible values:**
		- `tree`: Maximizes the probability sum by assigning (or potentially not assigning) individuals to objects in the frame. This returns the correct solution, but might take long for high quantities of individuals.
		- `approximate`: Simply assigns the highest probability edges (blob to individual) to all individuals - first come, first serve. Parameters have to be set very strictly (especially speed) in order to have as few objects to choose from as possible and limit the error.
		- `hungarian`: The hungarian algorithm (as implemented in O(n^3) by Mattias Andrée `https://github.com/maandree/hungarian-algorithm-n3`).
		- `benchmark`: Runs all algorithms and pits them against each other, outputting statistics every few frames.
		- `automatic`: Uses automatic selection based on density.

	Changes the default algorithm to be used for matching blobs in one frame with blobs in the next frame. The accurate algorithm performs best, but also scales less well for more individuals than the approximate one. However, if it is too slow (temporarily) in a few frames, the program falls back to using the approximate one that doesnt slow down.




.. function:: matching_probability_threshold(float)

	**default value:** 0.1


	The probability below which a possible connection between blob and identity is considered too low. The probability depends largely upon settings like ``track_max_speed``.

	.. seealso:: :func:`track_max_speed`, 


.. function:: meta_mass_mg(float)

	**default value:** 200


	Used for exporting event-energy levels.



.. function:: meta_real_width(float)

	**default value:** 0


	Used to calculate the ``cm_per_pixel`` conversion factor, relevant for e.g. converting the speed of individuals from px/s to cm/s (to compare to ``track_max_speed`` which is given in cm/s). By default set to 30 if no other values are available (e.g. via command-line). This variable should reflect actual width (in cm) of what is seen in the video image. For example, if the video shows a tank that is 50cm in X-direction and 30cm in Y-direction, and the image is cropped exactly to the size of the tank, then this variable should be set to 50.

	.. seealso:: :func:`cm_per_pixel`, :func:`track_max_speed`, 


.. function:: meta_source_path(path)

	**default value:** ""


	Path of the original video file for conversions (saved as debug info).



.. function:: midline_invert(bool)

	**default value:** false


	If enabled, all midlines will be inverted (tail/head swapped).



.. function:: midline_resolution(uint)

	**default value:** 25


	Number of midline points that are saved. Higher number increases detail.



.. function:: midline_samples(uint64)

	**default value:** 0


	The maximum number of samples taken for generating a ``median midline length``. Setting this to 0 removes the limit all together. A limit may be set for very long videos, or videos with lots of individuals, for memory reasons.



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



.. function:: outline_curvature_range_ratio(float)

	**default value:** 0.03


	Determines the ratio between number of outline points and distance used to calculate its curvature. Program will look at index +- ``ratio * size()`` and calculate the distance between these points (see posture window red/green color).



.. function:: outline_resample(float)

	**default value:** 0.5


	Spacing between outline points in pixels, after resampling (normalizing) the outline. A lower value here can drastically increase the number of outline points generated (and decrease speed).



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



.. function:: output_centered(bool)

	**default value:** false


	If set to true, the origin of all X and Y coordinates is going to be set to the center of the video.



.. function:: output_csv_decimals(uchar)

	**default value:** 0


	Maximum number of decimal places that is written into CSV files (a text-based format for storing data). A value of 0 results in integer values.



.. function:: output_default_options(map<string,array<string>>)

	**default value:** {"v_direction":["/10"],"NEIGHBOR_DISTANCE":["/10"],"X":["/100"],"outline_size":["/100"],"ANGULAR_A":["/1000","SMOOTH","CENTROID"],"threshold_reached":["POINTS"],"DOT_V":["/10"],"L_V":["/10"],"ANGULAR_V":["/10","SMOOTH","CENTROID"],"event_acceleration":["/10"],"midline_length":["/15"],"SPEED":["/10","SMOOTH"],"ACCELERATION":["/15","SMOOTH","CENTROID"],"NEIGHBOR_VECTOR_T":["/1"],"Y":["/100"],"tailbeat_threshold":["pm"],"tailbeat_peak":["pm"],"amplitude":["/100"],"global":["/10"]}


	Default scaling and smoothing options for output functions, which are applied to functions in ``output_graphs`` during export.

	.. seealso:: :func:`output_graphs`, 


.. function:: output_dir(path)

	**default value:** "C:\\Users\\tristan\\Videos"


	Default output-/input-directory. Change this in order to omit paths in front of filenames for open and save.



.. function:: output_format(output_format_t)

	**default value:** npz

	**possible values:**
		- `csv`: A standard data format, comma-separated columns for each data stream. Use `output_csv_decimals` to adjust the maximum precision for exported data.
		- `npz`: NPZ is basically a collection of binary arrays, readable by NumPy and other plugins (there are plugins available for Matlab and R).

	When pressing the S(ave) button or using ``auto_quit``, this setting allows to switch between CSV and NPZ output. NPZ files are recommended and will be used by default - some functionality (such as visual fields, posture data, etc.) will remain in NPZ format due to technical constraints.


	.. seealso:: :func:`auto_quit`, 


.. function:: output_frame_window(int)

	**default value:** 100


	If an individual is selected during CSV output, use these number of frames around it (or -1 for all frames).



.. function:: output_graphs(array<pair<string,array<string>>>)

	**default value:** [["X",["RAW","WCENTROID"]],["Y",["RAW","WCENTROID"]],["X",["RAW","HEAD"]],["Y",["RAW","HEAD"]],["VX",["RAW","HEAD"]],["VY",["RAW","HEAD"]],["AX",["RAW","HEAD"]],["AY",["RAW","HEAD"]],["ANGLE",["RAW"]],["ANGULAR_V",["RAW"]],["ANGULAR_A",["RAW"]],["MIDLINE_OFFSET",["RAW"]],["normalized_midline",["RAW"]],["midline_length",["RAW"]],["midline_x",["RAW"]],["midline_y",["RAW"]],["segment_length",["RAW"]],["SPEED",["RAW","WCENTROID"]],["SPEED",["SMOOTH","WCENTROID"]],["SPEED",["RAW","PCENTROID"]],["SPEED",["RAW","HEAD"]],["BORDER_DISTANCE",["PCENTROID"]],["time",[]],["timestamp",[]],["frame",[]],["missing",[]],["num_pixels",[]],["ACCELERATION",["RAW","PCENTROID"]],["ACCELERATION",["RAW","WCENTROID"]]]


	The functions that will be exported when saving to CSV, or shown in the graph. ``[['X',[option], ...]]``



.. function:: output_heatmaps(bool)

	**default value:** false


	When set to true, heatmaps are going to be saved to a separate file, or set of files '_p*' - with all the settings in heatmap_* applied.



.. function:: output_image_per_tracklet(bool)

	**default value:** false


	If set to true, the program will output one median image per tracklet (time-series segment) and save it alongside the npz/csv files.



.. function:: output_interpolate_positions(bool)

	**default value:** false


	If turned on this function will linearly interpolate X/Y, and SPEED values, for all frames in which an individual is missing.



.. function:: output_invalid_value(output_invalid_t)

	**default value:** inf

	**possible values:**
		- `inf`: Infinity (e.g. np.inf)
		- `nan`: NaN (e.g. np.nan)

	Determines, what is exported in cases where the individual was not found (or a certain value could not be calculated). For example, if an individual is found but posture could not successfully be generated, then all posture-based values (e.g. ``midline_length``) default to the value specified here. By default (and for historic reasons), any invalid value is marked by 'inf'.




.. function:: output_min_frames(uint64)

	**default value:** 1


	Filters all individual with less than N frames when exporting. Individuals with fewer than N frames will also be hidden in the GUI unless ``gui_show_inactive_individuals`` is enabled (default).

	.. seealso:: :func:`gui_show_inactive_individuals`, 


.. function:: output_normalize_midline_data(bool)

	**default value:** false


	If enabled: save a normalized version of the midline data saved whenever ``output_posture_data`` is set to true. Normalized means that the position of the midline points is normalized across frames (or the distance between head and point n in the midline array).

	.. seealso:: :func:`output_posture_data`, 


.. function:: output_posture_data(bool)

	**default value:** false


	Save posture data npz file along with the usual NPZ/CSV files containing positions and such. If set to true, a file called '``output_dir``/``fish_data_dir``/``<filename>_posture_fishXXX.npz``' will be created for each individual XXX.

	.. seealso:: :func:`output_dir`, 


.. function:: output_prefix(string)

	**default value:** ""


	A prefix that is prepended to all output files (csv/npz).



.. function:: output_recognition_data(bool)

	**default value:** false


	Save recognition / probability data npz file along with the usual NPZ/CSV files containing positions and such. If set to true, a file called '``output_dir``/``fish_data_dir``/``<filename>_recognition_fishXXX.npz``' will be created for each individual XXX.

	.. seealso:: :func:`output_dir`, 


.. function:: output_statistics(bool)

	**default value:** false


	Save an NPZ file containing an array with shape Nx16 and contents [``adding_seconds``, ``combined_posture_seconds``, ``number_fish``, ``loading_seconds``, ``posture_seconds``, ``match_number_fish``, ``match_number_blob``, ``match_number_edges``, ``match_stack_objects``, ``match_max_edges_per_blob``, ``match_max_edges_per_fish``, ``match_mean_edges_per_blob``, ``match_mean_edges_per_fish``, ``match_improvements_made``, ``match_leafs_visited``, ``method_used``] and an 1D-array containing all frame numbers. If set to true, a file called '``output_dir``/``fish_data_dir``/``<filename>_statistics.npz``' will be created. This will not output anything interesting, if the data was loaded instead of analysed.

	.. seealso:: :func:`output_dir`, 


.. function:: panic_button(int)

	**default value:** 0


	42



.. function:: peak_mode(peak_mode_t)

	**default value:** pointy

	**possible values:**
		- `pointy`: The head is broader than the tail.
		- `broad`: The tail is broader than the head.

	This determines whether the tail of an individual should be expected to be pointy or broad.




.. function:: pixel_grid_cells(uint64)

	**default value:** 25


	



.. function:: posture_closing_size(uchar)

	**default value:** 2


	The kernel size for erosion / dilation of the posture algorithm. Only has an effect with  ``posture_closing_steps`` > 0.

	.. seealso:: :func:`posture_closing_steps`, 


.. function:: posture_closing_steps(uchar)

	**default value:** 0


	When enabled (> 0), posture will be processed using a combination of erode / dilate in order to close holes in the shape and get rid of extremities. An increased number of steps will shrink the shape, but will also be more time intensive.



.. function:: posture_direction_smoothing(uint64)

	**default value:** 0


	Enables or disables smoothing of the posture orientation based on previous frames (not good for fast turns).



.. function:: posture_head_percentage(float)

	**default value:** 0.1


	The percentage of the midline-length that the head is moved away from the front of the body.



.. function:: postures_per_thread(float)

	**default value:** 1


	Number of individuals for which postures will be estimated per thread.



.. function:: python_path(path)

	**default value:** "C:\\Users\\tristan\\anaconda3\\envs\\trex\\python.EXE"


	Path to the python home folder (containing pythonXX.exe). If left empty, the user is required to make sure that all necessary libraries are in-scope the PATH environment variable.



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


	.. seealso:: :func:`recognition_shapes`, 


.. function:: recognition_border_shrink_percent(float)

	**default value:** 0.3


	The amount by which the recognition border is shrunk after generating it (roughly and depends on the method).



.. function:: recognition_border_size_rescale(float)

	**default value:** 0.5


	The amount that blob sizes for calculating the heatmap are allowed to go below or above values specified in ``blob_size_ranges`` (e.g. 0.5 means that the sizes can range between ``blob_size_ranges.min * (1 - 0.5)`` and ``blob_size_ranges.max * (1 + 0.5)``).

	.. seealso:: :func:`blob_size_ranges`, 


.. function:: recognition_coeff(uint64)

	**default value:** 50


	



.. function:: recognition_enable(bool)

	**default value:** true


	This enables internal training. Requires Python3 and Keras to be available.



.. function:: recognition_image_scale(float)

	**default value:** 1


	Scaling applied to the images before passing them to the network.



.. function:: recognition_image_size(size)

	**default value:** [80,80]


	Size of each image generated for network training.



.. function:: recognition_normalization(recognition_normalization_t)

	**default value:** posture

	**possible values:**
		- `none`: No normalization. Images will only be cropped out and used as-is.
		- `moments`: Images will be cropped out and aligned as in idtracker.ai using the main axis calculated using `image moments`.
		- `posture`: Images will be cropped out and rotated so that the head will be fixed in one position and only the tail moves.
		- `legacy`: Images will be aligned parallel to the x axis.

	This enables or disable normalizing the images before training. If set to ``none``, the images will be sent to the GPU raw - they will only be cropped out. Otherwise they will be normalized based on head orientation (posture) or the main axis calculated using ``image moments``.




.. function:: recognition_save_progress_images(bool)

	**default value:** false


	If set to true, an image will be saved for all training epochs, documenting the uniqueness in each step.



.. function:: recognition_save_training_images(bool)

	**default value:** false


	If set to true, the program will save the images used for a successful training of the recognition network to the output path.



.. function:: recognition_segment_add_factor(float)

	**default value:** 1.5


	This factor will be multiplied with the probability that would be pure chance, during the decision whether a segment is to be added or not. The default value of 1.5 suggests that the minimum probability for each identity has to be 1.5 times chance (e.g. 0.5 in the case of two individuals).



.. function:: recognition_shapes(array<array<vec>>)

	**default value:** []


	If ``recognition_border`` is set to 'shapes', then the identification network will only be applied to blobs within the convex shapes specified here.

	.. seealso:: :func:`recognition_border`, 


.. function:: recognition_smooth_amount(uint64)

	**default value:** 200


	



.. function:: settings_file(path)

	**default value:** ""


	Name of the settings file. By default, this will be set to ``filename``.settings in the same folder as ``filename``.

	.. seealso:: :func:`filename`, :func:`filename`, 


.. function:: smooth_window(uint)

	**default value:** 2


	Smoothing window used for exported data with the #smooth tag.



.. function:: speed_extrapolation(float)

	**default value:** 3


	Used for matching when estimating the next position of an individual. Smaller values are appropriate for lower frame rates. The higher this value is, the more previous frames will have significant weight in estimating the next position (with an exponential decay).



.. function:: tags_path(path)

	**default value:** ""


	If this path is set, the program will try to find tags and save them at the specified location.



.. function:: terminate(bool)

	**default value:** false


	If set to true, the application terminates.



.. function:: terminate_training(bool)

	**default value:** false


	Setting this to true aborts the training in progress.



.. function:: threshold_ratio_range(rangef)

	**default value:** [0.5,1]


	If ``track_threshold_2`` is not equal to zero, this ratio will be multiplied by the number of pixels present before the second threshold. If the resulting size falls within the given range, the blob is deemed okay.

	.. seealso:: :func:`track_threshold_2`, 


.. function:: track_consistent_categories(bool)

	**default value:** false


	Utilise categories (if present) when tracking. This may break trajectories in places with imperfect categorization, but only applies once categories have been applied.



.. function:: track_do_history_split(bool)

	**default value:** true


	If disabled, blobs will not be split automatically in order to separate overlapping individuals. This usually happens based on their history.



.. function:: track_end_segment_for_speed(bool)

	**default value:** true


	Sometimes individuals might be assigned to blobs that are far away from the previous position. This could indicate wrong assignments, but not necessarily. If this variable is set to true, consecutive frame segments will end whenever high speeds are reached, just to be on the safe side. For scenarios with lots of individuals (and no recognition) this might spam yellow bars in the timeline and may be disabled.



.. function:: track_ignore(array<array<vec>>)

	**default value:** []


	If this is not empty, objects within the given rectangles or polygons (>= 3 points) ``[[x0,y0],[x1,y1](, ...)], ...]`` will be ignored during tracking.



.. function:: track_include(array<array<vec>>)

	**default value:** []


	If this is not empty, objects within the given rectangles or polygons (>= 3 points) ``[[x0,y0],[x1,y1](, ...)], ...]`` will be the only objects being tracked. (overwrites ``track_ignore``)

	.. seealso:: :func:`track_ignore`, 


.. function:: track_intensity_range(rangel)

	**default value:** [-1,-1]


	When set to valid values, objects will be filtered to have an average pixel intensity within the given range.



.. function:: track_max_individuals(uint)

	**default value:** 0


	The maximal number of individual that are assigned at the same time (infinite if set to zero). If the given number is below the actual number of individual, then only a (random) subset of individual are assigned and a warning is shown.



.. function:: track_max_reassign_time(float)

	**default value:** 0.5


	Distance in time (seconds) where the matcher will stop trying to reassign an individual based on previous position. After this time runs out, depending on the settings, the tracker will try to find it based on other criteria, or generate a new individual.



.. function:: track_max_speed(float)

	**default value:** 10


	The maximum speed an individual can have (=> the maximum distance an individual can travel within one second) in cm/s. Uses and is influenced by ``meta_real_width`` and ``cm_per_pixel`` as follows: ``speed(px/s) * cm_per_pixel(cm/px) -> cm/s``.

	.. seealso:: :func:`meta_real_width`, :func:`cm_per_pixel`, 


.. function:: track_only_categories(array<string>)

	**default value:** []


	If this is a non-empty list, only objects that have previously been assigned one of the correct categories will be tracked. Note that this also excludes noise particles or very short segments with no tracking.



.. function:: track_posture_threshold(int)

	**default value:** 15


	Same as ``track_threshold``, but for posture estimation.

	.. seealso:: :func:`track_threshold`, 


.. function:: track_segment_max_length(float)

	**default value:** 0


	If set to something bigger than zero, this represents the maximum number of seconds that a consecutive segment can be.



.. function:: track_speed_decay(float)

	**default value:** 0.7


	The amount the expected speed is reduced over time when an individual is lost. When individuals collide, depending on the expected behavior for the given species, one should choose different values for this variable. If the individuals usually stop when they collide, this should be set to 1. If the individuals are expected to move over one another, the value should be set to ``0.7 > value > 0``.



.. function:: track_threshold(int)

	**default value:** 15


	Constant used in background subtraction. Pixels with grey values above this threshold will be interpreted as potential individuals, while pixels below this threshold will be ignored.



.. function:: track_threshold_2(int)

	**default value:** 0


	If not zero, a second threshold will be applied to all objects after they have been deemed do be theoretically large enough. Then they are compared to #before_pixels * ``threshold_ratio_range`` to see how much they have been shrunk).

	.. seealso:: :func:`threshold_ratio_range`, 


.. function:: track_time_probability_enabled(bool)

	**default value:** true


	



.. function:: track_trusted_probability(float)

	**default value:** 0.5


	If the probability, that is used to assign an individual to an object, is smaller than this value, the current segment will be ended (thus this will also not be a consecutive segment anymore for this individual).



.. function:: tracklet_max_images(uint64)

	**default value:** 0


	Maximum number of images that are being output per tracklet given that ``output_image_per_tracklet`` is true. If the number is 0, then every image will be exported that has been recognized as an individual.

	.. seealso:: :func:`output_image_per_tracklet`, 


.. function:: tracklet_normalize_orientation(bool)

	**default value:** true


	If enabled, all exported tracklet images are normalized according to the calculated posture orientation, so that all heads are looking to the left and only the body moves.



.. function:: tracklet_restore_split_blobs(bool)

	**default value:** true


	If enabled, all exported tracklet images are checked for missing pixels. When a blob is too close to another blob, parts of the other blob might be erased so the individuals can be told apart. If enabled, another mask will be saved, that contains only the blob in focus, without the rest-pixels.



.. function:: use_differences(bool)

	**default value:** false


	This should be set to false unless when using really old files.



.. function:: version(string)

	**default value:** "v1.1.6-18-gf49b7f7"


	Current application version.



.. function:: video_info(string)

	**default value:** ""


	Information on the current video as provided by PV.



.. function:: video_length(uint64)

	**default value:** 0


	The length of the video in frames



.. function:: video_size(size)

	**default value:** [-1,-1]


	The dimensions of the currently loaded video.



.. function:: visual_field_eye_offset(float)

	**default value:** 0.15


	A percentage telling the program how much the eye positions are offset from the start of the midline.



.. function:: visual_field_eye_separation(float)

	**default value:** 60


	Degrees of separation between the eye and looking straight ahead. Results in the eye looking towards head.angle +- .



.. function:: web_quality(int)

	**default value:** 75


	JPEG quality of images transferred over the web interface.



.. function:: web_time_threshold(float)

	**default value:** 0.05


	Maximum refresh rate in seconds for the web interface.



