.. include:: names.rst

.. toctree::
   :maxdepth: 2

Batch processing support
========================

|trex| and |grabs| both offer full batch processing support. All parameters that can be setup via the settings box (and even some that are read-only when the program is already started), can be appended to the command-line -- as mentioned above. For batch processing, special parameters are available::

	auto_quit			  # automatically saves all requested data to the output folder and quits the app
	auto_train			 # automatically attempts to train the visual identification if successfully tracked
	auto_apply			 # automatically attempts to load weights from a previous training and auto correct the video
	auto_no_results		# do not save a .results file
	auto_no_tracking_data  # do not save the data/file_fishX.npz files
	auto_no_memory_stats   # (enabled by default) do not save memory statistics