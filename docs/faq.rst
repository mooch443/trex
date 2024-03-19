.. include:: names.rst

.. toctree::
   :maxdepth: 2

Frequently Asked Questions
--------------------------


I am using Windows and Python cannot be initialized successfully!
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	If you compiled or installed |trex| in a conda environment and it is having trouble finding the necessary files, you can try to help it by editing the default.settings file inside the application folder (be careful if you're compiling the program on your own, the one in the build folder gets replaced by the one in the folder ``[root]/Application/default.settings`` after every build that altered anything). Just put a ``python_path = "C:\Users\[USERNAME]\Anaconda3\envs\[ENVNAME]"`` inside it and try starting again! Otherwise, you can be more savage and set the ``PYTHONHOME`` variable to the same folder and add it to ``PATH``, too (this breaks other anaconda stuff most likely).

Segmentation/objects barely visible or too large!
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	Is the background image good enough (if you're using background subtraction), e.g. are there artifacts from individuals visible in the background, or does the background change during the video?
		If the background is dynamic, you might have to disable subtraction by setting :func:`enable_difference` to false, and adjusting :func:`threshold` to a cut-off greyscale value. Otherwise, consider using a different :func:`averaging_method`. For example, max/mode are good for white backgrounds and short videos, or barely moving individuals.
	Is there a lot of magenta-colored noise outside of objects?
		Increase :func:`threshold` during recording/conversion.
	Objects are too small?
		Decrease :func:`threshold` during recording/conversion.
		
Trajectories jump around a lot for no particular reason!
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	Changing :func:`track_max_speed` might help to mitigate this problem. Generally this can be the symptom of many different problems that lead to individuals being lost: size, speed, visibility issues, etc.: Sometimes individuals are lost because they are moving too fast (faster than the maximally allowed speed), or because they are expected to move much faster. Try lowering or increasing that limit. To get a hint at which speed to set, open up |trex|, track a few frames and select an individual - if there are consecutive frames for that individual, it will display a cm/s speed in the top-left overlay.

I set :func:`track_max_individuals` to zero, but it still does not track all individuals!
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	Probably what's happening is that you have not created a ``.settings`` file for the given video yet (or not in the right folder). Try that, and then attach the command-line option again.

|trex| is really **laggy** and frequently complains about **too many combinations**!
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	Pause the analysis (``,`` key, this may take a few seconds). The matching algorithm (can be changed in :func:`match_mode`) has diffculty separating individuals into distinct cliques of neighboring individuals, or there are simply too many of them - which only happens if you changed the :func:`match_mode` from `automatic` to `tree`. This could be because your video contains too many trackable objects and no limit on the number of individuals has been set (:func:`track_max_individuals`), or there are significant time-jumps in the video. If the number of individuals should be much lower than detected, check your :func:`track_threshold`/:func:`track_size_filter` settings. If that does not fix anything, check your settings for :func:`track_max_speed`, which controls the size of the neighbourhood to be considered during matching, and reduce it until there are no further warnings. Alternatively, change the :func:`match_mode` to automatic, which should get rid of these problems.
	
|grabs| does not quit and only shows "[...] not properly closed [...]"!
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	You may have to forcibly quit the application, either using a task manager, or by finding and manually ending its process in a Unix terminal:

	.. code-block:: bash

		ps aux | grep tgrabs    # on unix systems
		kill <pid>              # insert the correct PID from the previous call here
		
		killall tgrabs          # alternative (kills all instances)

I have attached my Basler camera, and now |grabs| is stuck initializing the camera!
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	Most likely the camera driver crashed. Try restarting your computer to fix it.

I am getting a `GLIBCXX_3.X.XX` (e.g. `GLIBCXX_3.4.30`) error on Linux.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	There's a good chance you're using a Radeon GPU + driver. Those seem to require specific glibc versions - you can try to install a different libstdc++-ng version from conda-forge to work around this::

		conda install -c conda-forge libstdcxx-ng

	If this doesn't help, feel free to open a `new issue <https://github.com/mooch443/trex/issues/new?assignees=mooch443&labels=bug&template=bug_report.md&title=>`_ on GitHub!
