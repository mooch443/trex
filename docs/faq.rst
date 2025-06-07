.. include:: names.rst

.. toctree::
   :maxdepth: 2

Frequently Asked Questions
--------------------------


I am using Windows and Python cannot be initialized successfully!
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	There is a high likelihood that something went wrong during installation. Make sure you did not first create an environment and then installed |trex| into this environment - also make sure you did not manually add ``pip`` packages that overwrote, for example, ``numpy`` with a different version (it should be ``1.26.4``).
		
Trajectories jump around a lot for no particular reason!
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	Changing :param:`track_max_speed` might help mitigate this problem. Generally this can be the symptom of many different problems that lead to individuals being lost (e.g., size, speed, visibility issues). Sometimes individuals are lost because they are moving too fast (faster than :param:`track_max_speed`), or, oppositely, because they are expected to move much faster. Try lowering or increasing that limit. To get a hint at which speed to set, open up |trex|, track a few frames and select an individual - if there are consecutive frames for that individual, it will display a cm/s speed in the top-left overlay. Also note that :param:`track_max_speed` depends on your :param:`cm_per_pixel` setting, so if you change that you also need to adjust your filter sizes and maximum speeds.

Tracking is really slow!
~~~~~~~~~~~~~~~~~~~~~~~~
	Are you tracking too many individuals at the same time? There are many factors that play into how difficult the tracking becomes. This could be because :param:`track_max_individuals` is set to ``0`` and there are many individuals (or not properly filtered noise particles) to track. You can try limiting the maximum number of individuals, or improve your noise filtering. If the individuals are tracked and filtered properly, and you're tracking a lot of them, they might be a bit too close for the parameters you have set. Specifically :param:`track_max_speed` - try setting this as tightly as possible to remove unnecessary matching combinations that the tracker has to check.
	
|trex| does not quit and seems to be stuck!
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	You may have to forcibly quit the application, either using a task manager (on Windows), or by finding and manually ending its process in a Unix terminal:

	.. code-block:: bash

		ps aux | grep trex      # this is case-sensitive, so also try TRex
		kill <pid>              # insert the correct PID from the previous call here

	If you're running into this situation often, please collect some information about your process and how you got there and feel free to open a `new issue <https://github.com/mooch443/trex/issues/new?assignees=mooch443&labels=bug&template=bug_report.md&title=>`_ on GitHub!

I am getting a `GLIBCXX_3.X.XX` (e.g. `GLIBCXX_3.4.30`) error on Linux.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	There's a good chance you're using a Radeon GPU + driver. Those seem to require specific glibc versions - you can try to install a different libstdc++-ng version from conda-forge to work around this::

		conda install -c conda-forge libstdcxx-ng

	If this doesn't help, feel free to open a `new issue <https://github.com/mooch443/trex/issues/new?assignees=mooch443&labels=bug&template=bug_report.md&title=>`_ on GitHub!
