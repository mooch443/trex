.. include:: names.rst

.. toctree::
   :maxdepth: 3
   :numbered:

Tutorials
=========

|trex| is a versatile tracking software that can help you solve almost all tracking problems with minimal manual effort, computational power, and time. It's designed to be user-friendly for general use but also provides advanced features for more complex situations. This section provides tutorials on how to use |trex|, from setting up the software to analyzing videos and exporting data. 

You can follow along by reading the text or watching the video tutorials on the `YouTube channel <https://www.youtube.com/@TRex-f9g>`_.

Understanding Computational Complexity
--------------------------------------

If you are just planning to design your experiment or are new to tracking, it's essential to understand the technical implications of the specific data you're looking for. This can help you set realistic expectations and design your experiment accordingly.

Here are a few key factors we usually think about first:

1. **Number of Individuals**
   
   .. epigraph::

      *More individuals multiply tracking complexity.*

   Tracking more individuals in your video increases difficulty in many ways and the computational resources required, potentially also limiting your analysis options later [#f1]_. For example: Automatic visual identification is only feasible with smaller groups (typically fewer than 50 individuals) since it relies on relative visual differences in a *known* group. If it is not possible to automatically maintain perfect identities in too large a group, you may need to limit your analysis to more general information about the group's behavior rather than real identities. |trex| also subsections trajectory pieces into *tracklets*, which are shorter sequences per tracked object where identities are maintained with high confidence (see below). These can be used for more short-term analyses, even if the full video is too complex to maintain identities throughout.

2. **Scene Complexity**
   
   .. epigraph::

      *Eliminate visual clutter — use uniform, high-contrast backgrounds if possible.*

   Complex scenes make it difficult to see individuals. This does not just apply to human eyes, but also to computer vision. Various factors such as very small objects, heterochrome objects, occlusions, shadows, reflections, and other interfering objects can hinder detection (and thus tracking) performance. Varying or visually complex backgrounds with challenging contrast may require more advanced segmentation algorithms, often based on machine learning [#f2]_. These are more generic and often *better*, generally speaking, but require more manual labor to set up and more computational resources to apply. 

3. **Camera and Lighting Choice**

   .. epigraph::

      *Pay attention to your camera settings and their influence on visual quality. Do not store more data than you need (e.g., zoom in).*

   The camera and recording settings you use affect image and computational complexity. Low-resolution or low frame rate cameras may make it difficult to identify individuals visually — especially if you plan to use the visual identification algorithm. A slow shutter speed can introduce motion blur, complicating tracking efforts. Lighting conditions also play a significant role; when possible, prefer DC lights over AC lights to avoid flickering. Shiny fur or scales can also cause reflections that interfere with visual identification algorithms. High frame rates can prolong analyses since there are simply more images to process. Do not record more data than you need - zoom in if possible, and record only the area of interest. Point the camera straight down to avoid distortion and ensure that the camera is stable to avoid shaking.

4. **Research Question**

   .. epigraph::

      *Define what kind of data you need, and what it implies for your setup, before you start filming.*
   
   Aim to use the simplest approach that effectively answers your research question. If you're interested in the general behavior of a group, maintaining perfect identities may not be necessary. |trex| provides information on shorter sequences per tracked object (the aforementioned *tracklets*), which, even without identity corrections, often suffice. For instance, if you're interested in the average speed of a group or average speed of individuals in a certain area, then short *tracklets* are enough. However, if you are distinguishing between specific individuals (invisibly *parasitized*/*unparasitized* or *informed*/*uninformed*), you might need to use the visual identification algorithm. This, if successful, gives you a stronger guarantee for maintained identities, but requires you to invest more time and computational power.
   
   Your research question also influences the *type* of data output you need - most of the time the centroid position is enough (simply called *detection*), but there are other types: *Pose*/*Keypoint* data is only necessary if you're focusing on specific body parts (see :numref:`data_types`) in any way, and *outline* data is necessary if you're interested in the directionality of the individuals (e.g., front/back) or other aspects of their 2D shape. See also [#f2]_ and :numref:`data_types` for more information.

The following sections will guide you through increasingly complex problems and how to solve them with |trex|. Identify which category your research question falls into to find the right tools.

.. [#f1] The number of possible combinations of individuals at any given time increases exponentially with the number of individuals — a phenomenon known as the "curse of dimensionality" in computer vision and machine learning. With a large number of individuals, you may need to wait a bit longer, and potentially have more visual overlaps between individuals.

.. [#f2] Segmentation algorithms are used to fully separate individuals from the background in a video - meaning not just *position* but also a *"picture"*. Keypoint data, for example, sometimes only retrieves multiple points - not the actual shape. For complex scenes, advanced algorithms like deep learning models (e.g., YOLO) are often employed.

.. _data_types:

.. figure:: images/data_types.png

   Different data types that can be extracted from a video: just the centroid position, pose/keypoint data, and outline-type data including directionality (front/back).

Recommended System Requirements
-------------------------------

While |trex| is designed to be lightweight, ensuring a modern CPU, sufficient RAM (8GB or more), and a dedicated GPU (optional but beneficial for advanced machine learning tasks) can optimize performance. Our test systems cover a broad spectrum of operating systems and architectures, such as Windows, macOS, and Linux, and we recommend using a system with at least the following specifications:

- **Operating System**: Windows 10, macOS 10.15, or Ubuntu 20.04 LTS
- **Processor**: Intel Core i5 or AMD Ryzen 5
- **Memory**: 8GB RAM
- **Graphics**: dedicated NVIDIA GPU with 2GB VRAM (optional but recommended), or Apple Silicon's integrated GPU

This is a general recommendation, and |trex| can run on systems with lower specifications depending on the specific task at hand. However, the performance may be slower, especially for larger videos or more complex scenes. If you encounter any issues with the software that you think should not be happening, feel free to file a bug report on our `GitHub repository <https://github.com/mooch443/trex>`_.

Installation
------------

You can download the latest version of |trex| using `Miniforge` (conda). To install |trex|, you need to have `Miniforge` installed on your system.

If you're not familiar with `conda` or `Miniforge`, you can find more information on how to install them `here <https://conda-forge.org/miniforge/>`_. Miniforge is a minimal installer for `conda`, an open-source package manager that helps you create virtual environments and install software and libraries without installing them globally on your system [#f3]_. We do not support Anaconda's default channels, so please use `Miniforge` instead (in Anaconda you can also restrict your channels to `conda-forge` only [#f4]_).

Open your `Miniforge` Prompt and run:

.. code-block:: bash

   conda create -n track -c trex-beta trex

.. NOTE::

   This only works if `conda-forge` is the *only* channel you have added to your `conda` configuration. By default, this is the case if you're using `Miniforge`. If you have other channels added, or are using Anaconda, you can run the following command instead:

   .. code-block:: bash

      conda create -n track --override-channels -c trex-beta -c conda-forge trex

   If any other channels are used, the installation might not work as expected and may throw `package not found` errors.

This will create a new conda environment called ``track`` with |trex| installed. This could take a while, especially during conda's 'verifying transaction' phase. Activate the environment using:

.. code-block:: bash

   conda activate track

Then start |trex| by typing:

.. code-block:: bash

   trex

and pressing **Enter**. 

If a window showing a friendly T-Rex appears, you've successfully installed the software and can proceed to the next section.

.. _welcome_screen:

.. figure:: images/welcome_screen-1.mov
   :width: 100%

   The TRex graphical user interface (GUI) showing the welcome screen.

If you have any issues with the installation, please refer to the (more detailed) :doc:`installation guide <install>`.

.. [#f3] The advantage of this is that you can have different versions of the same software installed on your system without conflicts, and that they can be easily removed.

.. [#f4] We do not support Anaconda's default channels because forge has easier license agreements and is often more up-to-date. Anaconda's hosted channels can be problematic for you too, if your institution does not have a license agreement with them.

Tutorial: Basics
----------------

This tutorial will guide you through the basic steps of setting up |trex|, analyzing an example video, and exporting the data.

If you want to follow along exactly as described, you can download the example video from `here <https://trex.run/8guppies_20s.mp4>`_ and place it in any folder you'd like to use as your root analysis folder.

Setting Up a New Project
^^^^^^^^^^^^^^^^^^^^^^^^

When you first start |trex|, you'll see a window with a welcome message and a button to open a file or start the webcam. Click on the folder icon in the center of the screen to proceed to the initial settings screen (see :numref:`welcome_screen`).

There are a few tabs on top of your screen now (see :numref:`initialsettings`). You will be landed on the first one:

- **Locations**: Set the input and output files and related settings.
- **Detection**: Configure settings related to detecting individuals (or objects) in the raw image frames. This is the first real step in the pipeline, and settings here cannot be effectively changed afterward.
- **Tracking**: Adjust settings related to tracking the detected individuals. This is the second step in the pipeline, and settings here can be changed at any time, although it might require re-analysis.

.. _initialsettings:

.. figure:: images/configuration_screen-1.mov
   :width: 100%

   Initial settings screen where you can choose your input and output files, as well as various detection and tracking parameters.

Lets set an input file first and then go through a few more steps to get started:

1. **Select the Input File**: Click on the 📂 folder button next to the input file name at the top, or enter the path manually. Once you selected the video, the background of the dialog will change to a slowed down and slightly blurry version of it (you can hover it to see it more clearly).

2. **Setting Output Preferences** (optional):

   By default, |trex| will place generated outputs in the same folder as the input file, but you can choose a different folder in the output section below.

   - **Output Folder**: You can choose where to save the output files by clicking on the 📂 folder button next to the output file name. By default, |trex| saves the output files in the same folder as the input file.
   - **Prefix**: The ``prefix`` (or ``output_prefix``) can be optionally set. This creates a subfolder with the given name under the output root, redirecting all new outputs there while the original ``.pv`` file stays in the root folder. This helps organize different sessions for the same video (e.g., trying out different settings or separating tracking per species).

3. **Set the detection type**:

   Tracking can generally be defined as connecting the dots between detections across the temporal dimension. The first step in this process is detecting the individuals in each frame. |trex|, at the moment, offers two different detection types:

   - **Background subtraction**: This is the default detection type and works well for most videos. It's based on the difference between the current frame and a background model. This model is built from the first few frames of the video and is updated over time. It's a simple and fast method that works well for most videos.
   - **YOLO**: This is a more advanced detection type that uses a deep learning model (e.g. YOLO architecture) to detect individuals. It's often more accurate and can handle more complex scenes, but it requires more computational power and can be slower. It's recommended for videos with complex backgrounds, occlusions, or other challenging conditions. It may, however, require you to manually prepare a detection model specifically suited to your video.

   For this tutorial, we'll use the default *background subtraction* method. By default, YOLO will be selected - please navigate to the **Detection** tab to fix that. In the same tab we can also change the :param:`threshold` value, which is the minimum greyscale intensity value (0-255) for a pixel to be considered part of an individual. The default value is `15`, but you can adjust it to better fit your video. Unlike :param:`track_threshold`, this value acts *destructively* and could actually be regarded as a lower bound for :param:`track_threshold`.
   
   We will check back on this setting later.

4. **Set the Number of Individuals**:

   - Navigate to the **Tracking** tab.
   - We counted the number of individuals when we recorded it - so we should specify it here.
   - Set the *maximum* number of individuals to 8. 
     
   It's called a *maximum* number because in some frames (e.g. during overlap or occlusion) the software might not be able to detect all individuals - or more objects are detected than there are individuals in the scene. This is a common problem in tracking and not specific to |trex|. The software will try to resolve this by merging or splitting objects, but it's not always possible to get it right. If you set the maximum number too low, you might lose individuals in the analysis. If you set it too high, you might get more overdetection. The software will try to resolve this as best as it can, but it's always a good idea to check the results visually.

   .. _settings_pane:

   .. figure:: images/settings_pane.png
      :width: 100%

      The tracking settings tab allows you to set the number of individuals to track, as well as other tracking-related settings.

5. **Start the Conversion**:

  Once you're happy with how you've configured the settings, click the **Convert** button at the bottom right to begin processing. On most computers this will be done relatively quickly, but it can take longer for larger videos or more complex scenes. After conversion is done, you'll be dropped into the default tracking view.

   .. _tracking_view:

   .. figure:: images/tracking_view.png
      :width: 100%

      The tracking view shows the video with the detected individuals highlighted. You can adjust the tracking settings here to improve the results. You may have already spotted some problems here, but don't worry - we'll fix them now.

It is fairly normal to have to adjust the settings a few times to get the best results. Anything related to tracking can be changed at any time, but detection settings are fixed once the conversion starts. So if you're not happy with the initial **Detection**, you can always cancel an ongoing conversion, go back to the opening dialog and try again. **Tracking problems**, on the other hand, can be resolved more easily later on; the only requirement is that the detection is good enough to start with - meaning no individuals are completely undetected or deformed by irreversible settings such as the :param:`threshold` mentioned earlier.

.. epigraph::

   **Tip:** Feel free to explore other settings either here (:doc:`parameters_trex`) or inside the app. Simply hover over any setting with your mouse to read its documentation and understand how it affects the analysis.

Optimizations
^^^^^^^^^^^^^

As you can spot in :numref:`tracking_view`, the tracking is not perfect yet. We can improve it by adjusting a few settings - let's first go back to the welcome screen. You can do that by clicking on the **Menu** button in the top right corner of the screen, followed by a click on **Close file**.

You should see the guppy video now as one of the "Recent files". Click on it to open the settings again. This time, we'll adjust the threshold value in the **Detection** tab. Trust me here, the threshold was a bit too high - this led to some individuals not being detected fully or at all, as well as additional noise because of fragmented individuals. We'll set it to `9` this time. Initially this will produce slightly more particles floating around, but those are easily filtered out by changing the :param:`track_size_filter` setting later on.

Now click on **Convert** again and agree to overwrite the existing file.

Once it's done, and you're back in tracking view, you'll notice that not all individuals are tracked while some of the randomly floating particles are! Luckily, we can fix this rather easily by adjusting :param:`track_size_filter` - but what are the correct values here?

Press the ``D`` key, or alternatively click on **Raw** on the top-right, to switch to the **Raw view**. This view shows you the raw detections in each frame, and you can see the size of each detection when you hover it. Hover some of the individuals and particles, and you'll quickly see that all of the individuals are around a size of ``300`` pixels, while the particles are much smaller. We can use this information to filter out the particles by setting the :param:`track_size_filter` to ``[100,1000]``, for example. Typically, the lower bound should be a bit below the smallest individual size, and the upper bound a bit above the largest individual size. This will filter out most of the particles, but keep all individuals.

Now press ``D`` again to switch back to tracking view and click on **Reanalyse** 🔄 on the top-right to apply your changes by retracking the video.

Scrub through the video by clicking (or dragging) on the timeline at the top. Tracking should be nice now! At this point you could already press ``S`` to export what you have and use it elsewhere. We will be going for a few more optimizations, though.

.. NOTE::

   Note that sometimes the videos can lag behind slightly while scrubbing around. This does not affect analysis of course. In fact, |trex| does not use the original video anymore at this point and showing it is simply a visual aid.

Visual identification
^^^^^^^^^^^^^^^^^^^^^

While the results look pretty nice already, basic tracking does not give you any guarantees on their correctness. This is because there are many things that can happen in a video that make *perfect* tracking impossible, independently of the algorithm or program used. There are many examples, some of which are:

- **Overlaps**: Individuals may overlap for prolonged periods of time - this makes it impossible (without additional sources of information, such as "personally knowing the identity of individuals") to tell who is who after they separate again.
- **Obscured vision**: The camera might not be able to see the individuals for some time, potentially even multiple individuals at the same time. They might also be out of frame for a bit or merge with the background.
- **Speed bursts**: Sometimes the individuals move faster than the camera framerate, which would make it impossible to know who went where in some situations.

Often, basic temporary tracking suffices. For each uninterrupted sequence, |trex| saves the start and end point (in time) so that you can use shorter sequences to measure speed within a certain part of the arena, or similar metrics. If you do need a stronger guarantee on identities (and really only if you do, because this is usually time-intensive), you can consider using the "Visual Identification" algorithm (see also :doc:`identification` for more detail). It basically tries to become an expert on visually differentiating between the individuals of a fixed group and uses only individual images, no time-series.

In our example, you'd simply click on **Menu** and then *Train visual identification* to start the process. It will guide you through a few dialogs to define what you want to do and then start learning. In case it is successful it will automatically reanalyse the entire video given the new information - always visually confirm whether the results make sense. Mistakes will happen, but Visual identification gives it an independent source of information that prevents follow-up errors - meaning it can rediscover the correct individual again, even after short erroneous sequences. So that's nice.