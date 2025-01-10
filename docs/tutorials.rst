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

   Tracking more individuals in your video increases difficulty in many ways and the computational resources required, potentially also limiting your analysis options later [#f1]_. For example: Automatic visual identification is only feasible with smaller groups (typically fewer than 50 individuals) since it relies on relative visual differences in a *known* group. If it is not possible to automatically maintain perfect identities in too large a group you may need to limit your analysis on more general information about the group's behavior rather than real identities. |trex| also subsections trajectory pieces into *tracklets*, which are shorter sequences per tracked object where identities are maintained with high confidence (see below). These can be used for more shortterm analyses, even if the full video is too complex to maintain identities throughout.

2. **Scene Complexity**
   
   .. epigraph::

      *Eliminate visual clutter — use uniform, high-contrast backgrounds if possible.*

   Complex scenes make it difficult to see individuals. This does not just apply to human eyes, but also to computer vision. Various factors such as very small objects, heterochrome objects, occlusions, shadows, reflections, and other interfering objects can hinder detection (and thus tracking) performance. Varying or visually complex backgrounds with challenging contrast may require more advanced segmentation algorithms, often based on machine learning [#f2]_. These are more generic and often *better*, generally speaking, but require more manual labor to set up and more computational resources to apply. 

3. **Camera and Lighting Choice**

   .. epigraph::

      *Pay attention to your camera settings and their influence on visual quality. Do not store more data than you need (e.g. zoom in).*

   The camera and recording settings you use affect image and computational complexity. Low-resolution or low frame rate cameras may make it difficult to identify individuals visually — especially if you plan to use the visual identification algorithm. A slow shutter speed can introduce motion blur, complicating tracking efforts. Lighting conditions also play a significant role; when possible, prefer DC lights over AC lights to avoid flickering. Shiny fur or scales can also cause reflections that interfere with visual identification algorithms. High framerates can prolong analyses since there are simply more images to process. Do not record more data than you need - zoom in if possible, and record only the area of interest. Point the camera straight down to avoid distortion and ensure that the camera is stable to avoid shaking.

4. **Research Question**

   .. epigraph::

      *Define what kind of data you need, and what it implies for your setup, before you start filming.*
   
   Aim to use the simplest approach that effectively answers your research question. If you're interested in the general behavior of a group, maintaining perfect identities may not be necessary. |trex| provides information on shorter sequences per tracked object (the aforementioned *tracklets*), which, even without identity corrections, often suffice. For instance, if you're interested in the average speed of a group or average speed of individuals in a certain area, then short *tracklets* are enough. However, if you are distinguishing between specific individuals (invisibly *parasitized*/*unparasitized* or *informed*/*uninformed*) individuals, you might need to use the visual identification algorithm. This, if successful, gives you a stronger guarantee for maintained identities, but requires you to invest more time and computational power.
   
   Your research question also influences the *type* of data output you need - most of the time the centroid position is enough (simply called *detection*), but there are other types: *Pose*/*Keypoint* data is only necessary if you're focusing on specific body parts (see :numref:`data_types`) in any way, and *outline* data is necessary if you're interested in the directionality of the individuals (e.g., front/back) or other aspects of their 2D shape. See also [#f2]_ and :numref:`data_types` for more information.

The following sections will guide you through increasingly complex problems and how to solve them with |trex|. Identify which category your research question falls into to find the right tools.

.. [#f1] The number of possible combinations of individuals at any given time increases exponentially with the number of individuals — a phenomenon known as the "curse of dimensionality" in computer vision and machine learning. With a large number of individuals, you may need to wait a bit longer, and potentially have more visual overlaps between individuals.

.. [#f2] Segmentation algorithms are used to fully separate individuals from the background in a video - meaning not just *position* but also a *"picture"*. Keypoint data, for example, sometimes only retrieves multiple points - not the actual shape. For complex scenes, advanced algorithms like deep learning models (e.g., YOLO) are often employed.

.. _data_types:

.. figure:: images/data_types.png

   Different data types that can be extracted from a video: just the centroid position, pose/keypoint data, and outline-type data including directionality (front/back).

Basics
------

This tutorial will guide you through the basic steps of setting up |trex|, analyzing an example video, and exporting the data.

If you want to follow along exactly as described, you can download the example video from `here <https://trex.run/8guppies_20s.mp4>`_ and place it in any folder you'd like to use as your root analysis folder.

Installation
^^^^^^^^^^^^

You can download the latest version of |trex| using `Miniforge` (conda). To install |trex|, you need to have `Miniforge` installed on your system.

If you're not familiar with `conda` or `Miniforge`, you can find more information on how to install them `here <https://conda-forge.org/miniforge/>`_. Miniforge is a minimal installer for `conda`, an open-source package manager that helps you create virtual environments and install software and libraries without installing them globally on your system [#f3]_. We do not support Anaconda`s default channels, so please use `Miniforge` instead or restrict your channels to `conda-forge` only [#f4]_.

Open your `Miniforge` Prompt and run:

.. code-block:: bash

   conda create -n track -c trex-beta trex

.. NOTE::

   This only works if `conda-forge` is the *only* channel you have added to your `conda` configuration. By default, this is the case if you're using `Miniforge`. If you have other channels added, or are using Anaconda, you can run the following command instead:

   .. code-block:: bash

      conda create -n track --override-channels -c trex-beta -c conda-forge trex

This will create a new conda environment called ``track`` with |trex| installed. This could take a while, especially in conda´s 'verifying transaction'. Activate the environment using:

.. code-block:: bash

   conda activate track

Then start |trex| by typing:

.. code-block:: bash

   trex

and pressing **Enter**. 

If a window showing a friendly T-Rex appears, you've successfully installed the software.

Recommended System Requirements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

While |trex| is designed to be lightweight, ensuring a modern CPU, sufficient RAM (8GB or more), and a dedicated GPU (optional but beneficial for advanced machine learning tasks) can optimize performance.

.. _welcome_screen:

.. figure:: images/welcome_screen-1.mov
   :width: 100%

   The TRex graphical user interface (GUI) showing the welcome screen.

If you have any issues with the installation, please refer to the :doc:`installation guide <install>`.

.. [#f3] The advantage of this is that you can have different versions of the same software installed on your system without conflicts, and that they can be easily removed.

.. [#f4] We do not support Anaconda's default channels because forge has easier license agreements and is often more up-to-date. Anaconda`s hosted channels can be problematic for you too, if your institution does not have a license agreement with them.

Setting Up a New Project
^^^^^^^^^^^^^^^^^^^^^^^^

When you first start |trex|, you'll see a window with a welcome message and a button to open a file or start the webcam. Click on the folder icon in the center of the screen to proceed to the initial settings screen (see :numref:`initialsettings`).

.. _initialsettings:

.. figure:: images/configuration_screen-1.mov
   :width: 100%

   Initial settings screen where you can choose your input and output files, as well as various detection and tracking parameters.

Steps to Set Up
^^^^^^^^^^^^^^^

1. **Select the Input File**: Click on the 📂 folder button next to the input file name at the top, or enter the path manually. By default, |trex| will place generated outputs in the same folder as the input file, but you can choose a different folder in the output section below.

2. **Set Output Preferences**:

   - **Prefix**: The ``prefix`` (or ``output_prefix``) can be optionally set. This creates a subfolder with the given name under the output root, redirecting all new outputs there while the original ``.pv`` file stays in the root folder. This helps organize different sessions for the same video (e.g., trying out different settings or separating tracking per species).

3. **Adjust Settings Using Tabs**:

   - **Locations**: Set the input and output files and related settings.
   - **Detection**: Configure settings related to detecting individuals (or objects) in the raw image frames. This is the first real step in the pipeline, and settings here cannot be effectively changed afterward.
   - **Tracking**: Adjust settings related to tracking the detected individuals. This is the second step in the pipeline, and settings here can be changed at any time, although it might require re-analysis.


4. **Set the Number of Individuals**:

   - Navigate to the **Tracking** tab.
   - Set the number of individuals to 8, since we know this from the example video.

   .. _settings_pane:

   .. figure:: images/settings_pane.png
      :width: 100%

      The tracking settings tab allows you to set the number of individuals to track, as well as other tracking-related settings.


5. **Start the Conversion**:

   - Once you've configured the settings, click the **Convert** button at the bottom right to begin processing.

**Tip:** Feel free to explore other settings. Hover over any setting with your mouse to read its documentation and understand how it affects the analysis.