.. include:: names.rst

.. toctree::
   :maxdepth: 3
   :numbered:

Tutorials
=========

|trex| is a versatile tracking software that can help you solve almost all tracking problems with varying manual effort, computational power, and time. It's designed to be user-friendly for general use but also provides advanced features for more complex situations. This section provides tutorials on how to use |trex|, from setting up the software to analyzing videos and exporting data. 

You can follow along by reading the text or watching the video tutorials on the `YouTube channel <https://www.youtube.com/@TRex-f9g>`_.

Before diving in, I’d like to add a brief note — strictly from a computer vision standpoint — to highlight a few key considerations that can simplify your work and help you set realistic expectations.

Understanding Problem Complexity
--------------------------------

Before analyzing or recording videos, it's crucial to understand the complexity of the problem you're addressing. This insight will help you choose the appropriate settings and methods to achieve optimal results and guide you in designing experiments. The complexity can be split into several factors:

1. **Number of Individuals**
   
   Tracking more individuals in your video increases difficulty in many ways and the computational resources required, potentially also limiting your analysis options later [#f1]_. For example: Automatic visual identification is only feasible with smaller groups (typically fewer than 50 individuals) since it relies on relative visual differences in a *known* group. If you cannot automatically maintain perfect identities in a large group you may need to limit your analysis on more general information about the group's behavior rather than real identities. |trex| offers tools like *trajectory segments*, which are shorter sequences per tracked object where identities are maintained with high confidence.

2. **Scene Complexity**
   
   Complex scenes make it difficult to see individuals. This does not just apply to human eyes, but also (potentially to a lesser degree) to computer vision. Additional factors such as occlusions, shadows, reflections, and other interfering objects can further hinder tracking performance. Varying or visually complex backgrounds with challenging contrast may require more advanced segmentation algorithms, often based on machine learning [#f2]_. These are more generic and often *better*, but require more manual labor to set up and more computational resources to run. Always avoid visual clutter in your scene, and if possible, use a uniform and constrasting background.

3. **Camera and Lighting Choice**

   The camera and recording settings you use affect the problem's complexity. Low-resolution or low frame rate cameras may make it difficult to identify individuals visually—especially if you plan to use the visual identification algorithm. A slow shutter speed can introduce motion blur, complicating tracking efforts. Lighting conditions also play a significant role; when possible, prefer DC lights over AC lights to avoid flickering.

4. **Research Question**

   Aim to use the simplest approach that effectively answers your research question. If you're interested in the general behavior of a group, maintaining perfect identities may not be necessary. |trex| provides information on shorter sequences per tracked object, even without identity corrections, which can suffice for analysis. However, if you need perfectly maintained identities [#f3]_, you might need to use the visual identification algorithm, which requires more time and computational power. Your research question also influences the type of data output you need. For instance, if you're interested in the average speed of a group, trajectory data may be sufficient, while pose data is necessary if you're focusing on specific body parts (see image below).

The following sections will guide you through increasingly complex problems and how to solve them with |trex|. Identify which category your research question falls into to find the right tools.

.. [#f1] The number of possible combinations of individuals at any given time increases exponentially with the number of individuals — a phenomenon known as the "curse of dimensionality" in computer vision and machine learning. With a large number of individuals, you may need to wait a bit longer, and potentially have more visual overlaps between individuals.

.. [#f2] Segmentation algorithms are used to separate individuals from the background in a video. For complex scenes, advanced algorithms like deep learning models (e.g., YOLO) are often employed.

.. [#f3] For example, distinguishing between parasitized and unparasitized individuals or informed vs. uninformed individuals.

.. figure:: images/data_types.png
   :alt: Different data types that can be extracted from a video: just the centroid position, pose/keypoint data, and outline-type data including directionality (front/back).

   Different data types that can be extracted from a video: just the centroid position, pose/keypoint data, and outline-type data including directionality (front/back).

Basics
------

This tutorial will guide you through the basic steps of setting up |trex|, analyzing an example video, and exporting the data.

If you want to follow along exactly as described, you can download the example video from `here <https://trex.run/8guppies_20s.mp4>`_ and place it in any folder you'd like to use as your root analysis folder.

Installation
^^^^^^^^^^^^

You can download the latest version of |trex| using `Miniforge` (conda). To install |trex|, you need to have `Miniforge` installed on your system.

If you're not familiar with `conda` or `Miniforge`, you can find more information on how to install them `here <https://conda-forge.org/miniforge/>`_. Miniforge is a minimal installer for `conda`, an open-source package manager that helps you create virtual environments and install software and libraries without installing them globally on your system.

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

.. figure:: images/welcome_screen-1.mov
   :width: 100%
   :alt: The TRex graphical user interface (GUI) showing the welcome screen.

   The TRex graphical user interface (GUI) showing the welcome screen.

If you have any issues with the installation, please refer to the :doc:`installation guide <install>`.

Setting Up a New Project
^^^^^^^^^^^^^^^^^^^^^^^^

When you first start |trex|, you'll see a window with a welcome message and a button to open a file or start the webcam. Click on the folder icon in the center of the screen to proceed to the initial settings screen (see :ref:`Figure <initialsettings>`).

.. figure:: images/configuration_screen-1.mov
   :width: 100%
   :name: initialsettings
   :alt: Initial settings screen where you can choose your input and output files, as well as various detection and tracking parameters.

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

   .. figure:: images/settings_pane.png
      :width: 100%
      :alt: The tracking settings tab allows you to set the number of individuals to track, as well as other tracking-related settings.

      The tracking settings tab allows you to set the number of individuals to track, as well as other tracking-related settings.


5. **Start the Conversion**:

   - Once you've configured the settings, click the **Convert** button at the bottom right to begin processing.

**Tip:** Feel free to explore other settings. Hover over any setting with your mouse to read its documentation and understand how it affects the analysis.