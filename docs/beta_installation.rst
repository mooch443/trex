=============================================
TRex Installation and Usage Guide
=============================================

Introduction
============

Welcome to the TRex software guide. This document aims to assist you in navigating the installation process and utilizing TRex's capabilities for both segmentation and object detection. Designed to provide a robust solution for tracking and analyzing animal behavior, TRex brings the power of yolov8 networks to your fingertips, offering the flexibility to train these networks on any dataset you choose. From simple tracking tasks to more complex scenarios, including region proposal or tiling, TRex can accommodate various research needs. Whether you're outlining instances of objects or detecting objects within a frame, TRex ensures precision and efficiency in animal behavior analysis.

Installation
============

Use the following commands for installing TRex based on your system:

Linux and Windows
-------------------

.. code-block:: bash

    conda create -n beta -c trex-beta -c pytorch -c nvidia trex pytorch-cuda=11.7

macOS (M1/M2)
----------------

.. code-block:: bash

    conda create -n beta -c trex-beta -c apple -c pytorch-nightly trex

macOS (Intel)
----------------

.. code-block:: bash

    conda create -n beta -c trex-beta -c pytorch trex


Usage
=====

Once installed, you can execute TRex using the following command:

.. code-block:: bash

    trexa -i video.mp4 -m model.pt -detection_resolution 640

The `detection_resolution` parameter is 640 by default, so in most cases you won't need to specify it. This works for both segmentation and object detection.

Advanced Usage
==============

TRex equips you with a robust suite of features, perfect for handling intricate use cases. If there are specific regions of interest in your study, you can leverage a different network to predict these regions. This feature is especially useful when dealing with complex scenarios such as monitoring specific areas within an otherwise nearly vacant arena. Here, region proposals can help improving the accuracy of the detections by strategically increasing spatial resolution where it matters. This feature is also beneficial for analyzing high-resolution videos that do not fit in memory in one piece.

When examining densely crowded objects of interest where the entire image contains relevant data, TRex offers a tiling feature. Activated with `-tile_image N`, this functionality splits the image into an NxN grid, with each tile analyzed individually before being merged post-analysis. However, please note that networks may need to be specifically trained on this tiling size to assure accurate detection.

TRex is flexible and supports yolov8 networks trained on any user-defined dataset. You can customize your object detection or instance segmentation tasks based on your particular needs. For example, using `yolov8n-seg.pt` will generate outlines around objects (achieving instance segmentation), while `yolov8n.pt` will solely perform object detection (producing boxes centered on objects, without posture information).

Command Line Arguments
-----------------------

Below is a list of various command-line arguments that you can use with the TRex command, along with a brief explanation of each:

.. code-block:: bash

    - `meta_video_scale` (float, default=1)  # Scale of the video
    - `source` (string, no default)  # Source video file
    - `model` (file path, no default)  # Path to the object detection or instance segmentation model
    - `region_model` (file path, no default)  # Path to the region prediction model
    - `region_resolution` (uint16_t, default=320)  # Resolution for region prediction tasks
    - `detection_resolution` (uint16_t, default=640)  # Resolution for detection tasks
    - `filename` (file path, no default)  # Output file name
    - `meta_classes` (std::vector<std::string>{}, default=empty)  # Class names for object classification in video during conversion
    - `detection_type` (enum, default=yolo8)  # Type of object detection model. Possible values: yolo7, yolo7seg, yolo8, customseg
    - `tile_image` (size_t, default=0)  # Size for tiling the image
    - `batch_size` (uchar, default=10)  # Size of the batch for processing frames
    - `do_filter` (boolean, default=false)  # Enable or disable filtering of certain classes
    - `filter_classes` (vector of uint8_t, no default)  # Class numbers of the filtered objects if do_filter is on

For boolean parameters, you can omit their value (e.g. `-do_filter` instead of `-do_filter true`). In the case of strings, arguments with spaces or arrays, use quotes around the values.
