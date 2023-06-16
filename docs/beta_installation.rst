=============================================
TRex Installation and Usage Guide
=============================================

Introduction
============

This document provides an installation guide for the TRex software and its usage for both segmentation and object detection. This software supports yolov8 networks trained on user-defined datasets.


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

    trexa -i video.mp4 -m model.pt

Command Line Arguments
-----------------------

There are various command-line arguments that you can use with the TRex command:

- `meta_video_scale` (float, default=1)
- `source` (string, no default)
- `model` (file path, no default)
- `segmentation_resolution` (uint16_t, default=128)
- `segmentation_model` (file path, no default)
- `region_model` (file path, no default)
- `region_resolution` (uint16_t, default=320)
- `detection_resolution` (uint16_t, default=640)
- `filename` (file path, no default)
- `detection_type` (enum, default=yolo8)
- `tile_image` (size_t, default=0)
- `batch_size` (uchar, default=10)
- `do_filter` (boolean, default=false)
- `filter_classes` (vector of uint8_t, no default)

Boolean values can be omitted (e.g. `-do_filter` instead of `-do_filter true`). For arguments with spaces or arrays, use quotes around values. You can use the following shortcuts:

- `-m` for model
- `-bm` for region_model
- `-o` for filename
- `-i` for source
