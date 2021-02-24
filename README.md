<p align="center"><img src="https://github.com/mooch443/trex/blob/master/images/Icon1024.png" width="160px"></p>

[![CondaBuildLinux](https://github.com/mooch443/trex/actions/workflows/cmake-ubuntu.yml/badge.svg)](https://github.com/mooch443/trex/actions/workflows/cmake-ubuntu.yml) [![CondaBuildMacOS](https://github.com/mooch443/trex/actions/workflows/cmake-macos.yml/badge.svg)](https://github.com/mooch443/trex/actions/workflows/cmake-macos.yml) [![CondaBuildWindows](https://github.com/mooch443/trex/actions/workflows/cmake-windows.yml/badge.svg)](https://github.com/mooch443/trex/actions/workflows/cmake-windows.yml)

# Hey there

Welcome to the git repository of **TRex** (https://trex.run) -- a software designed to track and identify individuals and other moving entities using computer vision and machine learning. The work-load is split into two (not entirely separate) tools:

* **TGrabs**: Record or convert existing videos, perform live-tracking and closed-loop experiments
* **TRex**: Track converted videos (in PV format), use the automatic visual recognition, explore the data with visual helpers, export task-specific data, and adapt tracking parameters to specific use-cases

TRex can track 256 individuals in real-time, or up to 128 with all fancy features like posture estimation enabled, and for up to 100 individuals allows you to 
(when realtime speed is not required) visually recognize individuals and automatically correct potential tracking errors.

**TGrabs**, which is used to directly process already saved videos or to record directly from webcams and/or Basler machine-vision cameras with integrated and customizable closed-loop support. Camera support can be extended for other APIs with a bit of C++ knowledge, of course.

# Installation

TRex supports all major platforms. You can create a new virtual environment (named ``tracking`` here) using Anaconda or miniconda/miniforge by running:

	conda create -n tracking -c trexing trex                  # macOS (Intel), Windows
	conda create -n tracking -c conda-forge -c trexing trex   # Linux (Intel)

If you own a new Mac with an **Apple Silicone CPU**, the Intel version (above) works fine in Rosetta. However, I would strongly encourage installing TRex via ``miniforge``, which is like Anaconda but supports native arm64 packages. This way, hardware accelerated machine learning on your M1 Macbook is possible! Simply follow the instructions here for installing miniforge: https://github.com/apple/tensorflow_macos/issues/153#issue-799924913. Once you're done, you can run the same command as above (only that now everything will be all fast and native ``arm64`` code)::

	conda create -n tracking -c trexing trex  # macOS (arm64)

There is no official tensorflow package yet, which is why TRex will not allow you to use machine learning right away. But -- yay -- Apple provides their own version for macOS including a native ML Compute (https://blog.tensorflow.org/2020/11/accelerating-tensorflow-performance-on-mac.html) backend, which has shown quite a bit of potential. To install tensorflow inside your activated environment, just run:

	pip install --upgrade --force --no-dependencies https://github.com/apple/tensorflow_macos/releases/download/v0.1alpha2/tensorflow_macos-0.1a2-cp38-cp38-macosx_11_0_arm64.whl https://github.com/apple/tensorflow_macos/releases/download/v0.1alpha2/tensorflow_addons_macos-0.1a2-cp38-cp38-macosx_11_0_arm64.whl

Pre-built binaries are compiled with fewer optimzations and features than a manually compiled one (due to compatibility and licensing issues) and thus are slightly slower =(. For example, the conda versions do not offer support for Basler cameras. If you need to use TGrabs with machine vision cameras, or need as much speed as possible (or the newest version), please consider compiling the software yourself.

If you want compatibility with the Basler API (or other things with licensing/portability issues), please 
use one of the manual compilation options (see https://trex.run/docs/install.html).

# Contributors, Issues, etc.

This project has been developed, is still being updated, by [Tristan Walter](http://moochm.de).
If you want to contribute, please submit a pull request on github and I will be happy to credit you here, for any substantial contributions!

If you have any issues running the software please consult the documentation first (especially the FAQ section) 
and if this does not solve your problem, please file an issue using the [issue tracker](https://github.com/mooch443/trex/issues) here on github. 
If you experience problems with [Tensorflow](https://tensorflow.org), such as installing CUDA or cuDNN dependencies, then please direct issues to those development teams.

# License

Released under the GPLv3 License (see [LICENSE](https://github.com/mooch443/trex/blob/master/LICENSE)).

# Reference

If you use this software in your work, please cite our [pre-print](http://preprint.trex.run):
```
  @article{walter2020trex,
    author = {Walter, Tristan and Couzin, Iain D},
    title = {TRex, a fast multi-animal tracking system with markerless identification, 2D body posture estimation and visual field reconstruction},
    year = {2020},
    doi = {10.1101/2020.10.14.338996},
    publisher = {Cold Spring Harbor Laboratory},
    URL = {https://www.biorxiv.org/content/early/2020/10/15/2020.10.14.338996},
    journal = {bioRxiv}
  }
```
