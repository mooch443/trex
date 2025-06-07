<p align="center"><img src="https://github.com/mooch443/trex/blob/main/Application/src/tracker/gfx/TRex_1024.png" width="320px"></p>

`main`
[![CondaBuildLinux](https://github.com/mooch443/trex/actions/workflows/cmake-ubuntu.yml/badge.svg?branch=main)](https://github.com/mooch443/trex/actions/workflows/cmake-ubuntu.yml)
[![CondaBuildMacOS](https://github.com/mooch443/trex/actions/workflows/cmake-macos.yml/badge.svg?branch=main)](https://github.com/mooch443/trex/actions/workflows/cmake-macos.yml)
[![CondaBuildWindows](https://github.com/mooch443/trex/actions/workflows/cmake-windows.yml/badge.svg?branch=main)](https://github.com/mooch443/trex/actions/workflows/cmake-windows.yml)

`dev`
[![CondaBuildLinux](https://github.com/mooch443/trex/actions/workflows/cmake-ubuntu.yml/badge.svg?branch=dev)](https://github.com/mooch443/trex/actions/workflows/cmake-ubuntu.yml)
[![CondaBuildMacOS](https://github.com/mooch443/trex/actions/workflows/cmake-macos.yml/badge.svg?branch=dev)](https://github.com/mooch443/trex/actions/workflows/cmake-macos.yml)
[![CondaBuildWindows](https://github.com/mooch443/trex/actions/workflows/cmake-windows.yml/badge.svg?branch=dev)](https://github.com/mooch443/trex/actions/workflows/cmake-windows.yml)

*Documentation: https://trex.run/docs*
*YouTube: https://youtube.com/@TRexTracker*

# Hey there

Welcome to the git repository of **TRex** (https://trex.run) -- a software designed to track and identify individuals and other moving entities using computer vision and machine learning.

TRaditionally, using background-subtraction, TRex can track 256 fish faster than the video plays back, and for up to 100 individuals allows you to visually recognize individuals and automatically correct potential tracking errors using machine-learning based automatic classification.

Since version 2.0, TRex also supports machine-learning based detection models. Out of the box we support models compatible with the `ultralytics` (https://github.com/ultralytics) package, like YOLO. All common tasks, such as bounding-boxes, oriented bounding-boxes, keypoints, and segmentation as well as offering simple SAHI-like features (https://github.com/obss/sahi). This replaces background subtraction in situations where it is not sufficient - you can bring your own ``.pt`` model, of course!

It allows you, from the same interface, to load up existing videos or record directly from your webcam. Camera support (and other things) can be extended for other APIs with a bit of C++ or Python knowledge, of course.

# Installation

TRex supports all major platforms. Miniforge is the main package repository used for installing/building TRex. Simply follow the instructions here for installing miniforge: https://github.com/conda-forge/miniforge#download.
You can then create a new virtual environment (named ``tracking`` here) by running:

`Windows, Linux (NVIDIA graphics card recommended), macOS (Intel / ARM64)`
```bash
conda create -n tracking -c trexing trex
```

# Usage

Within the conda environment, simply run:

	trex

To convert a video to our custom pv format from the command-line:

	trex -i /full/path/to/video.mp4 -o funny_name

This will output to `/full/path/to/video.pv`. If you want to change the output destination root, add `-d /output/path`.

Opening a video directly and adjusting [parameters](https://trex.run/docs/parameters_trex.html):

	trex -i /full/path/to/video.mp4 -track_threshold 25 -track_max_individuals 10

If you don't want a graphical user interface and save/quit when tracking finishes, add `-nowindow` and `-auto_quit`:

	trex -i /full/path/to/video.mp4 -nowindow -auto_quit

Read [more](https://trex.run/docs/run.html) about parameters for TRex [here](https://trex.run/docs/parameters_trex.html).

# Contributors, Issues, etc.

This project has been developed, is still being updated, by [Tristan Walter](http://moochm.de).
If you want to contribute, please submit a pull request on GitHub and I will be happy to credit you here, for any substantial contributions!

If you have any issues running the software please consult the documentation first (especially the FAQ section) 
and if this does not solve your problem, please file an issue using the [issue tracker](https://github.com/mooch443/trex/issues) here on github. 
If you experience problems with [PyTorch](https://pytorch.org), such as installing CUDA or cuDNN dependencies, then please direct issues to those development teams.

Significant contributions by:
[Angela Albi](https://github.com/albiangela)

# License

Released under the GPLv3 License (see [LICENSE](https://github.com/mooch443/trex/blob/main/LICENSE)).

# Reference

If you use this software in your work, please cite our [open-access paper](https://elifesciences.org/articles/64000):
```
@article {walter2020trex,
  article_type = {journal},
  title = {TRex, a fast multi-animal tracking system with markerless identification, and 2D estimation of posture and visual fields},
  author = {Walter, Tristan and Couzin, Iain D},
  editor = {Lentink, David},
  volume = 10,
  year = 2021,
  month = {feb},
  pub_date = {2021-02-26},
  pages = {e64000},
  citation = {eLife 2021;10:e64000},
  doi = {10.7554/eLife.64000},
  url = {https://doi.org/10.7554/eLife.64000},
  journal = {eLife},
  issn = {2050-084X},
  publisher = {eLife Sciences Publications, Ltd},
}
```
