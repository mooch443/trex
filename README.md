<p align="center"><img src="https://github.com/mooch443/trex/blob/main/images/Icon1024.png" width="160px"></p>

[![CondaBuildLinux](https://github.com/mooch443/trex/actions/workflows/cmake-ubuntu.yml/badge.svg?branch=main)](https://github.com/mooch443/trex/actions/workflows/cmake-ubuntu.yml) [![CondaBuildMacOS](https://github.com/mooch443/trex/actions/workflows/cmake-macos.yml/badge.svg?branch=main)](https://github.com/mooch443/trex/actions/workflows/cmake-macos.yml) [![CondaBuildWindows](https://github.com/mooch443/trex/actions/workflows/cmake-windows.yml/badge.svg?branch=main)](https://github.com/mooch443/trex/actions/workflows/cmake-windows.yml)

*Now with native [Apple Silicone (M1)](https://www.apple.com/mac/m1/) and ML Compute support. [How to install TRex (arm64)](https://trex.run/docs/install.html#install-m1).*

*Documentation: https://trex.run/docs*

# Hey there

Welcome to the git repository of **TRex** (https://trex.run) -- a software designed to track and identify individuals and other moving entities using computer vision and machine learning. The work-load is split into two (not entirely separate) tools:

* **TGrabs**: Record or convert existing videos, perform live-tracking and closed-loop experiments
* **TRex**: Track converted videos (in PV format), use the automatic visual recognition, explore the data with visual helpers, export task-specific data, and adapt tracking parameters to specific use-cases

TRex can track 256 individuals in real-time, or up to 128 with all fancy features like posture estimation enabled, and for up to 100 individuals allows you to 
(when realtime speed is not required) visually recognize individuals and automatically correct potential tracking errors.

**TGrabs**, which is used to directly process already saved videos or to record directly from webcams and/or Basler machine-vision cameras with integrated and customizable closed-loop support. Camera support can be extended for other APIs with a bit of C++ knowledge, of course.

# Installation

TRex supports all major platforms. You can create a new virtual environment (named ``tracking`` here) using Anaconda or miniconda/miniforge by running:

```bash
# macOS (Intel/arm64 M1), Windows
conda create -n tracking -c trexing trex 
```

```bash
# Linux
conda create -n tracking -c defaults -c conda-forge -c trexing trex 
```

## macOS with an arm64 / M1 processor

If you own a new Mac with an **Apple Silicone CPU**, the Intel version (above) works fine in Rosetta. However, I would strongly encourage installing TRex via ``miniforge``, a flavor of miniconda that natively supports arm64 packages. Simply follow the instructions here for installing miniforge: https://github.com/conda-forge/miniforge#download.

Once you're done, you can run this command to create the virtual environment:

```bash
# macOS (arm64/M1)
conda create -n tracking -c trexing trex 
```

Installing tensorflow on the M1 is a bit more complicated, which is why TRex will not allow you to use machine learning unless you install the following extra packages manually. Instructions will be printed out after you created the environment Apple provides their own tensorflow version for macOS including a native METAL (https://developer.apple.com/metal/tensorflow-plugin/) plugin. To install tensorflow inside your environment, just run:

```bash
# activate the TRex environment
conda activate tracking

# install tensorflow dependencies and metal plugin
conda install -c apple -y tensorflow-deps==2.7.0
python -m pip install tensorflow-macos==2.7.0 tensorflow-metal==0.3.0
```

## Manual compilation

Pre-built binaries are compiled with fewer optimzations and features than a manually compiled one (due to compatibility and licensing issues) and thus are slightly slower =(. For example, the conda versions do not offer support for Basler cameras. If you need to use TGrabs with machine vision cameras, or need as much speed as possible (or the newest version), please consider compiling the software yourself.

If you want compatibility with the Basler API (or other things with licensing/portability issues), please 
use one of the manual compilation options (see https://trex.run/docs/install.html).

# Usage

Within the conda environment, simply run:

	trex

Opening a video directly and adjusting [parameters](https://trex.run/docs/parameters_trex.html):

	trex -i /path/to/video.pv -track_threshold 25 -track_max_individuals 10

If you don't want a graphical user interface and save/quit when tracking finishes:

	trex -i /path/to/video.pv -nowindow -auto_quit

To convert a video to our custom pv format (for usage in TRex) from the command-line:

	tgrabs -i /full/path/to/video.mp4 -o funny_name

Read [more](https://trex.run/docs/run.html) about parameters for TRex [here](https://trex.run/docs/parameters_trex.html) and for TGrabs [here](https://trex.run/docs/parameters_tgrabs.html).

# Contributors, Issues, etc.

This project has been developed, is still being updated, by [Tristan Walter](http://moochm.de).
If you want to contribute, please submit a pull request on github and I will be happy to credit you here, for any substantial contributions!

If you have any issues running the software please consult the documentation first (especially the FAQ section) 
and if this does not solve your problem, please file an issue using the [issue tracker](https://github.com/mooch443/trex/issues) here on github. 
If you experience problems with [Tensorflow](https://tensorflow.org), such as installing CUDA or cuDNN dependencies, then please direct issues to those development teams.

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
