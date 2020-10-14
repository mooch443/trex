<p align="center"><img src="https://github.com/mooch443/trex/blob/master/images/Icon1024.png" width="160px"></p>

# Hey there

Welcome to the git repository of **TRex** (https://trex.run) -- a software designed to track and identify individuals and other moving entities using computer vision and machine learning. The work-load is split into two (not entirely separate) tools:

* **TGrabs**: Record or convert existing videos, perform live-tracking and closed-loop experiments
* **TRex**: Track converted videos (in PV format), use the automatic visual recognition, explore the data with visual helpers, export task-specific data, and adapt tracking parameters to specific use-cases

TRex can track 256 individuals in real-time, or up to 128 with all fancy features like posture estimation enabled, and for up to 100 individuals allows you to 
(when realtime speed is not required) visually recognize individuals and automatically correct potential tracking errors.

**TGrabs**, which is used to directly process already saved videos or to record directly from webcams and/or Basler machine-vision cameras with integrated and customizable closed-loop support. Camera support can be extended for other APIs with a bit of C++ knowledge, of course.

# Installation

Create a new virtual conda environment on macOS/Windows, using:

```
conda create -n tracking -c trexing trex                           # macOS, Windows
```
or, add the `conda-forge` channel for Linux:
```
conda create -n tracking -c main -c conda-forge -c trexing trex    # Linux
```

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
