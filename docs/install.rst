.. include:: names.rst

.. toctree::
   :maxdepth: 2

.. NOTE::
	|trex| now supports the new Apple Silicone Macs with Apple's own M1 CPU, including hardware accelerated machine learning! Please follow the instructions below to obtain a native version with hardware acceleration: :ref:`install-m1`.

.. WARNING::
	On Windows and Linux a NVIDIA graphics-card is required for hardware accelerated machine learning.

Installation
############

The easy way (Windows, Linux and Intel macOS)
*********************************************

|trex| supports all major platforms. There is an easy way to install |trex| using Anaconda, by creating a new virtual environment (here named ``tracking``, which you can replace)::

	conda create -n tracking -c trexing trex                          # macOS (Intel), Windows
	conda create -n tracking -c main -c conda-forge -c trexing trex   # Linux (Intel)

The down-side is that pre-built binaries are compiled with fewer optimzations and features than a manually compiled one (due to compatibility and licensing issues) and thus are slightly slower =(. For example, the conda version does not offer support for Basler cameras. If you need to use |grabs| with machine vision cameras, or need as much speed as possible/the newest version, please consider compiling the software yourself.

.. _install-m1:

Apple Silicone (macOS arm64)
****************************

If you own a new Mac with an Apple Silicone CPU, the Intel version (above) works fine in Rosetta. However, I would strongly encourage installing |trex| via ``miniforge``, which is like Anaconda but supports native arm64 packages. This way, hardware accelerated machine learning on your M1 Macbook is possible! Simply follow the instructions here for installing miniforge: `github.com/apple/tensorflow_macos <https://github.com/apple/tensorflow_macos/issues/153#issue-799924913>`_. Once you're done, you can run the same command as above (only that now everything will be all fast and native ``arm64`` code)::

	conda create -n tracking -c trexing trex  # macOS (arm64)

There is no official tensorflow package yet, which is why |trex| will not allow you to use machine learning right away. But -- yay -- Apple provides their own version for macOS including a native ML Compute (`blog.tensorflow.com <https://blog.tensorflow.org/2020/11/accelerating-tensorflow-performance-on-mac.html>`_) backend, which has shown quite a bit of potential. An Apple Silicone MacBook (2020) only needs ~50ms/step and (with the same data and code) is not much slower than my fast i7 PC with an NVIDIA Geforce 1070 -- running at roughly ~21ms/step. To install tensorflow inside your activated environment, just run::

	pip install --upgrade --force --no-dependencies https://github.com/apple/tensorflow_macos/releases/download/v0.1alpha2/tensorflow_macos-0.1a2-cp38-cp38-macosx_11_0_arm64.whl https://github.com/apple/tensorflow_macos/releases/download/v0.1alpha2/tensorflow_addons_macos-0.1a2-cp38-cp38-macosx_11_0_arm64.whl

Now |trex|, if installed within the same environment, has the full power of your Mac at its disposal. Have fun!

Compile it yourself
*******************

There are two ways to get your own version of |trex|:

* creating a local conda channel, and installing from there
* running CMake/build manually with customized options

Both are obviously similar in result, but there *are* differences (the local channel is essentially a script for the manual procedure, with some caveats). For example, the conda build is limited to certain compiler and OS-SDK versions -- which is something that you may want to change in order to enable certain optimizations. We start out here by describing the more automated way using conda, followed by a description of how to do everything manually.

Local conda channel
===================

In order to get your own (local) conda channel, all you need to do is make sure you have Anaconda installed, as well as the ``conda-build`` package. This is a package that allows you to make your own packages from within the base environment (use ``conda deactivate``, until it says ``base`` on the left). It creates a virtual environment, within which it compiles/tests the software you are trying to build. You can install it using::

	conda install conda-build

After that, from within the conda ``base`` environment, clone the |trex| repository using::

	git clone --recursive https://github.com/mooch443/trex
	cd trex/conda

Now, from within that folder, run::

	./build_conda_package.bat # Windows
	./build_conda_package.sh  # Linux, macOS

This runs ``conda build .``, which builds the program according to all the settings inside ``meta.yaml`` (for dependencies), using ``build.sh`` (or ``bld.bat`` on Windows) to configure CMake. If you want to enable/disable certain features (e.g. use a locally installed OpenCV library, enable the Pylon SDK, etc.) this build script is the place where you can do that. Although beware that you may need to add absolute paths to the cmake call (e.g. adding folders to ``CMAKE_PREFIX_PATH``) so that it can find all your locally installed libraries -- in which case your conda package will probably not be portable.

After compilation was successful, |trex| can be installed using::

	conda create -n tracking -c trexing trex                          # macOS, Windows
	conda create -n tracking -c main -c conda-forge -c trexing trex   # Linux (Intel)

Notice there is a ``-c local``, instead of the ``-c trexing`` from the first section.

Finally, to run it simply switch to the environment you just created (tracking) using ``conda activate tracking`` and run ``trex`` to see if the window appears!

Compiling manually
==================

First, make sure that you fulfill the platform-specific requirements:

* **Windows**: Please make sure you have Visual Studio installed on your computer. It can be downloaded for free from https://visualstudio.microsoft.com. We have tested Visual Studio versions 2017 and 2019. We are using the Anaconda PowerShell here in our examples.
* **MacOS**: Make sure you have Xcode and the Xcode compiler tools installed. They can be downloaded for free from the App Store (Xcode includes the compiler tools). We used macOS 10.15 and Xcode 11.5.
* **Linux**: You should have build-essential installed, as well as ``g++ >=8`` or a different compiler with full C++17 support.

As well as the general requirements:

* **Python**: We use version ``>= 3.6``.
* **CMake**: Version ``>= 3.16``.

.. NOTE::
	We will be using Anaconda here. However, it is not *required* to use Anaconda when compiling |trex| -- it is just a straight-forward way to obtain dependencies. In case you do not want to use Anaconda, please make sure that all mentioned dependencies are installed in a way that can be detected by CMake. You may also add necessary paths to the CMake command-line, such as ``-DOpenCV_DIR=/path/to/opencv`` and use switches to compile certain libraries (such as OpenCV) statically with |trex|.

The easiest way to ensure that all requirements are met, is by using conda to create a new environment::

	conda create -n tracking -c conda-forge cmake ffmpeg tensorflow=2 cxx-compiler c-compiler glfw mesa-libgl-devel-cos6-x86_64 libxdamage-devel-cos6-x86_64 libxi-devel-cos6-x86_64 libxxf86vm-cos6-x86_64 libselinux-devel-cos6-x86_64

If your GPU is supported by TensorFlow, you can modify the above line by appending ``-gpu`` to ``tensorflow`` to get ``tensorflow-gpu=2``.
	
Next, switch to the conda environment using::

	conda activate tracking

You can now clone the repository and change your directory to a build folder::

	git clone --recursive https://github.com/mooch443/trex
	cd trex/Application
	mkdir build
	cd build
	
Now we have to generate the project files for the given platform and compiler. The required CMake command varies slightly depending on the operating system. Within the environment, go to the ``trex/Application/build`` repository (created in the previous step) and execute the compile script for your platform (on a Unix system ``../trex_build_unix.sh``, or on Windows ``../trex_build_windows.bat``) or execute cmake yourself with custom settings (have a look at the compile script for your platform for inspiration). You can also modify them, and add switches to the cmake commands.

Regarding switches, TRex offers a couple of additional options, with which you can decide to either compile libraries on your own or use existing ones in your system/environment path -- see next section.

The compile scripts will attempt to compile the software in Release mode. To compile in a different mode, simply run ``cmake --build . --config mode``. If compilation succeeds, you should now be able to run |trex| and |grabs| from the command-line, within the environment selected during compilation.

Special needs
=============

|trex| compilation using CMake offers switches to customize your build. Each of them can be appended to the above CMake command with ``-D<option>=<value>`` where ``value`` is usually either ``ON`` or ``OFF`` (unless it is a path, in which case it is a path). Below, we explain a couple of use-cases where these might come in handy -- but first, let's see a list of all CMake options available:

* **WITH_PYLON**: Activates Pylon compatibility, enabling support for machine vision cameras from Basler (using USB interfaces). We tested this with versions 5 and 6. See `Basler Pylon support`_ below.
* **WITH_FFMPEG**: Enabled by default, but can be forcibly turned off. This enables the streaming of MP4 video when recording from a camera in |grabs|. See `FFMPEG support`_.
* **WITH_HTTPD**: Disabled by default. Enables a web-server (see `Remote access`_ below).
* **TREX_BUILD_OPENCV**: If set to ``ON``, |trex| builds its own version of OpenCV with OpenCL support enabled, but otherwise limited features. Avoids using system provided binaries (or binaries in the conda environment) if enabled. See `Use an existing OpenCV distribution`_.
* **TREX_BUILD_ZIP**: Builds libzip and libz.
* **TREX_BUILD_PNG**: Builds libpng. If set to ``OFF``, then both libraries have to be provided in a way that CMake can find them.
* **TREX_BUILD_GLFW**: In order to display windows and graphics inside these windows, GLFW is required. You can use a custom build by enabling this option.
* **TREX_DONT_USE_PCH**: If you are getting errors from precompiled-headers, enable this option.
* **TREX_WITH_TESTS**: Build or don't build additional test executables.

Use an existing OpenCV distribution
-----------------------------------

|trex| likes to compile its own OpenCV distribution. However, you might want to use already existing OpenCV binaries to shorten compilation times, or specifically support a certain architecture. In this case, add the option ``-DTREX_COMPILE_OPENCV=OFF`` to your CMake command-line. You might need to specify ``-DOpenCV_DIR=/path/to/opencv`` in case the binaries are not in the global ``PATH``. After successful compilation, you may need to either append OpenCV's library path to the global ``PATH`` anyway -- or copy the shared library files to the correct location (beside trex' binary files).

Basler Pylon support
--------------------

In case you are planning to use |grabs| to record from Basler cameras directly, you have to compile the program with the additional option ``-DWITH_PYLON=ON``. Prior to this, you will also need to install the Basler Pylon SDK from their website at https://www.baslerweb.com/. We tested |trex| with version ``5.2.0``.

FFMPEG support
--------------

If you want to stream recorded videos directly to an MP4 container using |grabs|, then you need to enable FFMPEG support using ``-DWITH_FFMPEG=ON``.

Remote access
-------------

You might want to access the tracking software remotely. In case you have an exposed IP address that is accessible over the internet, you should not attempt this. However, if your computer is securely behind a firewall and only accessible via VPN, you can attach::

	cmake [...] -DWITH_HTTPD=ON

which enables HTTP support to |trex| and |grabs|. In order to successfully compile ``libmicrohttpd``, these additional libraries are needed to be available in ``PATH``::

	autoconf libtool automake

Now, whenever you start one of the programs, there will be a server accessible in your browser on port ``[IP]:8080`` on the computer |trex| is running on.
