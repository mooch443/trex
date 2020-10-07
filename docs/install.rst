.. include:: names.rst

.. toctree::
   :maxdepth: 2

.. WARNING::
	There are currently no GPU-enabled builds of tensorflow available for MacOS, so network training can only be accelerated by a GPU on Windows and Linux, given a NVIDIA graphics-card. Training still works on other systems, it's just slower.

Installation
############

The easy way
************

|trex| supports all major platforms. There is an easy way to install |trex| using Anaconda, by creating a new virtual environment (here named ``tracking``, which you can replace)::

	conda create -n tracking -c conda-forge -c trexing trex

The conda version does not offer support for Basler cameras. If you need to use |grabs| with machine vision cameras, please consider compiling the software yourself -- it has other advantages, too (such as enabling some Metal features on macOS and getting a squeaky new version)!

Compile it yourself
*******************

There are two ways to get your own version of |trex|:

* creating a local conda channel
* running CMake/build manually

Both are obviously connected (the local conda channel is essentially a script for the manual procedure), but there are differences. For example, the conda build is limited to certain compiler and OS-SDK versions -- which is something that you may want to change in order to enable certain OS features. We start out here by describing the easier way, followed by a description of how to do everything manually.

Local conda channel
===================

In order to get your own conda channel, all you need to do is make sure you have Anaconda installed, as well as the ``conda-build`` package. This is a package that allows you to make your own packages from within the base environment. It creates a virtual environment, within which it compiles/tests the software you are trying to build. You can install it using::

	conda install conda-build

After that, clone the |trex| repository using::

	git clone --recursive https://github.com/mooch443/trex
	cd FishTracker/conda

Now, from within that folder, run::

	conda build . -c conda-forge

This builds the program with all the settings inside ``meta.yaml`` (for dependencies) as well as ``build.sh`` (or ``bld.bat`` on Windows) for the CMake settings. If you want to enable/disable certain features (e.g. use the OpenCV from within the conda environment, etc.) the build script is the place where you can do that.

Compiling manually
==================

First, make sure that you fulfill the platform-specific requirements:

* **Windows**: Please make sure you have Visual Studio installed on your computer. It can be downloaded for free from https://visualstudio.microsoft.com. We have tested Visual Studio versions 2017 and 2019. We are using the Anaconda PowerShell here in our examples.
* **MacOS**: Make sure you have Xcode and the Xcode compiler tools installed. They can be downloaded for free from the App Store (Xcode includes the compiler tools). We used macOS 10.15 and Xcode 11.5.
* **Linux**: You should have build-essential installed, as well as ``g++ >=8`` or a different compiler with full C++17 support.

As well as the general requirements:

* **Python**: We use version ``3.6.7``.
* **CMake**: Version ``>= 3.16``.

.. NOTE::
	We will be using Anaconda here. However, it is not *required* to use Anaconda when compiling |trex| -- it is just a straight-forward way to obtain all required dependencies. In case you do not want to use Anaconda, please make sure that all mentioned dependencies are installed in a way that can be detected by CMake. You may also add necessary paths to the CMake command-line, such as ``-DOpenCV_DIR=/path/to/opencv`` and use switches to compile certain libraries (such as OpenCV) statically with |trex|.

The easiest way to ensure that all requirements are met, is by using conda to create a new environment::

	conda create -n tracking -c conda-forge cmake ffmpeg tensorflow=1.13 keras
	
under Linux, you may have to add the ``gtk2`` package as well.

If your GPU is supported by TensorFlow, you can modify the above line by appending ``-gpu`` to ``tensorflow`` to get ``tensorflow-gpu=1.13``.

Notice that we are omitting some runtime dependencies here, which will be added at the end of this section. If we (for example) install OpenCV for python *now*, then we might run into issues -- especially on Windows -- since the custom OpenCV version might interfere destructively with the existing version.
	
Next, switch to the conda environment using::

	conda activate tracking

You can now clone the repository and change your directory to a build folder::

	git clone --recursive https://github.com/mooch443/trex
	cd FishTracker/Application
	mkdir build
	cd build
	
Now we have to generate the project files for the given platform and compiler. The following CMake command varies slightly depending on the operating system. Within the environment, go to the ``FishTracker/Application/build`` repository (created previously) and execute:

* **Windows**::

	cmake .. -DPYTHON_INCLUDE_DIR:FILEPATH=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") `
	 -DPYTHON_LIBRARY:FILEPATH=$(python ../find_library.py) `
	 -DPYTHON_EXECUTABLE:FILEPATH=$(Get-Command python | Select-Object -ExpandProperty Definition) `
	 -DCMAKE_BUILD_TYPE=Release `
	 -DTREX_BUILD_OPENCV=ON `
	 -DTREX_BUILD_GLFW=ON `
	 -DWITH_FFMPEG=ON `
	 -DCMAKE_PREFIX_PATH=$CONDA_PREFIX

* **Linux**::

	CC=/usr/bin/gcc CXX=/usr/bin/g++ PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig \
	cmake .. -DPYTHON_INCLUDE_DIR:FILEPATH=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
	 -DPYTHON_LIBRARY:FILEPATH=$(python3 ../find_library.py) \
	 -DPYTHON_EXECUTABLE:FILEPATH=$(which python3) \
	 -DCMAKE_BUILD_TYPE=Release \
	 -DTREX_BUILD_OPENCV=ON \
	 -DTREX_BUILD_GLFW=ON \
	 -DWITH_FFMPEG=ON \
	 -DCMAKE_PREFIX_PATH=$CONDA_PREFIX
	 
.. NOTE::
	Under Linux, you might also have to install the OpenGL library, and further Xorg dependencies. For example::
	
		conda install -c conda-forge xorg-libxinerama xorg-libxcursor \
					xorg-libxi xorg-libxrandr xorg-libxdamage libxxf86vm-cos6-x86_64 \
					libselinux-cos6-x86_64 mesa-dri-drivers-cos6-x86_64
	 
.. 
	NOTE:: 
	Under Linux, you might also have to install the OpenGL library and Xorg dependencies -- depending on your platform. For example:: 
..	
		conda install mesa-libgl-devel-cos6-x86_64 xorg-libxinerama xorg-libxcursor \
			xorg-libxi xorg-libxrandr xorg-libxdamage libxxf86vm-cos6-x86_64 \
			libselinux-cos6-x86_64 mesa-dri-drivers-cos6-x86_64
	
* **macOS**::

	PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig \
	cmake .. -DPYTHON_INCLUDE_DIR:FILEPATH=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
	 -DPYTHON_LIBRARY:FILEPATH=$(python3 ../find_library.py) \
	 -DPYTHON_EXECUTABLE:FILEPATH=$(which python3) \
	 -DCMAKE_BUILD_TYPE=Release \
	 -G Xcode \
	 -DTREX_BUILD_OPENCV=ON \
	 -DTREX_BUILD_GLFW=ON \
	 -DWITH_FFMPEG=ON \
	 -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
	 -DPYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")

The CMake list offers a couple of options, with which you can decide to either compile libraries on your own or use existing ones in your system/environment path -- see next section.

Now that your project files are set-up, execute these commands in order (for Unix systems)::

	cmake --build . --target CustomOpenCV --config Release \
	 && cmake .. \
	 && cmake --build . --config Release

or in PowerShell::

	cmake --build . --target CustomOpenCV --config Release `
	 -and cmake .. `
	 -and cmake --build . --config Release
	 
To be able to run |trex|, you will need to install additional python dependencies, which have been omitted previously to avoid linking to the wrong libraries::

	conda install -c conda-forge matplotlib pillow opencv

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
