.. include:: names.rst

.. toctree::
   :maxdepth: 2

.. WARNING::
	On Windows and Linux a NVIDIA graphics-card is required for hardware accelerated machine learning. On macOS (Apple Silicone), the Metal backend is used for hardware acceleration.

Installation
############

The easy way (Windows, Linux and Intel macOS)
*********************************************

|trex| supports all major platforms. There is an easy way to install |trex| using Anaconda, by creating a new virtual environment (here named ``tracking``, but you can name it whatever you like) and installing |trex| into it:

:green:`# macOS (Apple Silicone)`::

	conda create -n tracking --override-channels -c trex-beta -c pytorch -c conda-forge trex

:green:`# macOS (Intel)`::

	conda create -n tracking --override-channels -c trex-beta -c pytorch -c local trex

:green:`# Windows, Linux`::

	conda create -n tracking --override-channels -c trex-beta -c pytorch -c nvidia -c defaults trex

After the installation is complete, you can activate the environment and run |trex| using the following commands::

	conda activate tracking
	trex

The down-side of installing through conda may be that pre-built binaries are compiled with fewer optimzations and features than a manually compiled one (due to compatibility and licensing issues) and thus are slightly slower =(. For example, the conda version does not offer support for Basler cameras. If you need to use machine vision cameras, or need as much speed as possible/the newest version, please consider compiling the software yourself. 

.. NOTE::
	
	You can also check out our beta channel for the latest features and bug fixes (but maybe also new bugs). Just replace the ``-c trexing`` with ``-c trex-beta`` in the above command.

Compile it yourself
*******************

There are two ways to get your own version of |trex|:

* creating a local conda channel, and installing from there
* running CMake/build manually with customized options

Both are obviously similar in result, but there *are* differences (the local channel is essentially a script for the manual procedure, with some caveats). For example, the conda build is limited to certain compiler and OS-SDK versions -- which is something that you may want to change in order to enable certain optimizations. There is also no straight-forward way to add options like enabling Pylon support, for which you'd have to go the manual way described after the next section. We start out here by describing the more rigid (but more automated) way using conda, followed by a description of how to do everything manually.

Local conda channel
===================

In order to get your own (local) conda channel, all you need to do is make sure you have conda installed, as well as the ``conda-build`` package. This is a package that allows you to make your own packages locally (use ``conda deactivate``, until it says ``base`` on the left). Now you should probably create a new build environment first, keeping your base environment clean::

	conda create -n build conda-build git python
	conda activate build

Once this is done, you can clone the |trex| repository and change your directory to the ``conda`` folder::

	git clone --recursive https://github.com/mooch443/trex
	cd trex/conda

Next, make sure you have Visual Studio 2019 installed (yes, this is an older version that you need to download from Microsoft's archive), or Xcode on macOS. Linux should work out of the box, and if not you could try to install `build-essential` first.

Finally, you can build the package using::

	./build_conda_package.bat # Windows
	./build_conda_package.sh  # Linux, macOS

This runs ``conda build .`` (+ possibly additional arguments for the channels, as in the ``conda create`` command), which builds the program according to all the settings inside ``meta.yaml`` (for dependencies), using ``build.sh`` (or ``bld.bat`` on Windows) to configure CMake. If you want to enable/disable certain features (e.g. use a locally installed OpenCV library, enable the Pylon SDK, etc.) this build script is the place where you can do that.

.. NOTE::
	Note that if you want to add Pylon SDKs etc., you may need to add absolute paths to the cmake call (e.g. adding folders to ``CMAKE_PREFIX_PATH``) so that it can find all your locally installed libraries -- in which case your conda package will probably not be portable.

After compilation was successful, |trex| can be installed using:

:green:`# macOS (Apple Silicone)`::

	conda create -n tracking --override-channels -c local -c pytorch -c conda-forge trex

:green:`# macOS (Intel)`::

	conda create -n tracking --override-channels -c local -c pytorch -c defaults trex

:green:`# Windows, Linux`::

	conda create -n tracking --override-channels -c local -c pytorch -c defaults trex

Notice there is a ``-c local``, instead of the ``-c trexing`` from the first section.

Finally, to run it simply switch to the environment you just created (tracking) using ``conda activate tracking`` and run ``trex`` to see if the window appears!

Compiling manually (TODO)
=========================

.. WARNING::
	Please note that the below instructions may not be up-to-date yet. I am working on it and will update them as soon as possible.

First, make sure that you fulfill the platform-specific requirements:

* **Windows**: Please make sure you have Visual Studio installed on your computer. It can be downloaded for free from https://visualstudio.microsoft.com. We have tested Visual Studio versions 2019 up to 2022. We are using the Anaconda PowerShell here in our examples.
* **MacOS**: Make sure you have Xcode and the Xcode compiler tools installed. They can be downloaded for free from the App Store (Xcode includes the compiler tools). We used ``macOS >=10.15`` and ``Xcode >=11.5``.
* **Linux**: You should have build-essential installed, as well as ``g++ >=11`` or a different compiler with C++23 support.

As well as the general requirements:

* **Python**: We use version ``>= 3.7,<4``.
* **CMake**: Version ``>= 3.22``.

.. NOTE::
	We will be using Anaconda here. However, it is not *required* to use Anaconda when compiling |trex| -- it is just a straight-forward way to obtain dependencies. In case you do not want to use Anaconda, please make sure that all mentioned dependencies are installed in a way that can be detected by CMake. You may also add necessary paths to the CMake command-line, such as ``-DOpenCV_DIR=/path/to/opencv`` and use switches to compile certain libraries (such as OpenCV) statically with |trex|.

The easiest way to ensure that all requirements are met, is by using conda to create a new environment::

	# Windows
	conda create -n trex git cmake ffmpeg tensorflow=2
	
	# Linux (minila)
	conda create -n trex git cmake ffmpeg=4 tensorflow=2 cxx-compiler c-compiler

	# Linux (graphics) - if compilation is missing graphics driver things, try recreating the environment like this and start over:
	conda create -n trex gcc git cmake ffmpeg=4 tensorflow=2 cxx-compiler c-compiler mesa-libgl-devel-cos6-x86_64 libxdamage-devel-cos6-x86_64 libxi-devel-cos6-x86_64 libxxf86vm-cos6-x86_64 libselinux-devel-cos6-x86_64 libuuid-devel-cos6-x86_64 mesa-libgl-devel-cos6-x86_64

	# on linux you may also need this, so that you don't need to set LD_LIBRARY_PATH every time you want to run trex:
	conda activate trex
	conda install -c conda-forge gcc pkg-config libxcursor-devel-cos6-x86_64 libxrender-devel-cos6-x86_64 libx11-devel-cos6-x86_64 libXfixes-devel-cos6-x86_64 libxcb-cos6-x86_64 libxrandr-devel-cos6-x86_64 libxi-devel-cos6-x86_64 libXfixes-devel-cos6-x86_64 libXxf86vm-devel-cos6-x86_64 xorg-x11-proto-devel-cos6-x86_64 libxext-devel-cos6-x86_64 libxdamage-devel-cos6-x86_64 libxinerama-devel-cos6-x86_64 libselinux-cos6-x86_64 libXau-devel-cos6-x86_64 libuuid-devel-cos6-x86_64 libdc1394


	conda create -n track -c pytorch-nightly -c nvidia pytorch-cuda=11.7 torchvision torchaudio cmake ffmpeg=4 git scikit-learn requests python 'tensorflow-gpu>=2.4,<3' pip pandas seaborn 'numpy=1.19'

If your GPU is supported by TensorFlow, you can modify the above line by appending ``-gpu`` to ``tensorflow`` to get ``tensorflow-gpu=2``.
	
Next, switch to the conda environment using::

	conda activate trex

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
