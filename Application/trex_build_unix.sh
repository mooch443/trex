if [ ! $(which python3) ]; then
    echo "Python is not installed. Please install python3 first, or simply use a conda environment that provides python3."
    exit 1
else
    echo "Using python at '$(which python3)'..."
fi

if [ ! $(which cmake) ]; then
    echo "CMake >=3 is required to build this project."
    exit 1
fi

if [ ! $(which git) ]; then
    echo "You need to have git installed to build this project."
    exit 1
fi

git submodule update --recursive --init

IN_CONDA=$(printenv CONDA_PREFIX_1)
if [ ! $IN_CONDA ]; then
    IN_CONDA=${CONDA_PREFIX}
fi

if [ "$(uname)" == "Linux" ]; then
    echo "Setting up for Linux."
    echo ""
    
    CC=$(which gcc)
    CXX=$(which g++)

    if [ $(printenv CC) ]; then
        CC=$(printenv CC)
    fi
    if [ $(printenv CXX) ]; then
        CXX=$(printenv CXX)
    fi
    
    if [ ! $CC ]; then
        echo "No gcc compiler found. Please provide it in PATH or as a CC environment variable."
        exit 1
    fi
    
    if [ ! $CXX ]; then
        echo "No g++ compiler found. Please provide it in PATH or as a CXX environment variable."
        exit 1
    fi
    
    echo "Building GLFW, ZIP, and ZLIB - checking for OpenCV."

    if [ ${IN_CONDA} ]; then
        echo "**************************************"
        echo "Using conda environment $CONDA_PREFIX"
        echo "If you dont want this, please deactivate the conda environment first."
        echo "**************************************"
        
        CC=${CC} CXX=${CXX} PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig cmake .. \
            -DPYTHON_INCLUDE_DIR:FILEPATH=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
            -DPYTHON_LIBRARY:FILEPATH=$(python3 ../find_library.py) \
            -DPYTHON_EXECUTABLE:FILEPATH=$(which python3) \
            -DCMAKE_BUILD_TYPE=Release \
            -DWITH_FFMPEG=ON \
            -DCOMMONS_BUILD_ZLIB=ON \
            -DCOMMONS_BUILD_ZIP=ON \
            -DCOMMONS_BUILD_PNG=ON \
            -DCOMMONS_BUILD_OPENCV=ON \
            -DCMAKE_PREFIX_PATH="$CONDA_PREFIX;$CONDA_PREFIX/lib/pkgconfig;$CONDA_PREFIX/lib" \
            -DWITH_PYLON=ON
    else
        echo "**************************************"
        echo "Not in a conda environment."
        echo "Trying to build everything on my own."
        echo "If you wish to use a conda environment, please activate it first."
        echo "**************************************"
        
        echo "If you want to specify an FFMPEG path, please set the PKG_CONFIG_PATH environment variable accordingly."
        echo ""
        
        CC=${CC} CXX=${CXX} cmake .. \
            -DPYTHON_INCLUDE_DIR:FILEPATH=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
            -DPYTHON_LIBRARY:FILEPATH=$(python3 ../find_library.py) \
            -DPYTHON_EXECUTABLE:FILEPATH=$(which python3) \
            -DCMAKE_BUILD_TYPE=Release \
            -DWITH_FFMPEG=ON \
            -DCOMMONS_BUILD_ZLIB=ON \
            -DCOMMONS_BUILD_ZIP=ON \
            -DCOMMONS_BUILD_PNG=ON \
            -DCOMMONS_BUILD_OPENCV=ON \
            -DCMAKE_PREFIX_PATH="$PKG_CONFIG_PATH" \
            -DWITH_PYLON=ON
    fi
    
else
    echo "Setting up for macOS."
    echo ""
    echo "Building GLFW, ZIP, and ZLIB - checking for OpenCV."
    
    if [ ${IN_CONDA} ]; then
        echo "**************************************"
        echo "Using conda environment $CONDA_PREFIX"
        echo "If you dont want this, please deactivate the conda environment first."
        echo "**************************************"
        
        PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig cmake .. \
            -DPYTHON_INCLUDE_DIR:FILEPATH=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
            -DPYTHON_LIBRARY:FILEPATH=$(python3 ../find_library.py) \
            -DPYTHON_EXECUTABLE:FILEPATH=$(which python3) \
            -DCMAKE_BUILD_TYPE=Release  \
            -G Xcode \
            -DWITH_FFMPEG=ON \
            -DPYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
            -DCOMMONS_BUILD_ZLIB=ON \
            -DCOMMONS_BUILD_ZIP=ON \
            -DCOMMONS_BUILD_PNG=ON \
            -DCOMMONS_BUILD_OPENCV=ON \
            -DCMAKE_PREFIX_PATH="$CONDA_PREFIX;$CONDA_PREFIX/lib/pkgconfig;$CONDA_PREFIX/lib"
    else
        echo "**************************************"
        echo "Not in a conda environment."
        echo "Trying to build everything on my own."
        echo "If you wish to use a conda environment, please activate it first."
        echo "**************************************"
        
        echo "If you want to specify an FFMPEG path, please set the PKG_CONFIG_PATH environment variable accordingly."
        echo ""
        
        cmake .. \
            -DPYTHON_INCLUDE_DIR:FILEPATH=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
            -DPYTHON_LIBRARY:FILEPATH=$(python3 ../find_library.py) \
            -DPYTHON_EXECUTABLE:FILEPATH=$(which python3) \
            -DCMAKE_BUILD_TYPE=Release  \
            -DCOMMONS_BUILD_ZLIB=ON \
            -DCOMMONS_BUILD_ZIP=ON \
            -DCOMMONS_BUILD_PNG=ON \
            -DCOMMONS_BUILD_OPENCV=ON \
            -G Xcode \
            -DWITH_FFMPEG=ON
    fi
fi

# Determine OS and set NPROC appropriately
if [ "$(uname)" == "Darwin" ]; then
    # macOS
    NPROC=$(sysctl -n hw.ncpu)
elif [ "$(uname)" == "Linux" ]; then
    # Linux
    NPROC=$(nproc)
else
    echo "Unsupported operating system"
    exit 1
fi

echo "NPROC=$NPROC"

# Build targets with cmake
CMAKE_BUILD_PARALLEL_LEVEL=$NPROC cmake --build . --target Z_LIB --config Release --parallel ${NPROC}
CMAKE_BUILD_PARALLEL_LEVEL=$NPROC cmake --build . --target libzip --config Release --parallel ${NPROC}
CMAKE_BUILD_PARALLEL_LEVEL=$NPROC cmake --build . --target libpng_custom --config Release --parallel ${NPROC}
cmake ..
CMAKE_BUILD_PARALLEL_LEVEL=$NPROC cmake --build . --target CustomOpenCV --config Release --parallel ${NPROC}
cmake ..
CMAKE_BUILD_PARALLEL_LEVEL=$NPROC cmake --build . --config Release --parallel ${NPROC}
