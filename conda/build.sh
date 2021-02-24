#!/bin/bash

_BUILD_PREFIX=$(cat build_env_setup.sh | grep 'export BUILD=' | cut -d'=' -f2 | cut -d'"' -f2)

cd Application
mkdir build
cd build

declare -a CMAKE_PLATFORM_FLAGS
BUILD_GLFW="OFF"

if [ "$(uname)" == "Linux" ]; then
    # Fix up CMake for using conda's sysroot
    # See https://docs.conda.io/projects/conda-build/en/latest/resources/compiler-tools.html?highlight=cmake#an-aside-on-cmake-and-sysroots
    CMAKE_PLATFORM_FLAGS+=("-DCMAKE_TOOLCHAIN_FILE=${RECIPE_DIR}/conda_sysroot.cmake")
    CMAKE_PLATFORM_FLAGS+=("-DCMAKE_SYSTEM_PROCESSOR=x86_64")
    BUILD_GLFW="ON"

    # new 9.3 compiler sets this to cos7, which does not exist
    #export _PYTHON_SYSCONFIGDATA_NAME="_sysconfigdata_x86_64_conda_cos6_linux_gnu"
    #echo "ARCH = ${ARCH}"
    #echo "_BUILD_PREFIX: ${_BUILD_PREFIX} BUILD_PREFIX: ${BUILD_PREFIX} PREFIX: ${PREFIX}"
    #CMAKE_PLATFORM_FLAGS+=("-DCMAKE_PREFIX_PATH=${PREFIX}/${_BUILD_PREFIX}/sysroot/usr/lib64;${PREFIX}/${_BUILD_PREFIX}/sysroot/usr/include;${PREFIX}")
else
    echo "ARCH = ${ARCH}"
    echo "CONDA_BUILD_SYSROOT=${CONDA_BUILD_SYSROOT} SDKROOT=${SDKROOT} MACOSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET}. forcing it."
    if [ "${ARCH}" == "arm64" ]; then
        echo "Using up-to-date sysroot for arm64 arch."
        export CONDA_BUILD_SYSROOT="/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk"
    else
        ARCH="x86_64"
        if [ -z "${GITHUB_WORKFLOW}" ]; then
            echo "Detected GITHUB_WORKFLOW environment: ${GITHUB_WORKFLOW}"
            ls -la /Applications/Xcode*.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs
            export CONDA_BUILD_SYSROOT="/Applications/Xcode_11.4.1.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.15.sdk"
            export SDKROOT="${CONDA_BUILD_SYSROOT}"
            #export MACOSX_DEPLOYMENT_TARGET="10.12"
            CMAKE_PLATFORM_FLAGS+=("-DCMAKE_OSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET}")
        else
            export CONDA_BUILD_SYSROOT="/opt/MacOSX10.12.sdk"
            CMAKE_PLATFORM_FLAGS+=("-DCMAKE_OSX_DEPLOYMENT_TARGET=10.12")
        fi
    fi
    CMAKE_PLATFORM_FLAGS+=("-DCMAKE_OSX_SYSROOT=${CONDA_BUILD_SYSROOT}")
    CMAKE_PLATFORM_FLAGS+=("-DCMAKE_OSX_ARCHITECTURES=${ARCH}")
    BUILD_GLFW="ON"
fi

echo "Using system flags: ${CMAKE_PLATFORM_FLAGS[@]}"
PKG_CONFIG_PATH="${PREFIX}/lib/pkgconfig:${BUILD_PREFIX}/${HOST}/sysroot/usr/lib64/pkgconfig" cmake .. \
    -DPYTHON_INCLUDE_DIR:FILEPATH=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
    -DPYTHON_LIBRARY:FILEPATH=$(python3 ../find_library.py) \
    -DPYTHON_EXECUTABLE:FILEPATH=$(which python3) \
    -DCMAKE_BUILD_TYPE=Release \
    -DWITH_GITSHA1=ON \
    -DCMAKE_INSTALL_PREFIX=$PREFIX \
    -DWITH_PYLON=OFF \
    -DTREX_BUILD_OPENCV=ON \
    -DTREX_BUILD_GLFW=${BUILD_GLFW} \
    -DTREX_BUILD_ZLIB=ON \
    -DTREX_BUILD_PNG=ON \
    -DTREX_BUILD_ZIP=ON \
    -DTREX_CONDA_PACKAGE_INSTALL=ON \
    -DTREX_DONT_USE_PCH=ON \
    -DCMN_USE_OPENGL2=OFF \
    -DTREX_WITH_TESTS=OFF \
    -DCMAKE_PREFIX_PATH=$PREFIX \
    -DCMAKE_INSTALL_RPATH_USE_LINK_PATH:BOOL=TRUE \
    -DTREX_WITH_TESTS=OFF \
    -DCMAKE_C_COMPILER=$CC \
    -DCMAKE_CXX_COMPILER=$CXX \
    ${CMAKE_PLATFORM_FLAGS[@]}

if [ "$(uname)" == "Linux" ]; then
    make -j$(( $(nproc) - 1 )) Z_LIB
else
    make -j$(( $(sysctl -n hw.ncpu) - 1 )) Z_LIB
fi

if [ "$(uname)" == "Linux" ]; then
    make -j$(( $(nproc) - 1 )) CustomOpenCV
else
    make -j$(( $(sysctl -n hw.ncpu) - 1 )) CustomOpenCV
fi

cmake ..

if [ "$(uname)" == "Linux" ]; then
    make -j$(( $(nproc) - 1 )) && make install
else
    make -j$(( $(sysctl -n hw.ncpu) - 1 )) && make install
fi
