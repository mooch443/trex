#!/bin/bash

_BUILD_PREFIX=$(cat build_env_setup.sh | grep 'export BUILD=' | cut -d'=' -f2 | cut -d'"' -f2)

cd Application
mkdir build
cd build

declare -a CMAKE_PLATFORM_FLAGS
BUILD_GLFW="OFF"
echo "GITHUB_WORKFLOW = ${GITHUB_WORKFLOW}"
echo "ARCH = ${ARCH}"
echo "CONDA_BUILD_SYSROOT=${CONDA_BUILD_SYSROOT}"
echo "SDKROOT=${SDKROOT}"
echo "MACOSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET}"

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
    if [ "${ARCH}" == "arm64" ]; then
        echo "Using up-to-date sysroot for arm64 arch."
        export MACOSX_DEPLOYMENT_TARGET="11.0"

        export CONDA_BUILD_SYSROOT=$(ls -d /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX13.1.sdk | tail -n1)
        export SDKROOT="${CONDA_BUILD_SYSROOT}"
        CMAKE_PLATFORM_FLAGS+=("-DCMAKE_OSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET}")
    else
        ARCH="x86_64"
        if [ ! -z ${GITHUB_WORKFLOW+x} ]; then
            echo "Detected GITHUB_WORKFLOW environment: ${GITHUB_WORKFLOW}"
            ls -la /Applications/Xcode*.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs
            export CONDA_BUILD_SYSROOT="/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX13.1.sdk"
            export SDKROOT="${CONDA_BUILD_SYSROOT}"
            export MACOSX_DEPLOYMENT_TARGET="11.0"
            CMAKE_PLATFORM_FLAGS+=("-DCMAKE_OSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET}")
        else
            echo "No GITHUB_WORKFLOW detected."
            #export CONDA_BUILD_SYSROOT=$(ls -d /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX12.3.sdk | tail -n1)
            export CONDA_BUILD_SYSROOT="/opt/MacOSX10.14.sdk"
            export MACOSX_DEPLOYMENT_TARGET="10.14"
            export SDKROOT="${CONDA_BUILD_SYSROOT}"
            CMAKE_PLATFORM_FLAGS+=("-DCMAKE_OSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET}")
            export CXXFLAGS="${CXXFLAGS} -isysroot ${CONDA_BUILD_SYSROOT}"  # [osx]
            export CFLAGS="${CFLAGS} -isysroot ${CONDA_BUILD_SYSROOT}"  # [osx]
        fi
    fi
    CMAKE_PLATFORM_FLAGS+=("-DCMAKE_OSX_SYSROOT=${CONDA_BUILD_SYSROOT}")
    CMAKE_PLATFORM_FLAGS+=("-DCMAKE_OSX_ARCHITECTURES=${ARCH}")
    BUILD_GLFW="ON"
fi

export PKG_CONFIG_PATH="${PREFIX}/lib/pkgconfig:${BUILD_PREFIX}/${HOST}/sysroot/usr/lib64/pkgconfig:${PKG_CONFIG_PATH}"
export PKG_CONFIG_LIBDIR="${PKG_CONFIG_PATH}"
echo "PKG_CONFIG_PATH=${PKG_CONFIG_PATH}"

echo "Using system flags: ${CMAKE_PLATFORM_FLAGS[@]}"
cmake .. \
    -DPython_EXECUTABLE:FILEPATH=${PREFIX}/bin/python3 \
    -DPython_ROOT_DIR:FILEPATH=${PREFIX} \
    -DCONDA_PREFIX:PATH=${PREFIX} \
    -DCMAKE_BUILD_TYPE=Release \
    -DWITH_GITSHA1=ON \
    -DCMAKE_INSTALL_PREFIX=$PREFIX \
    -DWITH_PYLON=OFF \
    -DCOMMONS_BUILD_OPENCV=ON \
    -DCOMMONS_BUILD_GLFW=${BUILD_GLFW} \
    -DCOMMONS_BUILD_ZLIB=ON \
    -DCOMMONS_BUILD_PNG=ON \
    -DCOMMONS_BUILD_ZIP=ON \
    -DTREX_CONDA_PACKAGE_INSTALL=ON \
    -DCOMMONS_CONDA_PACKAGE_INSTALL=ON \
    -DCOMMONS_DONT_USE_PCH=ON \
    -DCMN_USE_OPENGL2=OFF \
    -DTREX_WITH_TESTS=OFF \
    -DCMAKE_PREFIX_PATH=$PREFIX \
    -DCMAKE_INSTALL_RPATH_USE_LINK_PATH:BOOL=TRUE \
    -DCMAKE_C_COMPILER=$CC \
    -DCMAKE_CXX_COMPILER=$CXX \
    ${CMAKE_PLATFORM_FLAGS[@]}
    #-DPython_INCLUDE_DIRS:FILEPATH=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
    #-DPython_LIBRARIES:FILEPATH=$(python3 ../find_library.py) \
if [ "$(uname)" == "Linux" ]; then
    make -j$(( $(nproc) - 1 )) Z_LIB
else
    make -j$(( $(sysctl -n hw.ncpu) - 1 )) Z_LIB
fi

if [ "$(uname)" == "Linux" ]; then
    make -j$(( $(nproc) - 1 )) libzip
else
    make -j$(( $(sysctl -n hw.ncpu) - 1 )) libzip
fi  

if [ "$(uname)" == "Linux" ]; then
    make -j$(( $(nproc) - 1 )) libpng_custom
else
    make -j$(( $(sysctl -n hw.ncpu) - 1 )) libpng_custom
fi  

if [ "$(uname)" == "Linux" ]; then
    make -j$(( $(nproc) - 1 )) CustomOpenCV
else
    make -j$(( $(sysctl -n hw.ncpu) - 1 )) CustomOpenCV
fi

if [ "$(uname)" == "Linux" ]; then
    make -j$(( $(nproc) - 1 )) gladex
fi

if [ "$(uname)" == "Linux" ]; then
    make -j$(( $(nproc) - 1 )) imgui
else
    make -j$(( $(sysctl -n hw.ncpu) - 1 )) imgui
fi

cmake ..

if [ "$(uname)" == "Linux" ]; then
    make -j$(( $(nproc) - 1 )) && make install
else
    make -j$(( $(sysctl -n hw.ncpu) - 1 )) && make install
fi
