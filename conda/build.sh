#!/bin/bash

_BUILD_PREFIX=$(cat build_env_setup.sh | grep 'export BUILD=' | cut -d'=' -f2 | cut -d'"' -f2)

# --------------------------------------------------------------------------
# Inject git‑based version information so that meta.yaml still produces
# exactly the same version/build string it used to build with `os.popen`.
# --------------------------------------------------------------------------
if [ -z "${GIT_DESCRIBE_TAG}" ]; then
    # Get the most recent tag in the repository by commit date
    export GIT_DESCRIBE_TAG="$(git describe --tags $(git rev-list --tags --max-count=1) 2>/dev/null || echo vuntagged)"
fi

if [ -z "${TREX_DESCRIBE_TAG}" ]; then
    # Latest tag (equivalent to: git describe --tags --always --abbrev=0)
    export TREX_DESCRIBE_TAG="$(git describe --tags $(git rev-list --tags --max-count=1) 2>/dev/null || echo vuntagged)"
fi

echo "GIT_DESCRIBE_TAG=${GIT_DESCRIBE_TAG}"
echo "TREX_DESCRIBE_TAG=${TREX_DESCRIBE_TAG}"

if [ -z "${GIT_DESCRIBE_NUMBER}" ] || [ -z "${GIT_DESCRIBE_HASH}" ]; then
    # Full describe looks like: v1.2.3-4-gabcdef
    DESCRIBE_FULL="$(git describe --tags $(git rev-list --tags --max-count=1) 2>/dev/null || echo g0000000)"
    if [[ "${DESCRIBE_FULL}" == *-* ]]; then
        # Extract the "4" and the "gabcdef"
        export GIT_DESCRIBE_NUMBER="$(echo "${DESCRIBE_FULL}" | awk -F'-' '{print $(NF-1)}')"
        export GIT_DESCRIBE_HASH="$(echo "${DESCRIBE_FULL}"  | awk -F'-' '{print $NF}')"
    else
        # No tags present; treat the output as the hash
        export GIT_DESCRIBE_NUMBER="0"
        export GIT_DESCRIBE_HASH="${DESCRIBE_FULL}"
        echo "Warning: No tags found. Using hash as version."
    fi
fi

echo "GIT_DESCRIBE_NUMBER=${GIT_DESCRIBE_NUMBER}"
echo "GIT_DESCRIBE_HASH=${GIT_DESCRIBE_HASH}"
echo "TREX_DESCRIBE_NUMBER=${GIT_DESCRIBE_NUMBER}"

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
echo "CC=${CC}"
echo "CXX=${CXX}"

patch_ctime_header() {
  set -e

  # Ensure CONDA_PREFIX is set.
  if [ -z "$CONDA_PREFIX" ]; then
    echo "Error: CONDA_PREFIX is not set. Please activate your conda environment."
    return 1
  fi

  # Ensure CC is set.
  if [ -z "$CC" ]; then
    echo "Error: CC is not set. Please set CC to your C compiler (e.g., gcc or clang)."
    return 1
  fi

  # Use the compiler's verbose preprocessor output to list include directories.
  INCLUDE_DIRS=$("$CC" -x c++ -E -v - </dev/null 2>&1 | awk '
    /#include <\.\.\.> search starts here:/ {flag=1; next}
    /End of search list/ {flag=0}
    flag {print $1}
  ')

  echo "Compiler include directories found:"
  echo "$INCLUDE_DIRS"
  echo

  # Search the include directories for the <ctime> header file.
  CTIME_FILE=""
  for dir in $INCLUDE_DIRS; do
    if [ -f "$dir/ctime" ]; then
      CTIME_FILE="$dir/ctime"
      break
    fi
  done

  if [ -z "$CTIME_FILE" ]; then
    echo "Error: <ctime> header not found in any of the include directories."
    return 1
  fi

  echo "Found <ctime> file at: $CTIME_FILE"

  # Backup the original <ctime> file.
  BACKUP_FILE="${CTIME_FILE}.bak"
  cp "$CTIME_FILE" "$BACKUP_FILE"
  echo "Backup saved as $BACKUP_FILE"

  # Check if the header contains the pattern that enables timespec_get.
  if grep -q '#if __cplusplus >= 201703L && defined(_GLIBCXX_HAVE_TIMESPEC_GET)' "$CTIME_FILE"; then
    # Patch the file to disable that block.
    sed -i 's/#if __cplusplus >= 201703L && defined(_GLIBCXX_HAVE_TIMESPEC_GET)/#if 0  \/\/ patched to disable timespec_get block/' "$CTIME_FILE"
    echo "Patch applied successfully to <ctime>."
  else
    echo "Expected pattern not found in <ctime>."
    echo "Either the header is already updated or a different version is used."
    echo "Proceeding with the normal compilation procedure."
  fi
}

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

    patch_ctime_header

else
    SDKS=$(echo /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX*.*.sdk | sort | tail -n1)

    if [ "${ARCH}" == "arm64" ]; then
        echo "Using up-to-date sysroot for arm64 arch."
        export MACOSX_DEPLOYMENT_TARGET="11.0"

        export CONDA_BUILD_SYSROOT=$(ls -d $SDKS | tail -n1)
        export SDKROOT="${CONDA_BUILD_SYSROOT}"
        CMAKE_PLATFORM_FLAGS+=("-DCMAKE_OSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET}")
    else
        ARCH="x86_64"
        if [ ! -z ${GITHUB_WORKFLOW+x} ]; then
            echo "Detected GITHUB_WORKFLOW environment: ${GITHUB_WORKFLOW}"
            echo "CC=${CC} CXX=${CXX}"

            if [ ! -f ${CC} ]; then
                echo "Cannot find compiler ${CC}!"
            fi

            ls -la /Applications/Xcode*.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs
            export CONDA_BUILD_SYSROOT="/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX15.5.sdk"
            export SDKROOT="${CONDA_BUILD_SYSROOT}"
            export MACOSX_DEPLOYMENT_TARGET="11.0"
            CMAKE_PLATFORM_FLAGS+=("-DCMAKE_OSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET}")
        else
            echo "No GITHUB_WORKFLOW detected."
            export CONDA_BUILD_SYSROOT=$(ls -d $SDKS | tail -n1)
            export SDKROOT="${CONDA_BUILD_SYSROOT}"
            export MACOSX_DEPLOYMENT_TARGET="11.0"
            CMAKE_PLATFORM_FLAGS+=("-DCMAKE_OSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET}")
        fi
    fi
    CMAKE_PLATFORM_FLAGS+=("-DCMAKE_OSX_SYSROOT=${CONDA_BUILD_SYSROOT}")
    CMAKE_PLATFORM_FLAGS+=("-DCMAKE_OSX_ARCHITECTURES=${ARCH}")
    BUILD_GLFW="ON"
fi

echo "----------"

echo "GITHUB_WORKFLOW = ${GITHUB_WORKFLOW}"
echo "ARCH = ${ARCH}"
echo "CONDA_BUILD_SYSROOT=${CONDA_BUILD_SYSROOT}"
echo "SDKROOT=${SDKROOT}"
echo "MACOSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET}"
echo "CC=${CC}"
echo "CXX=${CXX}"

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
    -DWITH_FFMPEG=ON \
    -DCMAKE_INSTALL_PREFIX=$PREFIX \
    -DWITH_PYLON=OFF \
    -DCOMMONS_BUILD_OPENCV=ON \
    -DCOMMONS_BUILD_GLFW=${BUILD_GLFW} \
    -DCOMMONS_BUILD_ZLIB=ON \
    -DCOMMONS_BUILD_PNG=ON \
    -DCOMMONS_BUILD_ZIP=ON \
    -DCOMMONS_BUILD_EXAMPLES=OFF \
    -DTREX_CONDA_PACKAGE_INSTALL=ON \
    -DCOMMONS_CONDA_PACKAGE_INSTALL=ON \
    -DCOMMONS_DONT_USE_PCH=ON \
    -DCMN_USE_OPENGL2=OFF \
    -DTREX_WITH_TESTS=ON \
    -DCMAKE_PREFIX_PATH=$PREFIX \
    -DCMAKE_INSTALL_RPATH_USE_LINK_PATH:BOOL=TRUE \
    -DBUILD_LEGACY_TREX=OFF -DBUILD_LEGACY_TGRABS=OFF \
    -DCMAKE_C_COMPILER=$CC \
    -DCMAKE_CXX_COMPILER=$CXX \
    ${CMAKE_PLATFORM_FLAGS[@]}
    #-DPython_INCLUDE_DIRS:FILEPATH=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
    #-DPython_LIBRARIES:FILEPATH=$(python3 ../find_library.py) \

PROCS=8
if [ "$(uname)" == "Linux" ]; then
    if [ ! -z $(nproc) ]; then
        PROCS=$(( $(nproc) - 1 ))
    fi
    echo "Processors on Linux: ${PROCS}"
else
    if [ ! -z $(sysctl -n hw.ncpu) ]; then
        PROCS=$(( $(sysctl -n hw.ncpu) - 1 ))
    fi
    echo "Processors on macOS: $PROCS"
fi

echo "Choose processor number = ${PROCS}"

CMAKE_BUILD_PARALLEL_LEVEL=${PROCS} cmake --build . --target Z_LIB --parallel ${PROCS}
CMAKE_BUILD_PARALLEL_LEVEL=${PROCS} cmake --build . --target libzip --parallel ${PROCS}
CMAKE_BUILD_PARALLEL_LEVEL=${PROCS} cmake --build . --target libpng_custom --parallel ${PROCS}
CMAKE_BUILD_PARALLEL_LEVEL=${PROCS} cmake --build . --target CustomOpenCV --parallel ${PROCS}

if [ "$(uname)" == "Linux" ]; then
    CMAKE_BUILD_PARALLEL_LEVEL=${PROCS} cmake --build . --target gladex --parallel ${PROCS}
fi 
CMAKE_BUILD_PARALLEL_LEVEL=${PROCS} cmake --build . --target imgui --parallel ${PROCS}

cmake ..

CMAKE_BUILD_PARALLEL_LEVEL=${PROCS} cmake --build . --parallel ${PROCS} --target runAllTests --config Release

cmake .. -DTREX_WITH_TESTS=OFF
CMAKE_BUILD_PARALLEL_LEVEL=${PROCS} cmake --build . --parallel ${PROCS} && make install

echo "Build complete. Checking Git SHA1..."
if [ -f src/GitSHA1.cpp ]; then
    echo "Git SHA1 file exists."
    echo "Content:"
    cat src/GitSHA1.cpp
else
    echo "Git SHA1 file does not exist. Something went wrong."
fi

#make -j${PROCS} && make install
