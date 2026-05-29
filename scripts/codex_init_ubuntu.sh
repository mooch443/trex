#!/usr/bin/env bash
set -euo pipefail

# Bootstrap a fresh TRex clone inside an Ubuntu VM/container for Codex work.
# This is a manual CMake path (not conda-build).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="${ROOT_DIR}/Application"
BUILD_DIR="${APP_DIR}/build-codex-debug"
VENV_DIR="${ROOT_DIR}/.codex-venv"

INSTALL_SYSTEM_DEPS=1
INSTALL_POST_LINK_PYTHON=1
USE_VENV=1
PYTHON_CMD="python3"
LIGHTWEIGHT_MODE=0

echo "[info] root: ${ROOT_DIR}"

usage() {
    cat <<'EOF'
Usage: scripts/codex_init_ubuntu.sh [options]

Options:
  --skip-system-deps       Do not install apt packages.
  --skip-post-link-python  Do not install post-link Python extras.
    --lightweight            Install only lightweight Python extras.
    --full-post-link         Force full post-link Python extras.
  --no-venv                Use system Python directly (no .codex-venv).
  --python <exe>           Python executable to use (default: python3).
  --venv-dir <path>        Custom venv directory path.
  -h, --help               Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-system-deps)
            INSTALL_SYSTEM_DEPS=0
            ;;
        --skip-post-link-python)
            INSTALL_POST_LINK_PYTHON=0
            ;;
        --lightweight)
            LIGHTWEIGHT_MODE=1
            ;;
        --full-post-link)
            LIGHTWEIGHT_MODE=0
            ;;
        --no-venv)
            USE_VENV=0
            ;;
        --python)
            shift
            PYTHON_CMD="${1:-}"
            ;;
        --venv-dir)
            shift
            VENV_DIR="${1:-}"
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[error] unknown option: $1"
            usage
            exit 1
            ;;
    esac
    shift
done

require_cmd() {
    local cmd="$1"
    if ! command -v "${cmd}" >/dev/null 2>&1; then
        echo "[error] missing required command: ${cmd}"
        exit 1
    fi
}

require_cmd git
require_cmd cmake

install_system_deps() {
    if [ "${INSTALL_SYSTEM_DEPS}" -eq 0 ]; then
        echo "[step] skipping apt dependency installation"
        return 0
    fi

    if ! command -v apt-get >/dev/null 2>&1; then
        echo "[warn] apt-get not found; cannot auto-install Ubuntu dependencies"
        return 0
    fi

    local sudo_cmd=""
    if [ "$(id -u)" -ne 0 ]; then
        if command -v sudo >/dev/null 2>&1; then
            sudo_cmd="sudo"
        else
            echo "[error] need root or sudo to install apt dependencies"
            exit 1
        fi
    fi

    echo "[step] install Ubuntu build/host deps inspired by conda/meta.yaml"
    ${sudo_cmd} apt-get update
    ${sudo_cmd} apt-get install -y \
        build-essential \
        cmake \
        pkg-config \
        git \
        make \
        gcc \
        g++ \
        nasm \
        libicu-dev \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        python3-numpy \
        ffmpeg \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libswscale-dev \
        libswresample-dev \
        libvorbis-dev \
        libogg-dev \
        libavif-dev \
        libgl1-mesa-dev \
        libx11-dev \
        libxext-dev \
        libxrandr-dev \
        libxinerama-dev \
        libxcursor-dev \
        libxi-dev \
        libxfixes-dev \
        libxdamage-dev \
        libxxf86vm-dev \
        libxrender-dev \
        libxcb1-dev \
        libxau-dev \
        libselinux1-dev
}

setup_python() {
    require_cmd "${PYTHON_CMD}"

    if [ "${USE_VENV}" -eq 1 ]; then
        echo "[step] create/use local venv at ${VENV_DIR}"
        "${PYTHON_CMD}" -m venv "${VENV_DIR}"
        # shellcheck disable=SC1090
        source "${VENV_DIR}/bin/activate"
        python -m pip install --upgrade pip setuptools wheel
    fi

    PYTHON_BIN="$(command -v python)"
    if [ -z "${PYTHON_BIN}" ]; then
        PYTHON_BIN="$(command -v "${PYTHON_CMD}")"
    fi

    if [ -z "${PYTHON_BIN}" ]; then
        echo "[error] failed to resolve a Python executable"
        exit 1
    fi

    echo "[info] using python: ${PYTHON_BIN}"
}

install_post_link_python() {
    if [ "${INSTALL_POST_LINK_PYTHON}" -eq 0 ]; then
        echo "[step] skipping post-link Python extras"
        return 0
    fi

    if [ "${LIGHTWEIGHT_MODE}" -eq 1 ]; then
        echo "[step] install lightweight Python extras"
        ${PYTHON_BIN} -m pip install \
            numpy \
            requests \
            psutil \
            scikit-learn
        return 0
    fi

    echo "[step] install full Python extras aligned with conda/post-link.sh"
    local numpy_ver
    numpy_ver="$(${PYTHON_BIN} -c 'import numpy; print(numpy.__version__)' 2>/dev/null || true)"

    local packages=(
        "torch>=2.0.0,<2.9.0"
        "torchvision>=0.15.1,<0.24.0"
        "torchmetrics"
        "tqdm"
        "opencv-python>=4,<5"
        "ultralytics>=8.3.0,<9"
        "dill"
    )

    if [ -n "${numpy_ver}" ]; then
        packages+=("numpy==${numpy_ver}")
    fi

    ${PYTHON_BIN} -m pip install "${packages[@]}"
}

resolve_python_cmake_vars() {
    mapfile -t _pyvals < <(${PYTHON_BIN} - <<'PY'
import glob
import os
import sysconfig

inc = sysconfig.get_paths().get("include") or sysconfig.get_config_var("INCLUDEPY")
libdir = sysconfig.get_config_var("LIBDIR") or ""
ldlibrary = sysconfig.get_config_var("LDLIBRARY") or sysconfig.get_config_var("LIBRARY") or ""

candidates = []
if libdir and ldlibrary:
    candidates.append(os.path.join(libdir, ldlibrary))
if libdir:
    candidates.extend(sorted(glob.glob(os.path.join(libdir, "libpython*.so*"))))

lib = ""
for candidate in candidates:
    if candidate and os.path.exists(candidate):
        lib = candidate
        break

print(inc or "")
print(lib)
PY
    )

    PY_INCLUDE="${_pyvals[0]:-}"
    PY_LIBRARY="${_pyvals[1]:-}"

    if [ -z "${PY_INCLUDE}" ]; then
        echo "[error] failed to resolve Python include dir"
        exit 1
    fi

    if [ -z "${PY_LIBRARY}" ]; then
        echo "[warn] failed to resolve full libpython path; CMake may still detect Python automatically"
    fi
}

configure_cmake() {
    local cmake_args=(
        -S "${APP_DIR}"
        -B "${BUILD_DIR}"
        -DPYTHON_INCLUDE_DIR:FILEPATH="${PY_INCLUDE}"
        -DPYTHON_EXECUTABLE:FILEPATH="${PYTHON_BIN}"
        -DCMAKE_BUILD_TYPE=Debug
        -DWITH_FFMPEG=ON
        -DCOMMONS_BUILD_ZLIB=ON
        -DCOMMONS_BUILD_ZIP=ON
        -DCOMMONS_BUILD_PNG=ON
        -DTREX_WITH_TESTS=ON
        -DCOMMONS_BUILD_OPENCV=ON
        -DWITH_PYLON=OFF
    )

    if [ -n "${PY_LIBRARY}" ]; then
        cmake_args+=("-DPYTHON_LIBRARY:FILEPATH=${PY_LIBRARY}")
    fi

    cmake "${cmake_args[@]}"
}

cd "${ROOT_DIR}"

install_system_deps

echo "[step] sync git submodules"
git submodule update --init --recursive

setup_python
install_post_link_python

if command -v nproc >/dev/null 2>&1; then
    NPROC="$(nproc)"
else
    NPROC="2"
fi

if [ "${NPROC}" -gt 1 ]; then
    PARALLEL="$((NPROC - 1))"
else
    PARALLEL="1"
fi

echo "[info] parallel jobs: ${PARALLEL}"
resolve_python_cmake_vars

mkdir -p "${BUILD_DIR}"

echo "[step] configure Debug build"
configure_cmake

echo "[step] build dependency targets first (same principle as trex_build_unix.sh)"
cmake --build "${BUILD_DIR}" --target Z_LIB --config Debug --parallel "${PARALLEL}"
cmake --build "${BUILD_DIR}" --target libzip --config Debug --parallel "${PARALLEL}"
cmake --build "${BUILD_DIR}" --target libpng_custom --config Debug --parallel "${PARALLEL}"

echo "[step] refresh configure and build OpenCV target"
configure_cmake
cmake --build "${BUILD_DIR}" --target CustomOpenCV --config Debug --parallel "${PARALLEL}"

echo "[step] final Debug build"
configure_cmake
cmake --build "${BUILD_DIR}" --config Debug --parallel "${PARALLEL}"

echo "[done] Debug build initialized successfully"
echo "[done] build directory: ${BUILD_DIR}"