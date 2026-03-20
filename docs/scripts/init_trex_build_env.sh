#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Initialize a fresh conda environment for building TRex.

Usage:
  docs/scripts/init_trex_build_env.sh [--env-name NAME] [--recreate] [--skip-post-link]

Options:
  --env-name NAME   Conda environment name to create or update (default: trex-build)
  --recreate        Remove the target environment before creating it again
  --skip-post-link  Do not run conda/post-link.sh after installing packages
  -h, --help        Show this help text

Notes:
  - On Linux, this installs the Conda OpenGL/X11 development packages needed for
    manual CMake builds. The CDT/sysroot dependencies used by conda-build are
    still resolved from conda/meta.yaml when you run `conda build`.
  - On macOS, this sets up the compiler and Python/build tooling, but you may
    still need to adjust the SDK settings described in docs/install.rst.
EOF
}

env_name="trex-build"
recreate=0
skip_post_link=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-name)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --env-name" >&2
                exit 1
            fi
            env_name="$2"
            shift 2
            ;;
        --recreate)
            recreate=1
            shift
            ;;
        --skip-post-link)
            skip_post_link=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if ! command -v conda >/dev/null 2>&1; then
    echo "conda was not found in PATH. Install Miniforge first:" >&2
    echo "  https://conda-forge.org/miniforge/" >&2
    exit 1
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../.." && pwd)
post_link_script="${repo_root}/conda/post-link.sh"

if [[ ! -f "${post_link_script}" ]]; then
    echo "Expected post-link script at ${post_link_script}, but it was not found." >&2
    exit 1
fi

os_name=$(uname -s)

common_packages=(
    python=3.11
    pip
    conda-build
    git
    cmake
    make
    pkg-config
    nasm
    icu
    ffmpeg
    numpy=1.26
    scikit-learn
    requests
    psutil
    c-compiler
    cxx-compiler
)

linux_packages=(
    sysroot_linux-64=2.17
    pthread-stubs
    libgl-devel
    libegl-devel
    libopengl-devel
    xorg-libx11
    xorg-libxext
    xorg-libxdamage
    xorg-libxxf86vm
    xorg-libxcursor
    xorg-libxcb
    xorg-libxfixes
    xorg-libxinerama
    xorg-libxrandr
    xorg-libxi
    xorg-libxrender
    xorg-libxau
    xorg-xorgproto
    libselinux
    libuuid
)

channels=(--override-channels -c conda-forge)

env_exists=0
if conda env list | awk '{print $1}' | grep -Fxq "${env_name}"; then
    env_exists=1
fi

if [[ ${recreate} -eq 1 && ${env_exists} -eq 1 ]]; then
    echo "Removing existing environment ${env_name}..."
    conda env remove -n "${env_name}" -y
    env_exists=0
fi

packages=("${common_packages[@]}")
if [[ "${os_name}" == "Linux" ]]; then
    packages+=("${linux_packages[@]}")
elif [[ "${os_name}" != "Darwin" ]]; then
    echo "Unsupported platform: ${os_name}" >&2
    exit 1
fi

if [[ ${env_exists} -eq 0 ]]; then
    echo "Creating environment ${env_name}..."
    conda create -n "${env_name}" -y "${channels[@]}" "${packages[@]}"
else
    echo "Updating environment ${env_name}..."
    conda install -n "${env_name}" -y "${channels[@]}" "${packages[@]}"
fi

if [[ ${skip_post_link} -eq 0 ]]; then
    echo "Running TRex post-link setup inside ${env_name}..."
    env REPO_ROOT_FOR_TREX="${repo_root}" conda run -n "${env_name}" bash -lc 'export PREFIX="$CONDA_PREFIX"; bash "$REPO_ROOT_FOR_TREX/conda/post-link.sh"'
fi

cat <<EOF

TRex build environment is ready.

Next steps:
  conda activate ${env_name}
  cd ${repo_root}/Application
  mkdir -p build && cd build
  ../trex_build_unix.sh

For a local package build instead:
  conda activate ${env_name}
  cd ${repo_root}/conda
  conda build .
EOF