#!/usr/bin/env bash

set -euo pipefail

if [[ "$(uname)" != "Darwin" ]]; then
    echo "This script is only for macOS."
    exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "Conda is required."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

ENV_NAME="${TREX_MODULES_ENV_NAME:-trex-modules}"
BUILD_DIR="${TREX_MODULES_BUILD_DIR:-${SCRIPT_DIR}/tmp-modules-osx}"
DEPLOY_TARGET="${MACOSX_DEPLOYMENT_TARGET:-26.0}"

CREATE_ENV_CMD="conda create -y -n ${ENV_NAME} --clone trex"
INSTALL_TOOLS_CMD="conda install -y -n ${ENV_NAME} -c conda-forge clang_osx-arm64=19.1.7 clangxx_osx-arm64=19.1.7 clang-tools=19.1.7"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "Conda environment '${ENV_NAME}' does not exist."
    echo "Create it with:"
    echo "  ${CREATE_ENV_CMD}"
    echo "  ${INSTALL_TOOLS_CMD}"
    exit 1
fi

ENV_PREFIX="$(conda run -n "${ENV_NAME}" python -c 'import os; print(os.environ["CONDA_PREFIX"])')"

missing=0

check_pkg() {
    local package_regex="$1"
    if ! conda list -n "${ENV_NAME}" | awk 'NF >= 2 { print $1 " " $2 }' | grep -Eq "${package_regex}"; then
        missing=1
    fi
}

check_binary() {
    local binary_path="$1"
    if [[ ! -x "${binary_path}" ]]; then
        missing=1
    fi
}

check_pkg '^cmake 4\.'
check_pkg '^ninja '
check_pkg '^python '
check_pkg '^ffmpeg '
check_pkg '^icu '
check_pkg '^libpng '
check_pkg '^nasm '
check_pkg '^clang_osx-arm64 19\.1\.7$'
check_pkg '^clangxx_osx-arm64 19\.1\.7$'
check_pkg '^clang-tools 19\.1\.7$'

check_binary "${ENV_PREFIX}/bin/clang"
check_binary "${ENV_PREFIX}/bin/clang++-19"
check_binary "${ENV_PREFIX}/bin/clang-scan-deps"

if [[ ${missing} -ne 0 ]]; then
    echo "The '${ENV_NAME}' environment is missing the required modules-build toolchain or packages."
    echo "Install or repair it with:"
    echo "  ${INSTALL_TOOLS_CMD}"
    exit 1
fi

mkdir -p "${BUILD_DIR}"

CONFIGURE_CMD=(
    conda run -n "${ENV_NAME}"
    cmake
    -S "${SCRIPT_DIR}"
    -B "${BUILD_DIR}"
    -G Ninja
    -DTREX_ENABLE_SHARED_INTERNAL_LIBS=ON
    -DBUILD_SHARED_LIBS=ON
    -DCMAKE_OSX_DEPLOYMENT_TARGET="${DEPLOY_TARGET}"
    -DCOMMONS_ENABLE_MODULES=ON
    -DTREX_ENABLE_MODULES=ON
    -DCMAKE_C_COMPILER="${ENV_PREFIX}/bin/clang"
    -DCMAKE_CXX_COMPILER="${ENV_PREFIX}/bin/clang++-19"
    -DCMAKE_CXX_COMPILER_CLANG_SCAN_DEPS="${ENV_PREFIX}/bin/clang-scan-deps"
)

BUILD_CMD=(
    conda run -n "${ENV_NAME}"
    cmake
    --build "${BUILD_DIR}"
    -j
)

echo "Using conda environment: ${ENV_NAME}"
echo "Using build directory: ${BUILD_DIR}"
echo "Using deployment target: ${DEPLOY_TARGET}"
echo ""
echo "Configure command:"
printf '  %q' "${CONFIGURE_CMD[@]}"
echo ""
echo ""
echo "Build command:"
printf '  %q' "${BUILD_CMD[@]}"
echo ""
echo ""

git -C "${REPO_DIR}" submodule update --recursive --init

"${CONFIGURE_CMD[@]}"
"${BUILD_CMD[@]}"