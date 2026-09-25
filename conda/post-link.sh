#!/bin/bash

set +e  # Disable immediate exit on error so that failures don't abort the script

export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1
export PIP_PROGRESS_BAR=off
export GIT_TERMINAL_PROMPT=0
export ULTRALYTICS_HUB_NO_PROGRESS=1
export HF_HUB_DISABLE_PROGRESS_BAR=1
export DISABLE_TQDM=1
export RICH_NO_COLOR=1
export RICH_FORCE_TERMINAL=0
export FORCE_COLOR=0
# Package sources are part of the post-link policy. Ignore ambient pip indexes
# such as those added by nvidia-pyindex; callers can use the TREX_* URL
# overrides below when an explicit mirror is required.
export PIP_CONFIG_FILE=/dev/null
unset PIP_INDEX_URL PIP_EXTRA_INDEX_URL PIP_FIND_LINKS

echo "PREFIX=${PREFIX}"
# CI can request stdout so its caller can tee and validate the transaction log.
if [ "${TREX_POST_LINK_OUTPUT:-}" = "stdout" ]; then
    OUT_STREAM=""
elif [ -n "${TREX_POST_LINK_OUTPUT:-}" ]; then
    OUT_STREAM="${TREX_POST_LINK_OUTPUT}"
elif [ -z "${PREFIX}" ]; then
    echo "PREFIX is not set. Using stdout."
    OUT_STREAM=""
else
    OUT_STREAM="${PREFIX}/.messages.txt"
fi

POST_LINK_FAILED=0
LAST_COMMAND_STATUS=0
INSTALL_PROGRESS_PID=""
INSTALL_PROGRESS_LABEL=""
INSTALL_PROGRESS_STARTED=0

write_install_progress() {
    if [ -w /dev/tty ]; then
        printf '%s\n' "$1" >/dev/tty 2>/dev/null || printf '%s\n' "$1" >&2
    else
        printf '%s\n' "$1" >&2
    fi
}

install_progress_worker() {
    local label="$1"
    local progress_log="$2"
    local elapsed=0
    local detail=""
    local sleeper=""
    trap '[ -n "${sleeper}" ] && kill "${sleeper}" 2>/dev/null; exit 0' TERM INT

    while true; do
        if [ -n "${progress_log}" ] && [ -f "${progress_log}" ]; then
            detail=$(tail -n 20 "${progress_log}" 2>/dev/null \
                | awk 'NF { line=$0 } END { print substr(line, 1, 100) }')
        fi
        if [ -n "${detail}" ]; then
            write_install_progress "[post-link] ${label} (${elapsed}s): ${detail}"
        else
            write_install_progress "[post-link] ${label} (${elapsed}s)"
        fi
        sleep 10 &
        sleeper=$!
        wait "${sleeper}"
        sleeper=""
        elapsed=$((elapsed + 10))
    done
}

start_install_progress() {
    INSTALL_PROGRESS_LABEL="$1"
    INSTALL_PROGRESS_STARTED=$(date +%s)
    install_progress_worker "$1" "$2" &
    INSTALL_PROGRESS_PID=$!
}

stop_install_progress() {
    [ -n "${INSTALL_PROGRESS_PID}" ] || return 0
    kill "${INSTALL_PROGRESS_PID}" 2>/dev/null
    wait "${INSTALL_PROGRESS_PID}" 2>/dev/null
    write_install_progress "[post-link] ${INSTALL_PROGRESS_LABEL} finished after $(( $(date +%s) - INSTALL_PROGRESS_STARTED ))s."
    INSTALL_PROGRESS_PID=""
}

trap stop_install_progress EXIT

# Append a single log line to the conda post-link message stream.
log() {
    if [ -n "${OUT_STREAM}" ]; then
        printf '%s\n' "$1" >>"${OUT_STREAM}"
    else
        printf '%s\n' "$1"
    fi
}

record_failure() {
    POST_LINK_FAILED=1
    log "$1"
}

# Run a command while retaining its output and exit status. Long operations use
# a direct terminal heartbeat because Conda captures normal post-link output.
run_with_reporting() {
    local progress_log=""
    if [ -n "${TREX_PROGRESS_LABEL:-}" ] && [ -n "${OUT_STREAM}" ]; then
        progress_log=$(mktemp "${TMPDIR:-/tmp}/trex_post_link_progress.XXXXXX")
        if [ -n "${progress_log}" ]; then
            start_install_progress "${TREX_PROGRESS_LABEL}" "${progress_log}"
            "$@" >"${progress_log}" 2>&1
            LAST_COMMAND_STATUS=$?
            stop_install_progress
            cat "${progress_log}" >>"${OUT_STREAM}"
            rm -f "${progress_log}"
            return "${LAST_COMMAND_STATUS}"
        fi
    elif [ -n "${TREX_PROGRESS_LABEL:-}" ]; then
        start_install_progress "${TREX_PROGRESS_LABEL}" ""
        "$@"
        LAST_COMMAND_STATUS=$?
        stop_install_progress
        return "${LAST_COMMAND_STATUS}"
    fi

    if [ -n "${OUT_STREAM}" ]; then
        "$@" >>"${OUT_STREAM}" 2>&1
        LAST_COMMAND_STATUS=$?
    else
        "$@"
        LAST_COMMAND_STATUS=$?
    fi
    return "${LAST_COMMAND_STATUS}"
}

# Emit the exact command that will be executed for easier reproduction.
log_command() {
    local formatted=()
    local arg
    for arg in "$@"; do
        formatted+=("$(printf '%q' "${arg}")")
    done
    log "[post-link] Running: ${formatted[*]}"
}

# nvidia-smi reports the newest CUDA runtime accepted by the installed driver.
detect_driver_cuda_version() {
    detected_cuda_version=""
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        return 1
    fi

    local smi_output
    if [ -n "${OUT_STREAM}" ]; then
        smi_output=$(nvidia-smi 2>>"${OUT_STREAM}")
    else
        smi_output=$(nvidia-smi)
    fi
    local smi_status=$?
    if [ ${smi_status} -ne 0 ]; then
        log "[post-link] nvidia-smi failed while checking driver compatibility (exit ${smi_status})."
        return 1
    fi

    detected_cuda_version=$(printf '%s\n' "${smi_output}" \
        | sed -nE 's/.*CUDA Version:[[:space:]]*([0-9]+\.[0-9]+).*/\1/p' \
        | head -n 1)
    if [ -z "${detected_cuda_version}" ]; then
        log "[post-link] nvidia-smi did not report a maximum supported CUDA version."
        return 1
    fi
}

query_cuda_channel_metadata() {
    python - channels "$1" "$2" <<'PY'
# TREX_TORCH_CHANNEL_SELECTOR
import re
import sys
from urllib.request import Request, urlopen

root_url, driver_code_text = sys.argv[2:4]
channels = {"cu118", "cu121", "cu124", "cu126", "cu128", "cu129", "cu130", "cu132"}
try:
    request = Request(root_url.rstrip("/") + "/", headers={"User-Agent": "TRex post-link"})
    with urlopen(request, timeout=10) as response:
        contents = response.read().decode(errors="replace")
except Exception:
    contents = ""
for href in re.findall(r'''href=["']?([^ >"']+)''', contents, re.IGNORECASE):
    channel = href.rstrip("/").rsplit("/", 1)[-1]
    if re.fullmatch(r"cu[0-9]{3,}", channel): channels.add(channel)
code = lambda channel: int(channel[2:4]) * 100 + int(channel[4:])
print("\n".join(sorted((item for item in channels if 1108 <= code(item) <= int(driver_code_text)), key=code, reverse=True)))
PY
}

query_cuda_pair_metadata() {
    python - pair "$1" "$2" <<'PY'
# TREX_TORCH_PAIR_SELECTOR
import re
import subprocess
import sys
try:
    from packaging.version import Version
except ImportError:
    from pip._vendor.packaging.version import Version

index_url, flavor = sys.argv[2:4]
def available_versions(package, minimum):
    result = subprocess.run(
        [sys.executable, "-m", "pip", "index", "versions", package,
         "--index-url", index_url],
        capture_output=True, text=True, errors="replace",
    )
    match = re.search(r"^Available versions:\s*(.+)$", result.stdout, re.MULTILINE)
    if result.returncode or not match:
        sys.stderr.write(result.stdout + result.stderr)
        raise SystemExit(1)
    versions = (item.strip() for item in match.group(1).split(","))
    return sorted({item for item in versions if item and Version(item) >= minimum
                   and Version(item).local == flavor}, key=Version, reverse=True)

torch_versions = available_versions("torch", Version("2.2"))
vision_versions = available_versions("torchvision", Version("0.17"))
vision_by_release = {Version(item).release: item for item in vision_versions}
for torch_version in torch_versions:
    release = Version(torch_version).release
    if len(release) < 2 or release[0] != 2: continue
    patch = release[2] if len(release) > 2 else 0
    vision_version = vision_by_release.get((0, release[1] + 15, patch))
    if vision_version:
        print(f"{torch_version}|{vision_version}")
        raise SystemExit(0)

raise SystemExit(1)
PY
}

# Query package metadata without downloading wheels, then select the newest
# synchronized release pair carrying the requested CUDA local-version label.
discover_cuda_pair() {
    local index_url="$1"
    local flavor="$2"
    local pair=""

    if [ -n "${OUT_STREAM}" ]; then
        pair=$(query_cuda_pair_metadata "${index_url}" "${flavor}" 2>>"${OUT_STREAM}")
    else
        pair=$(query_cuda_pair_metadata "${index_url}" "${flavor}")
    fi
    local discovery_status=$?
    if [ ${discovery_status} -ne 0 ] || [[ "${pair}" != *"|"* ]]; then
        return 1
    fi

    local torch_version vision_version
    IFS='|' read -r torch_version vision_version <<EOF
${pair}
EOF
    if [ -z "${torch_version}" ] || [ -z "${vision_version}" ]; then
        return 1
    fi
    torch_packages=("torch===${torch_version}" "torchvision===${vision_version}")
    log "[post-link] Selected newest compatible ${flavor} pair: torch ${torch_version}, torchvision ${vision_version}."
}

select_pypi_target() {
    torch_target="PyPI"
    torch_index_url="${pypi_index_url}"
    torch_dependency_index_args=()
    torch_packages=("torch>=2.2" "torchvision>=0.17")
}

# Select one distribution source before pip installs anything. The NVIDIA
# driver is backward compatible with older CUDA runtimes, so choose the newest
# supported channel not newer than the driver's advertised CUDA API.
select_torch_target() {
    select_pypi_target

    if [ "${system}" = "Darwin" ]; then
        torch_target="macOS/PyPI"
        log "[post-link] macOS detected; using the normal PyPI distribution with native MPS support."
        return 0
    fi

    case "${arch}" in
        arm|arm64|aarch64)
            torch_target="native PyPI"
            log "[post-link] ${system} ${arch} detected; using its native PyPI distribution."
            return 0
            ;;
    esac

    if [ "${system}" = "Linux" ] && [[ "${arch}" == "x86_64" || "${arch}" == "amd64" ]]; then
        local driver_major driver_minor driver_cuda_code
        if detect_driver_cuda_version; then
            IFS=. read -r driver_major driver_minor <<EOF
${detected_cuda_version}
EOF
            if [[ "${driver_major}" =~ ^[0-9]+$ && "${driver_minor}" =~ ^[0-9]+$ ]]; then
                driver_cuda_code=$((driver_major * 100 + driver_minor))
                if [ "${driver_cuda_code}" -ge 1108 ]; then
                    local channel_candidates channel
                    channel_candidates=$(query_cuda_channel_metadata "${torch_index_root}" "${driver_cuda_code}")
                    while IFS= read -r channel; do
                        [ -n "${channel}" ] || continue
                        local cuda_index_url="${torch_index_root}/${channel}"
                        log "[post-link] NVIDIA driver accepts CUDA ${detected_cuda_version}; checking ${channel} package metadata."
                        if discover_cuda_pair "${cuda_index_url}" "${channel}"; then
                            torch_target="CUDA ${channel}"
                            torch_index_url="${cuda_index_url}"
                            torch_dependency_index_args=(--extra-index-url "${pypi_index_url}")
                            return 0
                        fi
                        log "[post-link] No compatible ${channel} pair was found; checking older compatible CUDA channels."
                    done <<EOF
${channel_candidates}
EOF
                    log "[post-link] WARNING: No compatible CUDA channel was discoverable; falling back to PyPI."
                    return 0
                fi
                log "[post-link] Driver compatibility is below CUDA 11.8; falling back to PyPI."
            else
                log "[post-link] Could not parse NVIDIA CUDA compatibility '${detected_cuda_version}'; falling back to PyPI."
            fi
        else
            log "[post-link] No usable NVIDIA driver detected; falling back to PyPI."
        fi
    else
        log "[post-link] ${system} ${arch} detected; selecting the default PyPI distribution."
        return 0
    fi
}

install_selected_torch() {
    local pip_install_command=(python -m pip install "${pip_flags[@]}"
        "${numpy_constraint_args[@]}" --index-url "${torch_index_url}"
        "${torch_dependency_index_args[@]}" "${torch_packages[@]}"
        "${common_packages[@]}")
    log "[post-link] Running one resolver transaction for ${torch_target}; no version or index retries are permitted."
    log_command "${pip_install_command[@]}"
    TREX_PROGRESS_LABEL="Installing Python ML packages" run_with_reporting "${pip_install_command[@]}"
}

arch=$(uname -m)
system=$(uname)
pypi_index_url="${TREX_PYPI_INDEX_URL:-https://pypi.org/simple}"
torch_index_root="${TREX_TORCH_INDEX_ROOT:-https://download.pytorch.org/whl}"
clip_requirement="${TREX_CLIP_REQUIREMENT:-git+https://github.com/ultralytics/CLIP.git}"

# A Conda-owned NumPy is immutable. If Conda does not own NumPy, add it to the
# complete pip solve with the rest of the Python ML stack.
numpy_version=""
numpy_constraint_file=""
numpy_constraint_args=()
conda_numpy_owned=false
conda_opencv_owned=false
setup_ready=true
torch_installed=false

configure_numpy_policy() {
    local conda_record=""
    for conda_record in "${PREFIX}"/conda-meta/numpy-*.json; do
        if [ -f "${conda_record}" ]; then
            conda_numpy_owned=true
            break
        fi
    done

    if ! ${conda_numpy_owned}; then
        log "[post-link] Conda does not own NumPy; pip will solve numpy>=1.26,<3 with the complete ML dependency set."
        return 0
    fi

    # Read Conda's package record instead of importing NumPy in production.
    # The exact constraint prevents pip from replacing a Conda-owned package;
    # runtime/import validation belongs to the real-install CI jobs.
    if [ -n "${OUT_STREAM}" ]; then
        numpy_version=$(python -c "import json,sys; print(json.load(open(sys.argv[1], encoding='utf-8'))['version'])" "${conda_record}" 2>>"${OUT_STREAM}")
    else
        numpy_version=$(python -c "import json,sys; print(json.load(open(sys.argv[1], encoding='utf-8'))['version'])" "${conda_record}")
    fi
    if [ $? -ne 0 ] || [ -z "${numpy_version}" ]; then
        return 1
    fi

    if [ -z "${numpy_constraint_file}" ]; then
        numpy_constraint_file=$(mktemp "${TMPDIR:-/tmp}/trex_numpy_constraint.XXXXXX") || return 1
    fi
    printf 'numpy==%s\n' "${numpy_version}" > "${numpy_constraint_file}" || return 1
    numpy_constraint_args=(--constraint "${numpy_constraint_file}")
    log "[post-link] Conda owns NumPy ${numpy_version}; every pip solve is constrained to that exact version."
}

# pip is a Conda run dependency. Do not invoke Conda recursively while its
# transaction is still linking this environment.
if ! python -m pip --version >/dev/null 2>&1; then
    record_failure "[post-link] pip is unavailable; Python ML packages were not installed."
    setup_ready=false
fi

if ${setup_ready} && ! configure_numpy_policy; then
    record_failure "[post-link] Conda owns NumPy but its exact version could not be read; refusing to let pip modify it."
    setup_ready=false
fi

for conda_record in "${PREFIX}"/conda-meta/py-opencv-*.json; do
    if [ -f "${conda_record}" ]; then
        conda_opencv_owned=true
        log "[post-link] Conda owns cv2 through py-opencv; pip will not install an OpenCV wheel."
        break
    fi
done

common_packages=("torchmetrics" "tqdm" "ultralytics>=8.4.52,<9" "rfdetr==1.8.3"
    "dill" "timm" "scikit-learn" "${clip_requirement}")

if ! ${conda_opencv_owned}; then
    common_packages+=("opencv-python>=4.6,<5")
    log "[post-link] No Conda py-opencv binding detected; pip will provide cv2 for the non-minimal profile."
fi

if ! ${conda_numpy_owned}; then
    common_packages+=("numpy>=1.26,<3")
fi

pip_flags=(--disable-pip-version-check --no-input --no-color --progress-bar off)

if ${setup_ready}; then
    select_torch_target
    log "[post-link] Selected ${torch_target} from ${torch_index_url}."
    if install_selected_torch; then
        torch_installed=true
    else
        record_failure "[post-link] The single ${torch_target} installation failed; no retry was attempted."
    fi
fi

if ${torch_installed}; then
    log "[post-link] The single ${torch_target} Python ML installation transaction completed successfully."
    TORCH_INFO_STRING="import torch; print(f'[post-link] Installed PyTorch {torch.__version__}; compiled CUDA {torch.version.cuda}; torch.cuda.is_available() -> {torch.cuda.is_available()}')"
    log_command python -c "${TORCH_INFO_STRING}"
    if ! run_with_reporting python -c "${TORCH_INFO_STRING}"; then
        log "[post-link] WARNING: Unable to report the installed PyTorch CUDA status; installation remains successful."
    fi
    log "[post-link] Warming the Ultralytics runtime and model cache."
    CMD_STRING="from ultralytics import YOLO; from rfdetr import RFDETR; from torchvision.ops import nms; import cv2, numpy as np, torch; assert cv2.__version__.split('.')[0] == '4'; assert nms(torch.tensor([[0.,0.,1.,1.]]), torch.tensor([1.]), 0.5).tolist() == [0]; YOLO('yolo26n.yaml').to('cpu').predict(np.zeros((640, 480, 3), dtype=np.uint8))"
    log_command python -c "${CMD_STRING}"
    if ! TREX_PROGRESS_LABEL="Warming the Python ML runtime" run_with_reporting python -c "${CMD_STRING}"; then
        log "[post-link] WARNING: YOLO runtime warm-up failed (exit ${LAST_COMMAND_STATUS}); installation remains successful."
    fi
fi

if [ "${POST_LINK_FAILED}" -ne 0 ]; then
    log "============================================================"
    log "WARNING: TRex PYTHON ML SETUP IS INCOMPLETE"
    log "The Conda package installation will continue successfully."
    log "TRex itself is installed, but Python ML features may be unavailable."
    log "After installation, inspect this log. Dependency diagnostic: python -m pip check"
    log "============================================================"
    if [ -n "${OUT_STREAM}" ]; then
        echo "WARNING: TRex Python ML setup is incomplete; Conda installation will continue. See ${OUT_STREAM}." >&2
    else
        echo "WARNING: TRex Python ML setup is incomplete; Conda installation will continue. See stdout." >&2
    fi
    if [ -n "${OUT_STREAM}" ] && [ -f "${OUT_STREAM}" ]; then
        echo "[post-link] Dumping post-link log due to incomplete Python ML setup:" >&2
        cat "${OUT_STREAM}" >&2
    fi
fi

if [ -n "${numpy_constraint_file}" ]; then
    rm -f "${numpy_constraint_file}"
fi

exit 0
