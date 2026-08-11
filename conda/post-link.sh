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
PROGRESS_PID=""
PROGRESS_STOP=""

progress_stream() {
    if [ -w /dev/tty ]; then
        printf '%s' "$1" >/dev/tty
    else
        printf '%s' "$1" >&2
    fi
}

announce_progress() {
    progress_stream "$(printf '\n[post-link] %s\n' "$1")"
}

last_progress_line() {
    local path="$1"
    if [ -f "${path}" ]; then
        tail -n 20 "${path}" 2>/dev/null | awk 'NF { line=$0 } END { if (line) print substr(line, 1, 90) }'
    fi
}

start_progress() {
    local label="$1"
    local log_path="$2"

    if [ -n "${PROGRESS_PID}" ]; then
        return 0
    fi

    PROGRESS_STOP="${TMPDIR:-/tmp}/trex_post_link_stop_$$_${RANDOM:-0}"
    rm -f "${PROGRESS_STOP}" 2>/dev/null

    progress_stream "$(printf '\033[?25l')"
    (
        frames=("⠋" "⠙" "⠚" "⠞" "⠖" "⠦" "⠴" "⠲" "⠳" "⠓")
        i=0
        start_time=$(date +%s)
        while [ ! -f "${PROGRESS_STOP}" ]; do
            now=$(date +%s)
            elapsed=$((now - start_time))
            minutes=$((elapsed / 60))
            seconds=$((elapsed % 60))
            frame="${frames[$((i % ${#frames[@]}))]}"
            info=$(last_progress_line "${log_path}")
            if [ -n "${info}" ]; then
                progress_stream "$(printf '\r\033[34m%s\033[0m %s  %02d:%02d   ' "${frame}" "${info}" "${minutes}" "${seconds}")"
            else
                progress_stream "$(printf '\r\033[34m%s\033[0m %s  %02d:%02d   ' "${frame}" "${label}" "${minutes}" "${seconds}")"
            fi
            i=$((i + 1))
            sleep 0.1
        done
        progress_stream "$(printf '\r\033[2K\033[?25h')"
    ) &
    PROGRESS_PID=$!
}

stop_progress() {
    if [ -n "${PROGRESS_PID}" ]; then
        touch "${PROGRESS_STOP}" 2>/dev/null
        wait "${PROGRESS_PID}" 2>/dev/null
        rm -f "${PROGRESS_STOP}" 2>/dev/null
        PROGRESS_PID=""
        PROGRESS_STOP=""
    fi
}

trap stop_progress EXIT

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

# Run a command while teeing stdout/stderr into the log file and retain exit status.
run_with_reporting() {
    if [ -z "${OUT_STREAM}" ]; then
        local stdout_progress_log=""
        if [ -n "${TREX_PROGRESS_LABEL:-}" ] && command -v tee >/dev/null 2>&1; then
            stdout_progress_log="${TMPDIR:-/tmp}/trex_post_link_$$_${RANDOM:-0}.log"
            : >"${stdout_progress_log}" 2>/dev/null
            start_progress "${TREX_PROGRESS_LABEL}" "${stdout_progress_log}"
            "$@" 2>&1 | tee "${stdout_progress_log}"
            LAST_COMMAND_STATUS=${PIPESTATUS[0]}
            stop_progress
            rm -f "${stdout_progress_log}" 2>/dev/null
        else
            "$@"
            LAST_COMMAND_STATUS=$?
        fi
        return "${LAST_COMMAND_STATUS}"
    fi

    local progress_log=""
    if [ -n "${TREX_PROGRESS_LABEL:-}" ]; then
        progress_log="${TMPDIR:-/tmp}/trex_post_link_$$_${RANDOM:-0}.log"
        : >"${progress_log}" 2>/dev/null
        start_progress "${TREX_PROGRESS_LABEL}" "${progress_log}"
    fi

    if command -v tee >/dev/null 2>&1; then
        if [ -n "${progress_log}" ]; then
            "$@" 2>&1 | tee -a "${OUT_STREAM}" "${progress_log}" >/dev/null
        else
            "$@" 2>&1 | tee -a "${OUT_STREAM}"
        fi
        LAST_COMMAND_STATUS=${PIPESTATUS[0]}
    else
        "$@" >>"${OUT_STREAM}" 2>&1
        LAST_COMMAND_STATUS=$?
    fi

    stop_progress
    if [ -n "${progress_log}" ]; then
        rm -f "${progress_log}" 2>/dev/null
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

# Select one distribution source before pip runs. The NVIDIA driver is backward
# compatible with older CUDA runtimes, so cap selection at a broadly published
# channel instead of probing or retrying every channel the driver could accept.
select_torch_target() {
    torch_dependency_index_args=()

    if [ "${system}" = "Darwin" ]; then
        torch_target="macOS/PyPI"
        torch_index_url="${pypi_index_url}"
        log "[post-link] macOS detected; using the normal PyPI distribution with native MPS support."
        return 0
    fi

    case "${arch}" in
        arm|arm64|aarch64)
            torch_target="native PyPI"
            torch_index_url="${pypi_index_url}"
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
                local channel=""
                if [ "${driver_cuda_code}" -ge 1302 ]; then channel="cu132"
                elif [ "${driver_cuda_code}" -ge 1300 ]; then channel="cu130"
                elif [ "${driver_cuda_code}" -ge 1209 ]; then channel="cu129"
                elif [ "${driver_cuda_code}" -ge 1208 ]; then channel="cu128"
                elif [ "${driver_cuda_code}" -ge 1206 ]; then channel="cu126"
                elif [ "${driver_cuda_code}" -ge 1204 ]; then channel="cu124"
                elif [ "${driver_cuda_code}" -ge 1201 ]; then channel="cu121"
                elif [ "${driver_cuda_code}" -ge 1108 ]; then channel="cu118"
                fi
                if [ -n "${channel}" ]; then
                    torch_target="CUDA ${channel}"
                    torch_index_url="${torch_index_root}/${channel}"
                    torch_dependency_index_args=(--extra-index-url "${pypi_index_url}")
                    log "[post-link] NVIDIA driver accepts CUDA ${detected_cuda_version}; selected the single ${torch_target} distribution."
                    return 0
                fi
                log "[post-link] Driver compatibility is below CUDA 11.8; selecting the CPU-only distribution."
            else
                log "[post-link] Could not parse NVIDIA CUDA compatibility '${detected_cuda_version}'; selecting the CPU-only distribution."
            fi
        else
            log "[post-link] No usable NVIDIA driver detected; selecting the CPU-only distribution."
        fi
    else
        log "[post-link] ${system} ${arch} detected; selecting the default PyPI distribution."
        torch_target="PyPI"
        torch_index_url="${pypi_index_url}"
        return 0
    fi

    torch_target="CPU-only"
    torch_index_url="${torch_index_root}/cpu"
    torch_dependency_index_args=(--extra-index-url "${pypi_index_url}")
}

install_selected_torch() {
    if [ "${system}" = "Darwin" ]; then
        # This pair is already proven on both supported macOS architectures.
        # macOS has no CUDA wheel choice, so it never needs version probing.
        torch_packages=("torch==2.6.0" "torchvision==0.21.0")
    else
        # Torchvision declares its exact compatible torch dependency. Let pip
        # solve that relationship once instead of zipping index version lists.
        torch_packages=("torch>=2.2" "torchvision>=0.17")
    fi
    torch_index_args=(--index-url "${torch_index_url}")
    log "[post-link] Running one resolver transaction for ${torch_target}; no version or index retries are permitted."
    log_command python -m pip install "${pip_flags[@]}" \
        "${numpy_constraint_args[@]}" "${torch_index_args[@]}" \
        "${torch_dependency_index_args[@]}" "${torch_packages[@]}" \
        "${common_packages[@]}"
    if ! TREX_PROGRESS_LABEL="pip install ${torch_target} PyTorch..." run_with_reporting \
        python -m pip install "${pip_flags[@]}" \
        "${numpy_constraint_args[@]}" "${torch_index_args[@]}" \
        "${torch_dependency_index_args[@]}" "${torch_packages[@]}" \
        "${common_packages[@]}"
    then
        return 1
    fi

    return 0
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

common_packages=(
    "torchmetrics"
    "tqdm"
    "ultralytics>=8.3.0,<9"
    "rfdetr==1.8.3"
    "dill"
    "timm"
    "scikit-learn"
    "${clip_requirement}"
)

if ! ${conda_opencv_owned}; then
    common_packages+=("opencv-python>=4.6,<5")
    log "[post-link] No Conda py-opencv binding detected; pip will provide cv2 for the non-minimal profile."
fi

if ! ${conda_numpy_owned}; then
    common_packages+=("numpy>=1.26,<3")
fi

pip_flags=(
    --disable-pip-version-check
    --no-input
    --no-color
    --progress-bar
    off
)

if ${setup_ready}; then
    select_torch_target
    log "[post-link] Selected ${torch_target} from ${torch_index_url}."
    announce_progress "TRex is installing Python ML packages in one resolver transaction."
    if install_selected_torch; then
        torch_installed=true
    else
        record_failure "[post-link] The single ${torch_target} installation failed; no retry was attempted."
    fi
fi

if ${torch_installed}; then
    log "[post-link] The single ${torch_target} Python ML installation transaction completed successfully."
    log "[post-link] Warming the Ultralytics runtime and model cache."
    CMD_STRING="from ultralytics import YOLO; from rfdetr import RFDETR; from torchvision.ops import nms; import cv2, numpy as np, torch; assert cv2.__version__.split('.')[0] == '4'; assert nms(torch.tensor([[0.,0.,1.,1.]]), torch.tensor([1.]), 0.5).tolist() == [0]; YOLO('yolo26n.yaml').to('cpu').predict(np.zeros((640, 480, 3), dtype=np.uint8))"
    log_command python -c "${CMD_STRING}"
    if ! TREX_PROGRESS_LABEL="YOLO runtime warm-up..." run_with_reporting python -c "${CMD_STRING}"; then
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
