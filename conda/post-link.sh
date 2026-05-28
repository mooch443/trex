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
OUT_STREAM="${PREFIX}/.messages.txt"
if [ -z "${PREFIX}" ]; then
    echo "PREFIX is not set. Using stdout."
    OUT_STREAM="/dev/stdout"
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

    (
        frames='|/-\'
        i=0
        start_time=$(date +%s)
        while [ ! -f "${PROGRESS_STOP}" ]; do
            now=$(date +%s)
            elapsed=$((now - start_time))
            minutes=$((elapsed / 60))
            seconds=$((elapsed % 60))
            frame=$(printf '%s' "${frames}" | cut -c $((i % 4 + 1)))
            info=$(last_progress_line "${log_path}")
            if [ -n "${info}" ]; then
                progress_stream "$(printf '\r  %s %s  %02d:%02d   ' "${frame}" "${info}" "${minutes}" "${seconds}")"
            else
                progress_stream "$(printf '\r  %s %s  %02d:%02d   ' "${frame}" "${label}" "${minutes}" "${seconds}")"
            fi
            i=$((i + 1))
            sleep 1
        done
        progress_stream "$(printf '\r%*s\r' 120 '')"
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
    printf '%s\n' "$1" >>"${OUT_STREAM}"
}

# Mark the script as having failed while still allowing execution to continue.
SUPPRESS_FAILURES=0

# Conda build/test prefixes operate without network and the conda CLI; just log issues.
if [ -n "${GITHUB_WORKSPACE:-}" ]; then
    SUPPRESS_FAILURES=1
    log "[post-link] Conda-build context detected; ignoring optional post-link failures."
fi

record_failure() {
    if [ "${SUPPRESS_FAILURES}" -eq 0 ]; then
        POST_LINK_FAILED=1
    fi
    log "$1"
}

# Run a command while teeing stdout/stderr into the log file and retain exit status.
run_with_reporting() {
    if [ -z "${OUT_STREAM}" ] || [ "${OUT_STREAM}" = "/dev/stdout" ]; then
        "$@"
        LAST_COMMAND_STATUS=$?
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

# After installations succeed, report CUDA and NVIDIA GPU availability.
check_nvidia_support() {
    if [ "$(uname)" = "Darwin" ]; then
        log "[post-link] Skipping NVIDIA GPU check on macOS."
        return 0
    fi

    log "[post-link] Checking NVIDIA GPU support after install..."

    cuda_result=$(python - <<'PY'
import sys
try:
    import torch
    available = torch.cuda.is_available()
except Exception:
    available = None

sys.stdout.write(
    "True" if available else ("False" if available is not None else "Unavailable")
)
PY
    2>>"${OUT_STREAM}")
    cuda_status=$?

    if [ ${cuda_status} -eq 0 ] && [ -n "${cuda_result}" ]; then
        log "[post-link] torch.cuda.is_available() after install -> ${cuda_result}"
    else
        log "[post-link] Unable to query torch CUDA availability after install (exit ${cuda_status})."
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        log_command nvidia-smi --query-gpu=name --format=csv,noheader
        gpu_output=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>>"${OUT_STREAM}")
        gpu_status=$?
        if [ ${gpu_status} -eq 0 ]; then
            if [ -n "${gpu_output}" ]; then
                gpu_output=${gpu_output//$'\r'/}
                printf '%s\n' "${gpu_output}" >>"${OUT_STREAM}"
                # Coalesce multiline GPU names into a single summary string.
                gpu_summary=${gpu_output//$'\n'/, }
                gpu_summary=${gpu_summary%, }
                log "[post-link] NVIDIA GPUs detected via nvidia-smi: ${gpu_summary}"
            else
                log "[post-link] nvidia-smi ran successfully but reported no GPUs."
            fi
        else
            log "[post-link] nvidia-smi query failed (exit ${gpu_status})."
        fi
    else
        log "[post-link] nvidia-smi not found; NVIDIA GPU likely unavailable."
    fi
}

# Ensure pip is present on all supported platforms before installation.
if [ "$(uname -p)" = "arm" ] || [ "${OSTYPE}" = "linux-gnu" ] || [ "$(uname)" = "Linux" ] || [ "$(uname)" = "Darwin" ]; then
    if ! command -v pip &>/dev/null; then
        log "pip could not be found, installing via conda..."
        if ! run_with_reporting conda install pip -y; then
            record_failure "[post-link] Unable to install pip via conda (exit ${LAST_COMMAND_STATUS}); continuing without pip-managed extras."
        fi
    fi
fi

# Detect the currently bundled numpy version so pip sees a compatible build.
numpy_requirement=()
numpy_version=$(python -c "import numpy; print(numpy.__version__)" 2>>"${OUT_STREAM}")
if [ $? -eq 0 ] && [ -n "${numpy_version}" ]; then
    numpy_requirement=("numpy==${numpy_version}")
    log "Installing pip packages (numpy=${numpy_version})..."
else
    log "[post-link] Could not determine numpy version; will install latest numpy."
    numpy_requirement=("numpy")
fi

common_packages=(
    "torch>=2.0.0,<3.0.0"
    "torchvision>=0.15.1"
    "torchmetrics"
    "tqdm"
    "opencv-python>=4,<5"
    "ultralytics>=8.3.0,<9"
    "dill"
    "timm"
    "scikit-learn"
    "git+https://github.com/ultralytics/CLIP.git"
)

if [ ${#numpy_requirement[@]} -gt 0 ]; then
    common_packages+=("${numpy_requirement[@]}")
fi

pip_flags=(
    --disable-pip-version-check
    --no-input
    --no-color
    --progress-bar
    off
)

arch=$(uname -p)
system=$(uname)

announce_progress "TRex is installing Python ML packages. This can take several minutes; progress below shows the latest pip activity."

# Explicitly describe channel decisions per platform.
if [ "${arch}" = "arm" ]; then
    log "ARM architecture detected; using default pip index (no custom channels)."
    log_command python -m pip install "${pip_flags[@]}" "${common_packages[@]}"
    if TREX_PROGRESS_LABEL="pip install..." run_with_reporting python -m pip install "${pip_flags[@]}" "${common_packages[@]}"; then
        check_nvidia_support
    else
        record_failure "[post-link] pip package installation failed on ARM (exit ${LAST_COMMAND_STATUS})."
    fi
elif [ "${system}" = "Darwin" ]; then
    log "macOS detected; using default pip index (no custom channels)."
    log_command python -m pip install "${pip_flags[@]}" "${common_packages[@]}"
    if TREX_PROGRESS_LABEL="pip install..." run_with_reporting python -m pip install "${pip_flags[@]}" "${common_packages[@]}"; then
        check_nvidia_support
    else
        record_failure "[post-link] pip package installation failed on macOS (exit ${LAST_COMMAND_STATUS})."
    fi
else
    log "Linux architecture detected; checking CUDA availability to document channel choice."
    gpu_check=$(python - <<'PY'
try:
    import torch
    print(torch.cuda.is_available())
except Exception:
    print("False")
PY
)
    if [ "${gpu_check}" = "True" ]; then
        log "[post-link] torch.cuda.is_available() -> True; installing from default pip channels to let torch pick CUDA wheels."
    else
        log "[post-link] torch.cuda.is_available() -> ${gpu_check:-False}; installing from default pip channels."
    fi

    log_command python -m pip install "${pip_flags[@]}" "${common_packages[@]}"
    if TREX_PROGRESS_LABEL="pip install..." run_with_reporting python -m pip install "${pip_flags[@]}" "${common_packages[@]}"; then
        check_nvidia_support
    else
        record_failure "[post-link] pip package installation failed on Linux (exit ${LAST_COMMAND_STATUS})."
    fi
fi

log "Testing installation..."
announce_progress "TRex is running a short YOLO smoke test to verify the Python install."

CMD_STRING="from ultralytics import YOLO; import numpy as np; YOLO('yolo26n.yaml').to('cpu').predict(np.zeros((640, 480, 3), dtype=np.uint8))"
log_command python -c "${CMD_STRING}"

if TREX_PROGRESS_LABEL="YOLO smoke test..." run_with_reporting python -c "${CMD_STRING}"; then
    log "[post-link] YOLO smoke test succeeded."
else
    record_failure "[post-link] YOLO smoke test failed (exit ${LAST_COMMAND_STATUS})."
fi

if [ "${POST_LINK_FAILED}" -ne 0 ]; then
    log "[post-link] Completed with issues; conda installation will continue."
    echo "post-link.sh completed with issues; see ${OUT_STREAM} for details." >&2
    if [ -n "${OUT_STREAM}" ] && [ "${OUT_STREAM}" != "/dev/stdout" ] && [ -f "${OUT_STREAM}" ]; then
        echo "[post-link] Dumping post-link log due to failures:" >&2
        cat "${OUT_STREAM}" >&2
    fi
fi

exit 0
