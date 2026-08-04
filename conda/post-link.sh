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
    printf '%s\n' "$1" >>"${OUT_STREAM}"
}

record_failure() {
    POST_LINK_FAILED=1
    log "$1"
}

# Run a command while teeing stdout/stderr into the log file and retain exit status.
run_with_reporting() {
    LAST_COMMAND_RESOLUTION_FAILURE=0
    if [ -z "${OUT_STREAM}" ] || [ "${OUT_STREAM}" = "/dev/stdout" ]; then
        local stdout_progress_log=""
        if [ -n "${TREX_PROGRESS_LABEL:-}" ] && command -v tee >/dev/null 2>&1; then
            stdout_progress_log="${TMPDIR:-/tmp}/trex_post_link_$$_${RANDOM:-0}.log"
            : >"${stdout_progress_log}" 2>/dev/null
            start_progress "${TREX_PROGRESS_LABEL}" "${stdout_progress_log}"
            "$@" 2>&1 | tee "${stdout_progress_log}"
            LAST_COMMAND_STATUS=${PIPESTATUS[0]}
            stop_progress
            if grep -q "ResolutionImpossible" "${stdout_progress_log}" 2>/dev/null; then
                LAST_COMMAND_RESOLUTION_FAILURE=1
            fi
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
        if grep -q "ResolutionImpossible" "${progress_log}" 2>/dev/null; then
            LAST_COMMAND_RESOLUTION_FAILURE=1
        fi
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
    smi_output=$(nvidia-smi 2>>"${OUT_STREAM}")
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

torch_index_candidates() {
    local index_url="$1"
    python - "${index_url}" <<'PY'
import re
import subprocess
import sys
try:
    from packaging.version import Version
except ImportError:
    from pip._vendor.packaging.version import Version

flavor = sys.argv[1].rstrip("/").rsplit("/", 1)[-1]
required_flavor = flavor if flavor == "cpu" or flavor.startswith("cu") else None
version_lists = []
for package, minimum in (("torch", Version("2.2")), ("torchvision", Version("0.17"))):
    result = subprocess.run(
        [sys.executable, "-m", "pip", "index", "versions", package,
         "--index-url", sys.argv[1]],
        capture_output=True,
        text=True,
        errors="replace",
    )
    match = re.search(r"^Available versions:\s*(.+)$", result.stdout, re.MULTILINE)
    if result.returncode or not match:
        raise SystemExit(1)
    versions = sorted(
        {item.strip() for item in match.group(1).split(",") if item.strip()},
        key=Version,
        reverse=True,
    )
    version_lists.append([
        item for item in versions
        if Version(item) >= minimum
        and (required_flavor is None or Version(item).local == required_flavor)
    ])

# Official Torch indexes publish torch and torchvision as synchronized release
# series. Pip validates each exact pair below; a rejected newest pair therefore
# advances to the next complete same-index release without changing CUDA flavor.
for torch_version, vision_version in zip(*version_lists):
    print(f"{torch_version}|{vision_version}")
PY
}

resolve_torch_target() {
    torch_candidate_pairs=$(torch_index_candidates "${torch_index_url}") || return 1
    [ -n "${torch_candidate_pairs}" ]
}

# Build an ordered target list. Each pip resolution still sees exactly one Torch
# index: compatible CUDA channels from newest to oldest, then CPU, then PyPI.
torch_target_names=()
torch_target_urls=()
torch_target_dependency_urls=()

add_torch_target() {
    torch_target_names+=("$1")
    torch_target_urls+=("$2")
    torch_target_dependency_urls+=("$3")
}

build_torch_target_candidates() {
    if [ "${system}" = "Darwin" ]; then
        add_torch_target "macOS/PyPI" "https://pypi.org/simple" ""
        log "[post-link] macOS detected; using the normal PyPI distribution with native MPS support."
        return 0
    fi

    if [ "${system}" = "Linux" ] && [[ "${arch}" == "x86_64" || "${arch}" == "amd64" ]]; then
        local driver_major driver_minor driver_cuda_code
        if detect_driver_cuda_version; then
            IFS=. read -r driver_major driver_minor <<EOF
${detected_cuda_version}
EOF
            if [[ "${driver_major}" =~ ^[0-9]+$ && "${driver_minor}" =~ ^[0-9]+$ ]]; then
                driver_cuda_code=$((driver_major * 100 + driver_minor))
                log "[post-link] NVIDIA driver accepts CUDA ${detected_cuda_version}; trying every compatible PyTorch CUDA channel newest-first."
                if [ "${driver_cuda_code}" -ge 1302 ]; then add_torch_target "CUDA 13.2" "https://download.pytorch.org/whl/cu132" "https://pypi.org/simple"; fi
                if [ "${driver_cuda_code}" -ge 1300 ]; then add_torch_target "CUDA 13.0" "https://download.pytorch.org/whl/cu130" "https://pypi.org/simple"; fi
                if [ "${driver_cuda_code}" -ge 1209 ]; then add_torch_target "CUDA 12.9" "https://download.pytorch.org/whl/cu129" "https://pypi.org/simple"; fi
                if [ "${driver_cuda_code}" -ge 1208 ]; then add_torch_target "CUDA 12.8" "https://download.pytorch.org/whl/cu128" "https://pypi.org/simple"; fi
                if [ "${driver_cuda_code}" -ge 1206 ]; then add_torch_target "CUDA 12.6" "https://download.pytorch.org/whl/cu126" "https://pypi.org/simple"; fi
                if [ "${driver_cuda_code}" -ge 1204 ]; then add_torch_target "CUDA 12.4" "https://download.pytorch.org/whl/cu124" "https://pypi.org/simple"; fi
                if [ "${driver_cuda_code}" -ge 1201 ]; then add_torch_target "CUDA 12.1" "https://download.pytorch.org/whl/cu121" "https://pypi.org/simple"; fi
                if [ "${driver_cuda_code}" -ge 1108 ]; then add_torch_target "CUDA 11.8" "https://download.pytorch.org/whl/cu118" "https://pypi.org/simple"; fi
                if [ "${driver_cuda_code}" -lt 1108 ]; then
                    log "[post-link] Driver compatibility is below CUDA 11.8; skipping CUDA distributions."
                fi
            else
                log "[post-link] Could not parse NVIDIA CUDA compatibility '${detected_cuda_version}'; skipping CUDA distributions."
            fi
        else
            log "[post-link] No usable NVIDIA driver detected; skipping CUDA distributions."
        fi
    elif [ "${system}" = "Linux" ]; then
        log "[post-link] Linux ${arch} detected; CUDA wheel channels are not applicable."
    else
        log "[post-link] ${system} ${arch} detected; CUDA wheel channels are not applicable."
    fi

    # Plain PyPI is not guaranteed CPU-only on Linux or Windows, so try the
    # dedicated CPU distribution first and keep default PyPI as the last resort.
    add_torch_target "CPU-only" "https://download.pytorch.org/whl/cpu" "https://pypi.org/simple"
    add_torch_target "PyPI fallback" "https://pypi.org/simple" ""
}

select_torch_candidate() {
    local candidate_index="$1"
    torch_target="${torch_target_names[${candidate_index}]}"
    torch_index_url="${torch_target_urls[${candidate_index}]}"
    torch_index_args=(--index-url "${torch_index_url}")
    torch_dependency_index_args=()
    if [ -n "${torch_target_dependency_urls[${candidate_index}]}" ]; then
        torch_dependency_index_args=(--extra-index-url "${torch_target_dependency_urls[${candidate_index}]}")
    fi
}

install_selected_torch() {
    local torch_version vision_version
    while IFS='|' read -r torch_version vision_version; do
        [ -n "${torch_version}" ] && [ -n "${vision_version}" ] || continue
        torch_packages=(
            "torch===${torch_version}"
            "torchvision===${vision_version}"
        )
        log "[post-link] Resolving ${torch_target} pair: torch ${torch_version}, torchvision ${vision_version}."
        log_command python -m pip install "${pip_flags[@]}" \
            "${numpy_constraint_args[@]}" "${torch_index_args[@]}" \
            "${torch_dependency_index_args[@]}" "${torch_packages[@]}" \
            "${common_packages[@]}"
        if TREX_PROGRESS_LABEL="pip install ${torch_target} PyTorch..." run_with_reporting \
            python -m pip install "${pip_flags[@]}" \
            "${numpy_constraint_args[@]}" "${torch_index_args[@]}" \
            "${torch_dependency_index_args[@]}" "${torch_packages[@]}" \
            "${common_packages[@]}"
        then
            if python -c "import torch, torchvision" >/dev/null 2>&1 \
                && python -m pip check >>"${OUT_STREAM}" 2>&1
            then
                return 0
            fi
            LAST_COMMAND_STATUS=1
            log "[post-link] Installed pair failed import or dependency verification; trying the next ${torch_target} release pair."
            continue
        fi
        if [ "${LAST_COMMAND_RESOLUTION_FAILURE}" -ne 1 ]; then
            return 1
        fi
        log "[post-link] Pair rejected with the pinned dependency set; trying the next ${torch_target} release pair."
    done <<<"${torch_candidate_pairs}"
    return 1
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

arch=$(uname -m)
system=$(uname)

# A Conda-owned NumPy is immutable. If Conda does not own NumPy, add it to the
# complete pip solve with the rest of the Python ML stack.
numpy_version=""
numpy_constraint_file=""
numpy_constraint_args=()
conda_numpy_owned=false
setup_ready=true
torch_installed=false

configure_numpy_policy() {
    local conda_record metadata_numpy_version
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

    numpy_version=$(python -c "import numpy; print(numpy.__version__)" 2>>"${OUT_STREAM}")
    local numpy_status=$?
    if [ ${numpy_status} -ne 0 ] || [ -z "${numpy_version}" ]; then
        return 1
    fi
    metadata_numpy_version=$(python -c "from importlib.metadata import version; print(version('numpy'))" 2>>"${OUT_STREAM}")
    if [ -z "${metadata_numpy_version}" ] || [ "${metadata_numpy_version}" != "${numpy_version}" ]; then
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

for conda_record in "${PREFIX}"/conda-meta/py-opencv-*.json; do
    if [ -f "${conda_record}" ]; then
        record_failure "[post-link] Conda py-opencv is still present; refusing to install a second cv2 provider."
        setup_ready=false
        break
    fi
done

if ${setup_ready} && ! configure_numpy_policy; then
    record_failure "[post-link] Conda owns NumPy but its exact version could not be read; refusing to let pip modify it."
    setup_ready=false
fi

common_packages=(
    "torchmetrics"
    "tqdm"
    "opencv-python>=4.6,<5"
    "ultralytics>=8.3.0,<9"
    "rfdetr==1.8.3"
    "dill"
    "timm"
    "scikit-learn"
    "git+https://github.com/ultralytics/CLIP.git"
)

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
    announce_progress "TRex is installing Python ML packages. This can take several minutes; progress below shows the latest pip activity."
    build_torch_target_candidates

    # Resolve and install from one target at a time. Missing, deprecated, or
    # incompatible channels simply advance to the next ordered target.
    for candidate_index in "${!torch_target_names[@]}"; do
        select_torch_candidate "${candidate_index}"
        log "[post-link] Trying ${torch_target} from ${torch_index_url}."
        if ! resolve_torch_target; then
            log "[post-link] ${torch_target} has no compatible torch/torchvision pair for this Python; trying the next target."
            continue
        fi
        if install_selected_torch; then
            torch_installed=true
            break
        fi
        log "[post-link] ${torch_target} could not be installed with a consistent dependency set; trying the next target."
    done
fi

if ${torch_installed}; then
    installed_torch_version=$(python -c "from importlib.metadata import version; print(version('torch'))")
    installed_torchvision_version=$(python -c "from importlib.metadata import version; print(version('torchvision'))")
    log "[post-link] Installed and verified PyTorch ${installed_torch_version} + torchvision ${installed_torchvision_version} from the single ${torch_target} target."
    check_nvidia_support
else
    if ${setup_ready}; then
        record_failure "[post-link] No CUDA, CPU-only, or default PyPI torch/torchvision pair could be installed with a consistent dependency set."
    fi
fi

if ${torch_installed}; then
    log "Testing installation..."
    announce_progress "TRex is running a short YOLO smoke test to verify the Python install."
    numpy_ownership_assert=""
    if ${conda_numpy_owned}; then
        numpy_ownership_assert="assert version('numpy') == '${numpy_version}'; "
    fi
    CMD_STRING="from ultralytics import YOLO; from rfdetr import RFDETR; from torchvision.ops import nms; from importlib.metadata import version; import cv2, numpy as np, torch; ${numpy_ownership_assert}assert cv2.__version__.split('.')[0] == '4'; assert nms(torch.tensor([[0.,0.,1.,1.]]), torch.tensor([1.]), 0.5).tolist() == [0]; YOLO('yolo26n.yaml').to('cpu').predict(np.zeros((640, 480, 3), dtype=np.uint8))"
    log_command python -c "${CMD_STRING}"

    if TREX_PROGRESS_LABEL="YOLO smoke test..." run_with_reporting python -c "${CMD_STRING}"; then
        log "[post-link] YOLO smoke test succeeded."
    else
        record_failure "[post-link] YOLO smoke test failed (exit ${LAST_COMMAND_STATUS})."
    fi
fi

if [ "${POST_LINK_FAILED}" -ne 0 ]; then
    log "============================================================"
    log "WARNING: TRex PYTHON ML SETUP IS INCOMPLETE"
    log "The Conda package installation will continue successfully."
    log "TRex itself is installed, but Python ML features may be unavailable."
    log "After installation, repair the environment and run: python -m pip check"
    log "============================================================"
    echo "WARNING: TRex Python ML setup is incomplete; Conda installation will continue. See ${OUT_STREAM}." >&2
    if [ -n "${OUT_STREAM}" ] && [ "${OUT_STREAM}" != "/dev/stdout" ] && [ -f "${OUT_STREAM}" ]; then
        echo "[post-link] Dumping post-link log due to incomplete Python ML setup:" >&2
        cat "${OUT_STREAM}" >&2
    fi
fi

if [ -n "${numpy_constraint_file}" ]; then
    rm -f "${numpy_constraint_file}"
fi

exit 0
