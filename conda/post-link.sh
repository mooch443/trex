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

# Select exactly one target before wheel installation. A missing or obsolete CUDA
# index falls back to CPU during this metadata-only probe, before either wheel is
# downloaded. Ordinary package resolution is never given multiple Torch indexes.
select_torch_target() {
    torch_target="CPU"
    torch_index_url="https://download.pytorch.org/whl/cpu"
    torch_index_args=(--index-url "${torch_index_url}")
    torch_dependency_index_args=(--extra-index-url "https://pypi.org/simple")

    if [ "${system}" = "Darwin" ]; then
        torch_target="macOS/PyPI"
        torch_index_url="https://pypi.org/simple"
        torch_index_args=(--index-url "${torch_index_url}")
        torch_dependency_index_args=()
        log "[post-link] macOS detected; selecting the native PyTorch distribution."
        return 0
    fi

    case "${arch}" in
        arm|arm64|aarch64)
            torch_target="Linux ARM/PyPI"
            torch_index_url="https://pypi.org/simple"
            torch_index_args=(--index-url "${torch_index_url}")
            torch_dependency_index_args=()
            log "[post-link] ARM architecture detected; selecting the native CPU PyTorch distribution."
            return 0
            ;;
    esac

    if [ "${system}" != "Linux" ]; then
        torch_target="PyPI CPU"
        torch_index_url="https://pypi.org/simple"
        torch_index_args=(--index-url "${torch_index_url}")
        torch_dependency_index_args=()
        log "[post-link] ${system} detected; selecting the native CPU PyTorch distribution."
        return 0
    fi

    local driver_cuda_version driver_major driver_minor driver_cuda_code channel=""
    if ! detect_driver_cuda_version; then
        log "[post-link] No usable NVIDIA driver detected; selecting the CPU-only PyTorch distribution."
        return 0
    fi
    driver_cuda_version="${detected_cuda_version}"

    IFS=. read -r driver_major driver_minor <<EOF
${driver_cuda_version}
EOF
    if ! [[ "${driver_major}" =~ ^[0-9]+$ && "${driver_minor}" =~ ^[0-9]+$ ]]; then
        log "[post-link] Could not parse NVIDIA CUDA compatibility '${driver_cuda_version}'; selecting CPU-only PyTorch."
        return 0
    fi
    driver_cuda_code=$((driver_major * 100 + driver_minor))

    if [ "${driver_cuda_code}" -ge 1302 ]; then channel="cu132"
    elif [ "${driver_cuda_code}" -ge 1300 ]; then channel="cu130"
    elif [ "${driver_cuda_code}" -ge 1209 ]; then channel="cu129"
    elif [ "${driver_cuda_code}" -ge 1208 ]; then channel="cu128"
    elif [ "${driver_cuda_code}" -ge 1206 ]; then channel="cu126"
    elif [ "${driver_cuda_code}" -ge 1204 ]; then channel="cu124"
    elif [ "${driver_cuda_code}" -ge 1201 ]; then channel="cu121"
    elif [ "${driver_cuda_code}" -ge 1108 ]; then channel="cu118"
    else
        log "[post-link] NVIDIA driver supports CUDA ${driver_cuda_version}, below the supported CUDA 11.8 baseline; selecting CPU-only PyTorch."
        return 0
    fi

    torch_target="CUDA ${channel#cu}"
    torch_index_url="https://download.pytorch.org/whl/${channel}"
    torch_index_args=(--index-url "${torch_index_url}")
    log "[post-link] NVIDIA driver accepts CUDA ${driver_cuda_version}; selecting the single ${torch_target} PyTorch distribution."
}

select_cpu_torch_target() {
    torch_target="CPU fallback"
    torch_index_url="https://download.pytorch.org/whl/cpu"
    torch_index_args=(--index-url "${torch_index_url}")
    torch_dependency_index_args=(--extra-index-url "https://pypi.org/simple")
}

select_pypi_torch_target() {
    torch_target="PyPI fallback"
    torch_index_url="https://pypi.org/simple"
    torch_index_args=(--index-url "${torch_index_url}")
    torch_dependency_index_args=()
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

# pip is a Conda run dependency. Do not invoke Conda recursively while its
# transaction is still linking this environment.
if ! python -m pip --version >/dev/null 2>&1; then
    record_failure "[post-link] pip is unavailable; skipping pip-managed extras."
    exit 0
fi

arch=$(uname -m)
system=$(uname)

# NumPy is Conda-owned. Constrain every pip resolution to the exact installed
# version without making NumPy a pip installation target.
numpy_version=""
numpy_constraint_file=""
numpy_constraint_args=()

configure_numpy_constraint() {
    numpy_version=$(python -c "import numpy; print(numpy.__version__)" 2>>"${OUT_STREAM}")
    local numpy_status=$?
    if [ ${numpy_status} -ne 0 ] || [ -z "${numpy_version}" ]; then
        return 1
    fi

    if [ -z "${numpy_constraint_file}" ]; then
        numpy_constraint_file=$(mktemp "${TMPDIR:-/tmp}/trex_numpy_constraint.XXXXXX") || return 1
    fi
    printf 'numpy==%s\n' "${numpy_version}" > "${numpy_constraint_file}" || return 1
    numpy_constraint_args=(--constraint "${numpy_constraint_file}")
    log "[post-link] Constraining pip to Conda-owned NumPy ${numpy_version}."
}

if ! configure_numpy_constraint; then
    record_failure "[post-link] Could not constrain pip to the Conda-installed NumPy; skipping pip-managed extras."
    exit 0
fi

torch_packages=()

common_packages=(
    "torchmetrics"
    "tqdm"
    "ultralytics>=8.3.0,<9"
    "rfdetr==1.8.3"
    "dill"
    "timm"
    "scikit-learn"
    "git+https://github.com/ultralytics/CLIP.git"
)

has_conda_py_opencv=false
for conda_record in "${PREFIX}"/conda-meta/py-opencv-*.json; do
    if [ -f "${conda_record}" ]; then
        has_conda_py_opencv=true
        break
    fi
done

if ${has_conda_py_opencv}; then
    log "Conda py-opencv detected; keeping its cv2 module and skipping the PyPI OpenCV wheel."
else
    log "Conda py-opencv not detected; adding opencv-python>=4,<5 for the buildall profile."
    common_packages+=("opencv-python>=4,<5")
fi

pip_flags=(
    --disable-pip-version-check
    --no-input
    --no-color
    --progress-bar
    off
)

announce_progress "TRex is installing Python ML packages. This can take several minutes; progress below shows the latest pip activity."

# Resolve exact variants from one target before installation. Exact pins prevent
# PyPI dependency lookup from substituting a different Torch distribution, while
# the normal resolver still validates and installs the pair's dependencies.
select_torch_target
torch_installed=false
torch_resolved=false
if resolve_torch_target; then
    torch_resolved=true
elif [[ "${torch_target}" == CUDA* ]]; then
    log "[post-link] The ${torch_target} index has no compatible torch/torchvision pair; selecting the newest CPU-only distribution."
    select_cpu_torch_target
    if resolve_torch_target; then torch_resolved=true; fi
fi
if ! ${torch_resolved} && [[ "${torch_target}" != *PyPI* ]]; then
    log "[post-link] The ${torch_target} index has no compatible pair; selecting the newest PyTorch distribution on PyPI."
    select_pypi_torch_target
    if resolve_torch_target; then torch_resolved=true; fi
fi

if ${torch_resolved} && install_selected_torch; then
    torch_installed=true
elif [[ "${torch_target}" == CUDA* ]]; then
    log "[post-link] The ${torch_target} install failed; falling back to the newest CPU-only PyTorch distribution."
    select_cpu_torch_target
    if resolve_torch_target && install_selected_torch; then
        torch_installed=true
    fi
fi

if ! ${torch_installed} && [[ "${torch_target}" != *PyPI* ]]; then
    log "[post-link] The ${torch_target} install failed; falling back to the newest PyTorch distribution on PyPI."
    select_pypi_torch_target
    if resolve_torch_target && install_selected_torch; then
        torch_installed=true
    fi
fi

if ${torch_installed}; then
    installed_torch_version=$(python -c "from importlib.metadata import version; print(version('torch'))")
    installed_torchvision_version=$(python -c "from importlib.metadata import version; print(version('torchvision'))")
    log "[post-link] Installed and verified PyTorch ${installed_torch_version} + torchvision ${installed_torchvision_version} from the single ${torch_target} target."
    check_nvidia_support
else
    record_failure "[post-link] The selected ${torch_target} PyTorch pair could not be installed or verified (exit ${LAST_COMMAND_STATUS})."
fi

log "Testing installation..."
announce_progress "TRex is running a short YOLO smoke test to verify the Python install."

CMD_STRING="from ultralytics import YOLO; from rfdetr import RFDETR; from torchvision.ops import nms; from importlib.metadata import version; import cv2, numpy as np, torch; assert version('numpy') == '${numpy_version}'; assert cv2.__version__.split('.')[0] == '4'; assert nms(torch.tensor([[0.,0.,1.,1.]]), torch.tensor([1.]), 0.5).tolist() == [0]; YOLO('yolo26n.yaml').to('cpu').predict(np.zeros((640, 480, 3), dtype=np.uint8))"
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

if [ -n "${numpy_constraint_file}" ]; then
    rm -f "${numpy_constraint_file}"
fi

exit 0
