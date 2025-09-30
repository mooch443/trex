@echo off
setlocal EnableDelayedExpansion

echo PREFIX=%PREFIX%

rem Decide where to stream post-link log messages.
if defined PREFIX (
    set "OUT_STREAM=%PREFIX%\.messages.txt"
) else (
    echo PREFIX is not set. Using stdout.
    set "OUT_STREAM="
)

set "POST_LINK_FAILED=0"
set "LAST_COMMAND_STATUS=0"

rem Ensure pip is available on Windows so subsequent installs succeed.
where /q pip
if errorlevel 1 (
    call :log "pip could not be found, installing via conda..."
    call :log_command conda install pip -y
    call :run_with_reporting conda install pip -y
    if errorlevel 1 (
        call :record_failure "[post-link] Unable to install pip via conda (exit !LAST_COMMAND_STATUS!); continuing without pip-managed extras."
    )
)

set "NUMPY_VERSION="
for /f "usebackq delims=" %%i in (`python -c "import numpy, sys; sys.stdout.write(numpy.__version__)" 2^>NUL`) do (
    set "NUMPY_VERSION=%%i"
)

if defined NUMPY_VERSION (
    call :log "Installing pip packages (numpy=!NUMPY_VERSION!)..."
) else (
    call :log "[post-link] Could not determine numpy version; proceeding without pinning."
)

rem Compose the pip command arguments that should be shared across platforms.
set "PIP_ARGS="
call :add_package "torch>=2.0.0,<2.9.0"
call :add_package "torchvision>=0.15.1,<0.24.0"
call :add_package "torchmetrics"
call :add_package "tqdm"
call :add_package "opencv-python>=4,<5"
call :add_package "ultralytics>=8.3.0,<9"
call :add_package "dill"

if defined NUMPY_VERSION (
    call :add_package "numpy==!NUMPY_VERSION!"
)

call :log "Windows detected; checking CUDA availability to document channel choice."

set "GPU_CHECK=False"
for /f "usebackq delims=" %%i in (`python -c "import sys; available = False; exec('try:\n    import torch\n    available = torch.cuda.is_available()\nexcept Exception:\n    available = False\n', globals()); sys.stdout.write('True' if available else 'False')"`) do (
    set "GPU_CHECK=%%i"
)

if /i "!GPU_CHECK!"=="True" (
    call :log "[post-link] torch.cuda.is_available() -> True; installing from default pip channels to let torch pick CUDA wheels."
) else (
    call :log "[post-link] torch.cuda.is_available() -> !GPU_CHECK!; installing from default pip channels."
)

call :log_command python -m pip install !PIP_ARGS!
call :run_with_reporting python -m pip install !PIP_ARGS!
if errorlevel 1 (
    call :record_failure "[post-link] pip package installation failed on Windows (exit !LAST_COMMAND_STATUS!)."
) else (
    call :check_nvidia_support
)

call :log "Testing installation..."
call :log_command python -c "from ultralytics import YOLO; import numpy as np; YOLO('yolo11n.pt').to('cpu').predict(np.zeros((640, 480, 3), dtype=np.uint8))"
call :run_with_reporting python -c "from ultralytics import YOLO; import numpy as np; YOLO('yolo11n.pt').to('cpu').predict(np.zeros((640, 480, 3), dtype=np.uint8))"
if errorlevel 1 (
    call :record_failure "[post-link] YOLO smoke test failed (exit !LAST_COMMAND_STATUS!)."
)

if not "!POST_LINK_FAILED!"=="0" (
    if defined OUT_STREAM (
        >&2 echo post-link.bat completed with issues; see "%OUT_STREAM%" for details.
    ) else (
        >&2 echo post-link.bat completed with issues; see stdout for details.
    )
    call :log "[post-link] Completed with issues; conda installation will continue."
)

exit /b 0

:log
setlocal EnableDelayedExpansion
set "message=%~1"
if defined OUT_STREAM (
    >>"%OUT_STREAM%" echo(!message!
) else (
    echo(!message!
)
endlocal
exit /b 0

:record_failure
set POST_LINK_FAILED=1
call :log %*
exit /b 0

:log_command
setlocal EnableDelayedExpansion
set "cmd="
:log_command_args
if "%~1"=="" goto log_command_emit
if defined cmd (
    set "cmd=!cmd! %~1"
) else (
    set "cmd=%~1"
)
shift
goto log_command_args
:log_command_emit
if not defined cmd set "cmd="
set "log_line=!cmd!"
if defined log_line (
    set "log_line=!log_line:^>=^>!"
    set "log_line=!log_line:^<=^<!"
    set "log_line=!log_line:&=^&!"
    set "log_line=!log_line:|=^|!"
)
call :log "[post-link] Running: !log_line!"
endlocal
exit /b 0
:add_package
rem Helper to accumulate quoted pip package arguments.
set "__PIP_PACKAGE=%~1"
set "__PIP_PACKAGE="!__PIP_PACKAGE!""
if defined PIP_ARGS (
    set "PIP_ARGS=!PIP_ARGS! !__PIP_PACKAGE!"
) else (
    set "PIP_ARGS=!__PIP_PACKAGE!"
)
set "__PIP_PACKAGE="
exit /b 0

:check_nvidia_support
call :log "[post-link] Checking NVIDIA GPU support after install..."

set "CUDA_RESULT="
for /f "usebackq delims=" %%i in (`python -c "exec('try:\n import torch\n available = torch.cuda.is_available()\nexcept Exception:\n available = None\nprint(True if available else (False if available is not None else None))')"` ) do (
    set "CUDA_RESULT=%%i"
)
if defined CUDA_RESULT (
    call :log "[post-link] torch.cuda.is_available() after install -> !CUDA_RESULT!"
) else (
    call :log "[post-link] Unable to query torch CUDA availability after install."
)

where /q nvidia-smi
if errorlevel 1 (
    call :log "[post-link] nvidia-smi not found; NVIDIA GPU likely unavailable."
    exit /b 0
)

set "GPU_TMP="
if defined TEMP (
    set "GPU_TMP=%TEMP%\trex_nvidia_gpu.txt"
) else (
    set "GPU_TMP=%CD%\trex_nvidia_gpu.txt"
)

del /q "!GPU_TMP!" >nul 2>&1
call :log_command nvidia-smi --query-gpu=name --format=csv,noheader
nvidia-smi --query-gpu=name --format=csv,noheader >"!GPU_TMP!" 2>&1
set "GPU_CMD_STATUS=!ERRORLEVEL!"

if defined OUT_STREAM (
    type "!GPU_TMP!" >>"%OUT_STREAM%"
) else (
    type "!GPU_TMP!"
)

if "!GPU_CMD_STATUS!"=="0" (
    set "GPU_NAMES="
    for /f "usebackq delims=" %%i in ("!GPU_TMP!") do (
        if defined GPU_NAMES (
            set "GPU_NAMES=!GPU_NAMES!, %%i"
        ) else (
            set "GPU_NAMES=%%i"
        )
    )
    if defined GPU_NAMES (
        call :log "[post-link] NVIDIA GPUs detected via nvidia-smi: !GPU_NAMES!"
    ) else (
        call :log "[post-link] nvidia-smi ran successfully but reported no GPUs."
    )
) else (
    call :log "[post-link] nvidia-smi query failed (exit !GPU_CMD_STATUS!)."
)

del /q "!GPU_TMP!" >nul 2>&1
exit /b 0

:run_with_reporting
setlocal EnableDelayedExpansion
if defined OUT_STREAM (
    >>"%OUT_STREAM%" 2>&1 cmd /c %*
) else (
    cmd /c %*
)
set "status=%ERRORLEVEL%"
endlocal & set "LAST_COMMAND_STATUS=%status%"
exit /b %status%
