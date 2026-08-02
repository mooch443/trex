@echo off
setlocal EnableDelayedExpansion

chcp 65001 >nul 2>&1
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "PIP_PROGRESS_BAR=off"
set "PIP_NO_INPUT=1"
set "PIP_INSTALL_FLAGS=--disable-pip-version-check --no-input --progress-bar off --no-color --quiet"
set "ULTRALYTICS_HUB_NO_PROGRESS=1"
set "HF_HUB_DISABLE_PROGRESS_BAR=1"
set "DISABLE_TQDM=1"
set "RICH_NO_COLOR=1"
set "RICH_FORCE_TERMINAL=0"
set "FORCE_COLOR=0"

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

rem pip is a Conda run dependency. Do not invoke Conda recursively while its
rem transaction is still linking this environment.
python -X utf8 -m pip --version >nul 2>&1
if errorlevel 1 (
    call :record_failure "[post-link] pip is unavailable; skipping pip-managed extras."
    goto post_link_finish
)

rem Compose pip argument lists. torch and torchvision are installed first from
rem compatible official CUDA indexes; non-torch packages are then installed from
rem PyPI so ordinary dependencies are never resolved against a CUDA-only index.
rem
rem NumPy is Conda-owned. Capture its exact installed version in a pip constraint
rem so no pip command can replace it with a wheel from PyPI or a PyTorch index.
set "NUMPY_CONSTRAINT_FILE=%TEMP%\trex_numpy_constraint_%RANDOM%.txt"
call :configure_numpy_constraint
if errorlevel 1 (
    call :record_failure "[post-link] Could not constrain pip to the Conda-installed NumPy; skipping pip-managed extras."
    goto post_link_finish
)

set "PIP_ARGS="
call :add_package "torchmetrics"
call :add_package "tqdm"
set "HAS_CONDA_PY_OPENCV=0"
if exist "%PREFIX%\conda-meta\py-opencv-*.json" set "HAS_CONDA_PY_OPENCV=1"
if "!HAS_CONDA_PY_OPENCV!"=="1" (
    call :log "Conda py-opencv detected; keeping its cv2 module and skipping the PyPI OpenCV wheel."
) else (
    call :log "Conda py-opencv not detected; adding opencv-python^>=4,^<5 for the buildall profile."
    call :add_package "opencv-python>=4,<5"
)
call :add_package "ultralytics>=8.3.0,<9"
call :add_package "rfdetr==1.8.3"
call :add_package "dill"
call :add_package "scikit-learn"
call :add_package "timm"
call :add_package "git+https://github.com/ultralytics/CLIP.git"
set "PIP_ARGS_SIMPLE=!PIP_ARGS!"

set "PIP_ARGS="
call :add_package "torch>=2.2.0,<3.0.0"
call :add_package "torchvision>=0.17.0"
set "PIP_ARGS_TORCH=!PIP_ARGS!"

set "PIP_ARGS="
call :add_package "torch==2.5.0"
call :add_package "torchvision==0.20.0"
set "PIP_ARGS_TORCH_FALLBACK=!PIP_ARGS!"

rem Remove only the pip-managed torch pair before resolving against the selected
rem CUDA indexes. This replaces an existing incompatible wheel without using
rem --force-reinstall, which would also overwrite Conda-owned NumPy dependencies.
call :log_command python -X utf8 -m pip uninstall --yes torch torchvision
call :run_with_reporting python -X utf8 -m pip uninstall --yes torch torchvision

call :log "Windows detected; selecting PyTorch against the installed NVIDIA driver."
call :detect_driver_cuda_version
if defined CUDA_MAX_VERSION (
    call :log "[post-link] NVIDIA driver accepts CUDA !CUDA_MAX_VERSION!; selecting the newest PyTorch release from compatible official indexes."
    call :log "[post-link] Compatible PyTorch CUDA channels: !CUDA_CHANNELS!."
) else (
    call :log "[post-link] No usable NVIDIA driver was detected for PyTorch package selection."
)

rem Spin up a background progress indicator that writes directly to CONOUT$ via
rem ctypes, bypassing conda's pipe that holds .messages.txt until the script exits.
rem The script polls a sentinel file; we create it when the pip install finishes.
rem Note: | and & must be escaped as ^| and ^& in echo outside a paren block.
set "PROGRESS_PY=%TEMP%\trex_pip_progress_%RANDOM%.py"
set "PROGRESS_STOP=%TEMP%\trex_pip_stop_%RANDOM%.flag"
set "PROGRESS_LOG=%TEMP%\trex_pip_log_%RANDOM%.txt"
if exist "%PROGRESS_STOP%" del "%PROGRESS_STOP%" 2>nul
if exist "%PROGRESS_LOG%" del "%PROGRESS_LOG%" 2>nul

rem Python progress script: reads the last non-empty line from the live pip log
rem and displays it on the console via CONOUT$ (bypasses conda's stdout pipe).
rem Indentation uses 1 space throughout to keep echo escaping simple.
rem Batch special chars in echo outside paren blocks: | -> ^|  & -> ^&
echo import ctypes,time,os,sys > "%PROGRESS_PY%"
echo k=ctypes.windll.kernel32 >> "%PROGRESS_PY%"
echo h=k.CreateFileW("CONOUT$",0x40000000,3,None,3,0,None) >> "%PROGRESS_PY%"
echo if h==-1:sys.exit(0) >> "%PROGRESS_PY%"
echo ENABLE_PROCESSED_OUTPUT=1 >> "%PROGRESS_PY%"
echo ENABLE_VIRTUAL_TERMINAL_PROCESSING=4 >> "%PROGRESS_PY%"
echo mode=ctypes.c_ulong(0) >> "%PROGRESS_PY%"
echo vt=False >> "%PROGRESS_PY%"
echo if k.GetConsoleMode(h,ctypes.byref(mode)):vt=bool(k.SetConsoleMode(h,mode.value ^| ENABLE_PROCESSED_OUTPUT ^| ENABLE_VIRTUAL_TERMINAL_PROCESSING)) >> "%PROGRESS_PY%"
echo stop=sys.argv[1] >> "%PROGRESS_PY%"
echo log=sys.argv[2] >> "%PROGRESS_PY%"
echo s=time.time() >> "%PROGRESS_PY%"
echo i=0 >> "%PROGRESS_PY%"
echo w=ctypes.c_ulong(0) >> "%PROGRESS_PY%"
echo frames=["⠋","⠙","⠚","⠞","⠖","⠦","⠴","⠲","⠳","⠓"] >> "%PROGRESS_PY%"
echo BLUE="\033[34m" if vt else "" >> "%PROGRESS_PY%"
echo RESET="\033[0m" if vt else "" >> "%PROGRESS_PY%"
echo HIDE="\033[?25l" if vt else "" >> "%PROGRESS_PY%"
echo CLEAR="\r\033[2K\033[?25h" if vt else "\r"+" "*120+"\r" >> "%PROGRESS_PY%"
echo if HIDE:k.WriteConsoleW(h,HIDE,len(HIDE),ctypes.byref(w),None) >> "%PROGRESS_PY%"
echo def last(p): >> "%PROGRESS_PY%"
echo  try: >> "%PROGRESS_PY%"
echo   with open(p,"rb") as f: >> "%PROGRESS_PY%"
echo    f.seek(0,2) >> "%PROGRESS_PY%"
echo    sz=f.tell() >> "%PROGRESS_PY%"
echo    f.seek(max(0,sz-2048)) >> "%PROGRESS_PY%"
echo    chunk=f.read(2048) >> "%PROGRESS_PY%"
echo   lines=chunk.decode("utf-8",errors="replace").splitlines() >> "%PROGRESS_PY%"
echo   for ln in reversed(lines): >> "%PROGRESS_PY%"
echo    ln=ln.strip() >> "%PROGRESS_PY%"
echo    if ln:return ln[:60] >> "%PROGRESS_PY%"
echo  except:pass >> "%PROGRESS_PY%"
echo  return "" >> "%PROGRESS_PY%"
echo while not os.path.exists(stop): >> "%PROGRESS_PY%"
echo  e=int(time.time()-s) >> "%PROGRESS_PY%"
echo  m,r=divmod(e,60) >> "%PROGRESS_PY%"
echo  info=last(log) >> "%PROGRESS_PY%"
echo  frame=frames[i%%len(frames)] >> "%PROGRESS_PY%"
echo  if info: >> "%PROGRESS_PY%"
echo   msg="\r"+BLUE+frame+RESET+" "+info+"  "+str(m).zfill(2)+":"+str(r).zfill(2)+"   " >> "%PROGRESS_PY%"
echo  else: >> "%PROGRESS_PY%"
echo   msg="\r"+BLUE+frame+RESET+" pip install...  "+str(m).zfill(2)+":"+str(r).zfill(2)+"   " >> "%PROGRESS_PY%"
echo  k.WriteConsoleW(h,msg,len(msg),ctypes.byref(w),None) >> "%PROGRESS_PY%"
echo  i=i+1 >> "%PROGRESS_PY%"
echo  time.sleep(0.1) >> "%PROGRESS_PY%"
echo k.WriteConsoleW(h,CLEAR,len(CLEAR),ctypes.byref(w),None) >> "%PROGRESS_PY%"
echo k.CloseHandle(h) >> "%PROGRESS_PY%"

start "" /b python -X utf8 "%PROGRESS_PY%" "%PROGRESS_STOP%" "%PROGRESS_LOG%"

rem Verbose flags for pip: no --quiet so Collecting/Downloading/Installing lines appear
rem in PROGRESS_LOG for the live display. The log is appended to OUT_STREAM afterwards.
set "PIP_FLAGS_LOG=--disable-pip-version-check --no-input --no-color --progress-bar off"

rem Step 1: offer pip every official CUDA index accepted by the driver. Pip then
rem chooses the newest PyTorch release available across those compatible indexes.
rem Installing torch first prevents ordinary PyPI dependencies from selecting an
rem incompatible default wheel.
set "TORCH_INSTALLED=0"
if defined CUDA_MAX_VERSION (
    call :log_command python -X utf8 -m pip install !PIP_FLAGS_LOG! --constraint "!NUMPY_CONSTRAINT_FILE!" !PIP_TORCH_INDEX_ARGS! !PIP_ARGS_TORCH!
    python -X utf8 -m pip install !PIP_FLAGS_LOG! --constraint "!NUMPY_CONSTRAINT_FILE!" !PIP_TORCH_INDEX_ARGS! !PIP_ARGS_TORCH! > "%PROGRESS_LOG%" 2>&1
    set "LAST_COMMAND_STATUS=!ERRORLEVEL!"
    if defined OUT_STREAM (
        type "%PROGRESS_LOG%" >> "%OUT_STREAM%" 2>nul
    )
    if "!LAST_COMMAND_STATUS!"=="0" (
        set "TORCH_INSTALLED=1"
        call :log "[post-link] PyTorch installation succeeded using driver-compatible CUDA indexes."
        call :check_nvidia_support
    ) else (
        call :log "[post-link] No PyTorch wheel could be resolved from the driver-compatible CUDA indexes (exit !LAST_COMMAND_STATUS!)."
    )
)

if "!TORCH_INSTALLED!"=="0" (
    set "PIP_INDEX_URL=https://download.pytorch.org/whl/cu118"
    call :log "[post-link] Falling back to the oldest CUDA pair compatible with the supported NumPy 1.x/2.x range: torch 2.5.0 + torchvision 0.20.0 (CUDA 11.8)."
    call :log_command python -X utf8 -m pip install !PIP_FLAGS_LOG! --constraint "!NUMPY_CONSTRAINT_FILE!" --index-url !PIP_INDEX_URL! !PIP_ARGS_TORCH_FALLBACK!
    python -X utf8 -m pip install !PIP_FLAGS_LOG! --constraint "!NUMPY_CONSTRAINT_FILE!" --index-url !PIP_INDEX_URL! !PIP_ARGS_TORCH_FALLBACK! > "%PROGRESS_LOG%" 2>&1
    set "LAST_COMMAND_STATUS=!ERRORLEVEL!"
    if defined OUT_STREAM (
        type "%PROGRESS_LOG%" >> "%OUT_STREAM%" 2>nul
    )
    if "!LAST_COMMAND_STATUS!"=="0" (
        set "TORCH_INSTALLED=1"
        call :log "[post-link] PyTorch CUDA 11.8 fallback installation succeeded."
        call :check_nvidia_support
    )
)
if "!TORCH_INSTALLED!"=="0" (
    call :record_failure "[post-link] PyTorch installation failed for both compatible CUDA indexes and the CUDA 11.8 fallback (exit !LAST_COMMAND_STATUS!)."
    goto pip_install_after
)

:torch_install_done

rem Step 2: install remaining packages from PyPI. torch is already present so pip
rem will not attempt to resolve a different (CPU-only) torch from PyPI.
call :log "[post-link] Installing non-torch packages from PyPI..."
call :log_command python -X utf8 -m pip install !PIP_FLAGS_LOG! --constraint "!NUMPY_CONSTRAINT_FILE!" --index-url https://pypi.org/simple !PIP_ARGS_SIMPLE!
python -X utf8 -m pip install !PIP_FLAGS_LOG! --constraint "!NUMPY_CONSTRAINT_FILE!" --index-url https://pypi.org/simple !PIP_ARGS_SIMPLE! > "%PROGRESS_LOG%" 2>&1
set "LAST_COMMAND_STATUS=!ERRORLEVEL!"
if defined OUT_STREAM (
    type "%PROGRESS_LOG%" >> "%OUT_STREAM%" 2>nul
)
if not "!LAST_COMMAND_STATUS!"=="0" (
    call :record_failure "[post-link] Non-torch pip install failed (exit !LAST_COMMAND_STATUS!)."
)

:pip_install_after

rem Signal the progress indicator to stop. The sentinel is left in %TEMP% (harmless random-named
rem file) so the Python polling loop cannot miss it by racing against a delete.
copy nul "%PROGRESS_STOP%" >nul 2>&1
timeout /t 1 /nobreak >nul 2>&1
del "%PROGRESS_PY%" "%PROGRESS_LOG%" 2>nul

call :log "Testing installation..."
call :log_command python -X utf8 -c "from ultralytics import YOLO; from rfdetr import RFDETR; from torchvision.ops import nms; from importlib.metadata import version; import cv2, numpy as np, torch; assert version('numpy') == '!NUMPY_VERSION!'; assert version('opencv-python').split('.')[0] == '4'; assert nms(torch.tensor([[0.,0.,1.,1.]]), torch.tensor([1.]), 0.5).tolist() == [0]; YOLO('yolo26n.yaml').to('cpu').predict(np.zeros((640, 480, 3), dtype=np.uint8))"
call :run_with_reporting python -X utf8 -c "from ultralytics import YOLO; from rfdetr import RFDETR; from torchvision.ops import nms; from importlib.metadata import version; import cv2, numpy as np, torch; assert version('numpy') == '!NUMPY_VERSION!'; assert version('opencv-python').split('.')[0] == '4'; assert nms(torch.tensor([[0.,0.,1.,1.]]), torch.tensor([1.]), 0.5).tolist() == [0]; YOLO('yolo26n.yaml').to('cpu').predict(np.zeros((640, 480, 3), dtype=np.uint8))"
if errorlevel 1 (
    call :record_failure "[post-link] YOLO smoke test failed (exit !LAST_COMMAND_STATUS!)."
)

:post_link_finish
if not "!POST_LINK_FAILED!"=="0" (
    call :log "[post-link] Completed with issues; conda installation will continue."
    if defined OUT_STREAM (
        >&2 echo post-link.bat completed with issues; see "%OUT_STREAM%" for details.
        if exist "%OUT_STREAM%" (
            >&2 echo [post-link] Dumping post-link log due to failures:
            type "%OUT_STREAM%" 1>&2
        )
    ) else (
        >&2 echo post-link.bat completed with issues; see stdout for details.
    )
)

if defined NUMPY_CONSTRAINT_FILE del /q "!NUMPY_CONSTRAINT_FILE!" >nul 2>&1

exit /b 0

:configure_numpy_constraint
set "NUMPY_VERSION="
for /f "usebackq delims=" %%i in (`python -X utf8 -c "import numpy,sys; sys.stdout.write(numpy.__version__)" 2^>NUL`) do set "NUMPY_VERSION=%%i"
if not defined NUMPY_VERSION exit /b 1
>"!NUMPY_CONSTRAINT_FILE!" echo numpy==!NUMPY_VERSION!
call :log "[post-link] Constraining pip to Conda-owned NumPy !NUMPY_VERSION!."
exit /b 0

:log
setlocal EnableDelayedExpansion

chcp 65001 >nul 2>&1
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "PIP_PROGRESS_BAR=off"
set "PIP_NO_INPUT=1"
set "PIP_INSTALL_FLAGS=--disable-pip-version-check --no-input --progress-bar off --no-color --quiet"
set "ULTRALYTICS_HUB_NO_PROGRESS=1"
set "HF_HUB_DISABLE_PROGRESS_BAR=1"
set "DISABLE_TQDM=1"
set "RICH_NO_COLOR=1"
set "RICH_FORCE_TERMINAL=0"
set "FORCE_COLOR=0"
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

chcp 65001 >nul 2>&1
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "PIP_PROGRESS_BAR=off"
set "PIP_NO_INPUT=1"
set "PIP_INSTALL_FLAGS=--disable-pip-version-check --no-input --progress-bar off --no-color --quiet"
set "ULTRALYTICS_HUB_NO_PROGRESS=1"
set "HF_HUB_DISABLE_PROGRESS_BAR=1"
set "DISABLE_TQDM=1"
set "RICH_NO_COLOR=1"
set "RICH_FORCE_TERMINAL=0"
set "FORCE_COLOR=0"
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

:detect_driver_cuda_version
set "CUDA_MAX_VERSION="
set "CUDA_MAX_CODE="
set "CUDA_CHANNELS="
set "PIP_TORCH_INDEX_ARGS="
set "CUDA_SMI_TMP=%TEMP%\trex_nvidia_smi_%RANDOM%.txt"

where /q nvidia-smi
if errorlevel 1 exit /b 1

nvidia-smi >"!CUDA_SMI_TMP!" 2>&1
if errorlevel 1 (
    if defined OUT_STREAM type "!CUDA_SMI_TMP!" >>"!OUT_STREAM!" 2>nul
    del /q "!CUDA_SMI_TMP!" >nul 2>&1
    exit /b 1
)

for /f "usebackq delims=" %%i in (`python -X utf8 -c "import re,sys; text=open(sys.argv[1],encoding='utf-8',errors='replace').read(); match=re.search(r'CUDA(?:\s+UMD)?\s+Version:\s*([0-9]+\.[0-9]+)',text); print(match.group(1) if match else '')" "!CUDA_SMI_TMP!"`) do (
    set "CUDA_MAX_VERSION=%%i"
)
if defined OUT_STREAM type "!CUDA_SMI_TMP!" >>"!OUT_STREAM!" 2>nul
del /q "!CUDA_SMI_TMP!" >nul 2>&1
if not defined CUDA_MAX_VERSION (
    call :log "[post-link] nvidia-smi did not report a maximum supported CUDA version."
    exit /b 1
)

for /f "tokens=1,2 delims=." %%a in ("!CUDA_MAX_VERSION!") do (
    set /a CUDA_MAX_CODE=%%a*100+%%b
)

rem Keep this list aligned with https://pytorch.org/get-started/previous-versions/.
call :add_compatible_cuda_channel cu132 1302
call :add_compatible_cuda_channel cu130 1300
call :add_compatible_cuda_channel cu129 1209
call :add_compatible_cuda_channel cu128 1208
call :add_compatible_cuda_channel cu126 1206
call :add_compatible_cuda_channel cu124 1204
call :add_compatible_cuda_channel cu121 1201
call :add_compatible_cuda_channel cu118 1108

if not defined CUDA_CHANNELS (
    set "CUDA_MAX_VERSION="
    exit /b 1
)
exit /b 0

:add_compatible_cuda_channel
if %2 GTR !CUDA_MAX_CODE! exit /b 0

if defined CUDA_CHANNELS (
    set "CUDA_CHANNELS=!CUDA_CHANNELS! %1"
    set "PIP_TORCH_INDEX_ARGS=!PIP_TORCH_INDEX_ARGS! --extra-index-url https://download.pytorch.org/whl/%1"
) else (
    set "CUDA_CHANNELS=%1"
    set "PIP_TORCH_INDEX_ARGS=--index-url https://download.pytorch.org/whl/%1"
)
exit /b 0

:check_nvidia_support
call :log "[post-link] Checking NVIDIA GPU support after install..."

set "CUDA_RESULT="
for /f "usebackq delims=" %%i in (`python -X utf8 -c "import torch; print(torch.cuda.is_available())" 2^>NUL`) do (
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

chcp 65001 >nul 2>&1
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "PIP_PROGRESS_BAR=off"
set "PIP_NO_INPUT=1"
set "PIP_INSTALL_FLAGS=--disable-pip-version-check --no-input --progress-bar off --no-color --quiet"
set "ULTRALYTICS_HUB_NO_PROGRESS=1"
set "HF_HUB_DISABLE_PROGRESS_BAR=1"
set "DISABLE_TQDM=1"
set "RICH_NO_COLOR=1"
set "RICH_FORCE_TERMINAL=0"
set "FORCE_COLOR=0"
if defined OUT_STREAM (
    >>"%OUT_STREAM%" 2>&1 cmd /c %*
) else (
    cmd /c %*
)
set "status=%ERRORLEVEL%"
endlocal & set "LAST_COMMAND_STATUS=%status%"
exit /b %status%
