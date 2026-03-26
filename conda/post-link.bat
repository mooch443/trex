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

rem Compose the pip command arguments that should be shared across platforms.
rem numpy and scikit-learn are intentionally excluded from conda run deps on Windows
rem (see meta.yaml) to prevent conda from installing a conda-managed numpy whose DLL
rem layout is incompatible with torch wheels from download.pytorch.org (WinError 127 /
rem fbgemm.dll). Both are installed here via pip so DLL ownership is consistent.
set "PIP_ARGS="
call :add_package "torch>=2.0.0,<2.9.0"
call :add_package "torchvision>=0.15.1,<0.24.0"
call :add_package "torchmetrics"
call :add_package "tqdm"
call :add_package "opencv-python>=4,<5"
call :add_package "ultralytics>=8.3.0,<9"
call :add_package "dill"
call :add_package "numpy>=1.26,<3"
call :add_package "scikit-learn"

call :log "Windows detected; checking CUDA availability to document channel choice."

set "GPU_CHECK=None"
for /f "usebackq delims=" %%i in (`python -c "try: import torch; print(torch.cuda.is_available())\nexcept: print('None')"` ) do (
    set "GPU_CHECK=%%i"
)

if /i "!GPU_CHECK!"=="True" (
    call :log "[post-link] torch.cuda.is_available() -> True; selecting PyTorch CUDA wheel channel accordingly."
) else (
    call :log "[post-link] torch.cuda.is_available() -> !GPU_CHECK!; still selecting a PyTorch CUDA wheel channel for Windows."
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
echo stop=sys.argv[1] >> "%PROGRESS_PY%"
echo log=sys.argv[2] >> "%PROGRESS_PY%"
echo s=time.time() >> "%PROGRESS_PY%"
echo i=0 >> "%PROGRESS_PY%"
echo w=ctypes.c_ulong(0) >> "%PROGRESS_PY%"
echo frames=["|","/","-","\\"] >> "%PROGRESS_PY%"
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
echo  if info: >> "%PROGRESS_PY%"
echo   msg="\r  "+frames[i ^& 3]+" "+info+"  "+str(m).zfill(2)+":"+str(r).zfill(2)+"   " >> "%PROGRESS_PY%"
echo  else: >> "%PROGRESS_PY%"
echo   msg="\r  "+frames[i ^& 3]+" pip install...  "+str(m).zfill(2)+":"+str(r).zfill(2)+"   " >> "%PROGRESS_PY%"
echo  k.WriteConsoleW(h,msg,len(msg),ctypes.byref(w),None) >> "%PROGRESS_PY%"
echo  i=i+1 >> "%PROGRESS_PY%"
echo  time.sleep(0.5) >> "%PROGRESS_PY%"
echo clear="\r"+" "*80+"\r" >> "%PROGRESS_PY%"
echo k.WriteConsoleW(h,clear,len(clear),ctypes.byref(w),None) >> "%PROGRESS_PY%"
echo k.CloseHandle(h) >> "%PROGRESS_PY%"

start "" /b python -X utf8 "%PROGRESS_PY%" "%PROGRESS_STOP%" "%PROGRESS_LOG%"

set "CUDA_CHANNEL_SUFFIX="
set "CUDA_CHANNELS=cu128 cu126 cu124 cu122 cu121 cu118"

rem Verbose flags for pip: no --quiet so Collecting/Downloading/Installing lines appear
rem in PROGRESS_LOG for the live display. The log is appended to OUT_STREAM afterwards.
set "PIP_FLAGS_LOG=--disable-pip-version-check --no-input --no-color --progress-bar off"

for %%C in (!CUDA_CHANNELS!) do (
    set "CUDA_CHANNEL_SUFFIX=%%C"
    set "PIP_INDEX_URL=https://download.pytorch.org/whl/%%C"
    call :log "[post-link] Trying PyTorch install with CUDA channel %%C (!PIP_INDEX_URL!)."
    call :log_command python -X utf8 -m pip install !PIP_FLAGS_LOG! --index-url !PIP_INDEX_URL! --extra-index-url https://pypi.org/simple !PIP_ARGS!
    python -X utf8 -m pip install !PIP_FLAGS_LOG! --index-url !PIP_INDEX_URL! --extra-index-url https://pypi.org/simple !PIP_ARGS! > "%PROGRESS_LOG%" 2>&1
    set "LAST_COMMAND_STATUS=!ERRORLEVEL!"
    if defined OUT_STREAM (
        type "%PROGRESS_LOG%" >> "%OUT_STREAM%" 2>nul
    )
    if "!LAST_COMMAND_STATUS!"=="0" (
        call :log "[post-link] pip install succeeded using CUDA channel %%C."
        call :check_nvidia_support
        goto pip_install_after
    )
    call :log "[post-link] pip install failed for CUDA channel %%C (exit !LAST_COMMAND_STATUS!); trying next option."
)

call :record_failure "[post-link] pip package installation failed for all CUDA channels (last exit !LAST_COMMAND_STATUS!)."

:pip_install_after

rem Signal the progress indicator to stop. The sentinel is left in %TEMP% (harmless random-named
rem file) so the Python polling loop cannot miss it by racing against a delete.
copy nul "%PROGRESS_STOP%" >nul 2>&1
timeout /t 1 /nobreak >nul 2>&1
del "%PROGRESS_PY%" "%PROGRESS_LOG%" 2>nul

call :log "Testing installation..."
call :log_command python -X utf8 -c "from ultralytics import YOLO; import numpy as np; YOLO('yolo26n.yaml').to('cpu').predict(np.zeros((640, 480, 3), dtype=np.uint8))"
call :run_with_reporting python -X utf8 -c "from ultralytics import YOLO; import numpy as np; YOLO('yolo26n.yaml').to('cpu').predict(np.zeros((640, 480, 3), dtype=np.uint8))"
if errorlevel 1 (
    call :record_failure "[post-link] YOLO smoke test failed (exit !LAST_COMMAND_STATUS!)."
)

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

:check_nvidia_support
call :log "[post-link] Checking NVIDIA GPU support after install..."

set "CUDA_RESULT="
for /f "usebackq delims=" %%i in (`python -c "try: import torch; print(torch.cuda.is_available())\nexcept: print('None')"` ) do (
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
