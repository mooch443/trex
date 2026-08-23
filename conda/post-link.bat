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
if not defined TREX_PYPI_INDEX_URL set "TREX_PYPI_INDEX_URL=https://pypi.org/simple"
if not defined TREX_TORCH_INDEX_ROOT set "TREX_TORCH_INDEX_ROOT=https://download.pytorch.org/whl"
if not defined TREX_CLIP_REQUIREMENT set "TREX_CLIP_REQUIREMENT=git+https://github.com/ultralytics/CLIP.git"
rem Ignore ambient pip indexes and find-links. Only the explicit TRex source
rem overrides above participate in metadata discovery and installation.
set "PIP_CONFIG_FILE=NUL"
set "PIP_INDEX_URL="
set "PIP_EXTRA_INDEX_URL="
set "PIP_FIND_LINKS="

echo PREFIX=%PREFIX%

rem CI can request stdout so its caller can tee and validate the transaction log.
if /I "%TREX_POST_LINK_OUTPUT%"=="stdout" (
    set "OUT_STREAM="
) else if defined TREX_POST_LINK_OUTPUT (
    set "OUT_STREAM=%TREX_POST_LINK_OUTPUT%"
) else if defined PREFIX (
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

rem Compose pip argument lists. A compatible torch/torchvision pair is selected
rem from exactly one platform-appropriate index while the complete ML dependency
rem set is resolved from PyPI in the same transaction.
rem
rem A Conda-owned NumPy is immutable. Otherwise NumPy joins the complete pip
rem transaction so the resolver can choose one version for the whole ML stack.
set "NUMPY_CONSTRAINT_FILE=%TEMP%\trex_numpy_constraint_%RANDOM%.txt"
set "NUMPY_CONSTRAINT_ARG="
set "CONDA_NUMPY_OWNED=0"
call :configure_numpy_policy
if errorlevel 1 (
    call :record_failure "[post-link] Conda owns NumPy but its exact version could not be read; refusing to let pip modify it."
    goto post_link_finish
)

set "PIP_ARGS="
call :add_package "torchmetrics"
call :add_package "tqdm"
call :add_package "ultralytics>=8.3.0,<9"
call :add_package "rfdetr==1.8.3"
call :add_package "dill"
call :add_package "scikit-learn"
call :add_package "timm"
call :add_package "%TREX_CLIP_REQUIREMENT%"
if not exist "%PREFIX%\conda-meta\py-opencv-*.json" (
    call :add_package "opencv-python>=4.6,<5"
    call :log "[post-link] No Conda py-opencv binding detected; pip will provide cv2 for the non-minimal profile."
) else (
    call :log "[post-link] Conda owns cv2 through py-opencv; pip will not install an OpenCV wheel."
)
if "!CONDA_NUMPY_OWNED!"=="0" call :add_package "numpy>=1.26,<3"
set "PIP_ARGS_SIMPLE=!PIP_ARGS!"

set "PIP_ARGS="
call :add_package "torch>=2.2"
call :add_package "torchvision>=0.17"
set "PIP_ARGS_TORCH=!PIP_ARGS!"
call :select_torch_target
call :log "[post-link] Selected !TORCH_TARGET! from !TORCH_INDEX_URL!."

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

set "TORCH_INSTALLED=0"
call :install_selected_torch
if not errorlevel 1 set "TORCH_INSTALLED=1"
if "!TORCH_INSTALLED!"=="0" (
    call :record_failure "[post-link] The single !TORCH_TARGET! installation failed; no retry was attempted."
    goto pip_install_after
)

call :log "[post-link] The single !TORCH_TARGET! Python ML installation transaction completed successfully."

:pip_install_after

rem Signal the progress indicator to stop. The sentinel is left in %TEMP% (harmless random-named
rem file) so the Python polling loop cannot miss it by racing against a delete.
copy nul "%PROGRESS_STOP%" >nul 2>&1
timeout /t 1 /nobreak >nul 2>&1
del "%PROGRESS_PY%" "%PROGRESS_LOG%" 2>nul

if "!TORCH_INSTALLED!"=="1" (
    call :log_command python -X utf8 -c "import torch; print('[post-link] Installed PyTorch:', torch.__version__); print('[post-link] Compiled CUDA:', torch.version.cuda); print('[post-link] GPU available:', torch.cuda.is_available())"
    call :run_with_reporting python -X utf8 -c "import torch; print('[post-link] Installed PyTorch:', torch.__version__); print('[post-link] Compiled CUDA:', torch.version.cuda); print('[post-link] GPU available:', torch.cuda.is_available())"
    if errorlevel 1 call :log "[post-link] WARNING: Could not inspect the installed PyTorch CUDA status."
    call :log "[post-link] Warming the Ultralytics runtime and model cache."
    call :log_command python -X utf8 -c "from ultralytics import YOLO; from rfdetr import RFDETR; from torchvision.ops import nms; import cv2, numpy as np, torch; assert cv2.__version__.split('.')[0] == '4'; assert nms(torch.tensor([[0.,0.,1.,1.]]), torch.tensor([1.]), 0.5).tolist() == [0]; YOLO('yolo26n.yaml').to('cpu').predict(np.zeros((640, 480, 3), dtype=np.uint8))"
    call :run_with_reporting python -X utf8 -c "from ultralytics import YOLO; from rfdetr import RFDETR; from torchvision.ops import nms; import cv2, numpy as np, torch; assert cv2.__version__.split('.')[0] == '4'; assert nms(torch.tensor([[0.,0.,1.,1.]]), torch.tensor([1.]), 0.5).tolist() == [0]; YOLO('yolo26n.yaml').to('cpu').predict(np.zeros((640, 480, 3), dtype=np.uint8))"
    if errorlevel 1 call :log "[post-link] WARNING: YOLO runtime warm-up failed; installation remains successful."
)

:post_link_finish
if not "!POST_LINK_FAILED!"=="0" (
    call :log "============================================================"
    call :log "WARNING: TRex PYTHON ML SETUP IS INCOMPLETE"
    call :log "The Conda package installation will continue successfully."
    call :log "TRex itself is installed, but Python ML features may be unavailable."
    call :log "After installation, inspect this log. Dependency diagnostic: python -m pip check"
    call :log "============================================================"
    if defined OUT_STREAM (
        >&2 echo WARNING: TRex Python ML setup is incomplete; Conda installation will continue. See "%OUT_STREAM%".
        if exist "%OUT_STREAM%" (
            >&2 echo [post-link] Dumping post-link log due to incomplete Python ML setup:
            type "%OUT_STREAM%" 1>&2
        )
    ) else (
        >&2 echo WARNING: TRex Python ML setup is incomplete; Conda installation will continue. See stdout.
    )
)

if defined NUMPY_CONSTRAINT_FILE del /q "!NUMPY_CONSTRAINT_FILE!" >nul 2>&1
exit /b 0

:configure_numpy_policy
if not exist "%PREFIX%\conda-meta\numpy-*.json" (
    set "CONDA_NUMPY_OWNED=0"
    call :log "[post-link] Conda does not own NumPy; pip will solve numpy^>=1.26,^<3 with the complete ML dependency set."
    exit /b 0
)
set "CONDA_NUMPY_OWNED=1"
set "NUMPY_VERSION="
set "NUMPY_CONDA_RECORD="
for %%f in ("%PREFIX%\conda-meta\numpy-*.json") do if not defined NUMPY_CONDA_RECORD set "NUMPY_CONDA_RECORD=%%~ff"
for /f "usebackq delims=" %%i in (`python -X utf8 -c "import json,sys; print(json.load(open(sys.argv[1], encoding='utf-8'))['version'],end='')" "!NUMPY_CONDA_RECORD!" 2^>NUL`) do set "NUMPY_VERSION=%%i"
if not defined NUMPY_VERSION exit /b 1
>"!NUMPY_CONSTRAINT_FILE!" echo numpy==!NUMPY_VERSION!
set NUMPY_CONSTRAINT_ARG=--constraint "!NUMPY_CONSTRAINT_FILE!"
call :log "[post-link] Conda owns NumPy !NUMPY_VERSION!; every pip solve is constrained to that exact version."
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

:select_torch_target
set "TORCH_TARGET=PyPI"
set "TORCH_INDEX_URL=%TREX_PYPI_INDEX_URL%"
set "TORCH_DEPENDENCY_INDEX_ARG="
call :detect_driver_cuda_version
if not defined CUDA_MAX_VERSION (
    call :log "[post-link] No usable NVIDIA driver detected; selected unqualified PyTorch from PyPI."
    exit /b 0
)

set "TORCH_CODE="
if !CUDA_MAX_CODE! GEQ 1302 (
    set "TORCH_CODE=cu132"
) else if !CUDA_MAX_CODE! GEQ 1300 (
    set "TORCH_CODE=cu130"
) else if !CUDA_MAX_CODE! GEQ 1209 (
    set "TORCH_CODE=cu129"
) else if !CUDA_MAX_CODE! GEQ 1208 (
    set "TORCH_CODE=cu128"
) else if !CUDA_MAX_CODE! GEQ 1206 (
    set "TORCH_CODE=cu126"
) else if !CUDA_MAX_CODE! GEQ 1204 (
    set "TORCH_CODE=cu124"
) else if !CUDA_MAX_CODE! GEQ 1201 (
    set "TORCH_CODE=cu121"
) else if !CUDA_MAX_CODE! GEQ 1108 (
    set "TORCH_CODE=cu118"
)
if not defined TORCH_CODE (
    call :log "[post-link] NVIDIA driver supports CUDA !CUDA_MAX_VERSION!, below the supported CUDA 11.8 baseline; selected unqualified PyTorch from PyPI."
    exit /b 0
)

set "CUDA_INDEX_URL=%TREX_TORCH_INDEX_ROOT%/!TORCH_CODE!"
call :discover_cuda_pair
if errorlevel 1 (
    call :log "[post-link] WARNING: No compatible PyTorch/torchvision !TORCH_CODE! pair was found at !CUDA_INDEX_URL!; falling back to unqualified PyTorch from PyPI."
    exit /b 0
)

set "TORCH_TARGET=CUDA !TORCH_CODE!"
set "TORCH_INDEX_URL=!CUDA_INDEX_URL!"
set "TORCH_DEPENDENCY_INDEX_ARG=--extra-index-url %TREX_PYPI_INDEX_URL%"
call :log "[post-link] NVIDIA driver accepts CUDA !CUDA_MAX_VERSION!; selected !TORCH_VERSION! with !TORCHVISION_VERSION! from the single !TORCH_TARGET! distribution."
exit /b 0

:discover_cuda_pair
set "TORCH_PAIR="
set "TORCH_VERSION="
set "TORCHVISION_VERSION="
set "TORCH_CANDIDATES_FILE=%TEMP%\trex_torch_candidates_%RANDOM%.txt"
set "TORCH_DISCOVERY_ERROR=%TEMP%\trex_torch_discovery_%RANDOM%.txt"
python -X utf8 -c "TREX_TORCH_PAIR_SELECTOR=1; import re,subprocess,sys; from pip._vendor.packaging.version import Version; index,flavor=sys.argv[1:3]; packages=('torch','torchvision'); minimum={'torch':Version('2.2'),'torchvision':Version('0.17')}; outputs=[subprocess.run([sys.executable,'-m','pip','index','versions',package,'--index-url',index,'--disable-pip-version-check','--no-color'],text=True,capture_output=True,check=True).stdout for package in packages]; matches=[re.search(r'^Available versions:\s*(.+)$',output,re.MULTILINE) for output in outputs]; assert all(matches),'malformed index metadata'; parsed=[[item.strip() for item in match.group(1).split(',') if item.strip() and Version(item.strip()).local==flavor and not Version(item.strip()) < minimum[package]] for package,match in zip(packages,matches)]; parsed=[sorted(set(items),key=Version,reverse=True) for items in parsed]; visions={Version(item).release:item for item in parsed[1]}; pairs=[(item,visions.get((0,Version(item).release[1]+15,Version(item).release[2] if len(Version(item).release)>2 else 0))) for item in parsed[0] if len(Version(item).release)>1 and Version(item).release[0]==2]; pairs=[pair for pair in pairs if pair[1]]; assert pairs,'no compatible flavored release pair'; print(pairs[0][0]+'|'+pairs[0][1])" "!CUDA_INDEX_URL!" "!TORCH_CODE!" >"!TORCH_CANDIDATES_FILE!" 2>"!TORCH_DISCOVERY_ERROR!"
set "TORCH_DISCOVERY_STATUS=!ERRORLEVEL!"
if not "!TORCH_DISCOVERY_STATUS!"=="0" (
    if defined OUT_STREAM (
        type "!TORCH_DISCOVERY_ERROR!" >>"!OUT_STREAM!" 2>nul
    ) else (
        type "!TORCH_DISCOVERY_ERROR!" 2>nul
    )
    del /q "!TORCH_CANDIDATES_FILE!" "!TORCH_DISCOVERY_ERROR!" >nul 2>&1
    exit /b 1
)
set /p "TORCH_PAIR=" <"!TORCH_CANDIDATES_FILE!"
del /q "!TORCH_CANDIDATES_FILE!" "!TORCH_DISCOVERY_ERROR!" >nul 2>&1
for /f "tokens=1,2 delims=|" %%a in ("!TORCH_PAIR!") do (
    set "TORCH_VERSION=%%a"
    set "TORCHVISION_VERSION=%%b"
)
if not defined TORCH_VERSION exit /b 1
if not defined TORCHVISION_VERSION exit /b 1
set "PIP_ARGS="
call :add_package "torch===!TORCH_VERSION!"
call :add_package "torchvision===!TORCHVISION_VERSION!"
set "PIP_ARGS_TORCH=!PIP_ARGS!"
exit /b 0

:install_selected_torch
call :log "[post-link] Running one resolver transaction for !TORCH_TARGET!; no version or index retries are permitted."
call :log_command python -X utf8 -m pip install !PIP_FLAGS_LOG! !NUMPY_CONSTRAINT_ARG! --index-url !TORCH_INDEX_URL! !TORCH_DEPENDENCY_INDEX_ARG! !PIP_ARGS_TORCH! !PIP_ARGS_SIMPLE!
python -X utf8 -m pip install !PIP_FLAGS_LOG! !NUMPY_CONSTRAINT_ARG! --index-url !TORCH_INDEX_URL! !TORCH_DEPENDENCY_INDEX_ARG! !PIP_ARGS_TORCH! !PIP_ARGS_SIMPLE! > "%PROGRESS_LOG%" 2>&1
set "LAST_COMMAND_STATUS=!ERRORLEVEL!"
if defined OUT_STREAM (
    type "%PROGRESS_LOG%" >> "%OUT_STREAM%" 2>nul
) else (
    type "%PROGRESS_LOG%" 2>nul
)
if not "!LAST_COMMAND_STATUS!"=="0" exit /b 1
exit /b 0

:detect_driver_cuda_version
set "CUDA_MAX_VERSION="
set "CUDA_MAX_CODE="
set "CUDA_SMI_TMP=%TEMP%\trex_nvidia_smi_%RANDOM%.txt"

where /q nvidia-smi
if errorlevel 1 exit /b 1

call nvidia-smi >"!CUDA_SMI_TMP!" 2>&1
if errorlevel 1 (
    if defined OUT_STREAM (
        type "!CUDA_SMI_TMP!" >>"!OUT_STREAM!" 2>nul
    ) else (
        type "!CUDA_SMI_TMP!" 2>nul
    )
    del /q "!CUDA_SMI_TMP!" >nul 2>&1
    exit /b 1
)

for /f "usebackq delims=" %%i in (`python -X utf8 -c "import re,sys; text=open(sys.argv[1],encoding='utf-8',errors='replace').read(); match=re.search(r'CUDA(?:\s+UMD)?\s+Version:\s*([0-9]+\.[0-9]+)',text); print(match.group(1) if match else '')" "!CUDA_SMI_TMP!"`) do (
    set "CUDA_MAX_VERSION=%%i"
)
if defined OUT_STREAM (
    type "!CUDA_SMI_TMP!" >>"!OUT_STREAM!" 2>nul
) else (
    type "!CUDA_SMI_TMP!" 2>nul
)
del /q "!CUDA_SMI_TMP!" >nul 2>&1
if not defined CUDA_MAX_VERSION (
    call :log "[post-link] nvidia-smi did not report a maximum supported CUDA version."
    exit /b 1
)

for /f "tokens=1,2 delims=." %%a in ("!CUDA_MAX_VERSION!") do (
    set /a CUDA_MAX_CODE=%%a*100+%%b
)
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
