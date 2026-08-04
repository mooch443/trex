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

if exist "%PREFIX%\conda-meta\py-opencv-*.json" (
    call :record_failure "[post-link] Conda py-opencv is still present; refusing to install a second cv2 provider."
    goto post_link_finish
)

set "PIP_ARGS="
call :add_package "torchmetrics"
call :add_package "tqdm"
call :add_package "opencv-python>=4.6,<5"
call :add_package "ultralytics>=8.3.0,<9"
call :add_package "rfdetr==1.8.3"
call :add_package "dill"
call :add_package "scikit-learn"
call :add_package "timm"
call :add_package "git+https://github.com/ultralytics/CLIP.git"
if "!CONDA_NUMPY_OWNED!"=="0" call :add_package "numpy>=1.26,<3"
set "PIP_ARGS_SIMPLE=!PIP_ARGS!"

rem Build the ordered list now; each actual pip command receives exactly one
rem Torch index from this list.
call :build_torch_target_candidates

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

rem Try every target in policy order. Missing or deprecated indexes, dependency
rem conflicts, and download failures all advance to the next target.
set "TORCH_INSTALLED=0"
for %%c in (!TORCH_TARGET_CODES!) do (
    if "!TORCH_INSTALLED!"=="0" (
        call :select_torch_candidate "%%c"
        call :log "[post-link] Trying !TORCH_TARGET! from !TORCH_INDEX_URL!."
        call :resolve_torch_target
        if errorlevel 1 (
            call :log "[post-link] !TORCH_TARGET! has no compatible torch/torchvision pair for this Python; trying the next target."
        ) else (
            call :install_resolved_torch_target
            if errorlevel 1 (
                call :log "[post-link] !TORCH_TARGET! could not be installed with a consistent dependency set; trying the next target."
            ) else (
                set "TORCH_INSTALLED=1"
            )
        )
    )
)
if "!TORCH_INSTALLED!"=="0" (
    call :record_failure "[post-link] No CUDA, CPU-only, or default PyPI torch/torchvision pair could be installed with a consistent dependency set."
    goto pip_install_after
)

set "INSTALLED_TORCH_VERSION="
set "INSTALLED_TORCHVISION_VERSION="
for /f "usebackq delims=" %%i in (`python -X utf8 -c "from importlib.metadata import version; print(version('torch'))"`) do set "INSTALLED_TORCH_VERSION=%%i"
for /f "usebackq delims=" %%i in (`python -X utf8 -c "from importlib.metadata import version; print(version('torchvision'))"`) do set "INSTALLED_TORCHVISION_VERSION=%%i"
call :log "[post-link] Installed and verified PyTorch !INSTALLED_TORCH_VERSION! + torchvision !INSTALLED_TORCHVISION_VERSION! from the single !TORCH_TARGET! target."
call :check_nvidia_support

:pip_install_after

rem Signal the progress indicator to stop. The sentinel is left in %TEMP% (harmless random-named
rem file) so the Python polling loop cannot miss it by racing against a delete.
copy nul "%PROGRESS_STOP%" >nul 2>&1
timeout /t 1 /nobreak >nul 2>&1
del "%PROGRESS_PY%" "%PROGRESS_LOG%" 2>nul

if not "!TORCH_INSTALLED!"=="1" goto post_link_finish

call :log "Testing installation..."
set "NUMPY_OWNERSHIP_ASSERT="
if "!CONDA_NUMPY_OWNED!"=="1" set "NUMPY_OWNERSHIP_ASSERT=assert version('numpy') == '!NUMPY_VERSION!'; "
call :log_command python -X utf8 -c "from ultralytics import YOLO; from rfdetr import RFDETR; from torchvision.ops import nms; from importlib.metadata import version; import cv2, numpy as np, torch; !NUMPY_OWNERSHIP_ASSERT!assert cv2.__version__.split('.')[0] == '4'; assert nms(torch.tensor([[0.,0.,1.,1.]]), torch.tensor([1.]), 0.5).tolist() == [0]; YOLO('yolo26n.yaml').to('cpu').predict(np.zeros((640, 480, 3), dtype=np.uint8))"
call :run_with_reporting python -X utf8 -c "from ultralytics import YOLO; from rfdetr import RFDETR; from torchvision.ops import nms; from importlib.metadata import version; import cv2, numpy as np, torch; !NUMPY_OWNERSHIP_ASSERT!assert cv2.__version__.split('.')[0] == '4'; assert nms(torch.tensor([[0.,0.,1.,1.]]), torch.tensor([1.]), 0.5).tolist() == [0]; YOLO('yolo26n.yaml').to('cpu').predict(np.zeros((640, 480, 3), dtype=np.uint8))"
if errorlevel 1 (
    call :record_failure "[post-link] YOLO smoke test failed (exit !LAST_COMMAND_STATUS!)."
)

:post_link_finish
if not "!POST_LINK_FAILED!"=="0" (
    call :log "============================================================"
    call :log "WARNING: TRex PYTHON ML SETUP IS INCOMPLETE"
    call :log "The Conda package installation will continue successfully."
    call :log "TRex itself is installed, but Python ML features may be unavailable."
    call :log "After installation, repair the environment and run: python -m pip check"
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
if defined TORCH_CANDIDATES_FILE del /q "!TORCH_CANDIDATES_FILE!" >nul 2>&1

exit /b 0

:configure_numpy_policy
if not exist "%PREFIX%\conda-meta\numpy-*.json" (
    set "CONDA_NUMPY_OWNED=0"
    call :log "[post-link] Conda does not own NumPy; pip will solve numpy^>=1.26,^<3 with the complete ML dependency set."
    exit /b 0
)
set "CONDA_NUMPY_OWNED=1"
set "NUMPY_VERSION="
for /f "usebackq delims=" %%i in (`python -X utf8 -c "import numpy,sys; sys.stdout.write(numpy.__version__)" 2^>NUL`) do set "NUMPY_VERSION=%%i"
if not defined NUMPY_VERSION exit /b 1
set "NUMPY_METADATA_VERSION="
for /f "usebackq delims=" %%i in (`python -X utf8 -c "from importlib.metadata import version; print(version('numpy'),end='')" 2^>NUL`) do set "NUMPY_METADATA_VERSION=%%i"
if not "!NUMPY_METADATA_VERSION!"=="!NUMPY_VERSION!" exit /b 1
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

:build_torch_target_candidates
set "TORCH_TARGET_CODES="
call :detect_driver_cuda_version
if not defined CUDA_MAX_VERSION (
    call :log "[post-link] No usable NVIDIA driver detected; skipping CUDA distributions."
) else (
    call :log "[post-link] NVIDIA driver accepts CUDA !CUDA_MAX_VERSION!; trying every compatible PyTorch CUDA channel newest-first."
    if !CUDA_MAX_CODE! GEQ 1302 set "TORCH_TARGET_CODES=!TORCH_TARGET_CODES! cu132"
    if !CUDA_MAX_CODE! GEQ 1300 set "TORCH_TARGET_CODES=!TORCH_TARGET_CODES! cu130"
    if !CUDA_MAX_CODE! GEQ 1209 set "TORCH_TARGET_CODES=!TORCH_TARGET_CODES! cu129"
    if !CUDA_MAX_CODE! GEQ 1208 set "TORCH_TARGET_CODES=!TORCH_TARGET_CODES! cu128"
    if !CUDA_MAX_CODE! GEQ 1206 set "TORCH_TARGET_CODES=!TORCH_TARGET_CODES! cu126"
    if !CUDA_MAX_CODE! GEQ 1204 set "TORCH_TARGET_CODES=!TORCH_TARGET_CODES! cu124"
    if !CUDA_MAX_CODE! GEQ 1201 set "TORCH_TARGET_CODES=!TORCH_TARGET_CODES! cu121"
    if !CUDA_MAX_CODE! GEQ 1108 set "TORCH_TARGET_CODES=!TORCH_TARGET_CODES! cu118"
    if !CUDA_MAX_CODE! LSS 1108 call :log "[post-link] Driver compatibility is below CUDA 11.8; skipping CUDA distributions."
)
set "TORCH_TARGET_CODES=!TORCH_TARGET_CODES! cpu pypi"
exit /b 0

:select_torch_candidate
set "TORCH_CODE=%~1"
set "TORCH_DEPENDENCY_INDEX_ARG=--extra-index-url https://pypi.org/simple"
if /I "!TORCH_CODE!"=="cu132" (
    set "TORCH_TARGET=CUDA 13.2"
    set "TORCH_INDEX_URL=https://download.pytorch.org/whl/cu132"
) else if /I "!TORCH_CODE!"=="cu130" (
    set "TORCH_TARGET=CUDA 13.0"
    set "TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130"
) else if /I "!TORCH_CODE!"=="cu129" (
    set "TORCH_TARGET=CUDA 12.9"
    set "TORCH_INDEX_URL=https://download.pytorch.org/whl/cu129"
) else if /I "!TORCH_CODE!"=="cu128" (
    set "TORCH_TARGET=CUDA 12.8"
    set "TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128"
) else if /I "!TORCH_CODE!"=="cu126" (
    set "TORCH_TARGET=CUDA 12.6"
    set "TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126"
) else if /I "!TORCH_CODE!"=="cu124" (
    set "TORCH_TARGET=CUDA 12.4"
    set "TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124"
) else if /I "!TORCH_CODE!"=="cu121" (
    set "TORCH_TARGET=CUDA 12.1"
    set "TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121"
) else if /I "!TORCH_CODE!"=="cu118" (
    set "TORCH_TARGET=CUDA 11.8"
    set "TORCH_INDEX_URL=https://download.pytorch.org/whl/cu118"
) else if /I "!TORCH_CODE!"=="cpu" (
    set "TORCH_TARGET=CPU-only"
    set "TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu"
) else if /I "!TORCH_CODE!"=="pypi" (
    set "TORCH_TARGET=PyPI fallback"
    set "TORCH_INDEX_URL=https://pypi.org/simple"
    set "TORCH_DEPENDENCY_INDEX_ARG="
) else (
    exit /b 1
)
exit /b 0

:resolve_torch_target
if not defined TORCH_CANDIDATES_FILE set "TORCH_CANDIDATES_FILE=%TEMP%\trex_torch_candidates_%RANDOM%.txt"
if exist "!TORCH_CANDIDATES_FILE!" del /q "!TORCH_CANDIDATES_FILE!" >nul 2>&1
python -X utf8 -c "import re,subprocess,sys; from pip._vendor.packaging.version import Version; flavor=sys.argv[1].rstrip('/').rsplit('/',1)[-1]; required_flavor=flavor if flavor == 'cpu' or flavor.startswith('cu') else None; results=[subprocess.run([sys.executable,'-m','pip','index','versions',package,'--index-url',sys.argv[1]],capture_output=True,text=True,errors='replace') for package in ('torch','torchvision')]; matches=[re.search(r'^Available versions:\s*(.+)$',result.stdout,re.M) for result in results]; assert all(result.returncode == 0 for result in results) and all(matches); lists=[sorted({item.strip() for item in match.group(1).split(',') if item.strip()},key=Version,reverse=True) for match in matches]; lists=[[item for item in items if Version(item) >= minimum and (required_flavor is None or Version(item).local == required_flavor)] for items,minimum in zip(lists,(Version('2.2'),Version('0.17')))]; print('\n'.join(torch_version+'|'+vision_version for torch_version,vision_version in zip(*lists)))" "!TORCH_INDEX_URL!" >"!TORCH_CANDIDATES_FILE!" 2>nul
if errorlevel 1 exit /b 1
for %%i in ("!TORCH_CANDIDATES_FILE!") do if %%~zi EQU 0 exit /b 1
if "!CONDA_NUMPY_OWNED!"=="1" (
    call :log "[post-link] Found candidates on !TORCH_TARGET!; pip will validate them newest-first with Conda NumPy !NUMPY_VERSION! fixed."
) else (
    call :log "[post-link] Found candidates on !TORCH_TARGET!; pip will solve NumPy and the complete dependency set newest-first."
)
exit /b 0

:install_resolved_torch_target
set "TORCH_INSTALL_SUCCEEDED=0"
for /f "usebackq tokens=1,2 delims=|" %%a in ("!TORCH_CANDIDATES_FILE!") do (
    if "!TORCH_INSTALL_SUCCEEDED!"=="0" (
        set "PIP_ARGS="
        call :add_package "torch===%%a"
        call :add_package "torchvision===%%b"
        set "PIP_ARGS_TORCH=!PIP_ARGS!"
        call :log "[post-link] Resolving !TORCH_TARGET! pair: torch %%a, torchvision %%b."
        call :log_command python -X utf8 -m pip install !PIP_FLAGS_LOG! !NUMPY_CONSTRAINT_ARG! --index-url !TORCH_INDEX_URL! !TORCH_DEPENDENCY_INDEX_ARG! !PIP_ARGS_TORCH! !PIP_ARGS_SIMPLE!
        python -X utf8 -m pip install !PIP_FLAGS_LOG! !NUMPY_CONSTRAINT_ARG! --index-url !TORCH_INDEX_URL! !TORCH_DEPENDENCY_INDEX_ARG! !PIP_ARGS_TORCH! !PIP_ARGS_SIMPLE! > "%PROGRESS_LOG%" 2>&1
        set "LAST_COMMAND_STATUS=!ERRORLEVEL!"
        if defined OUT_STREAM type "%PROGRESS_LOG%" >> "%OUT_STREAM%" 2>nul
        if "!LAST_COMMAND_STATUS!"=="0" (
            call :verify_torch_environment
            if not errorlevel 1 (
                set "TORCH_INSTALL_SUCCEEDED=1"
            ) else (
                set "LAST_COMMAND_STATUS=1"
                call :log "[post-link] Installed pair failed import or dependency verification; trying the next !TORCH_TARGET! release pair."
            )
        ) else (
            findstr /c:"ResolutionImpossible" "%PROGRESS_LOG%" >nul 2>&1
            if errorlevel 1 (
                set "TORCH_INSTALL_SUCCEEDED=-1"
            ) else (
                call :log "[post-link] Pair rejected with the pinned dependency set; trying the next !TORCH_TARGET! release pair."
            )
        )
    )
)
if "!TORCH_INSTALL_SUCCEEDED!"=="1" exit /b 0
exit /b 1

:verify_torch_environment
python -X utf8 -c "import torch,torchvision" >nul 2>&1
if errorlevel 1 exit /b 1
if "!CONDA_NUMPY_OWNED!"=="1" (
    python -X utf8 -c "from importlib.metadata import version; assert version('numpy') == '!NUMPY_VERSION!'" >nul 2>&1
    if errorlevel 1 exit /b 1
)
if defined OUT_STREAM (
    python -X utf8 -m pip check >> "%OUT_STREAM%" 2>&1
) else (
    python -X utf8 -m pip check
)
set "VERIFY_STATUS=!ERRORLEVEL!"
exit /b !VERIFY_STATUS!

:detect_driver_cuda_version
set "CUDA_MAX_VERSION="
set "CUDA_MAX_CODE="
set "CUDA_SMI_TMP=%TEMP%\trex_nvidia_smi_%RANDOM%.txt"

where /q nvidia-smi
if errorlevel 1 exit /b 1

call nvidia-smi >"!CUDA_SMI_TMP!" 2>&1
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
call nvidia-smi --query-gpu=name --format=csv,noheader >"!GPU_TMP!" 2>&1
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
