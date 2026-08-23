@echo off
setlocal EnableDelayedExpansion

chcp 65001 >nul 2>&1
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "PIP_PROGRESS_BAR=off"
set "PIP_NO_INPUT=1"
set "PIP_FLAGS=--disable-pip-version-check --no-input --progress-bar off --no-color"
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

set "TORCH_INSTALLED=0"
call :install_selected_torch
if not errorlevel 1 set "TORCH_INSTALLED=1"
if "!TORCH_INSTALLED!"=="0" (
    call :record_failure "[post-link] The single !TORCH_TARGET! installation failed; no retry was attempted."
    goto pip_install_after
)

call :log "[post-link] The single !TORCH_TARGET! Python ML installation transaction completed successfully."

:pip_install_after

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

:select_torch_target
set "TORCH_TARGET=PyPI"
set "TORCH_INDEX_URL=%TREX_PYPI_INDEX_URL%"
set "TORCH_DEPENDENCY_INDEX_ARG="
call :detect_driver_cuda_version
if not defined CUDA_MAX_VERSION (
    call :log "[post-link] No usable NVIDIA driver detected; selected unqualified PyTorch from PyPI."
    exit /b 0
)

if !CUDA_MAX_CODE! LSS 1108 (
    call :log "[post-link] NVIDIA driver supports CUDA !CUDA_MAX_VERSION!, below the supported CUDA 11.8 baseline; selected unqualified PyTorch from PyPI."
    exit /b 0
)

call :discover_cuda_channels
if errorlevel 1 (
    call :log "[post-link] WARNING: CUDA channel discovery failed; falling back to unqualified PyTorch from PyPI."
    exit /b 0
)
set "SELECTED_TORCH_CODE="
for /f "usebackq delims=" %%c in ("!CUDA_CHANNELS_FILE!") do (
    if not defined SELECTED_TORCH_CODE (
        set "TORCH_CODE=%%c"
        set "CUDA_INDEX_URL=%TREX_TORCH_INDEX_ROOT%/!TORCH_CODE!"
        call :log "[post-link] NVIDIA driver accepts CUDA !CUDA_MAX_VERSION!; checking !TORCH_CODE! package metadata."
        call :discover_cuda_pair
        if errorlevel 1 (
            call :log "[post-link] No compatible !TORCH_CODE! pair was found; checking older compatible CUDA channels."
        ) else (
            set "SELECTED_TORCH_CODE=!TORCH_CODE!"
        )
    )
)
del /q "!CUDA_CHANNELS_FILE!" >nul 2>&1
if not defined SELECTED_TORCH_CODE (
    call :log "[post-link] WARNING: No compatible CUDA channel was discoverable; falling back to unqualified PyTorch from PyPI."
    exit /b 0
)

set "TORCH_CODE=!SELECTED_TORCH_CODE!"
set "TORCH_TARGET=CUDA !TORCH_CODE!"
set "TORCH_INDEX_URL=!CUDA_INDEX_URL!"
set "TORCH_DEPENDENCY_INDEX_ARG=--extra-index-url %TREX_PYPI_INDEX_URL%"
call :log "[post-link] NVIDIA driver accepts CUDA !CUDA_MAX_VERSION!; selected !TORCH_VERSION! with !TORCHVISION_VERSION! from the single !TORCH_TARGET! distribution."
exit /b 0

:discover_cuda_channels
set "CUDA_CHANNELS_FILE=%TEMP%\trex_cuda_channels_%RANDOM%.txt"
python -X utf8 -c "TREX_TORCH_CHANNEL_SELECTOR=1; import re,sys; from urllib.request import Request,urlopen; root,driver=sys.argv[1:3]; channels={'cu118','cu121','cu124','cu126','cu128','cu129','cu130','cu132'}; data=''; exec('try:\n data=urlopen(Request(root.rstrip(chr(47))+chr(47),headers={chr(85)+chr(115)+chr(101)+chr(114)+chr(45)+chr(65)+chr(103)+chr(101)+chr(110)+chr(116):chr(84)+chr(82)+chr(101)+chr(120)}),timeout=10).read().decode(chr(117)+chr(116)+chr(102)+chr(45)+chr(56),errors=chr(114)+chr(101)+chr(112)+chr(108)+chr(97)+chr(99)+chr(101))\nexcept Exception:\n pass'); names={value.strip(chr(34)+chr(39)).rstrip(chr(47)).rsplit(chr(47),1)[-1] for value in re.findall(r'href=([^ >]+)',data,re.I)}; channels.update(name for name in names if re.fullmatch(r'cu[0-9]{3,}',name)); code=lambda channel:int(channel[2:4])*100+int(channel[4:]); print(chr(10).join(sorted((channel for channel in channels if 1108<=code(channel)<=int(driver)),key=code,reverse=True)))" "%TREX_TORCH_INDEX_ROOT%" "!CUDA_MAX_CODE!" >"!CUDA_CHANNELS_FILE!" 2>nul
if errorlevel 1 (
    del /q "!CUDA_CHANNELS_FILE!" >nul 2>&1
    exit /b 1
)
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
call :log_command python -X utf8 -m pip install !PIP_FLAGS! !NUMPY_CONSTRAINT_ARG! --index-url !TORCH_INDEX_URL! !TORCH_DEPENDENCY_INDEX_ARG! !PIP_ARGS_TORCH! !PIP_ARGS_SIMPLE!
call :run_with_reporting python -X utf8 -m pip install !PIP_FLAGS! !NUMPY_CONSTRAINT_ARG! --index-url !TORCH_INDEX_URL! !TORCH_DEPENDENCY_INDEX_ARG! !PIP_ARGS_TORCH! !PIP_ARGS_SIMPLE!
set "LAST_COMMAND_STATUS=!ERRORLEVEL!"
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
if defined OUT_STREAM (
    >>"%OUT_STREAM%" 2>&1 cmd /c %*
) else (
    cmd /c %*
)
set "status=%ERRORLEVEL%"
endlocal & set "LAST_COMMAND_STATUS=%status%"
exit /b %status%
