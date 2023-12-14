@echo off

set env_name=trex
set channels=pytorch nvidia defaults
echo "macos: set channels=conda-forge"
set packages=cmake python=3.10 numpy=1.23.5 "pytorch::pytorch-mutex=*=cuda" "pytorch::pytorch=2.*=py3.*_cuda11.7_cudnn*" "pytorch::torchvision=*=py3*" scikit-learn seaborn pip requests ffmpeg=4 git nasm
echo "macos: set packages=cmake python=3.11 numpy=1.26.2 pytorch torchvision scikit-learn seaborn pip requests ffmpeg=6 git nasm"

echo Creating Conda environment %env_name%...

setlocal enabledelayedexpansion

REM Set CONDA_EXE to your Conda executable if not already set
if "%CONDA_EXE%"=="" set "CONDA_EXE=C:\path\to\conda.exe"

set "prepended_channels="
for %%i in (%channels%) do (
    set "prepended_channels=!prepended_channels! -c %%i"
)

call %CONDA_EXE% create -n %env_name% --override-channels!prepended_channels! %packages%

if not %errorlevel% equ 0 goto error

echo Conda environment %env_name% created successfully.

echo Activating Conda environment %CONDA_EXE% %env_name%...
call conda.bat activate %env_name%

if not %errorlevel% equ 0 goto error

echo Conda environment %env_name% activated successfully.

echo Installing additional Python packages...
python -m pip install opencv-python ultralytics "numpy==1.23.5" "tensorflow-gpu>=2,<2.12"
echo "macos: python -m pip install opencv-python ultralytics tensorflow==2.14 tensorflow-estimator==2.14 numpy==1.26.2"

if not %errorlevel% equ 0 goto error

echo Additional Python packages installed successfully.
goto end

:error
echo Error occurred during script execution.
exit /b 1

:end
endlocal
