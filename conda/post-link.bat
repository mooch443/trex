@echo off
cls

REM Check if pip is installed
where /q pip
if ERRORLEVEL 1 (
    echo pip could not be found, installing...
    conda install pip -y
)

REM Define messages.txt file path
SET PREFIX=.
SET MESSAGES_FILE=%PREFIX%\.messages.txt

REM Install pip packages and write messages to messages.txt
python -m pip install torchmetrics "torch>=2.0.0,<2.5.0" "torchvision<0.20.0" "opencv-python>=4,<5" "ultralytics>=8,<9" "numpy==1.26.4" "dill" --index-url https://download.pytorch.org/whl/cu118 --extra-index-url https://pypi.org/simple >> %MESSAGES_FILE% 2>&1

REM Initialize ultralytics / YOLO (optional)
python -c "from ultralytics import YOLO; import numpy as np; YOLO('yolo11n.pt').predict(np.zeros((640, 480, 3), dtype=np.uint8))" >> %MESSAGES_FILE% 2>&1

echo. >> %MESSAGES_FILE%

REM Force a zero exit code regardless of any errors encountered above
exit /b 0