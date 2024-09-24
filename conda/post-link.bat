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
python -m pip install torchmetrics torch torchvision "opencv-python>=4,<4.10" "ultralytics>=8,<=8.2.73" "numpy==1.26.4" "dill" --index-url https://download.pytorch.org/whl/cu118 --extra-index-url https://pypi.org/simple >> %MESSAGES_FILE%

echo. >> %MESSAGES_FILE%
REM echo ============ TRex ============ >> %MESSAGES_FILE%
REM echo     conda activate %PREFIX% ^&^& python -m pip install "opencv-python>=4,<5" "ultralytics>=8,<=8.2.38" "numpy==1.26.4" >> %MESSAGES_FILE%
REM echo ============ /TRex ============ >> %MESSAGES_FILE%
