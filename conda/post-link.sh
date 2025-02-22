#!/bin/bash
set +e  # Disable immediate exit on error so that failures don't abort the script

# Check for pip for ARM, Linux, or macOS
if [ "$(uname -p)" == "arm" ] || [ "${OSTYPE}" == "linux-gnu" ] || [ "$(uname)" == "Linux" ] || [ "$(uname)" == "Darwin" ]; then
    if ! command -v pip &> /dev/null; 
	then
        echo "pip could not be found, installing..."
        conda install pip -y
    fi
fi

# Install pip packages
echo "Installing pip packages..." >> $PREFIX/.messages.txt
if [ "$(uname -p)" == "arm" ]; then
    echo "ARM architecture detected, installing packages..." >> $PREFIX/.messages.txt
    { python -m pip install 'torch>=2.0.0,<2.5.0' 'torchvision<0.20.0' torchmetrics 'opencv-python>=4,<5' 'ultralytics>=8.3.0,<9' numpy==1.26.4 dill 2>&1; } >> $PREFIX/.messages.txt;
elif [ "$(uname)" == "Darwin" ]; then
    echo "macOS detected, installing packages..." >> $PREFIX/.messages.txt
    { python -m pip install 'torch>=2.0.0,<2.5.0' 'torchvision<0.20.0' torchmetrics 'opencv-python>=4,<5' 'ultralytics>=8.3.0,<9' numpy==1.26.4 dill 2>&1; } >> $PREFIX/.messages.txt
else
    echo "Linux architecture detected, installing packages..." >> $PREFIX/.messages.txt
    { python -m pip install torchmetrics 'torch>=2.0.0,<2.5.0' 'torchvision<0.20.0' 'opencv-python>=4,<5' 'ultralytics>=8.3.0,<9' numpy==1.26.4 "dill" --index-url https://download.pytorch.org/whl/cu118 --extra-index-url https://pypi.org/simple 2>&1; } >> $PREFIX/.messages.txt
fi

echo "Testing installation..." >> $PREFIX/.messages.txt
{ python -c "from ultralytics import YOLO; import numpy as np; YOLO('yolo11n.pt').to('cpu').predict(np.zeros((640, 480, 3), dtype=np.uint8))" 2>&1; } >> $PREFIX/.messages.txt

# Ensure the script exits with a success code regardless of previous failures
exit 0