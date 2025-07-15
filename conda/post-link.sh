#!/bin/bash
set +e  # Disable immediate exit on error so that failures don't abort the script

echo "PREFIX=${PREFIX}"
OUT_STREAM="${PREFIX}/.messages.txt"
if [ -z "$PREFIX" ]; then
    echo "PREFIX is not set. Using stdout."
    OUT_STREAM="/dev/stdout"
fi

# Check for pip for ARM, Linux, or macOS
if [ "$(uname -p)" == "arm" ] || [ "${OSTYPE}" == "linux-gnu" ] || [ "$(uname)" == "Linux" ] || [ "$(uname)" == "Darwin" ]; then
    if ! command -v pip &> /dev/null; 
	then
        echo "pip could not be found, installing..."
        conda install pip -y
    fi
fi

# Install pip packages
numpy=$(python -c "import numpy; print(numpy.__version__)")
echo "Installing pip packages (numpy=${numpy})..." >> $OUT_STREAM
if [ "$(uname -p)" == "arm" ]; then
    echo "ARM architecture detected, installing packages..." >> $OUT_STREAM
    { python -m pip install 'torch>=2.0.0,<2.7.0' 'torchvision<0.22.0' torchmetrics 'opencv-python>=4,<5' 'ultralytics>=8.3.0,<9' numpy==${numpy} dill 2>&1; } >> $OUT_STREAM;
elif [ "$(uname)" == "Darwin" ]; then
    echo "macOS detected, installing packages..." >> $OUT_STREAM
    { python -m pip install 'torch>=2.0.0,<2.7.0' 'torchvision<0.22.0' torchmetrics 'opencv-python>=4,<5' 'ultralytics>=8.3.0,<9' numpy==${numpy} dill 2>&1; } >> $OUT_STREAM
else
    echo "Linux architecture detected, installing packages..." >> $OUT_STREAM
    { python -m pip install torchmetrics 'torch>=2.0.0,<2.7.0' 'torchvision<0.22.0' 'opencv-python>=4,<5' 'ultralytics>=8.3.0,<9' numpy==${numpy} "dill" --index-url https://download.pytorch.org/whl/cu118 --extra-index-url https://pypi.org/simple 2>&1; } >> $OUT_STREAM
fi

echo "Testing installation..." >> $OUT_STREAM
{ python -c "from ultralytics import YOLO; import numpy as np; YOLO('yolo11n.pt').to('cpu').predict(np.zeros((640, 480, 3), dtype=np.uint8))" 2>&1; } >> $OUT_STREAM

# Ensure the script exits with a success code regardless of previous failures
exit 0