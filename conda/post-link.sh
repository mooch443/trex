#!/bin/bash

# if arm or linux, or macos, check for pip
if [ "$(uname -p)" == "arm" ] || [ "${OSTYPE}" == "linux-gnu" ] || "$(uname)" == "Linux" || [ "$(uname)" == "Darwin" ]; then
	# Ensure pip is installed
	if ! command -v pip &> /dev/null
	then
		echo "pip could not be found, installing..."
		conda install pip -y
	fi
fi

# install pip packages
echo "Installing pip packages..." >> $PREFIX/.messages.txt
if [ "$(uname -p)" == "arm" ]; then
	echo "ARM architecture detected, installing torch, torchvision, torchaudio, opencv-python, ultralytics, numpy, dill..." >> $PREFIX/.messages.txt
	{ python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 2>&1; } >> $PREFIX/.messages.txt;
	{ python -m pip install 'opencv-python<4.10' 'ultralytics<=8.2.73' numpy==1.26.4 dill 2>&1; }  >> $PREFIX/.messages.txt;
	echo "" >> $PREFIX/.messages.txt;

elif [ "$(uname)" == "Darwin" ]; then
	echo "MacOS detected, installing torch, torchvision, torchaudio, opencv-python, ultralytics, numpy, dill..." >> $PREFIX/.messages.txt
    { python -m pip install torch torchvision torchaudio 'opencv-python<4.10' 'ultralytics<=8.2.73' numpy==1.26.4 dill 2>&1; } >> $PREFIX/.messages.txt;
    echo "" >> $PREFIX/.messages.txt;

else
	echo "Linux architecture detected, installing torch, torchvision, torchaudio, opencv-python, ultralytics, numpy, dill..." >> $PREFIX/.messages.txt
	{ python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 2>&1; } >> $PREFIX/.messages.txt;
	{ python -m pip install 'opencv-python<4.10' 'ultralytics<=8.2.73' numpy==1.26.4 dill 2>&1; }  >> $PREFIX/.messages.txt;
	echo "" >> $PREFIX/.messages.txt;
fi
