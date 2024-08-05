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
if [ "$(uname -p)" == "arm" ]; then
	{ python -m pip install 'tensorflow-macos' 'tensorflow-metal' 'opencv-python<4.10' 'ultralytics<=8.2.57' tensorflow==2.14 tensorflow-estimator==2.14 numpy==1.26.2 dill 2>&1; }  >> $PREFIX/.messages.txt;
	echo "" >> $PREFIX/.messages.txt;

elif [ "$(uname)" == "Darwin" ]; then
    { python -m pip install 'opencv-python<4.10' 'ultralytics<=8.2.57' tensorflow==2.14 tensorflow-estimator==2.14 numpy==1.26.2 dill 2>&1; } >> $PREFIX/.messages.txt;
    echo "" >> $PREFIX/.messages.txt;

else
	{ python -m pip install 'opencv-python<4.10' 'ultralytics<=8.2.57' tensorflow-gpu==2.10 tensorflow-estimator==2.10 dill 2>&1; }  >> $PREFIX/.messages.txt;
	echo "" >> $PREFIX/.messages.txt;
fi
