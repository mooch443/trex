#!/bin/bash

# if arm or linux, or macos, check for pip
if [ "$(uname -p)" == "arm" ] || [ "${OSTYPE}" == "linux-gnu" ] || [ "$(uname)" == "Darwin" ]; then
	# Ensure pip is installed
	if ! command -v pip &> /dev/null
	then
		echo "pip could not be found, installing..."
		conda install pip -y
	fi
fi

if [ "$(uname -p)" == "arm" ]; then
	# Install pip package
	python -m pip install 'tensorflow-macos==2.12' 'tensorflow-metal==0.8.0' opencv-python ultralytics 'numpy>=1.23,<1.24'  >> $PREFIX/.messages.txt;

	echo "" >> $PREFIX/.messages.txt;

elif [ "${OSTYPE}" == "linux-gnu" ]; then
	python -m pip install opencv-python ultralytics 'numpy>=1.23,<1.24'  >> $PREFIX/.messages.txt;

	echo "" >> $PREFIX/.messages.txt;
else
	python -m pip install opencv-python ultralytics 'numpy>=1.23,<1.24'  >> $PREFIX/.messages.txt;

	echo "" >> $PREFIX/.messages.txt;
fi
