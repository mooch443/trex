if [ "$(uname -p)" == "arm" ]; then
	echo "" >> $PREFIX/.messages.txt;
	echo "============ TRex ===========" >> $PREFIX/.messages.txt;
	echo "Please consider installing the macos-native tensorflow package from https://developer.apple.com/metal/tensorflow-plugin" >> $PREFIX/.messages.txt;
	echo "Quick-start (maybe not up-to-date, see https://trex.run/docs/install.html#apple-silicone-macos-arm64):" >> $PREFIX/.messages.txt;
	echo "    conda activate $(basename ${PREFIX}) && python -m pip install 'tensorflow-macos==2.12' 'tensorflow-metal==0.8.0' opencv-python 'git+https://github.com/facebookresearch/detectron2.git@v0.6'"  >> $PREFIX/.messages.txt;
	echo "============ /TRex ==========" >> $PREFIX/.messages.txt;
elif [ "${OSTYPE}" == "linux-gnu" ]; then
	echo "" >> $PREFIX/.messages.txt;
	echo "============ TRex ===========" >> $PREFIX/.messages.txt;
	echo "    conda activate $(basename ${PREFIX}) && python -m pip install opencv-python pybind11[global] && pip install --global-option=build_ext --global-option=\"-I$(python -c 'import pybind11;print(pybind11.get_include())')\" 'git+https://github.com/facebookresearch/detectron2.git@v0.6'"  >> $PREFIX/.messages.txt;
	echo "============ /TRex ==========" >> $PREFIX/.messages.txt;
else
	echo "" >> $PREFIX/.messages.txt;
	echo "============ TRex ===========" >> $PREFIX/.messages.txt;
	echo "    conda activate $(basename ${PREFIX}) && python -m pip install opencv-python 'git+https://github.com/facebookresearch/detectron2.git@v0.6'"  >> $PREFIX/.messages.txt;
	echo "============ /TRex ==========" >> $PREFIX/.messages.txt;
fi
