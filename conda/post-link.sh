if [ "$(uname -p)" == "arm" ]; then
	echo "" >> $PREFIX/.messages.txt;
	echo "============ TRex ===========" >> $PREFIX/.messages.txt;
	echo "Please consider installing the macos-native tensorflow package from https://developer.apple.com/metal/tensorflow-plugin" >> $PREFIX/.messages.txt;
	echo "Quick-start (maybe not up-to-date, see https://trex.run/docs/install.html#apple-silicone-macos-arm64):" >> $PREFIX/.messages.txt;
	echo "    conda activate $(basename ${PREFIX}) && conda install -c apple -y 'tensorflow-deps=2.9' && python -m pip install 'tensorflow-macos>=2,<3' tensorflow-metal"  >> $PREFIX/.messages.txt;
	echo "============ /TRex ==========" >> $PREFIX/.messages.txt;
fi
