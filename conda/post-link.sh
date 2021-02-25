if [ "$(uname -p)" == "arm" ]; then
	echo "" >> $PREFIX/.messages.txt;
	echo "============ TRex ===========" >> $PREFIX/.messages.txt;
	echo "Please consider installing the macos-native tensorflow package from github.com/apple/tensorflow_macos" >> $PREFIX/.messages.txt;
	echo "Quick-start (maybe not up-to-date, see https://trex.run/docs/install.html#apple-silicone-macos-arm64):" >> $PREFIX/.messages.txt;
	echo "    pip install --upgrade --force --no-dependencies https://github.com/apple/tensorflow_macos/releases/download/v0.1alpha2/tensorflow_macos-0.1a2-cp38-cp38-macosx_11_0_arm64.whl https://github.com/apple/tensorflow_macos/releases/download/v0.1alpha2/tensorflow_addons_macos-0.1a2-cp38-cp38-macosx_11_0_arm64.whl"  >> $PREFIX/.messages.txt;
	echo "============ /TRex ==========" >> $PREFIX/.messages.txt;
fi
