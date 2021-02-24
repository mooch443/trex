if [ "$(uname -p)" == "arm" ]; then
	echo "" >> $PREFIX/.messages.txt;
	echo "============ TRex ===========" >> $PREFIX/.messages.txt;
	echo "Please consider installing the macos-native tensorflow package from github.com/apple/tensorflow_macos" >> $PREFIX/.messages.txt;
	echo "See also: https://trex.run/docs/run.html#run-software-directly-using-shortcuts" >> $PREFIX/.messages.txt;
	echo "============ /TRex ==========" >> $PREFIX/.messages.txt;
fi
