if [ "$(uname)" == "Linux" ]; then
	conda build . --override-channels -c pytorch -c nvidia -c conda-forge
else
	if [ "$(uname)" == "Darwin" ]; then
		if [ "$(uname -m)" == "arm64" ]; then
			conda build . --override-channels -c pytorch -c conda-forge
		else
			conda build . --override-channels -c pytorch -c defaults
		fi
	else
		conda build . --override-channels -c pytorch -c nvidia -c defaults
	fi
fi
