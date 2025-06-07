if [ "$(uname)" == "Linux" ]; then
	conda build . --override-channels -c conda-forge
else
	if [ "$(uname)" == "Darwin" ]; then
		if [ "$(uname -m)" == "arm64" ]; then
			conda build . --override-channels -c conda-forge
		else
			conda build . --override-channels -c conda-forge
		fi
	else
		conda build . --override-channels -c conda-forge
	fi
fi
