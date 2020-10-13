if [ "$(uname)" == "Linux" ]; then
	conda build . -c conda-forge
else
	conda build .
fi
