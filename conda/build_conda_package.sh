if [ "$(uname)" == "Linux" ]; then
	conda build . -c main -c conda-forge
else
	conda build .
fi
