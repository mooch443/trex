if [ "$(uname)" == "Linux" ]; then
	conda build -c defaults -c conda-forge .
else
	conda build .
fi
