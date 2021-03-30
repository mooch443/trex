if [ "$(uname)" == "Linux" ]; then
	conda build .
else
	conda build .
fi
