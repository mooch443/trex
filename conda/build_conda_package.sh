conda build purge

if [ "$(uname)" == "Linux" ]; then
	conda build . -c conda-forge
else
	conda build . -c anaconda
fi
