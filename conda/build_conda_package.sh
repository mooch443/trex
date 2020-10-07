conda build purge

if [ "$(uname)" == "Linux" ]; then
	conda build . -c anaconda -c tom.schoonjans
else
	conda build . -c anaconda
fi
