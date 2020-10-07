cmake .. -DPython_ADDITIONAL_VERSIONS=3.6 -DPYBIND11_PYTHON_VERSION=3.6

# recommend doing this as well in your ~/.virtualenvs/postactivate script
# export PYTHONPATH="$(python -m site --user-site)"

cmake .. -DCMAKE_BUILD_TYPE=Release -DWITH_HTTPD=ON -DWITH_PYLON=ON -DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") -DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; import os; print(os.path.join(sysconfig.get_config_var('LIBDIR'), sysconfig.get_config_var('LDLIBRARY')))") -DPYTHON_EXECUTABLE:FILEPATH=`which python`
