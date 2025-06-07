@setlocal enableextensions enabledelayedexpansion
@echo off

for /f %%w in ('python -c "from shutil import which; print(which(\"python\"))"') do set var=%%w
echo var is %var%

for /f %%w in ('python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())"') do set pythoninclude=%%w
echo pythoninclude is %pythoninclude%

for /f %%w in ('python ../find_library.py') do set findlib=%%w
echo findlib is %findlib%

echo Generator %CMAKE_GENERATOR%
echo Python %PYTHON%

git submodule update --recursive --init


cmake .. -DTREX_ENABLE_CPP20=ON -DWITH_GITSHA1=ON -DPYTHON_INCLUDE_DIR:FILEPATH=%pythoninclude% -DPYTHON_LIBRARY:FILEPATH=%findlib% -DPYTHON_EXECUTABLE:FILEPATH=%var% -DWITH_PYLON=OFF -DCOMMONS_BUILD_OPENCV=ON -DCMAKE_SKIP_RPATH=ON -DCOMMONS_BUILD_PNG=ON -DCOMMONS_BUILD_ZIP=ON -DCMAKE_INSTALL_RPATH_USE_LINK_PATH:BOOL=TRUE -DTREX_WITH_TESTS:BOOL=OFF -DCOMMONS_BUILD_GLFW=ON -DCOMMONS_BUILD_ZLIB=ON
echo -G "Visual Studio 16"

cmake --build . --target Z_LIB --config Release
cmake --build . --target libzip --config Release
cmake --build . --target libpng_custom --config Release
cmake --build . --target CustomOpenCV --config Release
cmake ..
cmake --build . --config Release

endlocal