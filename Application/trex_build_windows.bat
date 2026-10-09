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

if not defined TREX_CONFIGURE set "TREX_CONFIGURE=buildall"
if /I not "%TREX_CONFIGURE%"=="buildall" if /I not "%TREX_CONFIGURE%"=="minimal" (
    echo Invalid TREX_CONFIGURE='%TREX_CONFIGURE%'; expected buildall or minimal.
    exit /b 2
)
echo TREX_CONFIGURE=%TREX_CONFIGURE%

cmake .. -DTREX_ENABLE_CPP20=ON -DWITH_GITSHA1=ON -DPYTHON_INCLUDE_DIR:FILEPATH=%pythoninclude% -DPYTHON_LIBRARY:FILEPATH=%findlib% -DPYTHON_EXECUTABLE:FILEPATH=%var% -DWITH_PYLON=OFF -DTREX_CONFIGURE=%TREX_CONFIGURE% -DCMAKE_SKIP_RPATH=ON -DCMAKE_INSTALL_RPATH_USE_LINK_PATH:BOOL=TRUE -DTREX_WITH_TESTS:BOOL=OFF -DCOMMONS_BUILD_GLFW=ON
echo -G "Visual Studio 16"

cmake --build . --config Release

endlocal
