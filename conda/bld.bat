cd Application
mkdir build
cd build

@setlocal enableextensions enabledelayedexpansion
@echo off

set MENU_DIR=%PREFIX%\Menu
mkdir %MENU_DIR%

echo copying %RECIPE_DIR%\..\Application\src\tracker\gfx\TRexA.ico to %MENU_DIR%
copy %RECIPE_DIR%\..\Application\src\tracker\gfx\TRexA.ico %MENU_DIR%
if errorlevel 1 exit 1

echo copying %RECIPE_DIR%\menu-windows.json to %MENU_DIR%\trex.json
copy %RECIPE_DIR%\menu-windows.json %MENU_DIR%\trex.json
if errorlevel 1 exit 1

for /f %%w in ('%PREFIX%\python -c "from shutil import which; print(which(\"python\"))"') do set var=%%w
echo var is %var%

for /f %%w in ('%PREFIX%\python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())"') do set pythoninclude=%%w
echo pythoninclude is %pythoninclude%

for /f %%w in ('%PREFIX%\python ../find_library.py') do set findlib=%%w
echo findlib is %findlib%

echo Generator %CMAKE_GENERATOR%
echo Python %PYTHON%
echo GITHUB_WORKFLOW %GITHUB_WORKFLOW%
set GENERATOR=-G "Visual Studio 16 2019"
#set GENERATOR=-G "Visual Studio 17 2022"

if "%CMAKE_GENERATOR%" == "Visual Studio 16 2019 Win64" set CMAKE_GENERATOR=Visual Studio 16 2019
if "%CMAKE_GENERATOR%" == "Visual Studio 17 2022 Win64" set CMAKE_GENERATOR=Visual Studio 17 2022
if "%GITHUB_WORKFLOW%" == "" set GENERATOR=-G "%CMAKE_GENERATOR%"
echo GENERATOR %GENERATOR%

cmake .. %GENERATOR% -DWITH_GITSHA1=ON -DPYTHON_INCLUDE_DIR:FILEPATH=%pythoninclude% -DPYTHON_LIBRARY:FILEPATH=%findlib% -DPYTHON_EXECUTABLE:FILEPATH=%PREFIX%\python -DWITH_PYLON=OFF -DCOMMONS_BUILD_OPENCV=ON -DCMAKE_INSTALL_PREFIX=%PREFIX% -DCMAKE_SKIP_RPATH=ON -DCOMMONS_BUILD_PNG=ON -DCOMMONS_BUILD_ZIP=ON -DTREX_CONDA_PACKAGE_INSTALL=ON -DCMAKE_INSTALL_RPATH_USE_LINK_PATH:BOOL=TRUE -DTREX_WITH_TESTS:BOOL=OFF -DCOMMONS_BUILD_GLFW=ON -DCOMMONS_BUILD_ZLIB=ON -DCMAKE_BUILD_TYPE=Release -DBUILD_LEGACY_TREX:BOOL=OFF -DBUILD_LEGACY_TGRABS:BOOL=OFF -DCOMMONS_BUILD_EXAMPLES:BOOL=OFF

cmake --build . --target Z_LIB --config Release
cmake --build . --target libzip --config Release
cmake --build . --target libpng_custom --config Release
cmake --build . --target CustomOpenCV --config Release
cmake ..
cmake --build . --target INSTALL --config Release

endlocal