@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

set BUILD_TYPE=%1
set SPHINX_PATH=%2
set DOCS_PATH=%3
set TREX_PATH=%4
set TGRABS_PATH=%5

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=%SPHINX_PATH%/sphinx-build
)
set SOURCEDIR=.
set BUILDDIR=_build

echo "exec %TREX_PATH% -d %DOCS_PATH% -h rst"
echo "exec %TGRABS_PATH% -d %DOCS_PATH% -h rst"

"%TREX_PATH%" -d "%DOCS_PATH%" -h rst
"%TGRABS_PATH%" -d "%DOCS_PATH%" -h rst

if "%1" == "" goto help

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
	echo.installed, then set the SPHINXBUILD environment variable to point
	echo.to the full path of the 'sphinx-build' executable. Alternatively you
	echo.may add the Sphinx directory to PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.http://sphinx-doc.org/
	exit /b 1
)

%SPHINXBUILD% -M %BUILD_TYPE% %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%

:end
popd
