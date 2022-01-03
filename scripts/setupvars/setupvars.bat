@echo off

:: Copyright (C) 2018-2021 Intel Corporation
:: SPDX-License-Identifier: Apache-2.0

set SCRIPT_NAME=%~nx0

set "INTEL_OPENVINO_DIR=%~dp0"

set "python_version="

:: command line arguments parsing
:input_arguments_loop
if not "%1"=="" (
    if "%1"=="-pyver" (
        set "python_version=%2"
        shift
    )
    shift
    goto :input_arguments_loop
)

:: OpenCV
if exist "%INTEL_OPENVINO_DIR%\opencv\setupvars.bat" (
call "%INTEL_OPENVINO_DIR%\opencv\setupvars.bat"
goto :opencv_done
)

if exist "%INTEL_OPENVINO_DIR%\extras\opencv\setupvars.bat" (
call "%INTEL_OPENVINO_DIR%\extras\opencv\setupvars.bat"
goto :opencv_done
)
:opencv_done

:: OpenVINO runtime
set "InferenceEngine_DIR=%INTEL_OPENVINO_DIR%\runtime\cmake"
set "ngraph_DIR=%INTEL_OPENVINO_DIR%\runtime\cmake"
set "OpenVINO_DIR=%INTEL_OPENVINO_DIR%\runtime\cmake"
set "HDDL_INSTALL_DIR=%INTEL_OPENVINO_DIR%\runtime\3rdparty\hddl"
set "OPENVINO_LIB_PATHS=%INTEL_OPENVINO_DIR%\runtime\bin\intel64\Release;%INTEL_OPENVINO_DIR%\runtime\bin\intel64\Debug;%HDDL_INSTALL_DIR%\bin;%OPENVINO_LIB_PATHS%"

:: TBB
if exist %INTEL_OPENVINO_DIR%\runtime\3rdparty\tbb (
set "OPENVINO_LIB_PATHS=%INTEL_OPENVINO_DIR%\runtime\3rdparty\tbb\bin;%OPENVINO_LIB_PATHS%"
set "TBB_DIR=%INTEL_OPENVINO_DIR%\runtime\3rdparty\tbb\cmake"
)

:: Compile tool
if exist %INTEL_OPENVINO_DIR%\tools\compile_tool (
set "PATH=%INTEL_OPENVINO_DIR%\tools\compile_tool;%PATH%"
)

:: Add libs dirs to the PATH
set "PATH=%OPENVINO_LIB_PATHS%;%PATH%"

:: Check if Python is installed
set PYTHON_VERSION_MAJOR=3
set MIN_REQUIRED_PYTHON_VERSION_MINOR=6
set MAX_SUPPORTED_PYTHON_VERSION_MINOR=9

python --version 2>NUL
if errorlevel 1 (
   echo Error^: Python is not installed. Please install one of Python %PYTHON_VERSION_MAJOR%.%MIN_REQUIRED_PYTHON_VERSION_MINOR% - %PYTHON_VERSION_MAJOR%.%MAX_SUPPORTED_PYTHON_VERSION_MINOR% ^(64-bit^) from https://www.python.org/downloads/
   exit /B 1
)

:: Check Python version if user did not pass -pyver
if "%python_version%" == "" (
    for /F "tokens=* USEBACKQ" %%F IN (`python -c "import sys; print(str(sys.version_info[0])+'.'+str(sys.version_info[1]))" 2^>^&1`) DO (
       set python_version=%%F
    )
)

for /F "tokens=1,2 delims=. " %%a in ("%python_version%") do (
   set pyversion_major=%%a
   set pyversion_minor=%%b
)

if "%pyversion_major%" equ "%PYTHON_VERSION_MAJOR%" (
   if "%pyversion_minor%" geq "%MIN_REQUIRED_PYTHON_VERSION_MINOR%" (
      if "%pyversion_minor%" leq "%MAX_SUPPORTED_PYTHON_VERSION_MINOR%" (
         set "check_pyversion=true"
      )
   )   
)

if not "%check_pyversion%"=="true" (
   echo Unsupported Python version. Please install one of Python %PYTHON_VERSION_MAJOR%.%MIN_REQUIRED_PYTHON_VERSION_MINOR% - %PYTHON_VERSION_MAJOR%.%MAX_SUPPORTED_PYTHON_VERSION_MINOR% ^(64-bit^) from https://www.python.org/downloads/
   exit /B 1
)

:: Check Python bitness
python -c "import sys; print(64 if sys.maxsize > 2**32 else 32)" 2 > NUL
if errorlevel 1 (
   echo Error^: Error during installed Python bitness detection
   exit /B 1
)

for /F "tokens=* USEBACKQ" %%F IN (`python -c "import sys; print(64 if sys.maxsize > 2**32 else 32)" 2^>^&1`) DO (
   set bitness=%%F
)

if not "%bitness%"=="64" (
   echo Unsupported Python bitness. Please install one of Python %PYTHON_VERSION_MAJOR%.%MIN_REQUIRED_PYTHON_VERSION_MINOR% - %PYTHON_VERSION_MAJOR%.%MAX_SUPPORTED_PYTHON_VERSION_MINOR%^(64-bit^) from https://www.python.org/downloads/
   exit /B 1
)

set PYTHONPATH=%INTEL_OPENVINO_DIR%\python\python%pyversion_major%.%pyversion_minor%;%INTEL_OPENVINO_DIR%\python\python3;%PYTHONPATH%

echo [setupvars.bat] OpenVINO environment initialized

exit /B 0

:GetFullPath
SET %2=%~f1

GOTO :EOF
