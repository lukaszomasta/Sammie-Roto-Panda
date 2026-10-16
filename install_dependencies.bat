@echo off
setlocal
pushd %~dp0

rem Python: the embedded Python from the Sammie-Roto release zip if present,
rem otherwise a venv made from an installed Python 3.12 (python.org installer).
set "EMBED=python-3.12.8-embed-amd64\python.exe"
if exist "%EMBED%" (
    set "PY=%EMBED%"
    set "PIPFLAGS=--no-warn-script-location"
    goto :have_python
)
if not exist "venv\Scripts\python.exe" (
    echo Creating Python virtual environment...
    py -3.12 -m venv venv 2>nul || python -m venv venv
    if errorlevel 1 (
        echo ERROR: Python 3.12 not found. Install it from https://www.python.org/ and run this again.
        pause
        exit /b 1
    )
)
set "PY=venv\Scripts\python.exe"
set "PIPFLAGS="

:have_python
echo Using %PY%

echo Select version to install:
echo 1. CUDA 13.0 (For modern NVIDIA GPUs, RTX)
echo 2. CUDA 12.6 (For old NVIDIA GPUs, GTX)
echo 3. Intel Arc (beta, may not work)
echo 4. CPU (Smaller download)
echo 5. Exit

choice /C 12345 /N /M "Enter your choice (1, 2, 3, 4 or 5):"

if errorlevel 5 (
    echo Exiting...
    exit /b 0
) else if errorlevel 4 (
    set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
) else if errorlevel 3 (
    set "TORCH_INDEX=https://download.pytorch.org/whl/xpu"
) else if errorlevel 2 (
    set "TORCH_INDEX=https://download.pytorch.org/whl/cu126"
) else (
    set "TORCH_INDEX=https://download.pytorch.org/whl/cu130"
)

"%PY%" -m pip install --upgrade pip %PIPFLAGS%
"%PY%" -m pip install wheel %PIPFLAGS%
echo Uninstalling existing Pytorch if found
"%PY%" -m pip uninstall -y torch torchvision torchaudio
echo Installing PyTorch from %TORCH_INDEX%
"%PY%" -m pip install torch==2.10.0 torchvision torchaudio --index-url %TORCH_INDEX% %PIPFLAGS%
"%PY%" -m pip install -r .\requirements.txt %PIPFLAGS%

rem detectron2 (ViTMatte / MEMatte only) must be compiled, which on Windows needs
rem Visual Studio Build Tools with C++. Sammie works without it.
echo.
choice /C YN /N /M "Build detectron2 for ViTMatte/MEMatte (needs Visual Studio C++ Build Tools and git)? [Y/N]"
if not errorlevel 2 (
    "%PY%" -m pip install --no-build-isolation "git+https://github.com/facebookresearch/detectron2.git@02b5c4e295e990042a714712c21dc79b731e8833" %PIPFLAGS%
    if errorlevel 1 echo WARNING: detectron2 failed to build. Only ViTMatte and MEMatte are unavailable.
)

rem LTX-2 code for LTX Alpha matting (LTX-2 Community License, not part of this repository).
rem --no-deps: its dependencies are in requirements.txt; pip must not swap the CUDA torch.
echo.
choice /C YN /N /M "Install Lightricks' LTX-2 code for LTX Alpha matting (needs git; LTX-2 Community License)? [Y/N]"
if not errorlevel 2 (
    "%PY%" -m pip install --no-deps "ltx-core @ git+https://github.com/Lightricks/LTX-2.git@be6d6d0a69d11a0a399c462ed92606692e5f8697#subdirectory=packages/ltx-core" "ltx-pipelines @ git+https://github.com/Lightricks/LTX-2.git@be6d6d0a69d11a0a399c462ed92606692e5f8697#subdirectory=packages/ltx-pipelines" %PIPFLAGS%
)

echo Completed. Models are not downloaded automatically - see README.md.
pause
