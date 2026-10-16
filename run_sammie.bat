@echo off
rem Launch Sammie-Roto with the embedded Python (release zip) or the venv (install_dependencies.bat).
pushd %~dp0
if exist "python-3.12.8-embed-amd64\python.exe" (
    "python-3.12.8-embed-amd64\python.exe" launcher.py --rpath "%~dp0." %*
) else if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" launcher.py --rpath "%~dp0." %*
) else (
    echo No Python environment found. Run install_dependencies.bat first.
    pause
)
