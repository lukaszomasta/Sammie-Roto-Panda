#!/bin/bash
# Launch Sammie-Roto from a source install (venv/ created by install_dependencies.sh)
# or from a portable build (python_mac/ on macOS, python_env/ on Linux). The portable
# builds call the real python3.12 binary, not the python3 symlink, so they still start
# after a copy through a filesystem without symlinks (exFAT, FAT, some network shares).
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"
if [ -x "$DIR/venv/bin/python" ]; then
    PY="$DIR/venv/bin/python"
elif [ "$(uname)" = "Darwin" ] && [ -x "$DIR/python_mac/bin/python3.12" ]; then
    PY="$DIR/python_mac/bin/python3.12"
elif [ "$(uname)" = "Linux" ] && [ -x "$DIR/python_env/bin/python3.12" ]; then
    PY="$DIR/python_env/bin/python3.12"
else
    echo "No Python environment found. Run: bash install_dependencies.sh"
    exit 1
fi
exec "$PY" "$DIR/launcher.py" --rpath "$DIR" "$@"
