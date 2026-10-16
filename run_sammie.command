#!/bin/bash
# Launch Sammie-Roto from a source install (venv/ created by install_dependencies.sh)
# or from the portable build (python_env/).
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"
if [ -x "$DIR/venv/bin/python" ]; then
    PY="$DIR/venv/bin/python"
elif [ -x "$DIR/python_env/bin/python" ]; then
    PY="$DIR/python_env/bin/python"
else
    echo "No Python environment found. Run: bash install_dependencies.sh"
    exit 1
fi
exec "$PY" "$DIR/launcher.py" --rpath "$DIR" "$@"
