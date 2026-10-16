#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"
"$DIR/python_env/bin/python" "$DIR/launcher.py" --rpath "$DIR"
