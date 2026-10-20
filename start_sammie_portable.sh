#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"
"$DIR/python_env/bin/python3.12" "$DIR/launcher.py" --rpath "$DIR"
