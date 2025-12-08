#!/usr/bin/env bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the script's directory
cd "$SCRIPT_DIR" || exit
cd ../simplegep || exit

echo "Current Working Directory"
pwd

echo 'Run DP-SGD sweep'
poetry run python sweepers/sweep.py -p --dp_method dp_sgd
echo 'DP-SGD sweep finished'