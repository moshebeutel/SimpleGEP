#!/usr/bin/env bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the simplegep directory
cd "$SCRIPT_DIR" || exit
cd ../simplegep || exit

echo "Current Working Directory"
pwd

echo 'Run GEP sweep'
poetry run python sweepers/sweep.py -p --dp_method gep
echo 'GEP sweep finished'