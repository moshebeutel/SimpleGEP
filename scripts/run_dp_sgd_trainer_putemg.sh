#!/usr/bin/env bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the script's directory
cd "$SCRIPT_DIR" || exit
cd ../simplegep || exit

echo "Current Working Directory"
pwd

echo 'Run no dp trainer putEMG'
poetry run train_dp_sgd_putemg
echo 'no dp trainer finished'