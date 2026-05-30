#!/usr/bin/env bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the script's directory
cd "$SCRIPT_DIR" || exit
cd ../simplegep || exit

echo "Current Working Directory"
pwd

echo 'Run no dp trainer'
poetry run train_no_dp_cifar10
echo 'no dp trainer finished'