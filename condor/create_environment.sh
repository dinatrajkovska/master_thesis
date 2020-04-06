#!/usr/bin/env bash

# Read .bashrc file from your home directory to load necessary packages
source ~/.bashrc

# Execute file specified as first argument to this shell file
python $1
