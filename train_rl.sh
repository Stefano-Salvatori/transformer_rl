#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate transformer_rl

python main.py "$@" 