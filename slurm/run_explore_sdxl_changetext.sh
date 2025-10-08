#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --partition=all
#SBATCH --mem=35G
#SBATCH --ntasks 1
#SBATCH -o slurm_out/slurm-%j.log
#SBATCH --job-name=sdxl-change

# mitigates activation problems
eval "$(conda shell.bash hook)"
source .bashrc

conda activate t2i_env
# debug print-outs
echo USER: $USER
which conda
which python

# run the code
PYTHONPATH=. python3 src/explore_sdxl_layers_changetext.py
