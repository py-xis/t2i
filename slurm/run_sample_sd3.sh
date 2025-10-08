#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --partition=all
#SBATCH --mem=35G
#SBATCH --ntasks 1
#SBATCH --nodelist=sprint2

# mitigates activation problems
eval "$(conda shell.bash hook)"
source .bashrc

# activate the correct environment
conda activate t2i

# debug print-outs
echo USER: $USER
which conda
which python

# run the code
PYTHONPATH=. python sample_sd3.py