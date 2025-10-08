#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --partition=all
#SBATCH --mem=256G
#SBATCH --ntasks 1
#SBATCH -o slurm_out/slurm-%j.log
#SBATCH --job-name=sd3-attn

# mitigates activation problems
eval "$(conda shell.bash hook)"
source .bashrc

conda activate env2
# debug print-outs
echo USER: $USER
which conda
which python

# run the code
PYTHONPATH=. python3 src/attention_localization_sd3_glyph_simple.py
