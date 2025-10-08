#!/bin/bash

#SBATCH --gres=gpu:2
#SBATCH --partition=all
#SBATCH --mem=300G
#SBATCH --ntasks 1
#SBATCH --nodelist=sprint2
#SBATCH -o slurm_out/slurm-%j.log
#SBATCH --job-name=sdxl-mario

# mitigates activation problems
eval "$(conda shell.bash hook)"
source .bashrc

conda activate t2i_env
# debug print-outs
echo USER: $USER
which conda
which python

# run the code
PYTHONPATH=. accelerate launch --mixed_precision fp16 --num_processes 2 src/explore_sdxl_layers_mario.py --experiment default --prompt_file data/MARIOEval/ChineseDrawText/ChineseDrawText.txt --output_dir results_mario/sdxl
