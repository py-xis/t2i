#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --partition=all
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --ntasks 1
#SBATCH -o slurm_out/ddim-sdxl-glyphsimple.log
#SBATCH --job-name=ddim-sdxl-glyphsimple
#SBATCH --nodelist=sprint3


# mitigates activation problems
eval "$(conda shell.bash hook)"
source ~/.bashrc

conda activate t2i
# debug print-outs
echo USER: $USER
which conda
which python

# run the code
PYTHONPATH=. python3 src/inversion/exp__edit_sdxl_glyphsimple_ddim.py
