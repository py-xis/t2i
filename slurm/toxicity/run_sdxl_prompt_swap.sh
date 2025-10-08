#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=60
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --account=plgdiffusion-gpu-gh200 
#SBATCH --partition=plgrid-gpu-gh200 
#SBATCH --output=slurm_out/toxicity_sdxl_prompt_swap.out
#SBATCH --job-name=toxicity_sdxl_prompt_swap

cd /net/scratch/hscra/plgrid/plglukaszst/projects/t2i-detoxify
module load ML-bundle/24.06a
source ./venv/bin/activate
export PYTHONPATH=$PWD

echo USER: $USER
which python

# run the code
accelerate launch --num_processes 4 src/toxic_prompt_swap/edit_sdxl_glyph_toxic_distr.py