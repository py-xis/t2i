#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=60
#SBATCH --mem=400G
#SBATCH --time=04:00:00
#SBATCH --account=plgdiffusion-gpu-gh200 
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=slurm_out/sd3_text_style_attn_5.out
#SBATCH --job-name=sd3_text_style_attn_5

cd /net/scratch/hscra/plgrid/plglukaszst/projects/t2i-detoxify
module load ML-bundle/24.06a
source ./venv/bin/activate
export PYTHONPATH=$PWD

echo USER: $USER
which python

# run the code
accelerate launch --num_processes 4 src/text_styles/edit_style.py --attn_to_patch 5
