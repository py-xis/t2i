#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=60
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --account=plgdiffusion-gpu-gh200 
#SBATCH --partition=plgrid-gpu-gh200 
#SBATCH --output=slurm_out/predict_toxic_sd3.out
#SBATCH --job-name=predict_toxic_sd3

cd /net/scratch/hscra/plgrid/plglukaszst/projects/t2i-detoxify
module load ML-bundle/24.06a
source ./venv/bin/activate
export PYTHONPATH=$PWD

echo USER: $USER
which python

# run the code
accelerate launch --num_processes 4 src/predict_toxic.py --path results_sd3/glyph_toxic/prompt_swap/20241017_183005_seed_42_n_samples_per_prompt_4_n_inference_steps_28_guidance_scale_7.0/metrics.csv