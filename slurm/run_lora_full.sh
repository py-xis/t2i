#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --partition=all
#SBATCH --mem=50G
#SBATCH --ntasks 1
#SBATCH -o slurm_out/slurm-%j.log
#SBATCH --job-name=loraFull

# mitigates activation problems
eval "$(conda shell.bash hook)"
source .bashrc

conda activate env2
# debug print-outs
echo USER: $USER
which conda
which python

export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
export PYTHONPATH=$(pwd):$PYTHONPATH

# run the code
accelerate launch /storage2/bartosz/code/t2i2/diffusers/examples/text_to_image/train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --train_data_dir=/storage3/datasets/laion-ocr/train \
  --enable_xformers_memory_efficient_attention \
  --resolution=512 --center_crop \
  --train_batch_size=64 \
  --gradient_accumulation_steps=1 --gradient_checkpointing \
  --max_train_steps=10000 \
  --learning_rate=1e-06 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --report_to="wandb" \
  --checkpointing_steps=5000 \
  --output_dir="sdxl-lora-all-layers" \
  --validation_steps=500 \
  --output_dir=/storage2/bartosz/code/t2i2/checkpoints/sdxl-lora-all-layers \
  --seed=42
