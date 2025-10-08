#!/bin/bash

export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
export PYTHONPATH=$(pwd):$PYTHONPATH
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200


# run the code
accelerate launch --num-processes=4 /net/scratch/hscra/plgrid/plgbcywinski/code/t2i-detoxify/diffusers/examples/text_to_image/train_text_to_image_lora_sdxl_full.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --train_data_dir=/net/storage/pr3/plgrid/plggdiffusion/datasets/laion_ocr/train \
  --enable_xformers_memory_efficient_attention \
  --resolution=512 --center_crop \
  --train_batch_size=128 \
  --val_batch_size=16 \
  --gradient_accumulation_steps=1 --gradient_checkpointing \
  --max_train_steps=7254 \
  --learning_rate=1e-06 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --report_to="wandb" \
  --checkpointing_steps=2000 \
  --validation_epochs=10 \
  --output_dir=/net/scratch/hscra/plgrid/plgbcywinski/code/t2i-detoxify/checkpoints/sdxl-lora-full-val-final2 \
  --seed=42 \
