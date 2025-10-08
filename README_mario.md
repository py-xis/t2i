# Prepare MARIO dataset for finetuning

1. Download googledrive folder and unzip with `mario-laion-unzip.py`
2. Create subset of links for dataset downloading using `random_subset.py`
3. Download dataset `img2dataset --url_list=url.txt --output_folder=laion_ocr --thread_count=64  --resize_mode=no`
4. Run `prepare_train.py` to prepare `train` directory with all images and json files
6. Run `prepare_metadata.py` to prepare `metadata.jsonl` file with names of all pairs (image, caption)

# Finetune LoRA 2 layers
```bash
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
export PYTHONPATH=$(pwd):$PYTHONPATH

accelerate launch /storage2/bartosz/code/t2i2/diffusers/examples/text_to_image/train_text_to_image_lora_sdxl_2L.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --train_data_dir=/storage3/datasets/laion-ocr/train \
  --enable_xformers_memory_efficient_attention \
  --resolution=512 --center_crop \
  --train_batch_size=64 \
  --gradient_accumulation_steps=1 --gradient_checkpointing \
  --max_train_steps=10000 \
  --use_8bit_adam \
  --learning_rate=1e-06 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --report_to="wandb" \
  --validation_prompt="A sign that says 'education'" \
  --checkpointing_steps=5000 \
  --output_dir="sdxl-lora-2-layers" \
  --validation_steps=200 \
  --output_dir=/storage2/bartosz/code/t2i2/checkpoints
```


# Finetune LoRA full
```bash
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
export PYTHONPATH=$(pwd):$PYTHONPATH

accelerate launch /storage2/bartosz/code/t2i2/diffusers/examples/text_to_image/train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --train_data_dir=/storage3/datasets/laion-ocr/train \
  --enable_xformers_memory_efficient_attention \
  --resolution=512 --center_crop \
  --train_batch_size=64 \
  --gradient_accumulation_steps=1 --gradient_checkpointing \
  --max_train_steps=10000 \
  --use_8bit_adam \
  --learning_rate=1e-06 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --report_to="wandb" \
  --validation_prompt="A sign that says 'education'" \
  --checkpointing_steps=5000 \
  --output_dir="sdxl-lora-2-layers" \
  --validation_steps=200 \
  --output_dir=/storage2/bartosz/code/t2i2/checkpoints
```
