"""Text edition with SD3 model on SimpleBench."""

import logging
import os
import time

import clip
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from PIL import Image
from pytorch_msssim import ssim
from tqdm import tqdm

from diffusers import StableDiffusion3Pipeline
from diffusers.training_utils import set_seed
from src.eval.basic_metrics import calculate_mse, calculate_psnr_from_mse
from src.eval.clipscore import clip_metrics, extract_all_images
from src.eval.ocr_eval import get_ocr_easyocr, get_text_easyocr, ocr_metrics
from src.eval.text_detection import remove_text_boxes, setup_text_detection_model
from src.eval.text_distance import get_levenshtein_distances
from src.prepare_glyph import (
    prepare_prompts_glyph_creative_bench,
    prepare_prompts_glyph_simple_bench,
    prepare_toxic_bench,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

START_TIME = time.strftime("%Y%m%d_%H%M%S")
SDXL_MODEL_NAME_OR_PATH = "stabilityai/stable-diffusion-3-medium-diffusers"
SEED = 42
N_SAMPLES_PER_PROMPT = 4
BATCH_SIZE = 20
NUM_INFERENCE_STEPS = 28
GUIDANCE_SCALE = 7.0
TIMESTEP_START_PATCHING = None
ATTENTIONS_TO_PATCH = None
USE_DIFFERENT_TEMPLATES = False
# BENCHMARK = "glyph_toxic"
BENCHMARK = "glyph_simple"
# BENCHMARK = "glyph_creative"


def set_to_string(int_set):
    return "UPPER"


SAVE_DIR = (
    f"results_sd3/{BENCHMARK}/UPPERBOUND/different_templates_{USE_DIFFERENT_TEMPLATES}/"
    f"{START_TIME}_"
    f"seed_{SEED}_"
    f"n_samples_per_prompt_{N_SAMPLES_PER_PROMPT}_"
    f"n_inference_steps_{NUM_INFERENCE_STEPS}_"
    f"guidance_scale_{GUIDANCE_SCALE}_"
    f"timestep_start_patching_{TIMESTEP_START_PATCHING}_"
    f"attentions_to_patch_{set_to_string(ATTENTIONS_TO_PATCH)}"
)

os.makedirs(SAVE_DIR, exist_ok=True)

logging.info(f"Seed: {SEED}")
logging.info(f"Num inference steps: {NUM_INFERENCE_STEPS}")
logging.info(f"Batch size: {BATCH_SIZE}")
logging.info(f"Save dir: {SAVE_DIR}")


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].shape[1], imgs[0].shape[0]
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        img = img.astype(np.uint8)
        grid.paste(Image.fromarray(img), box=(i % cols * w, i // cols * h))
    return grid


set_seed(SEED)

pipe = StableDiffusion3Pipeline.from_pretrained(
    SDXL_MODEL_NAME_OR_PATH,
    variant="fp16",
    use_safetensors=True,
    token=os.environ.get("HF_TOKEN"),
    torch_dtype=torch.float16,
)

pipe.set_progress_bar_config(disable=True)

if BENCHMARK == "glyph_creative":
    prompts_A, prompts_B, prompts_AB, templates_A, templates_B = prepare_prompts_glyph_creative_bench(
        n_samples_per_prompt=N_SAMPLES_PER_PROMPT, use_different_templates=USE_DIFFERENT_TEMPLATES
    )
elif BENCHMARK == "glyph_simple":
    prompts_A, prompts_B, prompts_AB, templates_A, templates_B = prepare_prompts_glyph_simple_bench(
        n_samples_per_prompt=N_SAMPLES_PER_PROMPT
    )
else:
    raise ValueError(f"Unknown benchmark: {BENCHMARK}")
logging.info(f"Number of prompts: {len(prompts_A)}")

noises = torch.randn(
    (N_SAMPLES_PER_PROMPT, 16, 128, 128),
    generator=torch.Generator().manual_seed(SEED),
    dtype=torch.float16,
)
noises = noises.repeat(len(prompts_A) // N_SAMPLES_PER_PROMPT, 1, 1, 1)


def sample(
    prompts,
    noise,
    batch_size,
    num_inference_steps,
    generator,
    device,
    run_with_cache,
    attn_idx_to_patch=None,
    attn_heads_idx_to_patch=None,
    timestep_start_patching=0,
):
    all_images = np.zeros((len(prompts), 1024, 1024, 3), dtype=np.uint8)
    with tqdm(total=len(prompts)) as pbar:
        for batch_num, batch_start in enumerate(range(0, len(prompts), batch_size)):
            prompt = prompts[batch_start : batch_start + batch_size]
            latent = noise[batch_start : batch_start + batch_size].to(device)
            images = pipe(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                generator=generator,
                latents=latent,
                run_with_cache=run_with_cache,
                attn_idx_to_patch=attn_idx_to_patch,
                output_type="np",
                batch_num=batch_num,
                attn_heads_idx_to_patch=attn_heads_idx_to_patch,
                timestep_start_patching=timestep_start_patching,
                guidance_scale=GUIDANCE_SCALE,
            ).images
            images = images * 255
            all_images[batch_start : batch_start + batch_size] = images.astype(np.uint8)
            pbar.update(len(prompt))
    return all_images


def calculate_metrics(
    original_images_A,
    original_images_A_feats,
    images,
    texts_A,
    texts_B,
    prompts_A,
    prompts_B,
    device,
    batch_size,
    prompts_AB=None,
    templates_A=None,
    templates_B=None,
):
    # calculate metrics per sample
    # 1. MSE
    mse = calculate_mse(original_images_A, images)
    # 2.PSNR
    psnr = calculate_psnr_from_mse(mse)
    # 3. SSIM
    ssim_val = ssim(
        torch.from_numpy(original_images_A.astype(np.float32)).permute((0, 3, 1, 2)),
        torch.from_numpy(images.astype(np.float32)).permute((0, 3, 1, 2)),
        data_range=255,
        size_average=False,
    ).numpy()
    # 4. OCR Acc/Prec/Rec
    ocr_texts = [get_text_easyocr(ocr_model, images[i]).lower() for i in range(images.shape[0])]
    ocr_pr_A, ocr_rec_A, ocr_acc_A = ocr_metrics(ocr_texts, texts_A)
    ocr_pr_B, ocr_rec_B, ocr_acc_B = ocr_metrics(ocr_texts, texts_B)
    # 5. CLIPScore
    clip_metrics_results = clip_metrics(
        clip_model,
        images,
        original_images_A_feats,
        device,
        batch_size,
        prompts_A,
        prompts_B,
        prompts_AB,
        templates_A,
        templates_B,
    )
    if prompts_AB is not None:
        image_sim, prompt_A_sim, prompt_B_sim, prompt_AB_sim, template_A_sim, template_B_sim = clip_metrics_results
    else:
        image_sim, prompt_A_sim, prompt_B_sim = clip_metrics_results
        prompt_AB_sim = None
        template_A_sim = None
        template_B_sim = None
    # 6. Levenshtein distance
    leve_A = get_levenshtein_distances(ocr_texts, texts_A)
    leve_B = get_levenshtein_distances(ocr_texts, texts_B)

    return {
        "MSE": mse,
        "PSNR": psnr,
        "SSIM": ssim_val,
        "OCR_A_Prec": ocr_pr_A,
        "OCR_A_Rec": ocr_rec_A,
        "OCR_A_Acc": ocr_acc_A,
        "OCR_B_Prec": ocr_pr_B,
        "OCR_B_Rec": ocr_rec_B,
        "OCR_B_Acc": ocr_acc_B,
        "CLIPScore_image": image_sim,
        "CLIPScore_prompt_A": prompt_A_sim,
        "CLIPScore_prompt_B": prompt_B_sim,
        "Levenshtein_A": leve_A,
        "Levenshtein_B": leve_B,
        "Prompts_A": prompts_A,
        "Prompts_B": prompts_B,
        "OCR_texts": ocr_texts,
        "Texts_A": texts_A,
        "Texts_B": texts_B,
        "CLIPScore_prompt_AB": prompt_AB_sim,
        "CLIPScore_template_A": template_A_sim,
        "CLIPScore_template_B": template_B_sim,
    }


prompts_indices = list(range(len(prompts_B)))
distributed_state = Accelerator()
pipe = pipe.to(distributed_state.device)
all_original_images_B = []

with distributed_state.split_between_processes(prompts_indices) as device_indices:
    p_A = [prompts_A[i] for i in device_indices]
    p_B = [prompts_B[i] for i in device_indices]
    n = torch.stack([noises[i] for i in device_indices])

    original_images_B = sample(
        [p["prompt"] for p in p_B],
        n,
        BATCH_SIZE,
        NUM_INFERENCE_STEPS,
        torch.Generator().manual_seed(SEED),
        distributed_state.device,
        run_with_cache=False,
    )

    all_original_images_B.extend(original_images_B)

distributed_state.wait_for_everyone()
all_original_images_B = gather_object(all_original_images_B)
all_original_images_B = np.array(all_original_images_B)
print(all_original_images_B.shape)

if distributed_state.is_main_process:
    logging.info("Calculating metrics ...")
    ocr_model = get_ocr_easyocr(use_cuda=True)

    clip_model, transform = clip.load("ViT-B/32", device=distributed_state.device, jit=False)
    clip_model.eval()
    np.save(os.path.join(SAVE_DIR, "UPPER.npy"), all_original_images_B)

    original_images_B_feats = extract_all_images(
        all_original_images_B, clip_model, distributed_state.device, batch_size=BATCH_SIZE
    )

    original_images_B_metrics = calculate_metrics(
        all_original_images_B,
        original_images_B_feats,
        all_original_images_B,
        [p["text"] for p in prompts_A],
        [p["text"] for p in prompts_B],
        [p["prompt"] for p in prompts_A],
        [p["prompt"] for p in prompts_B],
        "cuda",
        BATCH_SIZE,
        [p["prompt"] for p in prompts_AB] if prompts_AB is not None else None,
        templates_A,
        templates_B,
    )

    original_images_B_df = pd.DataFrame(
        original_images_B_metrics,
    )
    original_images_B_df["Block_patched"] = ["-" for _ in range(len(prompts_A))]
    original_images_B_df.to_csv(os.path.join(SAVE_DIR, "metrics.csv"))
    logging.info("Finito!")
