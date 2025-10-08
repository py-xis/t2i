"""Text edition with DeepFloyd-IF model on SimpleBench."""

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

from diffusers import IFPipeline, IFSuperResolutionPipeline
from diffusers.training_utils import set_seed
from src.eval.basic_metrics import calculate_mse, calculate_psnr_from_mse
from src.eval.clipscore import clip_metrics, extract_all_images
from src.eval.ocr_eval import get_ocr_easyocr, get_text_easyocr, ocr_metrics
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
SDXL_MODEL_NAME_OR_PATH = "DeepFloyd/IF-I-XL-v1.0"
SR_MODEL_NAME_OR_PATH = "DeepFloyd/IF-II-L-v1.0"
SEED = 42
N_SAMPLES_PER_PROMPT = 4
BATCH_SIZE = 16
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.0
TIMESTEP_START_PATCHING = None
ATTENTIONS_TO_PATCH = None
USE_DIFFERENT_TEMPLATES = False
# BENCHMARK = "glyph_toxic"
BENCHMARK = "glyph_simple"
# BENCHMARK = "glyph_creative"


def set_to_string(int_set):
    return "UPPER"


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].shape[1], imgs[0].shape[0]
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        img = img.astype(np.uint8)
        grid.paste(Image.fromarray(img), box=(i % cols * w, i // cols * h))
    return grid


def sample(
    pipe,
    pipe_upsample,
    prompts,
    noise,
    noise_upsample,
    batch_size,
    num_inference_steps,
    generator,
    device,
    run_with_cache,
    attn_idx_to_patch=None,
    attn_heads_idx_to_patch=None,
    timestep_start_patching=0,
    guidance_scale=GUIDANCE_SCALE,
):
    if run_with_cache is False or attn_idx_to_patch is None:
        all_images = np.zeros((len(prompts), 256, 256, 3), dtype=np.uint8)
    else:
        all_images = np.zeros((len(prompts), 64, 64, 3), dtype=np.uint8)
    with tqdm(total=len(prompts)) as pbar:
        for batch_num, batch_start in enumerate(range(0, len(prompts), batch_size)):
            prompt = prompts[batch_start : batch_start + batch_size]
            latent = noise[batch_start : batch_start + batch_size].to(device)
            pipe_upsample.to("cpu", silence_dtype_warnings=True)
            pipe.to(device)
            images = pipe(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                generator=generator,
                latents=latent,
                run_with_cache=run_with_cache,
                attn_idx_to_patch=attn_idx_to_patch,
                batch_num=batch_num,
                attn_heads_idx_to_patch=attn_heads_idx_to_patch,
                timestep_start_patching=timestep_start_patching,
                guidance_scale=guidance_scale,
            ).images
            pipe.to("cpu", silence_dtype_warnings=True)

            if run_with_cache is False or attn_idx_to_patch is None:
                pipe_upsample.to(device)
                latent = noise_upsample[batch_start : batch_start + batch_size].to(device)
                images = pipe_upsample(
                    image=images,
                    prompt=prompt,
                    generator=generator,
                    output_type="np",
                    latents=latent,
                ).images
                pipe_upsample.to("cpu", silence_dtype_warnings=True)
                images = images * 255
            all_images[batch_start : batch_start + batch_size] = images
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
    ocr_model,
    clip_model,
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
        "Templates_A": templates_A,
        "Templates_B": templates_B,
    }


if __name__ == "__main__":
    SAVE_DIR = (
        f"results_if/{BENCHMARK}/UPPERBOUND/diff_templ_{USE_DIFFERENT_TEMPLATES}/"
        f"{START_TIME}_"
        f"seed_{SEED}_"
        f"n_samples_per_prompt_{N_SAMPLES_PER_PROMPT}_"
        f"n_inference_steps_{NUM_INFERENCE_STEPS}_"
        f"guidance_scale_{GUIDANCE_SCALE}_"
        f"timestep_start_patching_{TIMESTEP_START_PATCHING}_"
        f"attentions_to_patch_{set_to_string(ATTENTIONS_TO_PATCH)}"
    )

    accelerator = Accelerator()
    if accelerator.is_main_process:
        os.makedirs(SAVE_DIR, exist_ok=True)
        logging.info(f"Seed: {SEED}")
        logging.info(f"Device: {accelerator.device}")
        logging.info(f"Num inference steps: {NUM_INFERENCE_STEPS}")
        logging.info(f"Batch size: {BATCH_SIZE}")
        logging.info(f"Save dir: {SAVE_DIR}")
    set_seed(SEED)

    pipe = IFPipeline.from_pretrained(
        SDXL_MODEL_NAME_OR_PATH,
        variant="fp16",
        use_safetensors=True,
        token=os.environ.get("HF_TOKEN"),
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
        watermarker=None,
    )
    pipe.set_progress_bar_config(disable=True)

    pipe_upsample = IFSuperResolutionPipeline.from_pretrained(
        SR_MODEL_NAME_OR_PATH,
        variant="fp16",
        use_safetensors=True,
        token=os.environ.get("HF_TOKEN"),
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
        watermarker=None,
    )
    pipe_upsample.set_progress_bar_config(disable=True)

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
        (N_SAMPLES_PER_PROMPT, 3, 64, 64),
        generator=torch.Generator().manual_seed(SEED),
        dtype=torch.float16,
    )
    noises = noises.repeat(len(prompts_A) // N_SAMPLES_PER_PROMPT, 1, 1, 1)
    noises_upsample = torch.randn(
        (N_SAMPLES_PER_PROMPT, 3, 256, 256),
        generator=torch.Generator().manual_seed(SEED),
        dtype=torch.float16,
    )
    noises_upsample = noises_upsample.repeat(len(prompts_A) // N_SAMPLES_PER_PROMPT, 1, 1, 1)

    if accelerator.is_main_process:
        ocr_model = get_ocr_easyocr(use_cuda=True)
        clip_model, transform = clip.load("ViT-B/32", device=accelerator.device, jit=False)
        clip_model.eval()
    else:
        ocr_model = None
        clip_model = None
        transform = None

    prompts_indices = list(range(len(prompts_A)))
    all_original_images_B = []
    all_patched_images = []
    if accelerator.is_main_process:
        logging.info("Sampling original images B ...")
    with accelerator.split_between_processes(prompts_indices) as device_indices:
        p_A = [prompts_A[i] for i in device_indices]
        p_B = [prompts_B[i] for i in device_indices]
        n = torch.stack([noises[i] for i in device_indices])
        n_u = torch.stack([noises_upsample[i] for i in device_indices])
        if accelerator.is_main_process:
            print(f"Sampling | {(len(device_indices)/len(prompts_indices)):.2f}")
        original_images_B = sample(
            pipe,
            pipe_upsample,
            [p["prompt"] for p in p_B],
            n,
            n_u,
            BATCH_SIZE,
            NUM_INFERENCE_STEPS,
            torch.Generator().manual_seed(SEED),
            accelerator.device,
            run_with_cache=False,
        )
        all_original_images_B.extend(original_images_B)
    accelerator.wait_for_everyone()

    all_original_images_B = gather_object(all_original_images_B)
    all_original_images_B = np.array(all_original_images_B)
    if accelerator.is_main_process:
        np.save(os.path.join(SAVE_DIR, "None.npy"), all_original_images_B)

    if accelerator.is_main_process:
        logging.info("Calculating metrics ...")
        original_images_B_feats = extract_all_images(
            all_original_images_B, clip_model, accelerator.device, batch_size=BATCH_SIZE
        )
        original_images_B_metrics = calculate_metrics(
            all_original_images_B,
            original_images_B_feats,
            all_original_images_B,
            [p["text"] for p in prompts_A],
            [p["text"] for p in prompts_B],
            [p["prompt"] for p in prompts_A],
            [p["prompt"] for p in prompts_B],
            accelerator.device,
            BATCH_SIZE,
            ocr_model,
            clip_model,
            [p["prompt"] for p in prompts_AB] if prompts_AB is not None else None,
            templates_A,
            templates_B,
        )
        original_images_B_df = pd.DataFrame(
            original_images_B_metrics,
        )
        original_images_B_df["Block_patched"] = ["-" for _ in range(len(prompts_B))]
        all_metrics_df = original_images_B_df
        all_metrics_df.to_csv(os.path.join(SAVE_DIR, "metrics.csv"))
    accelerator.wait_for_everyone()
    logging.info("Finito!")
