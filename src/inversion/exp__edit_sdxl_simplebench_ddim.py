"""Text edition with SD-XL model on Real Images."""

import json
import logging
import math
import os
import time
from datetime import timedelta
from pathlib import Path

import clip
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, gather_object
from PIL import Image
from pytorch_msssim import ssim
from tqdm import tqdm

from diffusers import DDIMScheduler, StableDiffusionXLPipeline
from diffusers.training_utils import set_seed
from src.eval.basic_metrics import calculate_mse, calculate_psnr_from_mse
from src.eval.clipscore import clip_metrics, extract_all_images
from src.eval.ocr_eval import get_ocr_easyocr, get_text_easyocr, ocr_metrics
from src.eval.text_distance import get_levenshtein_distances
from src.inversion.ddim.sdxl_ddim import SDXLDDIMPipeline
from src.prepare_glyph import prepare_prompts_glyph_simple_bench

torch.backends.cuda.matmul.allow_tf32 = True
torch._inductor.config.conv_1x1_as_mm = True  # type: ignore
torch._inductor.config.coordinate_descent_tuning = True  # type: ignore
torch._inductor.config.epilogue_fusion = False  # type: ignore
torch._inductor.config.coordinate_descent_check_all_directions = True  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=1800 * 6))
accelerator = Accelerator(kwargs_handlers=[kwargs])


START_TIME = time.strftime("%Y%m%d_%H%M%S")
SDXL_MODEL_NAME_OR_PATH = "stabilityai/stable-diffusion-xl-base-1.0"
SEED = 42
N_SAMPLES_PER_PROMPT = 4
BATCH_SIZE = 20
NUM_INFERENCE_STEPS = 50
TIMESTEP_START_PATCHING = 0
DS_PATH = Path("/net/storage/pr3/plgrid/plggdiffusion/ds_ls/real_images_400").resolve()
INVERSION_GUIDANCE_SCALE = 1.0
DDIM_GUIDANCE_SCALE = 5.0
GUIDANCE_SCALE_CACHE = 5.0
GUIDANCE_SCALE_PATCH = 5.0
ATTENTIONS_TO_PATCH = [55, 56, 57]


def set_to_string(int_set):
    return "_A".join(str(num) for num in int_set)


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
    guidance_scale=5.0,
):
    all_images = np.zeros((len(prompts), 1024, 1024, 3), dtype=np.uint8)
    loop = (
        tqdm(
            enumerate(range(0, len(prompts), batch_size)), total=math.ceil(len(prompts) / batch_size), desc="Sampling"
        )
        if accelerator.is_main_process
        else enumerate(range(0, len(prompts), batch_size))
    )
    for batch_num, idx_start in loop:
        prompt = prompts[idx_start : idx_start + batch_size]
        latent = noise[idx_start : idx_start + batch_size].to(device)
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
            guidance_scale=guidance_scale,
        ).images
        images = images * 255
        all_images[idx_start : idx_start + batch_size] = images.astype(np.uint8)
    return all_images


def reverse_sample(pipe_rev, prompts, images, batch_size, num_inference_steps, generator, inversion_guidance_scale):
    all_latents = np.zeros((len(prompts), 4, 128, 128))
    loop = (
        tqdm(range(0, len(prompts), batch_size), total=math.ceil(len(prompts) / batch_size), desc="Reverse Sampling")
        if accelerator.is_main_process
        else range(0, len(prompts), batch_size)
    )
    for idx in loop:
        p_A = prompts[idx : idx + batch_size]
        imgs = images[idx : idx + batch_size] / 255
        lats = pipe_rev(
            prompt=p_A,
            num_inference_steps=num_inference_steps,
            generator=generator,
            image=imgs,
            guidance_scale=inversion_guidance_scale,
        )
        latents = lats.images.cpu().numpy()
        all_latents[idx : idx + batch_size] = latents
    return all_latents


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
):
    # calculate metrics per sample
    # 1. MSE
    mse = calculate_mse(original_images_A, images)
    # 2.PSNR
    psnr = calculate_psnr_from_mse(mse)
    logging.info(f"MSE, PSNR completed")
    # 3. SSIM
    ssim_vals = []
    for i in tqdm(range(0, images.shape[0], 100), total=math.ceil(images.shape[0] / 100), desc="Calculating SSIM"):
        ssim_val = ssim(
            torch.from_numpy(original_images_A[i : i + 100].astype(np.float32)).permute((0, 3, 1, 2)),
            torch.from_numpy(images[i : i + 100].astype(np.float32)).permute((0, 3, 1, 2)),
            data_range=255,
            size_average=False,
        ).numpy()
        ssim_vals.append(ssim_val)
    ssim_val = np.concatenate(ssim_vals).mean()
    logging.info(f"SSIM completed")
    # 4. OCR Acc/Prec/Rec
    ocr_texts = [get_text_easyocr(ocr_model, images[i]).lower() for i in range(images.shape[0])]
    ocr_pr_A, ocr_rec_A, ocr_acc_A = ocr_metrics(ocr_texts, texts_A)
    ocr_pr_B, ocr_rec_B, ocr_acc_B = ocr_metrics(ocr_texts, texts_B)
    logging.info(f"OCR completed")
    # 5. CLIPScore
    image_sim, prompt_A_sim, prompt_B_sim = clip_metrics(
        clip_model,
        images,
        original_images_A_feats,
        device,
        batch_size,
        prompts_A,
        prompts_B,
    )
    logging.info(f"CLIPScore completed")
    # 6. Levenshtein distance
    leve_A = get_levenshtein_distances(ocr_texts, texts_A)
    leve_B = get_levenshtein_distances(ocr_texts, texts_B)
    logging.info(f"Levenshtein distance completed")
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
    }


SAVE_DIR = (
    f"results_sdxl/inversion/glyph_simplebench/ddim_edit/"
    f"{START_TIME}_"
    f"seed_{SEED}_"
    f"n_inference_steps_{NUM_INFERENCE_STEPS}_"
    f"timestep_start_patching_{TIMESTEP_START_PATCHING}_"
    f"inv_guidance_scale_{INVERSION_GUIDANCE_SCALE}"
    f"ddim_guidance_scale_{DDIM_GUIDANCE_SCALE}_"
    f"cache_guidance_scale_{GUIDANCE_SCALE_CACHE}_"
    f"patch_guidance_scale_{GUIDANCE_SCALE_PATCH}_"
    f"attentions_to_patch_A{set_to_string(ATTENTIONS_TO_PATCH)}"
)
if __name__ == "__main__":

    if accelerator.is_main_process:
        os.makedirs(SAVE_DIR, exist_ok=True)
        logging.info(f"Seed: {SEED}")
        logging.info(f"Device: {accelerator.device}")
        logging.info(f"Num inference steps: {NUM_INFERENCE_STEPS}")
        logging.info(f"Batch size: {BATCH_SIZE}")
        logging.info(f"Save dir: {SAVE_DIR}")

    set_seed(SEED)
    pipe_forward = StableDiffusionXLPipeline.from_pretrained(
        SDXL_MODEL_NAME_OR_PATH,
        variant="fp16",
        use_safetensors=True,
        token=os.environ.get("HF_TOKEN"),
        torch_dtype=torch.float16,
        silent=True,
    )
    pipe_forward.scheduler = DDIMScheduler.from_config(pipe_forward.scheduler.config)
    pipe_forward.set_progress_bar_config(disable=True)
    pipe_forward.to(accelerator.device)

    if accelerator.is_main_process:
        logging.info("Preparing data...")
    prompts_A, prompts_B = prepare_prompts_glyph_simple_bench(n_samples_per_prompt=N_SAMPLES_PER_PROMPT)
    noises = torch.randn(
        (N_SAMPLES_PER_PROMPT, 4, 128, 128),
        generator=torch.Generator().manual_seed(SEED),
        dtype=torch.float16,
    )
    noises = noises.repeat(len(prompts_A) // N_SAMPLES_PER_PROMPT, 1, 1, 1)

    pipe_rev = SDXLDDIMPipeline.from_pretrained(
        SDXL_MODEL_NAME_OR_PATH,
        variant="fp16",
        use_safetensors=True,
        token=os.environ.get("HF_TOKEN"),
        torch_dtype=torch.float16,
        silent=True,
    ).to(accelerator.device)
    pipe_rev.set_progress_bar_config(disable=True)

    indices_split = list(range(len(prompts_A)))
    all_samples_A = []
    with accelerator.split_between_processes(indices_split) as index_split:
        if accelerator.is_main_process:
            logging.info(f"Sampling Real Images...")
            logging.info(
                f"[{accelerator.process_index}/{accelerator.num_processes}] Split:{len(index_split)/len(prompts_A):.3f}"
            )
        p_A = [prompts_A[i] for i in index_split]
        p_B = [prompts_B[i] for i in index_split]
        n = torch.stack([noises[i] for i in index_split])
        original_images_A = sample(
            pipe_forward,
            [p["prompt"] for p in p_A],
            n,
            BATCH_SIZE,
            NUM_INFERENCE_STEPS,
            torch.Generator().manual_seed(SEED),
            accelerator.device,
            run_with_cache=False,
            guidance_scale=DDIM_GUIDANCE_SCALE,
        )
        all_samples_A.extend(original_images_A)
    accelerator.wait_for_everyone()

    all_samples_A = gather_object(all_samples_A)
    all_samples_A = np.array(all_samples_A)
    if accelerator.is_main_process:
        np.save(os.path.join(SAVE_DIR, "Real_images.npy"), all_samples_A)
    del noises

    if accelerator.is_main_process:
        logging.info("Extracting CLIP features...")
        clip_model, transform = clip.load("ViT-B/32", device=accelerator.device, jit=False)
        clip_model.eval()
        all_samples_A_feats = extract_all_images(all_samples_A, clip_model, accelerator.device, batch_size=BATCH_SIZE)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logging.info("Calculating metrics...")
        ocr_model = get_ocr_easyocr(use_cuda=True)
        logging.info(f"OCR loaded")
        all_samples_A_metrics = calculate_metrics(
            all_samples_A,
            all_samples_A_feats,
            all_samples_A,
            [p["text"] for p in prompts_A],
            [p["text"] for p in prompts_B],
            [p["prompt"] for p in prompts_A],
            [p["prompt"] for p in prompts_B],
            "cuda",
            BATCH_SIZE,
            ocr_model,
            clip_model,
        )
        all_samples_A_df = pd.DataFrame(
            all_samples_A_metrics,
        )
        all_samples_A_df["Block_patched"] = ["-" for _ in range(len(prompts_A))]
        all_metrics_df = all_samples_A_df
        all_metrics_df.to_csv(os.path.join(SAVE_DIR, "metrics.csv"))
    accelerator.wait_for_everyone()

    all_latents = []
    with accelerator.split_between_processes(indices_split) as index_split:
        images_A_device = np.array([all_samples_A[i] for i in index_split])
        prompts_A_device = [prompts_A[i] for i in index_split]
        prompts_B_device = [prompts_B[i] for i in index_split]
        if accelerator.is_main_process:
            logging.info("A Images Inversion...")

        latents_A_device = reverse_sample(
            pipe_rev,
            [p["prompt"] for p in prompts_A_device],
            images_A_device,
            BATCH_SIZE,
            NUM_INFERENCE_STEPS,
            generator=torch.Generator().manual_seed(SEED),
            inversion_guidance_scale=INVERSION_GUIDANCE_SCALE,
        )
        all_latents.extend(latents_A_device)
    accelerator.wait_for_everyone()
    all_latents = gather_object(all_latents)
    all_latents = np.array(all_latents)
    del latents_A_device
    if accelerator.is_main_process:
        np.save(os.path.join(SAVE_DIR, "Real_latents.npy"), all_latents)
    del pipe_rev
    accelerator.free_memory()

    latents_from_A = torch.from_numpy(all_latents).half()
    all_ddim_images = []
    with accelerator.split_between_processes(indices_split) as index_split:
        prompts_A_device = [prompts_A[i] for i in index_split]
        prompts_B_device = [prompts_B[i] for i in index_split]
        noises_device = torch.stack([latents_from_A[i] for i in index_split])

        if accelerator.is_main_process:
            logging.info("Editing with DDIM ...")
        samples_ddim = sample(
            pipe_forward,
            [p["prompt"] for p in prompts_B_device],
            noises_device,
            BATCH_SIZE,
            NUM_INFERENCE_STEPS,
            torch.Generator().manual_seed(SEED),
            accelerator.device,
            run_with_cache=False,
            attn_idx_to_patch=None,
            guidance_scale=DDIM_GUIDANCE_SCALE,
        )
        all_ddim_images.extend(samples_ddim)
    accelerator.wait_for_everyone()

    all_ddim_images = gather_object(all_ddim_images)
    all_ddim_images = np.array(all_ddim_images)
    if accelerator.is_main_process:
        np.save(os.path.join(SAVE_DIR, "DDIM_edit.npy"), all_ddim_images)
    if accelerator.is_main_process:
        logging.info("Calculating metrics (DDIM)...")
        ddim_edit_metrics = calculate_metrics(
            all_samples_A,
            all_samples_A_feats,
            all_ddim_images,
            [p["text"] for p in prompts_A],
            [p["text"] for p in prompts_B],
            [p["prompt"] for p in prompts_A],
            [p["prompt"] for p in prompts_B],
            accelerator.device,
            BATCH_SIZE,
            ocr_model,
            clip_model,
        )
        ddim_edit_df = pd.DataFrame(ddim_edit_metrics)
        ddim_edit_df["Block_patched"] = ["DDIM" for _ in range(len(prompts_A))]
        all_metrics_df = pd.concat([all_metrics_df, ddim_edit_df])
        all_metrics_df.to_csv(os.path.join(SAVE_DIR, "metrics.csv"))
    del all_ddim_images
    del samples_ddim
    accelerator.wait_for_everyone()
    accelerator.free_memory()

    all_patched_images = []
    with accelerator.split_between_processes(indices_split) as index_split:
        prompts_A_device = [prompts_A[i] for i in index_split]
        prompts_B_device = [prompts_B[i] for i in index_split]
        noises_device = torch.stack([latents_from_A[i] for i in index_split])

        if accelerator.is_main_process:
            logging.info("Caching on images B ...")

        original_images_B = sample(
            pipe_forward,
            [p["prompt"] for p in prompts_B_device],
            noises_device,
            BATCH_SIZE,
            NUM_INFERENCE_STEPS,
            torch.Generator().manual_seed(SEED),
            accelerator.device,
            run_with_cache=True,
            attn_idx_to_patch=ATTENTIONS_TO_PATCH,
            guidance_scale=GUIDANCE_SCALE_CACHE,
        )

        if accelerator.is_main_process:
            logging.info("Patching on images A...")
        images_patched = sample(
            pipe_forward,
            [p["prompt"] for p in prompts_A_device],
            noises_device,
            BATCH_SIZE,
            NUM_INFERENCE_STEPS,
            torch.Generator().manual_seed(SEED),
            accelerator.device,
            run_with_cache=False,
            attn_idx_to_patch=ATTENTIONS_TO_PATCH,
            guidance_scale=GUIDANCE_SCALE_PATCH,
            timestep_start_patching=TIMESTEP_START_PATCHING,
        )
        all_patched_images.extend(images_patched)
    accelerator.wait_for_everyone()
    all_patched_images = gather_object(all_patched_images)
    all_patched_images = np.array(all_patched_images)
    if accelerator.is_main_process:
        np.save(os.path.join(SAVE_DIR, f"A{set_to_string(ATTENTIONS_TO_PATCH)}.npy"), all_patched_images)
    del original_images_B
    del images_patched
    accelerator.wait_for_everyone()
    accelerator.free_memory()

    if accelerator.is_main_process:
        logging.info("Calculating metrics (patched)...")
        patched_images_metrics = calculate_metrics(
            all_samples_A,
            all_samples_A_feats,
            all_patched_images,
            [p["text"] for p in prompts_A],
            [p["text"] for p in prompts_B],
            [p["prompt"] for p in prompts_A],
            [p["prompt"] for p in prompts_B],
            accelerator.device,
            BATCH_SIZE,
            ocr_model,
            clip_model,
        )

        patched_images_df = pd.DataFrame(
            patched_images_metrics,
        )
        patched_images_df["Block_patched"] = [f"OURS" for _ in range(len(prompts_A))]
        all_metrics_df = pd.concat([all_metrics_df, patched_images_df])
        all_metrics_df.to_csv(os.path.join(SAVE_DIR, "metrics.csv"))
    accelerator.wait_for_everyone()
    accelerator.free_memory()

    if accelerator.is_main_process:
        logging.info(f"Metrics saved to {os.path.join(SAVE_DIR, 'metrics.csv')}")
        logging.info("Finito!")
