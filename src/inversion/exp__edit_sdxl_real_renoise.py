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

from diffusers import DDIMScheduler, StableDiffusionXLImg2ImgPipeline
from diffusers.training_utils import set_seed
from src.eval.basic_metrics import calculate_mse, calculate_psnr_from_mse
from src.eval.clipscore import clip_metrics, extract_all_images
from src.eval.ocr_eval import get_ocr_easyocr, get_text_easyocr, ocr_metrics
from src.eval.text_distance import get_levenshtein_distances
from src.inversion.renoise_inversion.sampling import (
    sdxl_invert_to_latent,
    sdxl_sample_from_latent,
    sdxl_sample_from_latent_our,
)
from src.inversion.renoise_inversion.src.eunms import Model_Type, Scheduler_Type
from src.inversion.renoise_inversion.src.utils.enums_utils import get_pipes

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
BATCH_SIZE = 20
NUM_INFERENCE_STEPS = 50
TIMESTEP_START_PATCHING = 0
DS_PATH = Path("/net/storage/pr3/plgrid/plggdiffusion/ds_ls/real_images_400").resolve()
INVERSION_GUIDANCE_SCALE = 1.0
NUM_RENOISE_STEPS = 1
INVERSION_MAX_STEP = 1.0
RENOISE_GUIDANCE_SCALE = 3.0
GUIDANCE_SCALE_CACHE = 6.0
GUIDANCE_SCALE_PATCH = 6.0
ATTENTIONS_TO_PATCH = [55, 56, 57]


def prepare_data():
    data = [json.loads(line) for line in open(DS_PATH / "metadata_edit.jsonl").readlines()]
    prompts_A = [{"text": line["text"], "prompt": line["prompt"]} for line in data]
    prompts_B = [{"text": line["text_edit"], "prompt": line["prompt_edit"]} for line in data]
    images = [Image.open((DS_PATH / line["file_name"])).convert("RGB").resize((1024, 1024)) for line in data]
    images_np = np.array([np.array(img) for img in images])
    return images_np, prompts_A, prompts_B


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


def sample_renoise(
    pipe,
    prompts,
    latents,
    batch_size,
    num_inference_steps,
    device,
    guidance_scale,
    inversion_max_step,
):
    all_images = np.zeros((len(prompts), 1024, 1024, 3), dtype=np.uint8)
    loop = (
        tqdm(enumerate(range(0, len(prompts), 1)), total=math.ceil(len(prompts) / 1), desc="Sampling")
        if accelerator.is_main_process
        else enumerate(range(0, len(prompts), 1))
    )
    for _, idx_start in loop:
        b_prompt = prompts[idx_start]
        b_latents = latents[idx_start].to(device)
        images = sdxl_sample_from_latent(
            pipe_forward=pipe,
            latent=b_latents,
            prompt_real=b_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            inversion_max_step=inversion_max_step,
            output_type="np",
        ).images
        images = images * 255
        all_images[idx_start] = images.astype(np.uint8)
    return all_images


def sample_ours(
    pipe,
    prompts_original,
    prompts_edit,
    latents,
    batch_size,
    num_inference_steps,
    device,
    guidance_scale_edit,
    guidance_scale_original,
    inversion_max_step,
    attn_idx_to_patch,
):
    all_images_patched = np.zeros((len(prompts_original), 1024, 1024, 3), dtype=np.uint8)
    loop = (
        tqdm(
            enumerate(range(0, len(prompts_original), 1)), total=math.ceil(len(prompts_original) / 1), desc="Sampling"
        )
        if accelerator.is_main_process
        else enumerate(range(0, len(prompts_original), 1))
    )
    logging.info("Patching images...")
    for _, idx_start in loop:
        pipe.clean_cache()
        b_prompt_orig = prompts_original[idx_start]
        b_latents = latents[idx_start].to(device)
        _ = sdxl_sample_from_latent_our(
            pipe_forward=pipe,
            latent=b_latents,
            prompt=b_prompt_orig,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale_edit,
            inversion_max_step=inversion_max_step,
            output_type="np",
            run_with_cache=True,
            batch_num=0,
            attn_idx_to_patch=attn_idx_to_patch,
        )
        b_prompt_edit = prompts_edit[idx_start]
        images = sdxl_sample_from_latent_our(
            pipe_forward=pipe,
            latent=b_latents,
            prompt=b_prompt_edit,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale_original,
            inversion_max_step=inversion_max_step,
            output_type="np",
            run_with_cache=False,
            batch_num=0,
            attn_idx_to_patch=attn_idx_to_patch,
        )
        images = images * 255
        all_images_patched[idx_start] = images.astype(np.uint8)
    return all_images_patched


def reverse_renoise(pipe_rev, prompts, images, num_inversion_steps, num_renoise_steps, inversion_max_step):
    all_latents = np.zeros((len(prompts), 4, 128, 128))
    loop = tqdm(range(len(prompts)), desc="ReNoise inversion") if accelerator.is_main_process else range(len(prompts))
    for img_idx in loop:
        b_prompt = prompts[img_idx]
        b_img = images[img_idx] / 255
        latent = (
            sdxl_invert_to_latent(
                pipe_inversion=pipe_rev,
                image=b_img,
                prompt_real=b_prompt,
                num_inversion_steps=num_inversion_steps,
                num_renoise_steps=num_renoise_steps,
                inversion_max_step=inversion_max_step,
            )
            .cpu()
            .numpy()
        )
        all_latents[img_idx] = latent
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
    image_sim, prompt_A_sim, prompt_B_sim = clip_metrics(
        clip_model,
        images,
        original_images_A_feats,
        device,
        batch_size,
        prompts_A,
        prompts_B,
    )
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
    }


SAVE_DIR = (
    f"results_sdxl/inversion/real/renoise_edit/"
    f"{START_TIME}_"
    f"seed_{SEED}_"
    f"n_inference_steps_{NUM_INFERENCE_STEPS}_"
    f"timestep_start_patching_{TIMESTEP_START_PATCHING}_"
    f"inv_guidance_scale_{INVERSION_GUIDANCE_SCALE}"
    f"renoise_guidance_scale_{RENOISE_GUIDANCE_SCALE}_"
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

    model_type = Model_Type.SDXL
    scheduler_type = Scheduler_Type.DDIM
    pipe_inversion, _ = get_pipes(model_type, scheduler_type, device=accelerator.device)
    pipe_inversion.set_progress_bar_config(disable=True)
    pipe_forward = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        SDXL_MODEL_NAME_OR_PATH,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        token=os.environ.get("HF_TOKEN"),
    )
    pipe_forward.scheduler = DDIMScheduler.from_config(pipe_forward.scheduler.config)
    pipe_forward.set_progress_bar_config(disable=True)
    pipe_forward = pipe_forward.to(accelerator.device)

    original_images_A, prompts_A, prompts_B = prepare_data()

    indices_split = list(range(len(prompts_A)))

    all_latents = []
    with accelerator.split_between_processes(indices_split) as index_split:
        images_A_device = np.array([original_images_A[i] for i in index_split])
        prompts_A_device = [prompts_A[i] for i in index_split]
        prompts_B_device = [prompts_B[i] for i in index_split]

        if accelerator.is_main_process:
            logging.info("A Images Inversion...")

        if accelerator.is_main_process:
            logging.info(
                f"[{accelerator.process_index}/{accelerator.num_processes}] Split:{len(index_split)/len(indices_split):.3f}"
            )

        latents_A_device = reverse_renoise(
            pipe_inversion,
            [p["prompt"] for p in prompts_A_device],
            images=images_A_device,
            num_inversion_steps=NUM_INFERENCE_STEPS,
            num_renoise_steps=NUM_RENOISE_STEPS,
            inversion_max_step=INVERSION_MAX_STEP,
        )
        all_latents.extend(latents_A_device)

    accelerator.wait_for_everyone()
    all_latents = gather_object(all_latents)
    all_latents = np.array(all_latents)

    if accelerator.is_main_process:
        np.save(os.path.join(SAVE_DIR, "Real_latents.npy"), all_latents)
        np.save(os.path.join(SAVE_DIR, "Real_images.npy"), original_images_A)

    if accelerator.is_main_process:
        logging.info("Calculating metrics...")
        ocr_model = get_ocr_easyocr(use_cuda=True)

        clip_model, transform = clip.load("ViT-B/32", device=accelerator.device, jit=False)
        clip_model.eval()
        original_images_A_feats = extract_all_images(
            original_images_A, clip_model, accelerator.device, batch_size=BATCH_SIZE
        )
        original_images_A_metrics = calculate_metrics(
            original_images_A,
            original_images_A_feats,
            original_images_A,
            [p["text"] for p in prompts_A],
            [p["text"] for p in prompts_B],
            [p["prompt"] for p in prompts_A],
            [p["prompt"] for p in prompts_B],
            "cuda",
            BATCH_SIZE,
        )
        original_images_A_df = pd.DataFrame(
            original_images_A_metrics,
        )
        original_images_A_df["Block_patched"] = ["-" for _ in range(len(prompts_A))]
        all_metrics_df = original_images_A_df
        all_metrics_df.to_csv(os.path.join(SAVE_DIR, "metrics.csv"))

    accelerator.wait_for_everyone()

    latents_from_A = torch.from_numpy(all_latents).half()
    all_patched_images = []
    all_ddim_images = []
    with accelerator.split_between_processes(indices_split) as index_split:
        prompts_A_device = [prompts_A[i] for i in index_split]
        prompts_B_device = [prompts_B[i] for i in index_split]
        noises_device = torch.stack([latents_from_A[i] for i in index_split])

        if accelerator.is_main_process:
            logging.info("Editing with ReNoise ...")
        samples_ddim = sample_renoise(
            pipe=pipe_forward,
            prompts=[p["prompt"] for p in prompts_B_device],
            latents=noises_device,
            batch_size=BATCH_SIZE,
            num_inference_steps=NUM_INFERENCE_STEPS,
            device=accelerator.device,
            guidance_scale=RENOISE_GUIDANCE_SCALE,
            inversion_max_step=INVERSION_MAX_STEP,
        )
        all_ddim_images.extend(samples_ddim)
    accelerator.wait_for_everyone()
    all_ddim_images = gather_object(all_ddim_images)
    all_ddim_images = np.array(all_ddim_images)
    if accelerator.is_main_process:
        np.save(os.path.join(SAVE_DIR, "DDIM_edit.npy"), all_ddim_images)
    del samples_ddim

    if accelerator.is_main_process:
        logging.info("Calculating metrics (DDIM)...")
        ddim_edit_metrics = calculate_metrics(
            original_images_A,
            original_images_A_feats,
            all_ddim_images,
            [p["text"] for p in prompts_A],
            [p["text"] for p in prompts_B],
            [p["prompt"] for p in prompts_A],
            [p["prompt"] for p in prompts_B],
            accelerator.device,
            BATCH_SIZE,
        )
        ddim_edit_df = pd.DataFrame(ddim_edit_metrics)
        ddim_edit_df["Block_patched"] = ["ReNoise" for _ in range(len(prompts_A))]
        all_metrics_df = pd.concat([all_metrics_df, ddim_edit_df])
        all_metrics_df.to_csv(os.path.join(SAVE_DIR, "metrics.csv"))
    accelerator.wait_for_everyone()
    accelerator.free_memory()
    del all_ddim_images

    with accelerator.split_between_processes(indices_split) as index_split:
        prompts_A_device = [prompts_A[i] for i in index_split]
        prompts_B_device = [prompts_B[i] for i in index_split]
        noises_device = torch.stack([latents_from_A[i] for i in index_split])

        if accelerator.is_main_process:
            logging.info("Editing with OURS ...")
        images_patched = sample_ours(
            pipe=pipe_forward,
            prompts_original=[p["prompt"] for p in prompts_A_device],
            prompts_edit=[p["prompt"] for p in prompts_B_device],
            latents=noises_device,
            batch_size=BATCH_SIZE,
            num_inference_steps=NUM_INFERENCE_STEPS,
            device=accelerator.device,
            guidance_scale_edit=GUIDANCE_SCALE_CACHE,
            guidance_scale_original=GUIDANCE_SCALE_PATCH,
            inversion_max_step=INVERSION_MAX_STEP,
            attn_idx_to_patch=ATTENTIONS_TO_PATCH,
        )
        all_patched_images.extend(images_patched)
    accelerator.wait_for_everyone()
    all_patched_images = gather_object(all_patched_images)
    all_patched_images = np.array(all_patched_images)
    if accelerator.is_main_process:
        np.save(os.path.join(SAVE_DIR, f"A{set_to_string(ATTENTIONS_TO_PATCH)}.npy"), all_patched_images)
    del images_patched

    if accelerator.is_main_process:
        logging.info("Calculating metrics (patched)...")
        patched_images_metrics = calculate_metrics(
            original_images_A,
            original_images_A_feats,
            all_patched_images,
            [p["text"] for p in prompts_A],
            [p["text"] for p in prompts_B],
            [p["prompt"] for p in prompts_A],
            [p["prompt"] for p in prompts_B],
            accelerator.device,
            BATCH_SIZE,
        )
        patched_images_df = pd.DataFrame(
            patched_images_metrics,
        )
        patched_images_df["Block_patched"] = [f"OURS" for _ in range(len(prompts_A))]
        all_metrics_df = pd.concat([all_metrics_df, patched_images_df])

        all_metrics_df.to_csv(os.path.join(SAVE_DIR, "metrics.csv"))
        logging.info(f"Metrics saved to {os.path.join(SAVE_DIR, 'metrics.csv')}")
        logging.info("Finito!")

    accelerator.wait_for_everyone()
    accelerator.free_memory()
