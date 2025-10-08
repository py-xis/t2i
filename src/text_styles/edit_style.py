"""Text edition with SD3 model on CreativeBench - experiments with text length."""

import argparse
import logging
import os
import time
from typing import Any

import clip
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from pytorch_msssim import ssim
from tqdm import tqdm

from diffusers import StableDiffusion3Pipeline
from diffusers.training_utils import set_seed
from src.eval.clipscore import clip_metrics_style
from src.eval.ocr_eval import get_ocr_easyocr, get_text_easyocr, ocr_metrics
from src.eval.text_distance import get_levenshtein_distances
from src.prepare_glyph import prepare_prompts_glyph_creative_text_len

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
TIMESTEP_START_PATCHING = 0

edit_pairs = [
    ("handwritten", "neon"),
    # ("neon", "handwritten"),
    # ("handwritten", "graffiti"),
    ("graffiti", "handwritten"),
    # ("handwritten", "comic"),
    ("comic", "handwritten"),
    ("neon", "comic"),
    # ("comic", "neon"),
]
text_styles = ["handwritten", "neon", "graffiti", "comic"]
words = ["hello world!", "happy new year", "i love you", "Welcome to Asia"]
templates = [
    "Road sign with a {style} text saying {sentence}, text {sentence} with a {style} style",
    "Notebook page with a {style} text saying {sentence}, text {sentence} with a {style} style",
    "Street wall covered in {style} text saying {sentence}, text {sentence} with a {style} style",
    "Bus stop advertisement with {style} text saying {sentence}, text {sentence} with a {style} style",
    "Urban skatepark ramp with {style} text saying {sentence}, text {sentence} with a {style} style",
]


def set_to_string(int_set: list[int]) -> str:
    return "A" + "_A".join(str(num) for num in int_set)


def get_save_dir(attn_idx: list[int]) -> str:
    save_dir = (
        f"results_sd3/text_style_more_layers/attn_{set_to_string(attn_idx)}/"
        f"{START_TIME}_"
        f"seed_{SEED}_"
        f"n_inference_steps_{NUM_INFERENCE_STEPS}_"
        f"guidance_scale_{GUIDANCE_SCALE}_"
        f"timestep_start_patching_{TIMESTEP_START_PATCHING}_"
        f"attentions_to_patch_{set_to_string(attn_idx)}"
    )
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def get_model() -> StableDiffusion3Pipeline:
    pipe = StableDiffusion3Pipeline.from_pretrained(
        SDXL_MODEL_NAME_OR_PATH,
        variant="fp16",
        use_safetensors=True,
        token=os.environ.get("HF_TOKEN"),
        torch_dtype=torch.float16,
    )
    pipe.set_progress_bar_config(disable=True)
    return pipe


def sample_batched(
    pipe: StableDiffusion3Pipeline,
    prompts: list[str],
    noise: torch.Tensor,
    batch_size: int,
    num_inference_steps: int,
    generator: torch.Generator,
    device: str,
    run_with_cache: bool,
    attn_idx_to_patch: list[int] | None = None,
    attn_heads_idx_to_patch: list[int] | None = None,
    timestep_start_patching: int = 0,
) -> np.ndarray:
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


def prepare_input():
    prompts_A, prompts_B = [], []

    for style_A, style_B in edit_pairs:
        for template in templates:
            for word in words:
                prompts_A.append(
                    {
                        "prompt": template.format(style=style_A, sentence=word),
                        "style": f"text in style {style_A}",
                        "text": f'text "{word}"',
                    }
                )
                prompts_B.append(
                    {
                        "prompt": template.format(style=style_B, sentence=word),
                        "style": f"text in style {style_B}",
                        "text": f'text "{word}"',
                    }
                )

    prompts_A = [p_A for p_A in prompts_A for _ in range(N_SAMPLES_PER_PROMPT)]
    prompts_B = [p_B for p_B in prompts_B for _ in range(N_SAMPLES_PER_PROMPT)]
    prompts_AB = None
    templates_A = None
    templates_B = None

    logging.info(f"Number of prompts: {len(prompts_A)}")

    noises = torch.randn(
        (N_SAMPLES_PER_PROMPT, 16, 128, 128),
        generator=torch.Generator().manual_seed(SEED),
        dtype=torch.float16,
    )
    noises = noises.repeat(len(prompts_A) // N_SAMPLES_PER_PROMPT, 1, 1, 1)
    logging.info(f"Noises shape: {noises.shape}")
    return prompts_A, prompts_B, prompts_AB, templates_A, templates_B, noises


def calculate_metrics(
    images: np.ndarray,
    text_styles_A: list[str],
    text_styles_B: list[str],
    text_contents_A: list[str],
    text_contents_B: list[str],
    prompts_A: list[str],
    prompts_B: list[str],
    device: str | torch.device,
    batch_size: int,
    clip_model: Any = None,
) -> dict[str, Any]:
    prompt_A_sim, prompt_B_sim, prompt_A_style_sim, prompt_B_style_sim, prompt_A_content_sim, prompt_B_content_sim = clip_metrics_style(
        clip_model, images, device, batch_size, prompts_A, prompts_B, text_styles_A, text_styles_B, text_contents_A, text_contents_B
    )

    return {
        "CLIPScore_prompt_A": prompt_A_sim,
        "CLIPScore_prompt_B": prompt_B_sim,
        "CLIPScore_prompt_A_style": prompt_A_style_sim,
        "CLIPScore_prompt_B_style": prompt_B_style_sim,
        "CLIPScore_prompt_A_content": prompt_A_content_sim,
        "CLIPScore_prompt_B_content": prompt_B_content_sim,
        "Text_contents_A": text_contents_A,
        "Text_contents_B": text_contents_B,
        "Prompts_A": prompts_A,
        "Prompts_B": prompts_B,
        "Text_styles_A": text_styles_A,
        "Text_styles_B": text_styles_B,
    }


def main(args: argparse.Namespace):
    set_seed(SEED)
    save_dir = get_save_dir(args.attn_to_patch)
    logging.info(f"Seed: {SEED}")
    logging.info(f"Num inference steps: {NUM_INFERENCE_STEPS}")
    logging.info(f"Batch size: {BATCH_SIZE}")
    logging.info(f"Save dir: {save_dir}")

    prompts_A, prompts_B, prompts_AB, templates_A, templates_B, noises = prepare_input()
    prompts_indices = list(range(len(prompts_A)))

    distributed_state = Accelerator()
    pipe = get_model()
    pipe = pipe.to(distributed_state.device)

    all_original_images_A = []
    all_original_images_B = []
    all_patched_images = []
    with distributed_state.split_between_processes(prompts_indices) as device_indices:
        p_A = [prompts_A[i] for i in device_indices]
        p_B = [prompts_B[i] for i in device_indices]
        n = torch.stack([noises[i] for i in device_indices])

        if distributed_state.is_main_process:
            logging.info("Sampling original images (prompt A) ...")
        original_images_A = sample_batched(
            pipe=pipe,
            prompts=[p["prompt"] for p in p_A],
            noise=n,
            batch_size=BATCH_SIZE,
            num_inference_steps=NUM_INFERENCE_STEPS,
            generator=torch.Generator().manual_seed(SEED),
            device=distributed_state.device,
            run_with_cache=False,
        )

        if distributed_state.is_main_process:
            logging.info("Sampling original images (prompt B) ...")
        original_images_B = sample_batched(
            pipe=pipe,
            prompts=[p["prompt"] for p in p_B],
            noise=n,
            batch_size=BATCH_SIZE,
            num_inference_steps=NUM_INFERENCE_STEPS,
            generator=torch.Generator().manual_seed(SEED),
            device=distributed_state.device,
            run_with_cache=True,
            attn_idx_to_patch=args.attn_to_patch,
            timestep_start_patching=TIMESTEP_START_PATCHING,
        )

        if distributed_state.is_main_process:
            logging.info(f"Sampling with patching (ours, {set_to_string(args.attn_to_patch)}) ...")
        patched_images = sample_batched(
            pipe=pipe,
            prompts=[p["prompt"] for p in p_A],
            noise=n,
            batch_size=BATCH_SIZE,
            num_inference_steps=NUM_INFERENCE_STEPS,
            generator=torch.Generator().manual_seed(SEED),
            device=distributed_state.device,
            run_with_cache=False,
            attn_idx_to_patch=args.attn_to_patch,
            timestep_start_patching=TIMESTEP_START_PATCHING,
        )

        all_original_images_A.extend(original_images_A)
        all_original_images_B.extend(original_images_B)
        all_patched_images.extend(patched_images)

    distributed_state.wait_for_everyone()
    all_original_images_A = gather_object(all_original_images_A)
    all_original_images_A = np.array(all_original_images_A)
    all_original_images_B = gather_object(all_original_images_B)
    all_original_images_B = np.array(all_original_images_B)
    all_patched_images = gather_object(all_patched_images)
    all_patched_images = np.array(all_patched_images)

    if distributed_state.is_main_process:
        np.save(os.path.join(save_dir, "original_images_A.npy"), all_original_images_A)
        np.save(os.path.join(save_dir, "original_images_B.npy"), all_original_images_B)
        np.save(os.path.join(save_dir, "patched_images.npy"), all_patched_images)
        logging.info(f"Saved original images (A) shape: {all_original_images_A.shape}")
        logging.info(f"Saved original images (B) shape: {all_original_images_B.shape}")
        logging.info(f"Saved patched images shape: {all_patched_images.shape}")

    if distributed_state.is_main_process:
        logging.info("Calculating metrics ...")
        clip_model, transform = clip.load("ViT-B/32", device=distributed_state.device, jit=False)
        clip_model.eval()
        logging.info("Calculating metrics for original images (A) ...")
        original_images_A_metrics = calculate_metrics(
            images=all_original_images_A,
            text_styles_A=[p["style"] for p in prompts_A],
            text_styles_B=[p["style"] for p in prompts_B],
            text_contents_A=[p["text"] for p in prompts_A],
            text_contents_B=[p["text"] for p in prompts_B],
            prompts_A=[p["prompt"] for p in prompts_A],
            prompts_B=[p["prompt"] for p in prompts_B],
            device=distributed_state.device,
            batch_size=BATCH_SIZE,
            clip_model=clip_model,
        )
        original_images_A_df = pd.DataFrame(original_images_A_metrics)
        original_images_A_df["Block_patched"] = ["A" for _ in range(len(prompts_A))]
        all_metrics_df = original_images_A_df

        logging.info("Calculating metrics for original images (B) ...")
        original_images_B_metrics = calculate_metrics(
            images=all_original_images_B,
            text_styles_A=[p["style"] for p in prompts_A],
            text_styles_B=[p["style"] for p in prompts_B],
            text_contents_A=[p["text"] for p in prompts_A],
            text_contents_B=[p["text"] for p in prompts_B],
            prompts_A=[p["prompt"] for p in prompts_A],
            prompts_B=[p["prompt"] for p in prompts_B],
            device=distributed_state.device,
            batch_size=BATCH_SIZE,
            clip_model=clip_model,
        )
        original_images_B_df = pd.DataFrame(original_images_B_metrics)
        original_images_B_df["Block_patched"] = ["B" for _ in range(len(prompts_B))]
        all_metrics_df = pd.concat([all_metrics_df, original_images_B_df])

        logging.info("Calculating metrics for patched images (ours) ...")
        patched_images_metrics = calculate_metrics(
            images=all_patched_images,
            text_styles_A=[p["style"] for p in prompts_A],
            text_styles_B=[p["style"] for p in prompts_B],
            text_contents_A=[p["text"] for p in prompts_A],
            text_contents_B=[p["text"] for p in prompts_B],
            prompts_A=[p["prompt"] for p in prompts_A],
            prompts_B=[p["prompt"] for p in prompts_B],
            device=distributed_state.device,
            batch_size=BATCH_SIZE,
            clip_model=clip_model,
        )
        patched_images_df = pd.DataFrame(patched_images_metrics)
        patched_images_df["Block_patched"] = [f"OURS" for _ in range(len(prompts_A))]
        all_metrics_df = pd.concat([all_metrics_df, patched_images_df])
        all_metrics_df.to_csv(os.path.join(save_dir, "metrics.csv"))
        logging.info("Finito!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attn_to_patch",
        nargs="+",
        type=int,
        help="List of indexes representing attention layers to patch",
        default=[10],
    )
    args = parser.parse_args()
    main(args=args)
