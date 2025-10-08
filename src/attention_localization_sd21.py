"""Localization of cross-attention layers in SD 2.1 model responsible for text generation."""

import logging
import os
import sys
import time

import clip
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

"""
Make this script runnable when invoked directly (e.g., on Kaggle):
- Add the repo root to sys.path so `import src.*` works.
- Add local diffusers src to sys.path if present (so our patched pipelines are used).
"""
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
DIFFUSERS_SRC = os.path.join(REPO_ROOT, "diffusers", "src")
for _p in (REPO_ROOT, DIFFUSERS_SRC):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from diffusers import StableDiffusionPipeline
from diffusers.training_utils import set_seed
from src.eval.basic_metrics import calculate_mse, calculate_psnr_from_mse
from src.eval.clipscore import clip_metrics, extract_all_images
from src.eval.ocr_eval import get_ocr_easyocr, get_text_easyocr, ocr_metrics
from src.eval.text_distance import get_levenshtein_distances
from src.prepare_glyph import (
    prepare_prompts_glyph_simple_bench_top_100,
    prepare_prompts_glyph_creative_bench_top_100,
)

# Perf flags
torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch._inductor.config.conv_1x1_as_mm = True
    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.epilogue_fusion = False
    torch._inductor.config.coordinate_descent_check_all_directions = True
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

START_TIME = time.strftime("%Y%m%d_%H%M%S")
SD21_MODEL_NAME_OR_PATH = "stabilityai/stable-diffusion-2-1"
SEED = 42
N_SAMPLES_PER_PROMPT = 1
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5


def set_to_string(int_set):
    return "_A".join(str(num) for num in int_set)


SAVE_DIR = (
    f"results_sd21/glyph_val/loc/"
    f"{START_TIME}_"
    f"seed_{SEED}_"
    f"n_samples_per_prompt_{N_SAMPLES_PER_PROMPT}_"
    f"n_inference_steps_{NUM_INFERENCE_STEPS}_"
    f"guidance_scale_{GUIDANCE_SCALE}"
)

os.makedirs(SAVE_DIR, exist_ok=True)

logging.info(f"Seed: {SEED}")
logging.info(f"Device: {DEVICE}")
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

dtype = torch.float32 if DEVICE == "cpu" else torch.float16
pipe = StableDiffusionPipeline.from_pretrained(
    SD21_MODEL_NAME_OR_PATH,
    variant="fp16" if dtype == torch.float16 else None,
    use_safetensors=True,
    token=os.environ.get("HF_TOKEN"),
    torch_dtype=dtype,
).to(DEVICE)

pipe.set_progress_bar_config(disable=True)

prompts_A, prompts_B = prepare_prompts_glyph_simple_bench_top_100(
    n_samples_per_prompt=N_SAMPLES_PER_PROMPT
)
creative_prompts = prepare_prompts_glyph_creative_bench_top_100(
    n_samples_per_prompt=N_SAMPLES_PER_PROMPT
)
prompts_A.extend(creative_prompts[0])
prompts_B.extend(creative_prompts[1])
logging.info(f"Number of prompts: {len(prompts_A)}")

# SD2.1 default 512x512 latents
noises = torch.randn(
    (N_SAMPLES_PER_PROMPT, 4, 64, 64),
    generator=torch.Generator().manual_seed(SEED),
    dtype=dtype,
)
noises = noises.repeat(len(prompts_A) // N_SAMPLES_PER_PROMPT, 1, 1, 1)

ocr_model = get_ocr_easyocr(use_cuda=(DEVICE == "cuda"))

clip_model, transform = clip.load("ViT-B/32", device=DEVICE, jit=False)
clip_model.eval()


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
    all_images = np.zeros((len(prompts), 512, 512, 3), dtype=np.uint8)
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
                device_map="balanced"  # Automatically distribute across available GPUs
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
):
    mse = calculate_mse(original_images_A, images)
    psnr = calculate_psnr_from_mse(mse)
    ocr_texts = [get_text_easyocr(ocr_model, images[i]).lower() for i in range(images.shape[0])]
    ocr_pr_A, ocr_rec_A, ocr_acc_A = ocr_metrics(ocr_texts, texts_A)
    ocr_pr_B, ocr_rec_B, ocr_acc_B = ocr_metrics(ocr_texts, texts_B)
    image_sim, prompt_A_sim, prompt_B_sim = clip_metrics(
        clip_model,
        images,
        original_images_A_feats,
        device,
        batch_size,
        prompts_A,
        prompts_B,
    )
    from src.eval.text_distance import get_levenshtein_distances
    leve_A = get_levenshtein_distances(ocr_texts, texts_A)
    leve_B = get_levenshtein_distances(ocr_texts, texts_B)

    return {
        "MSE": mse,
        "PSNR": psnr,
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


logging.info("Sampling original images A ...")
original_images_A = sample(
    [p["prompt"] for p in prompts_A],
    noises,
    BATCH_SIZE,
    NUM_INFERENCE_STEPS,
    torch.Generator().manual_seed(SEED),
    DEVICE,
    run_with_cache=False,
)
np.save(os.path.join(SAVE_DIR, "None.npy"), original_images_A)

original_images_A_feats = extract_all_images(
    original_images_A, clip_model, DEVICE, batch_size=BATCH_SIZE
)

logging.info("Sampling original images B ...")
original_images_B = sample(
    [p["prompt"] for p in prompts_B],
    noises,
    BATCH_SIZE,
    NUM_INFERENCE_STEPS,
    torch.Generator().manual_seed(SEED),
    DEVICE,
    run_with_cache=True,
)
np.save(os.path.join(SAVE_DIR, "Model.npy"), original_images_B)

original_images_A_metrics = calculate_metrics(
    original_images_A,
    original_images_A_feats,
    original_images_A,
    [p["text"] for p in prompts_A],
    [p["text"] for p in prompts_B],
    [p["prompt"] for p in prompts_A],
    [p["prompt"] for p in prompts_B],
    DEVICE,
    BATCH_SIZE,
)

original_images_B_metrics = calculate_metrics(
    original_images_A,
    original_images_A_feats,
    original_images_B,
    [p["text"] for p in prompts_A],
    [p["text"] for p in prompts_B],
    [p["prompt"] for p in prompts_A],
    [p["prompt"] for p in prompts_B],
    DEVICE,
    BATCH_SIZE,
)

original_images_A_df = pd.DataFrame(
    original_images_A_metrics,
)
original_images_A_df["Block_patched"] = ["-" for _ in range(len(prompts_A))]
original_images_B_df = pd.DataFrame(
    original_images_B_metrics,
    index=["Model" for _ in range(len(prompts_A))],
)
original_images_B_df["Block_patched"] = ["Model" for _ in range(len(prompts_B))]

all_metrics_df = pd.concat([original_images_A_df, original_images_B_df])

# Try to infer the number of cross-attention blocks by counting Attention processors with cross-attn
num_blocks = 0
for name, module in pipe.unet.named_modules():
    if hasattr(module, "__class__") and module.__class__.__name__ in ("BasicTransformerBlock",):
        # Basic heuristic: SD v1/v2 CrossAttn blocks include BasicTransformerBlock with context
        num_blocks += 1

logging.info(f"Estimated cross-attention blocks: {num_blocks}")

for attn_idx in tqdm(range(num_blocks)):
    patched_images = sample(
        [p["prompt"] for p in prompts_A],
        noises,
        BATCH_SIZE,
        NUM_INFERENCE_STEPS,
        torch.Generator().manual_seed(SEED),
        DEVICE,
        run_with_cache=False,
        attn_idx_to_patch=attn_idx,
    )
    patched_images_metrics = calculate_metrics(
        original_images_A,
        original_images_A_feats,
        patched_images,
        [p["text"] for p in prompts_A],
        [p["text"] for p in prompts_B],
        [p["prompt"] for p in prompts_A],
        [p["prompt"] for p in prompts_B],
        DEVICE,
        BATCH_SIZE,
    )

    patched_images_df = pd.DataFrame(
        patched_images_metrics,
    )
    patched_images_df["Block_patched"] = [
        f"A{set_to_string([attn_idx])}" for _ in range(len(prompts_A))
    ]

    all_metrics_df = pd.concat([all_metrics_df, patched_images_df])

    np.save(
        os.path.join(
            SAVE_DIR,
            f"A{set_to_string([attn_idx])}.npy",
        ),
        patched_images,
    )

# Save DataFrame to CSV file
all_metrics_df.to_csv(os.path.join(SAVE_DIR, "metrics.csv"))

logging.info("Finito!")
