"""Text edition with SD3 model on SimpleBench."""

import logging
import os
import time
from datetime import timedelta

import clip
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, gather_object
from PIL import Image
from pytorch_msssim import ssim
from tqdm import tqdm

from diffusers import StableDiffusionXLPipeline
from diffusers.training_utils import set_seed
from src.eval.basic_metrics import calculate_mse, calculate_psnr_from_mse
from src.eval.clipscore import clip_metrics, extract_all_images
from src.eval.ocr_eval import get_ocr_easyocr, get_text_easyocr, ocr_metrics
from src.eval.text_distance import get_levenshtein_distances
from src.prepare_glyph import prepare_toxic_bench

torch.backends.cuda.matmul.allow_tf32 = True
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

START_TIME = time.strftime("%Y%m%d_%H%M%S")
SDXL_MODEL_NAME_OR_PATH = "stabilityai/stable-diffusion-xl-base-1.0"
SEED = 42
N_SAMPLES_PER_PROMPT = 4
BATCH_SIZE = 20
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 5.0
BENCHMARK = "glyph_toxic"


def set_to_string(int_set):
    return "A" + "_A".join(str(num) for num in int_set)


SAVE_DIR = (
    f"results_sdxl/{BENCHMARK}/NEG_PROMPT/"
    f"{START_TIME}_"
    f"seed_{SEED}_"
    f"n_samples_per_prompt_{N_SAMPLES_PER_PROMPT}_"
    f"n_inference_steps_{NUM_INFERENCE_STEPS}_"
    f"guidance_scale_{GUIDANCE_SCALE}"
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
pipe = StableDiffusionXLPipeline.from_pretrained(
    SDXL_MODEL_NAME_OR_PATH,
    variant="fp16",
    use_safetensors=True,
    token=os.environ.get("HF_TOKEN"),
    torch_dtype=torch.float16,
)
pipe.set_progress_bar_config(disable=True)
prompts_A, prompts_B = prepare_toxic_bench(n_samples_per_prompt=N_SAMPLES_PER_PROMPT)
prompts_neg = [f'text "{p["text"]}"' for p in prompts_A]

logging.info(f"Number of prompts: {len(prompts_A)}")
noises = torch.randn(
    (N_SAMPLES_PER_PROMPT, 4, 128, 128),
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
    negative_prompts=None,
):
    all_images = np.zeros((len(prompts), 1024, 1024, 3), dtype=np.uint8)
    with tqdm(total=len(prompts)) as pbar:
        for batch_num, batch_start in enumerate(range(0, len(prompts), batch_size)):
            prompt = prompts[batch_start : batch_start + batch_size]
            latent = noise[batch_start : batch_start + batch_size].to(device)
            neg_prompt = (
                negative_prompts[batch_start : batch_start + batch_size] if negative_prompts is not None else None
            )
            images = pipe(
                prompt=prompt,
                negative_prompt=neg_prompt,
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
        clip_model, images, original_images_A_feats, device, batch_size, prompts_A, prompts_B, prompts_AB
    )
    if prompts_AB is not None:
        image_sim, prompt_A_sim, prompt_B_sim, _ = clip_metrics_results
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
    }


prompts_indices = list(range(len(prompts_A)))
kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=1800 * 6))
accelerator = Accelerator(kwargs_handlers=[kwargs])
pipe = pipe.to(accelerator.device)
all_original_images_A = []
all_neg_images = []

with accelerator.split_between_processes(prompts_indices) as device_indices:
    p_A = [prompts_A[i] for i in device_indices]
    p_neg = [prompts_neg[i] for i in device_indices] if prompts_neg is not None else None
    n = torch.stack([noises[i] for i in device_indices])

    original_images_A = sample(
        [p["prompt"] for p in p_A],
        n,
        BATCH_SIZE,
        NUM_INFERENCE_STEPS,
        torch.Generator().manual_seed(SEED),
        accelerator.device,
        run_with_cache=False,
    )
    all_original_images_A.extend(original_images_A)
accelerator.wait_for_everyone()
all_original_images_A = gather_object(all_original_images_A)
all_original_images_A = np.array(all_original_images_A)
print(all_original_images_A.shape)
if accelerator.is_main_process:
    logging.info("Calculating metrics ...")
    ocr_model = get_ocr_easyocr(use_cuda=True)

    clip_model, transform = clip.load("ViT-B/32", device=accelerator.device, jit=False)
    clip_model.eval()
    np.save(os.path.join(SAVE_DIR, "None.npy"), all_original_images_A)

    original_images_A_feats = extract_all_images(
        all_original_images_A,
        clip_model,
        accelerator.device,
        batch_size=BATCH_SIZE,
    )

    original_images_A_metrics = calculate_metrics(
        all_original_images_A,
        original_images_A_feats,
        all_original_images_A,
        [p["text"] for p in prompts_A],
        [p["text"] for p in prompts_B],
        [p["prompt"] for p in prompts_A],
        [p["prompt"] for p in prompts_B],
        accelerator.device,
        BATCH_SIZE,
    )

    original_images_A_df = pd.DataFrame(
        original_images_A_metrics,
    )
    original_images_A_df["Block_patched"] = ["-" for _ in range(len(prompts_A))]
    all_metrics_df = original_images_A_df
    all_metrics_df.to_csv(os.path.join(SAVE_DIR, "metrics.csv"))

    np.save(
        os.path.join(
            SAVE_DIR,
            f"neg.npy",
        ),
        all_neg_images,
    )
accelerator.wait_for_everyone()
del original_images_A

with accelerator.split_between_processes(prompts_indices) as device_indices:
    p_A = [prompts_A[i] for i in device_indices]
    p_neg = [prompts_neg[i] for i in device_indices] if prompts_neg is not None else None
    n = torch.stack([noises[i] for i in device_indices])

    neg_B = sample(
        [p["prompt"] for p in p_A],
        n,
        BATCH_SIZE,
        NUM_INFERENCE_STEPS,
        torch.Generator().manual_seed(SEED),
        accelerator.device,
        run_with_cache=False,
        negative_prompts=p_neg,
    )
    all_neg_images.extend(neg_B)
accelerator.wait_for_everyone()

all_neg_images = gather_object(all_neg_images)
all_neg_images = np.array(all_neg_images)
print(all_neg_images.shape)

if accelerator.is_main_process:
    logging.info("Calculating metrics ...")
    neg_images_metrics = calculate_metrics(
        all_original_images_A,
        original_images_A_feats,
        all_neg_images,
        [p["text"] for p in prompts_A],
        [p["text"] for p in prompts_B],
        [p["prompt"] for p in prompts_A],
        [p["prompt"] for p in prompts_B],
        accelerator.device,
        BATCH_SIZE,
    )

    neg_images_df = pd.DataFrame(
        neg_images_metrics,
    )
    neg_images_df["Block_patched"] = [f"NEG" for _ in range(len(prompts_A))]

    all_metrics_df = pd.concat([all_metrics_df, neg_images_df])

    np.save(
        os.path.join(
            SAVE_DIR,
            f"neg.npy",
        ),
        all_neg_images,
    )

    all_metrics_df.to_csv(os.path.join(SAVE_DIR, "metrics.csv"))
    logging.info("Finito!")
