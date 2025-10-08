"""Text edition with SD-XL model on SimpleBench."""

import logging
import os
import time

import clip
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_msssim import ssim
from tqdm import tqdm

from diffusers import StableDiffusionXLPipeline
from diffusers.training_utils import set_seed
from src.eval.basic_metrics import calculate_mse, calculate_psnr_from_mse
from src.eval.clipscore import clip_metrics, extract_all_images
from src.eval.ocr_eval import get_ocr_easyocr, get_text_easyocr, ocr_metrics
from src.eval.text_detection import remove_text_boxes, setup_text_detection_model
from src.eval.text_distance import get_levenshtein_distances
from src.prepare_glyph import prepare_prompts_glyph_creative_bench

torch.backends.cuda.matmul.allow_tf32 = True
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

START_TIME = time.strftime("%Y%m%d_%H%M%S")
SDXL_MODEL_NAME_OR_PATH = "stabilityai/stable-diffusion-xl-base-1.0"
SEED = 42
N_SAMPLES_PER_PROMPT = 4
BATCH_SIZE = 8
DEVICE = "cuda"
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 5.0
TIMESTEP_START_PATCHING = 4
ATTENTIONS_TO_PATCH = [
    55,
    56,
]


def set_to_string(int_set):
    return "_A".join(str(num) for num in int_set)


SAVE_DIR = (
    f"results_sdxl/glyph_creative/edit/"
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

pipe = StableDiffusionXLPipeline.from_pretrained(
    SDXL_MODEL_NAME_OR_PATH,
    variant="fp16",
    use_safetensors=True,
    token=os.environ.get("HF_TOKEN"),
    torch_dtype=torch.float32 if DEVICE == "cpu" else torch.float16,
).to(DEVICE)

pipe.set_progress_bar_config(disable=True)

prompts_A, prompts_B = prepare_prompts_glyph_creative_bench(
    n_samples_per_prompt=N_SAMPLES_PER_PROMPT
)
logging.info(f"Number of prompts: {len(prompts_A)}")

noises = torch.randn(
    (N_SAMPLES_PER_PROMPT, 4, 128, 128),
    generator=torch.Generator().manual_seed(SEED),
    dtype=torch.float32 if DEVICE == "cpu" else torch.float16,
)
noises = noises.repeat(len(prompts_A) // N_SAMPLES_PER_PROMPT, 1, 1, 1)

ocr_model = get_ocr_easyocr(use_cuda=(DEVICE == "cuda"))

clip_model, transform = clip.load("ViT-B/32", device=DEVICE, jit=False)
clip_model.eval()

text_detection_model = setup_text_detection_model(
    "/storage2/bartosz/code/t2i2/DB_IC15_resnet50.onnx"
)


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
    ocr_texts = [
        get_text_easyocr(ocr_model, images[i]).lower() for i in range(images.shape[0])
    ]
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
    # calculate visual metrics with removed text
    images_no_text = remove_text_boxes(images, text_detection_model)
    original_images_A_no_text = remove_text_boxes(
        original_images_A, text_detection_model
    )
    # MSE
    mse_no_text = calculate_mse(original_images_A_no_text, images_no_text)
    # PSNR
    psnr_no_text = calculate_psnr_from_mse(mse_no_text)
    # SSIM
    ssim_val_no_text = ssim(
        torch.from_numpy(original_images_A_no_text.astype(np.float32)).permute(
            (0, 3, 1, 2)
        ),
        torch.from_numpy(images_no_text.astype(np.float32)).permute((0, 3, 1, 2)),
        data_range=255,
        size_average=False,
    ).numpy()

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
        "MSE_no_text": mse_no_text,
        "PSNR_no_text": psnr_no_text,
        "SSIM_no_text": ssim_val_no_text,
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
    attn_idx_to_patch=ATTENTIONS_TO_PATCH,
    timestep_start_patching=TIMESTEP_START_PATCHING,
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

original_images_A_df = pd.DataFrame(
    original_images_A_metrics,
)
original_images_A_df["Block_patched"] = ["-" for _ in range(len(prompts_A))]
all_metrics_df = original_images_A_df

patched_images = sample(
    [p["prompt"] for p in prompts_A],
    noises,
    BATCH_SIZE,
    NUM_INFERENCE_STEPS,
    torch.Generator().manual_seed(SEED),
    DEVICE,
    run_with_cache=False,
    attn_idx_to_patch=ATTENTIONS_TO_PATCH,
    timestep_start_patching=TIMESTEP_START_PATCHING,
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
    f"A{set_to_string(ATTENTIONS_TO_PATCH)}" for _ in range(len(prompts_A))
]

all_metrics_df = pd.concat([all_metrics_df, patched_images_df])

np.save(
    os.path.join(
        SAVE_DIR,
        f"A{set_to_string(ATTENTIONS_TO_PATCH)}.npy",
    ),
    patched_images,
)

# Save DataFrame to CSV file
all_metrics_df.to_csv(os.path.join(SAVE_DIR, "metrics.csv"))

logging.info("Finito!")
