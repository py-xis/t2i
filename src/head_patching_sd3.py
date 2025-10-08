import copy
import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from diffusers.training_utils import set_seed
from src.prepare_glyph import prepare_prompts_glyph_simple_bench

torch.backends.cuda.matmul.allow_tf32 = True
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True

import logging
import time

import clip
import pandas as pd
from pytorch_msssim import ssim

from diffusers import StableDiffusion3Pipeline
from src.eval.basic_metrics import calculate_mse, calculate_psnr_from_mse
from src.eval.clipscore import clip_metrics, extract_all_images
from src.eval.ocr_eval import get_ocr_easyocr, get_text_easyocr, ocr_metrics
from src.eval.text_distance import get_levenshtein_distances

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
SDXL_MODEL_NAME_OR_PATH = "stabilityai/stable-diffusion-3-medium-diffusers"
SEED = 42
N_SAMPLES_PER_PROMPT = 1
BATCH_SIZE = 10
DEVICE = "cuda"
NUM_INFERENCE_STEPS = 28
GUIDANCE_SCALE = 9.0
TIMESTEP_START_PATCHING = 0
ATTENTIONS_TO_PATCH = [
    10,
]  # NOTE: During patching we cache all activations thus it consumes a lot of memory. Try to patch only limited number of attentions.


def set_to_string(int_set):
    return "+".join(str(num) for num in int_set)


SAVE_DIR = (
    f"results_sd3/glyph_simple/heads/"
    f"model_{SDXL_MODEL_NAME_OR_PATH.split('/')[-1]}_"
    f"seed_{SEED}_"
    f"samples_{N_SAMPLES_PER_PROMPT}_"
    f"batch_{BATCH_SIZE}_"
    f"device_{DEVICE}_"
    f"steps_{NUM_INFERENCE_STEPS}_"
    f"guidance_scale_{GUIDANCE_SCALE}_"
    f"timestep_start_patching_{TIMESTEP_START_PATCHING}_"
    f"attentions_to_patch_{set_to_string(ATTENTIONS_TO_PATCH)}_"
    f"{START_TIME}"
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

pipe = StableDiffusion3Pipeline.from_pretrained(
    SDXL_MODEL_NAME_OR_PATH,
    variant="fp16",
    use_safetensors=True,
    token=os.environ.get("HF_TOKEN"),
    torch_dtype=torch.float32 if DEVICE == "cpu" else torch.float16,
).to(DEVICE)

pipe.set_progress_bar_config(disable=True)

generator = torch.Generator().manual_seed(SEED)

prompts_A, prompts_B = prepare_prompts_glyph_simple_bench(
    n_samples_per_prompt=N_SAMPLES_PER_PROMPT
)
logging.info(f"Number of prompts: {len(prompts_A)}")
ocr_model = get_ocr_easyocr(use_cuda=(DEVICE == "cuda"))

clip_model, transform = clip.load("ViT-B/32", device=DEVICE, jit=False)
clip_model.eval()


noises = torch.randn(
    (N_SAMPLES_PER_PROMPT, 16, 128, 128),
    generator=generator,
    dtype=torch.float32 if DEVICE == "cpu" else torch.float16,
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
    all_images = np.zeros(
        (len(prompts) * N_SAMPLES_PER_PROMPT, 1024, 1024, 3), dtype=np.float32
    )
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
):
    # calculate metrics per sample
    # 1. MSE
    mse = calculate_mse(original_images_A, images)
    # 2.PSNR
    psnr = calculate_psnr_from_mse(mse)
    # 3. SSIM
    ssim_val = ssim(
        torch.from_numpy(original_images_A).permute((0, 3, 1, 2)),
        torch.from_numpy(images).permute((0, 3, 1, 2)),
        data_range=255,
        size_average=False,
    ).numpy()
    # 4. OCR Acc/Prec/Rec
    ocr_texts = [
        get_text_easyocr(ocr_model, images[i].astype(np.uint8)).lower()
        for i in range(images.shape[0])
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


logging.info("Sampling original images A ...")
original_images_A = sample(
    [p["prompt"] for p in prompts_A],
    noises,
    BATCH_SIZE,
    NUM_INFERENCE_STEPS,
    generator,
    DEVICE,
    run_with_cache=False,
)
np.save(os.path.join(SAVE_DIR, "None.npy"), original_images_A.astype(np.int8))


original_images_A_feats = extract_all_images(
    original_images_A, clip_model, DEVICE, batch_size=BATCH_SIZE
)

logging.info("Sampling original images B ...")
original_images_B = sample(
    [p["prompt"] for p in prompts_B],
    noises,
    BATCH_SIZE,
    NUM_INFERENCE_STEPS,
    generator,
    DEVICE,
    run_with_cache=True,
    attn_idx_to_patch=ATTENTIONS_TO_PATCH,
)
np.save(os.path.join(SAVE_DIR, "Model.npy"), original_images_B.astype(np.int8))


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

patched_images = sample(
    [p["prompt"] for p in prompts_A],
    noises,
    BATCH_SIZE,
    NUM_INFERENCE_STEPS,
    generator,
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
    f"T{ATTENTIONS_TO_PATCH[0]}" for _ in range(len(prompts_A))
]
np.save(
    os.path.join(
        SAVE_DIR,
        f"T{ATTENTIONS_TO_PATCH[0]}.npy",
    ),
    patched_images.astype(np.int8),
)

all_metrics_df = pd.concat([all_metrics_df, patched_images_df])


baseline_f1_score = (
    2
    * np.mean(patched_images_metrics["OCR_B_Prec"])
    * np.mean(patched_images_metrics["OCR_B_Rec"])
) / (
    np.mean(patched_images_metrics["OCR_B_Prec"])
    + np.mean(patched_images_metrics["OCR_B_Rec"])
    + 1e-6
)
print(f"Baseline F1 score: {baseline_f1_score}")
curr_f1_score = -float("inf")

active_heads_dict = {attn_idx: set(range(24)) for attn_idx in ATTENTIONS_TO_PATCH}
removed_heads_dict = {attn_idx: set() for attn_idx in ATTENTIONS_TO_PATCH}

while curr_f1_score < baseline_f1_score:
    iter_f1_scores = {attn_idx: [] for attn_idx in ATTENTIONS_TO_PATCH}
    # remove each active head one by one
    for attn_idx, active_heads in active_heads_dict.items():
        for i, active_head_idx in enumerate(active_heads):
            attn_heads_idx_to_patch = copy.deepcopy(active_heads_dict)
            attn_heads_idx_to_patch[attn_idx].remove(active_head_idx)
            patched_images = sample(
                [p["prompt"] for p in prompts_A],
                noises,
                BATCH_SIZE,
                NUM_INFERENCE_STEPS,
                generator,
                DEVICE,
                run_with_cache=False,
                attn_idx_to_patch=ATTENTIONS_TO_PATCH,
                attn_heads_idx_to_patch={
                    ai: list(hi) for ai, hi in attn_heads_idx_to_patch.items()
                },
                timestep_start_patching=TIMESTEP_START_PATCHING,
            )
            patched_images_metrics_active = calculate_metrics(
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
            f1_score = (
                2
                * np.mean(patched_images_metrics_active["OCR_B_Prec"])
                * np.mean(patched_images_metrics_active["OCR_B_Rec"])
            ) / (
                np.mean(patched_images_metrics_active["OCR_B_Prec"])
                + np.mean(patched_images_metrics_active["OCR_B_Rec"])
                + 1e-6
            )
            iter_f1_scores[attn_idx].append(f1_score)
    # find the head with the lowest f1 score
    all_f1_scores = [x for ai in ATTENTIONS_TO_PATCH for x in iter_f1_scores[ai]]
    min_f1_score = min(all_f1_scores)
    min_f1_score_idx = all_f1_scores.index(min_f1_score)
    if min_f1_score_idx < len(active_heads_dict[ATTENTIONS_TO_PATCH[0]]):
        min_f1_score_head = list(active_heads_dict[ATTENTIONS_TO_PATCH[0]])[
            min_f1_score_idx
        ]
        active_heads_dict[ATTENTIONS_TO_PATCH[0]].remove(min_f1_score_head)
        removed_heads_dict[ATTENTIONS_TO_PATCH[0]].add(min_f1_score_head)
        print(
            f"Removed head: T{ATTENTIONS_TO_PATCH[0]}H{min_f1_score_head}, F1 score: {min_f1_score}"
        )
    else:
        min_f1_score_head = list(active_heads_dict[ATTENTIONS_TO_PATCH[1]])[
            min_f1_score_idx - len(active_heads_dict[ATTENTIONS_TO_PATCH[0]])
        ]
        active_heads_dict[ATTENTIONS_TO_PATCH[1]].remove(min_f1_score_head)
        removed_heads_dict[ATTENTIONS_TO_PATCH[1]].add(min_f1_score_head)
        print(
            f"Removed head: T{ATTENTIONS_TO_PATCH[1]}H{min_f1_score_head}, F1 score: {min_f1_score}"
        )

    # patch with only removed heads
    removed_patched_images = sample(
        [p["prompt"] for p in prompts_A],
        noises,
        BATCH_SIZE,
        NUM_INFERENCE_STEPS,
        generator,
        DEVICE,
        run_with_cache=False,
        attn_idx_to_patch=ATTENTIONS_TO_PATCH,
        attn_heads_idx_to_patch={ai: list(hi) for ai, hi in removed_heads_dict.items()},
        timestep_start_patching=TIMESTEP_START_PATCHING,
    )
    np.save(
        os.path.join(
            SAVE_DIR,
            f"T{ATTENTIONS_TO_PATCH[0]}H{set_to_string(removed_heads_dict[ATTENTIONS_TO_PATCH[0]])}.npy",
        ),
        removed_patched_images.astype(np.int8),
    )

    removed_patched_images_metrics = calculate_metrics(
        original_images_A,
        original_images_A_feats,
        removed_patched_images,
        [p["text"] for p in prompts_A],
        [p["text"] for p in prompts_B],
        [p["prompt"] for p in prompts_A],
        [p["prompt"] for p in prompts_B],
        DEVICE,
        BATCH_SIZE,
    )

    patched_images_df = pd.DataFrame(
        removed_patched_images_metrics,
    )
    patched_images_df["Block_patched"] = [
        f"T{ATTENTIONS_TO_PATCH[0]}H{set_to_string(removed_heads_dict[ATTENTIONS_TO_PATCH[0]])}"
        for _ in range(len(prompts_A))
    ]

    all_metrics_df = pd.concat([all_metrics_df, patched_images_df])

    curr_f1_score = (
        2
        * np.mean(removed_patched_images_metrics["OCR_B_Prec"])
        * np.mean(removed_patched_images_metrics["OCR_B_Rec"])
    ) / (
        np.mean(removed_patched_images_metrics["OCR_B_Prec"])
        + np.mean(removed_patched_images_metrics["OCR_B_Rec"])
        + 1e-6
    )
    print(
        f"Current F1 for patched T{ATTENTIONS_TO_PATCH[0]}H{set_to_string(removed_heads_dict[ATTENTIONS_TO_PATCH[0]])}: {curr_f1_score}"
    )

# Save DataFrame to CSV file
all_metrics_df.to_csv(os.path.join(SAVE_DIR, "metrics.csv"))

logging.info("Finito!")
