import csv
import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from diffusers import StableDiffusionXLPipeline
from diffusers.training_utils import set_seed
from src.eval.ocr import get_idefics2, get_text_idefics2
from src.eval.text_distance import get_levenshtein_distances, plot_hist_levenshtein

SDXL_MODEL_NAME_OR_PATH = "stabilityai/stable-diffusion-xl-base-1.0"
SEED = 42
# N_PROMPTS = 15
N_SAMPLES_PER_PROMPT = 1
N_STEPS = 27
BATCH_SIZE = 6
DEVICE = "cuda"
TEXTS = ["diffusion", "text to image", "ICLR", "deep learning", "love"]

prompt_A_template = "a photo of a bear holding a board saying 'love'"
prompt_B_template = "hello world"
DIR_PATH = f"./results/attns_heads_logits/{''.join(c for c in prompt_A_template.replace(' ', '_') if c not in {'<', '>'})}_B_template_{''.join(c for c in prompt_B_template.replace(' ', '_') if c not in {'<', '>'})}"


CAUSAL_BLOCKS_TRACING = [
    {},
    {"down1": {0: list(range(2)), 1: list(range(2))}},
    {"down2": {0: list(range(10)), 1: list(range(10))}},
    {"mid": {0: list(range(10))}},
    {"up0": {0: list(range(10)), 1: list(range(10)), 2: list(range(10))}},
    {"up1": {0: list(range(2)), 1: list(range(2)), 2: list(range(2))}},
]

CAUSAL_TRANSFORMERS_TRACING = [
    {},
    # {"up0": {0: list(range(10))}},
    # {"up0": {1: list(range(10))}},
    {"up0": {2: list(range(10))}},
]

CAUSAL_LAYERS_TRACING = [
    {},
    {"up0": {2: [0]}},
    {"up0": {2: [1]}},
    {"up0": {2: [2]}},
    {"up0": {2: [3]}},
    {"up0": {2: [4]}},
    {"up0": {2: [5]}},
    {"up0": {2: [7]}},
    {"up0": {2: [8]}},
    {"up0": {2: [9]}},
]

CAUSAL_2LAYERS_TRACING = [
    {},
    {"up0": {2: [0, 2]}},
    {"up0": {2: [1, 2]}},
    {"up0": {2: [2]}},
    {"up0": {2: [2, 3]}},
    {"up0": {2: [2, 4]}},
    {"up0": {2: [2, 5]}},
    {"up0": {2: [2, 6]}},
    {"up0": {2: [2, 7]}},
    {"up0": {2: [2, 8]}},
    {"up0": {2: [2, 9]}},
]

CAUSAL_LAST_TRACING = [
    # {},
    # {"up0": {0: list(range(10)), 1: list(range(10)), 2: list(range(10))}},
    # {"up0": {2: list(range(10))}},
    # {"up0": {2: [2]}},
    # {"up0": {2: [1, 2]}},
    {"up0": {2: [1, 2, 3]}},
]

B_HEAD_MASKS = [
    # {},
    {
        "up0": {
            2: [
                [1, 14],
                [1, 8],
                [1, 4],
                [1, 7],
                [1, 15],
                [1, 18],
                [1, 5],
                [1, 13],
                [1, 16],
                [1, 17],
                [1, 0],
                [1, 9],
                [1, 3],
                [1, 1],
                [1, 11],
                [2, 15],
                [2, 17],
                [2, 2],
                [2, 19],
                [2, 8],
                [2, 4],
                [2, 13],
                [2, 12],
                [2, 10],
                [2, 3],
                [2, 16],
                [2, 9],
                [3, 6],
                [3, 7],
                [3, 19],
                [3, 11],
                [3, 18],
                [3, 5],
                [3, 15],
                [3, 0],
                [3, 17],
                [3, 14],
                [3, 9],
                [3, 10],
                [3, 4],
            ]
        }
    },
]


def get_ocrs(pil_images, device):
    processor, model = get_idefics2(use_cuda=(device == "cuda"))
    ocr_outs = []
    for pil_image in tqdm(pil_images, desc="OCR Idefics2"):
        ret = get_text_idefics2(
            processor,
            model,
            pil_image,
            (device == "cuda"),
        )
        ocr_outs.append(ret)
    return ocr_outs


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def image_grid_with_title(imgs, rows, cols, title="", font_size=80):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    font = ImageFont.load_default(size=font_size)
    title_height = font.getbbox(title)[3] + 10  # Adding some padding for the title
    grid = Image.new(
        "RGB", size=(cols * w, rows * h + title_height), color=(255, 255, 255)
    )

    draw = ImageDraw.Draw(grid)
    draw.text((10, 5), title, font=font, fill=(0, 0, 0))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h + title_height))

    return grid


def batch_sample(
    pipeline,
    noise,
    prompts_A,
    generator,
    num_inference_steps=N_STEPS,
    prompts_B=None,
    batch_size=BATCH_SIZE,
    B_layers=None,
    B_heads_mask=None,
):
    all_images = []
    for idx_start in tqdm(
        range(0, noise.shape[0], batch_size), desc="Batch generation", miniters=10
    ):
        idx_end = idx_start + batch_size
        images = pipeline(
            prompt=prompts_A[idx_start:idx_end],
            num_inference_steps=num_inference_steps,
            generator=generator,
            latents=noise[idx_start:idx_end].to(device),
            prompt_B=(prompts_B[idx_start:idx_end] if len(prompts_B) > 0 else None),
            B_layers=B_layers,
            B_heads_mask=B_heads_mask,
        ).images
        all_images.append(images)
    return sum(all_images, [])


set_seed(SEED)
A_prompts = [prompt_A_template]
B_prompts = [prompt_B_template]
device = DEVICE
generator = torch.Generator().manual_seed(SEED)

noise = torch.randn(
    (1, 4, 128, 128),
    generator=generator,
    dtype=torch.float32 if device == "cpu" else torch.float16,
)
noise = noise.repeat(len(B_prompts) // N_SAMPLES_PER_PROMPT, 1, 1, 1)
print(noise.shape)
# DIR_PATH = "./results/sdxl_blocks_transformers"
# DIR_PATH = "./results/sdxl_blocks_transformers_layers"
# DIR_PATH = "./results/sdxl_blocks_transformers_two_layers"
# DIR_PATH = "./results/sdxl_text_notext2"
os.makedirs(DIR_PATH, exist_ok=True)
layers_stats = {}
# for idx, b_layers in enumerate(CAUSAL_BLOCKS_TRACING):
# for idx, b_layers in enumerate(CAUSAL_TRANSFORMERS_TRACING):
# for idx, b_layers in enumerate(CAUSAL_LAYERS_TRACING):
# for b_layers in CAUSAL_2LAYERS_TRACING:
for idx, b_layers in enumerate(CAUSAL_LAST_TRACING):
    if len(b_layers.keys()) == 0:
        name = "_NO_INJECT"
    else:
        # bl_id = str(list(b_layers.keys())[0])
        # bl_id = str(list(b_layers["up0"].keys())[0])
        # bl_id = str(b_layers["up0"][2][0])
        # bl_id = "".join([str(i) for i in b_layers["up0"][2]])
        # name = f"UNet__B{bl_id.upper()}"
        # name = f"UNet__BUp0__T{bl_id.upper()}"
        # name = f"UNet__BUp0__T2__L{bl_id.upper()}"
        name = f"UNet_BUp0_T2_n_heads_{len(B_HEAD_MASKS[idx]['up0'][2])}_asc"
    LAYERS_PATH = f"{DIR_PATH}/{name}"
    os.makedirs(LAYERS_PATH, exist_ok=True)
    print(f"STARTING: {name}")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        SDXL_MODEL_NAME_OR_PATH,
        torch_dtype=torch.float16 if device == "cuda" else None,
        variant="fp16",
        use_safetensors=True,
        add_watermarker=False,
        token=os.environ.get("HF_TOKEN"),
    ).to(device)
    b_heads_mask = B_HEAD_MASKS[idx] if len(B_HEAD_MASKS) > 0 else None

    images = batch_sample(
        pipeline=pipe,
        noise=noise,
        prompts_A=A_prompts,
        generator=generator,
        num_inference_steps=N_STEPS,
        prompts_B=B_prompts,
        batch_size=BATCH_SIZE,
        B_layers=b_layers,
        B_heads_mask=b_heads_mask,
    )

    # texts_A = [prompt["text"] for prompt in A_prompts]
    # texts_B = [prompt["text"] for prompt in B_prompts]
    # prompts_A = [prompt["prompt"] for prompt in A_prompts]
    # prompts_B = [prompt["prompt"] for prompt in B_prompts]

    # gen_outs = get_ocrs(images, device)
    # distances_A = get_levenshtein_distances(gen_outs, texts_A)
    # distances_B = get_levenshtein_distances(gen_outs, texts_B)

    grid = image_grid(images, 1, 1)
    grid.save(f"{LAYERS_PATH}/bear.png")

    torch.cuda.empty_cache()
    # with open(f"{LAYERS_PATH}/labels.txt", "w", newline="") as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(["idx", "text_A", "text_B", "prompt_A", "prompt_B", "ocr_out"])
    #     for idx in range(len(texts_A)):
    #         writer.writerow(
    #             [
    #                 idx,
    #                 texts_A[idx],
    #                 texts_B[idx],
    #                 prompts_A[idx],
    #                 prompts_B[idx],
    #                 gen_outs[idx],
    #             ]
    #         )
    # stats = {
    #     "min_dist_A": distances_A.min(),
    #     "max_dist_A": distances_A.max(),
    #     "median_dist_A": np.median(distances_A),
    #     "mean_dist_A": distances_A.mean(),
    #     "min_dist_B": distances_B.min(),
    #     "max_dist_B": distances_B.max(),
    #     "median_dist_B": np.median(distances_B),
    #     "mean_dist_B": distances_B.mean(),
    #     "A_preference": np.sum(distances_A < distances_B),
    #     "B_preference": np.sum(distances_A > distances_B),
    #     "AB_equal": np.sum(distances_A == distances_B),
    # }
    # plot_hist_levenshtein(distances_A, save_path=f"{LAYERS_PATH}/toA_distances.png")
    # plot_hist_levenshtein(distances_B, save_path=f"{LAYERS_PATH}/toB_distances.png")
    # layers_stats[name] = stats

# with open(f"{DIR_PATH}/metrics.txt", "w", newline="") as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(
#         [
#             "Model Name",
#             "Min Dist A",
#             "Max Dist A",
#             "Median Dist A",
#             "Mean Dist A",
#             "Min Dist B",
#             "Max Dist B",
#             "Median Dist B",
#             "Mean Dist B",
#             "A Preference",
#             "B Preference",
#             "AB Equal",
#         ]
#     )
#     for k, v in layers_stats.items():
#         writer.writerow(
#             [
#                 k,
#                 v["min_dist_A"],
#                 v["max_dist_A"],
#                 v["median_dist_A"],
#                 v["mean_dist_A"],
#                 v["min_dist_B"],
#                 v["max_dist_B"],
#                 v["median_dist_B"],
#                 v["mean_dist_B"],
#                 v["A_preference"],
#                 v["B_preference"],
#                 v["AB_equal"],
#             ]
#         )
