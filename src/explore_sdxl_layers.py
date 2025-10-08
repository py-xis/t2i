import argparse
import csv
import math
import os
import random
from itertools import chain, combinations
from typing import List

import numpy as np
import torch
from accelerate import PartialState
from accelerate.utils import gather_object
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from diffusers import AutoencoderKL, StableDiffusionXLPipeline
from src.eval.ocr import get_idefics2, get_text_idefics2
from src.eval.prompts_template import generate_prompt_text_dataset
from src.eval.text_distance import get_levenshtein_distances, plot_hist_levenshtein

SDXL_MODEL_NAME_OR_PATH = "stabilityai/stable-diffusion-xl-base-1.0"
VAE_MODEL_NAME_OR_PATH = "madebyollin/sdxl-vae-fp16-fix"
LORA_MODEL_NAME_OR_PATH = "res/lora-trained-xl"
SEED = 42
N_PROMPTS = 200
N_SAMPLES_PER_PROMPT = 5
N_STEPS = 27
BATCH_SIZE = 10


def get_experiment(experiment):
    if experiment == "blocks":
        return [
            {},
            {"down1": {0: list(range(2)), 1: list(range(2))}},
            {"down2": {0: list(range(10)), 1: list(range(10))}},
            {"mid": {0: list(range(10))}},
            {"up0": {0: list(range(10)), 1: list(range(10)), 2: list(range(10))}},
            {"up1": {0: list(range(2)), 1: list(range(2)), 2: list(range(2))}},
        ], [
            "no_inject",
            "down1",
            "down2",
            "mid",
            "up0",
            "up1",
        ]
    elif experiment == "transformers":
        return [
            {},
            {"up0": {0: list(range(10))}},
            {"up0": {1: list(range(10))}},
            {"up0": {2: list(range(10))}},
        ], [
            "no_inject",
            "up0_T0",
            "up0_T1",
            "up0_T2",
        ]
    elif experiment == "layers":
        return [
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
        ], [
            "no_inject",
            "up0_T2_L0",
            "up0_T2_L1",
            "up0_T2_L2",
            "up0_T2_L3",
            "up0_T2_L4",
            "up0_T2_L5",
            "up0_T2_L6",
            "up0_T2_L7",
            "up0_T2_L8",
            "up0_T2_L9",
        ]
    elif experiment == "2layers":
        return [
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
        ], [
            "no_inject",
            "up0_T2_L02",
            "up0_T2_L12",
            "up0_T2_L2",
            "up0_T2_L23",
            "up0_T2_L24",
            "up0_T2_L25",
            "up0_T2_L26",
            "up0_T2_L27",
            "up0_T2_L28",
            "up0_T2_L29",
        ]
    elif experiment == "causal":
        return [
            {},
            {"up0": {0: list(range(10)), 1: list(range(10)), 2: list(range(10))}},
            {"up0": {2: list(range(10))}},
            {"up0": {2: [2]}},
            {"up0": {2: [1, 2]}},
            {"up0": {2: [1, 2, 3]}},
        ], [
            "no_inject",
            "up0",
            "up0_T2",
            "up0_T2_L2",
            "up0_T2_L12",
            "up0_T2_L123",
        ]
    else:
        raise NotImplementedError(f"Wrong experiment: {experiment}")


def get_noise_per_prompt(device):
    generator = torch.Generator(device=device).manual_seed(SEED)
    return torch.randn(
        (N_SAMPLES_PER_PROMPT, 4, 128, 128),
        generator=generator,
        device=device,
        dtype=torch.float32 if device == "cpu" else torch.float16,
    )


def get_ocrs(pil_images, distributed_state, device):
    processor, model = get_idefics2(device=device)
    ocr_outs = []
    ranks_indices = list(range(len(pil_images)))
    with distributed_state.split_between_processes(ranks_indices) as rank_ids:
        loop = (
            tqdm(rank_ids, total=len(rank_ids), desc="OCR")
            if distributed_state.is_local_main_process
            else rank_ids
        )
        for idx in loop:
            ret = get_text_idefics2(processor, model, pil_images[idx], device)
            ocr_outs.append(ret)
    distributed_state.wait_for_everyone()
    ocr_outs = gather_object(ocr_outs)
    return ocr_outs


def all_sublists(l, min_el=None, limit=None):
    if limit is None:
        limit = len(l)
    if min_el is None:
        min_el = 0
    return chain(*[combinations(l, i) for i in range(min_el, limit + 1)])


def get_device():
    distributed_state = PartialState()
    return distributed_state, distributed_state.device


def load_model(sdxl_path, vae_path, lora_path=None, device="cpu"):
    vae = AutoencoderKL.from_pretrained(
        vae_path, torch_dtype=torch.float32 if device == "cpu" else torch.float16
    )
    pipe = StableDiffusionXLPipeline.from_pretrained(
        sdxl_path,
        vae=vae,
        torch_dtype=torch.float32 if device == "cpu" else torch.float16,
    )
    if lora_path is not None:
        pipe.load_lora_weights(lora_path)
    pipe = pipe.to(device)
    return pipe


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


def sample_batched(
    pipeline,
    A_prompts,
    noises,
    generator,
    distributed_state,
    num_inference_steps=N_STEPS,
    B_prompts=None,
    batch_size=BATCH_SIZE,
    B_layers=None,
):
    all_images = []
    loop = (
        tqdm(
            range(0, noises.shape[0], batch_size),
            total=math.ceil(noises.shape[0] / batch_size),
            desc="Batched sampling",
        )
        if distributed_state.is_local_main_process
        else range(0, noises.shape[0], batch_size)
    )
    for idx_start in loop:
        idx_end = idx_start + batch_size
        images = pipeline(
            prompt=A_prompts[idx_start:idx_end],
            num_inference_steps=num_inference_steps,
            generator=generator,
            latents=noises[idx_start:idx_end],
            prompt_B=(B_prompts[idx_start:idx_end] if B_prompts else None),
            B_layers=B_layers,
        ).images
        all_images.append(images)
    return sum(all_images, [])


def get_trace_stats(
    a_prompts, b_prompts, layers_path, distributed_state: PartialState, device
):
    images = []
    texts_A = []
    texts_B = []
    prompts_A = []
    prompts_B = []
    prompt_indices = range(len(a_prompts))
    ids = []
    for prompt_idx in prompt_indices:
        A_prompt = a_prompts[prompt_idx]
        B_prompt = b_prompts[prompt_idx]
        for noise_idx in range(N_SAMPLES_PER_PROMPT):
            images.append(Image.open(f"{layers_path}/p{prompt_idx}_n{noise_idx}.png"))
            texts_A.append(A_prompt["text"])
            texts_B.append(B_prompt["text"])
            prompts_A.append(A_prompt["prompt"])
            prompts_B.append(B_prompt["prompt"])
            ids.append(f"p{prompt_idx}_n{noise_idx}")

    gen_outs = get_ocrs(images, distributed_state, device)
    distances_A = get_levenshtein_distances(gen_outs, texts_A)
    distances_B = get_levenshtein_distances(gen_outs, texts_B)

    if distributed_state.is_local_main_process:
        with open(f"{layers_path}/labels.txt", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                ["idx", "text_A", "text_B", "prompt_A", "prompt_B", "ocr_out"]
            )
            for idx in range(len(texts_A)):
                writer.writerow(
                    [
                        ids[idx],
                        texts_A[idx],
                        texts_B[idx],
                        prompts_A[idx],
                        prompts_B[idx],
                        gen_outs[idx],
                    ]
                )
    non_empty_ids = np.array([len(s) > 0 for s in gen_outs])
    neq_dist_ids = distances_A != distances_B
    corr_ids = np.logical_and(non_empty_ids, neq_dist_ids)
    corr_dist_A = distances_A[corr_ids]
    corr_dist_B = distances_B[corr_ids]
    stats = {
        "min_dist_A": corr_dist_A.min(),
        "max_dist_A": corr_dist_A.max(),
        "median_dist_A": np.median(corr_dist_A),
        "mean_dist_A": corr_dist_A.mean(),
        "min_dist_B": corr_dist_B.min(),
        "max_dist_B": corr_dist_B.max(),
        "median_dist_B": np.median(corr_dist_B),
        "mean_dist_B": corr_dist_B.mean(),
        "A_preference": np.sum(corr_dist_A < corr_dist_B),
        "B_preference": np.sum(corr_dist_A > corr_dist_B),
        "AB_equal": np.sum(distances_A == distances_B),
    }
    if distributed_state.is_local_main_process:
        plot_hist_levenshtein(corr_dist_A, save_path=f"{layers_path}/toA_distances.png")
        plot_hist_levenshtein(corr_dist_B, save_path=f"{layers_path}/toB_distances.png")
    return stats


def write_experiments_results(layers_stats, dir_path):
    with open(f"{dir_path}/metrics.txt", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "Model Name",
                "Min Dist A",
                "Max Dist A",
                "Median Dist A",
                "Mean Dist A",
                "Min Dist B",
                "Max Dist B",
                "Median Dist B",
                "Mean Dist B",
                "A Preference",
                "B Preference",
                "AB Equal",
            ]
        )
        for k, v in layers_stats.items():
            writer.writerow(
                [
                    k,
                    v["min_dist_A"],
                    v["max_dist_A"],
                    v["median_dist_A"],
                    v["mean_dist_A"],
                    v["min_dist_B"],
                    v["max_dist_B"],
                    v["median_dist_B"],
                    v["mean_dist_B"],
                    v["A_preference"],
                    v["B_preference"],
                    v["AB_equal"],
                ]
            )


def sample_trace(
    b_layers,
    a_prompts,
    b_prompts,
    pipe,
    distributed_state: PartialState,
    device,
    layers_path,
):
    generator = torch.Generator(device=device).manual_seed(SEED)
    prompt_indices = list(range(len(a_prompts)))
    with distributed_state.split_between_processes(prompt_indices) as rank_prompts:
        all_noises = torch.cat(
            [get_noise_per_prompt(device=device) for _ in range(len(rank_prompts))]
        )
        all_a_prompts = [
            a_prompts[prompt_idx]["prompt"]
            for prompt_idx in rank_prompts
            for _ in range(N_SAMPLES_PER_PROMPT)
        ]
        all_b_prompts = [
            b_prompts[prompt_idx]["prompt"]
            for prompt_idx in rank_prompts
            for _ in range(N_SAMPLES_PER_PROMPT)
        ]
        ids = [
            f"p{prompt_idx}_n{noise_idx}"
            for prompt_idx in rank_prompts
            for noise_idx in range(N_SAMPLES_PER_PROMPT)
        ]
        images = sample_batched(
            pipeline=pipe,
            A_prompts=all_a_prompts,
            noises=all_noises,
            generator=generator,
            distributed_state=distributed_state,
            num_inference_steps=N_STEPS,
            B_prompts=all_b_prompts,
            batch_size=BATCH_SIZE,
            B_layers=b_layers,
        )
        for idx, img in enumerate(images):
            img.save(f"{layers_path}/{ids[idx]}.png")
    if distributed_state.is_local_main_process:
        print("Rank 0 finished, waiting for other ranks...")
    distributed_state.wait_for_everyone()


def run_experiment(experiment_name):
    # loading
    tracing_layers, tracing_names = get_experiment(experiment=experiment_name)
    random.seed(SEED)

    # models/data
    a_prompts = generate_prompt_text_dataset(limit=N_PROMPTS)
    b_prompts = generate_prompt_text_dataset(limit=N_PROMPTS)
    distributed_state, device = get_device()
    if distributed_state.is_main_process:
        print(f"> Starting: {experiment_name}")
    pipe: StableDiffusionXLPipeline = load_model(
        sdxl_path=SDXL_MODEL_NAME_OR_PATH,
        vae_path=VAE_MODEL_NAME_OR_PATH,
        lora_path=None,
        device=device,
    )
    pipe.set_progress_bar_config(disable=True)

    # output dir
    dir_path = f"./results1k/sdxl_{experiment_name}"
    os.makedirs(dir_path, exist_ok=True)

    # experiment
    layers_stats = {}
    for idx, b_layers in enumerate(tracing_layers):
        bl_id = tracing_names[idx]
        name = bl_id.upper()
        layers_path = f"{dir_path}/{name}"
        os.makedirs(layers_path, exist_ok=True)
        if distributed_state.is_main_process:
            print(f">>>> Injection: {experiment_name} -> {name}")

        sample_trace(
            b_layers, a_prompts, b_prompts, pipe, distributed_state, device, layers_path
        )
        stats = get_trace_stats(
            a_prompts,
            b_prompts,
            layers_path=layers_path,
            distributed_state=distributed_state,
            device=device,
        )
        layers_stats[name] = stats

    if distributed_state.is_main_process:
        write_experiments_results(layers_stats, dir_path=dir_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        help="Specify the experiment to run",
        choices=["blocks", "transformers", "layers", "2layers", "causal"],
    )
    args = parser.parse_args()
    run_experiment(experiment_name=args.experiment)
