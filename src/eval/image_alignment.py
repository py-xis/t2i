import os
from functools import partial
from typing import List

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
DINO_MODEL_NAME = "facebook/dino-vits16"
BS = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_clip(device=DEVICE):
    clip_processor = AutoProcessor.from_pretrained(CLIP_MODEL_NAME)
    clip_model = AutoModel.from_pretrained(CLIP_MODEL_NAME).to(device)
    return clip_processor, clip_model


def get_dino(device=DEVICE):
    dino_model = AutoModel.from_pretrained(DINO_MODEL_NAME, add_pooling_layer=False).to(
        device
    )
    return dino_model


def get_clip_features(
    imgs: List[Image.Image], clip_processor, clip_model, device=DEVICE
):
    outs = []
    for batch_ids in tqdm(range(0, len(imgs), BS), desc="Calculating CLIP embeddings"):
        batch = imgs[batch_ids : batch_ids + BS]
        clip_batch_in = clip_processor(
            images=batch, return_tensors="pt"
        ).pixel_values.to(device)
        feats = clip_model.get_image_features(clip_batch_in)
        outs.append(feats.detach().cpu())
    return torch.cat(outs)


def get_dino_features(imgs: List[Image.Image], dino_model, device=DEVICE):
    T = transforms.Compose(
        [
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    outs = []
    for batch_ids in tqdm(range(0, len(imgs), BS), desc="Calculating DINO embeddings"):
        batch = imgs[batch_ids : batch_ids + BS]
        pred_imgs_processed = torch.stack([T(img).to(device) for img in batch])
        pred_features = dino_model(pred_imgs_processed).last_hidden_state[:, 0, :]
        outs.append(pred_features.detach().cpu())
    return torch.cat(outs)


def load_image(file_path):
    with Image.open(file_path) as img:
        return np.array(img)


def mean_pixel_distance(img1, img2):
    return np.mean(np.abs(img1 - img2))


def calculate_average_distance(dir1, dir2, n_prompts, n_noises):
    total_distance = 0
    num_pairs = 0

    with tqdm(total=n_prompts * n_noises, desc="Pixel distance calculation") as pbar:
        for p_idx in range(n_prompts):
            for n_idx in range(n_noises):
                file_name = f"p{p_idx}_n{n_idx}.png"
                file_path1 = os.path.join(dir1, file_name)
                file_path2 = os.path.join(dir2, file_name)

                if os.path.exists(file_path1) and os.path.exists(file_path2):
                    img1 = load_image(file_path1)
                    img2 = load_image(file_path2)
                    total_distance += mean_pixel_distance(img1, img2)
                    num_pairs += 1
                pbar.update(1)

    if num_pairs == 0:
        return 0
    else:
        return total_distance / num_pairs


def calculate_average_cos_sim(dir1, dir2, emb_model, n_prompts, n_noises):
    if emb_model["name"] == "dino":
        features_foo = partial(
            get_dino_features, dino_model=emb_model["model"], device=DEVICE
        )
    elif emb_model["name"] == "clip":
        features_foo = partial(
            get_clip_features,
            clip_processor=emb_model["processor"],
            clip_model=emb_model["model"],
            device=DEVICE,
        )
    else:
        raise NotImplementedError(f"Unknown embedding model name: {emb_model['name']=}")

    files_dir1 = []
    files_dir2 = []
    for p_idx in range(n_prompts):
        for n_idx in range(n_noises):
            file_name = f"p{p_idx}_n{n_idx}.png"
            file_path1 = os.path.join(dir1, file_name)
            file_path2 = os.path.join(dir2, file_name)
            files_dir1.append(Image.open(file_path1))
            files_dir2.append(Image.open(file_path2))

    features_dir1: torch.Tensor = features_foo(files_dir1)
    features_dir2: torch.Tensor = features_foo(files_dir2)

    cos_sim_foo = torch.nn.CosineSimilarity(dim=1)
    similarity_scores = cos_sim_foo(features_dir1, features_dir2)
    average_similarity = torch.mean(similarity_scores).item()
    return average_similarity
