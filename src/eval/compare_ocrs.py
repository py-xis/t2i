import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.eval.ocr import (
    get_idefics2,
    get_ocr_easyocr,
    get_text_easyocr,
    get_text_idefics2,
)
from src.eval.text_distance import get_levenshtein_distances, plot_hist_levenshtein

USE_CUDA = True
PATH_TO_GENS = "results/sample_sd3_no_inject"


def get_idefics2_ocrs(image_paths):
    print("OCRing Idefics2")
    processor, model = get_idefics2(USE_CUDA)
    ocr_outs = []
    for image in tqdm(image_paths, desc="OCR Idefics2"):
        ret = get_text_idefics2(
            processor,
            model,
            Image.open(image),
            USE_CUDA,
        )
        ocr_outs.append(ret)
    return ocr_outs


def get_easyocr_ocrs(image_paths):
    print("Testing EasyOCR:")
    model = get_ocr_easyocr(USE_CUDA)
    ocr_outs = []
    for image in tqdm(image_paths, desc="OCR EasyOCR"):
        ret = get_text_easyocr(model, Image.open(image))
        ocr_outs.append(ret)
    return ocr_outs


def benchmark_ocrs():
    df = pd.read_csv(PATH_TO_GENS + "/labels.txt", delimiter=",", header=0)
    image_paths = [
        f"{PATH_TO_GENS}/{img_idx}.png" for img_idx in df["img_idx"].tolist()
    ]
    target_texts = df["text"].tolist()
    idefics2_outs = get_idefics2_ocrs(image_paths)
    easyocr_outs = get_easyocr_ocrs(image_paths)

    distances_idefics2 = get_levenshtein_distances(idefics2_outs, target_texts)
    distances_easyocr = get_levenshtein_distances(easyocr_outs, target_texts)

    print(
        f"Idefics2 | Mean={distances_idefics2.mean()} Median={np.median(distances_idefics2)}, Min={distances_idefics2.min()}, Max={distances_idefics2.max()}"
    )
    plot_hist_levenshtein(
        distances_idefics2, save_path=f"{PATH_TO_GENS}/hist_idefics2.png"
    )

    print(
        f"EasyOCR | Mean={distances_easyocr.mean()} Median={np.median(distances_easyocr)}, Min={distances_easyocr.min()}, Max={distances_easyocr.max()}"
    )
    plot_hist_levenshtein(
        distances_idefics2, save_path=f"{PATH_TO_GENS}/hist_easyocr.png"
    )


if __name__ == "__main__":
    benchmark_ocrs()
