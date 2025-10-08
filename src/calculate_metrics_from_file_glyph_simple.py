import argparse
from src.eval.basic_metrics import calculate_mse, calculate_psnr_from_mse
from src.eval.clipscore import clip_metrics, extract_all_images
from src.eval.ocr_eval import get_ocr_easyocr, get_text_easyocr, ocr_metrics
from src.eval.text_distance import get_levenshtein_distances
from pytorch_msssim import ssim
import numpy as np
import torch
from diffusers.training_utils import set_seed
import random
import clip
import pandas as pd
from src.prepare_glyph import prepare_prompts_glyph_simple_bench
from src.eval.text_detection import setup_text_detection_model, remove_text_boxes

torch.backends.cuda.matmul.allow_tf32 = True
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
ocr_model = get_ocr_easyocr(use_cuda=torch.cuda.is_available())

clip_model, transform = clip.load("ViT-B/32", device=device, jit=False)
clip_model.eval()

text_detection_model = setup_text_detection_model(
    "/storage2/bartosz/code/t2i2/DB_IC15_resnet50.onnx"
)

N_SAMPLES_PER_PROMPT = 4
BATCH_SIZE = 1024
SEED = 42


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original_images_A",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--patched_images",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    set_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    original_images_A = np.load(args.original_images_A)
    print("loaded original images A")
    print("shape: ", original_images_A.shape)
    patched_images = np.load(args.patched_images)
    print("loaded patched images")
    print("shape: ", patched_images.shape)

    original_images_A_feats = extract_all_images(
        original_images_A, clip_model, device, batch_size=BATCH_SIZE
    )
    print("extracted features for original images A")
    prompts_A, prompts_B = prepare_prompts_glyph_simple_bench(
        n_samples_per_prompt=N_SAMPLES_PER_PROMPT
    )
    print(f"Number of prompts: {len(prompts_A)}")

    original_images_A_metrics = calculate_metrics(
        original_images_A,
        original_images_A_feats,
        original_images_A,
        [p["text"] for p in prompts_A],
        [p["text"] for p in prompts_B],
        [p["prompt"] for p in prompts_A],
        [p["prompt"] for p in prompts_B],
        device,
        BATCH_SIZE,
    )
    print("calculated metrics for original images A")
    patched_images_metrics = calculate_metrics(
        original_images_A,
        original_images_A_feats,
        patched_images,
        [p["text"] for p in prompts_A],
        [p["text"] for p in prompts_B],
        [p["prompt"] for p in prompts_A],
        [p["prompt"] for p in prompts_B],
        device,
        BATCH_SIZE,
    )
    print("calculated metrics for patched images")
    original_images_A_df = pd.DataFrame(
        original_images_A_metrics,
    )
    original_images_A_df["Block_patched"] = ["-" for _ in range(len(prompts_A))]
    patched_images_df = pd.DataFrame(
        patched_images_metrics,
    )
    patched_images_df["Block_patched"] = ["patched" for _ in range(len(prompts_A))]
    all_metrics_df = pd.concat([original_images_A_df, patched_images_df])
    all_metrics_df.to_csv(args.output_file, index=False)
