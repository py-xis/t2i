import sys
import numpy as np
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


def load_image(path):
    # Load numpy array
    img = np.load(path)
    img = np.transpose(img, (0, 3, 1, 2))
    # Convert to torch tensor
    img_tensor = torch.from_numpy(img).float()

    # Ensure the image has 3 channels (RGB)
    if img_tensor.shape[1] == 1:
        img_tensor = img_tensor.repeat(1, 3, 1, 1)

    return img_tensor


def calculate_lpips(img1_path, img2_path):
    # Load images
    img1 = load_image(img1_path) / 255
    img2 = load_image(img2_path) / 255

    # Initialize LPIPS metric
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True)

    # Calculate LPIPS
    with torch.no_grad():
        score = lpips(img1, img2)

    return score.item()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <path_to_image1.npy> <path_to_image2.npy>")
        sys.exit(1)

    img1_path = sys.argv[1]
    img2_path = sys.argv[2]

    lpips_score = calculate_lpips(img1_path, img2_path)
    print(f"LPIPS score: {lpips_score}")
