import torch
from PIL import Image

from diffusers import StableDiffusion3Pipeline
from diffusers.utils import load_image

SD3_MODEL_NAME_OR_PATH = "stabilityai/stable-diffusion-3-medium-diffusers"
LORA_MODEL_NAME_OR_PATH = "../res/trained-sd3"
SEED = 42
N_SAMPLES = 8
N_STEPS = 28


def get_device():
    return "cpu"


def load_model(sd3_path, lora_path=None, device="cpu"):
    pipe = StableDiffusion3Pipeline.from_pretrained(sd3_path, torch_dtype=torch.float16)
    if lora_path is not None:
        pipe.load_lora_weights(lora_path)
    pipe = pipe.to(device)
    return pipe


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


device = get_device()
generator = torch.Generator(device=device).manual_seed(SEED)
all_prompts = [
    "A photo of a dog with a text 'love' in his paws" for _ in range(N_SAMPLES)
]

noise = torch.randn(
    (N_SAMPLES, 16, 128, 128),
    generator=generator,
    device=device,
    dtype=torch.float16,
)

pipe = load_model(sd3_path=SD3_MODEL_NAME_OR_PATH, lora_path=LORA_MODEL_NAME_OR_PATH)

# images = pipe(
#     prompt=all_prompts, num_inference_steps=N_STEPS, generator=generator, latents=noise
# ).images

# grid = image_grid(images, rows=2, cols=4)
# grid.save("sample_sd3_grid.jpg")
