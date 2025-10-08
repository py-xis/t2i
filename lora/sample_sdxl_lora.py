import torch
from PIL import Image

from diffusers import AutoencoderKL, StableDiffusionXLPipeline

SDXL_MODEL_NAME_OR_PATH = "stabilityai/stable-diffusion-xl-base-1.0"
VAE_MODEL_NAME_OR_PATH = "madebyollin/sdxl-vae-fp16-fix"
LORA_MODEL_NAME_OR_PATH = "./res/lora-trained-xl"
SEED = 42
N_SAMPLES = 8
N_STEPS = 28


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(sdxl_path, vae_path, lora_path=None):
    vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float16)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        sdxl_path, vae=vae, torch_dtype=torch.float16
    )
    if lora_path is not None:
        pipe.load_lora_weights(lora_path)
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
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
    (N_SAMPLES, 4, 128, 128),
    generator=generator,
    device=device,
    dtype=torch.float16 if device == "cuda" else torch.float32,
)

pipe = load_model(
    sdxl_path=SDXL_MODEL_NAME_OR_PATH,
    vae_path=VAE_MODEL_NAME_OR_PATH,
    lora_path=LORA_MODEL_NAME_OR_PATH,
)

# pipe = load_model(
#     sdxl_path=SDXL_MODEL_NAME_OR_PATH, vae_path=VAE_MODEL_NAME_OR_PATH, lora_path=None
# )

images = pipe(
    prompt=all_prompts, num_inference_steps=N_STEPS, generator=generator, latents=noise
).images

grid = image_grid(images, rows=2, cols=4)
grid.save("sample_sdxllora_grid.jpg")
