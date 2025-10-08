import torch
from src.inversion.renoise_inversion.src.eunms import Model_Type, Scheduler_Type
from src.inversion.renoise_inversion.src.pipes.sdxl_inversion_pipeline import SDXLDDIMPipeline
from src.inversion.renoise_inversion.src.schedulers.ddim_scheduler import MyDDIMScheduler

from diffusers import StableDiffusionXLImg2ImgPipeline


def scheduler_type_to_class(scheduler_type):
    if scheduler_type == Scheduler_Type.DDIM:
        return MyDDIMScheduler
    else:
        raise ValueError("Unknown scheduler type")


def is_stochastic(scheduler_type):
    if scheduler_type == Scheduler_Type.DDIM:
        return False
    else:
        raise ValueError("Unknown scheduler type")


def model_type_to_class(model_type):
    if model_type == Model_Type.SDXL:
        return StableDiffusionXLImg2ImgPipeline, SDXLDDIMPipeline
    else:
        raise ValueError("Unknown model type")


def model_type_to_model_name(model_type):
    if model_type == Model_Type.SDXL:
        return "stabilityai/stable-diffusion-xl-base-1.0"
    else:
        raise ValueError("Unknown model type")


def model_type_to_size(model_type):
    if model_type == Model_Type.SDXL:
        return (1024, 1024)
    else:
        raise ValueError("Unknown model type")


def is_float16(model_type):
    if model_type == Model_Type.SDXL:
        return True
    else:
        raise ValueError("Unknown model type")


def is_sd(model_type):
    if model_type == Model_Type.SDXL:
        return False
    else:
        raise ValueError("Unknown model type")


def _get_pipes(model_type, device):
    model_name = model_type_to_model_name(model_type)
    pipeline_inf, pipeline_inv = model_type_to_class(model_type)

    if is_float16(model_type):
        pipe_inference = pipeline_inf.from_pretrained(
            model_name, torch_dtype=torch.float16, use_safetensors=True, variant="fp16", safety_checker=None
        ).to(device)
    else:
        pipe_inference = pipeline_inf.from_pretrained(model_name, use_safetensors=True, safety_checker=None).to(device)

    pipe_inversion = pipeline_inv(**pipe_inference.components)

    return pipe_inversion, pipe_inference


def get_pipes(model_type, scheduler_type, device="cuda"):
    scheduler_class = scheduler_type_to_class(scheduler_type)

    pipe_inversion, pipe_inference = _get_pipes(model_type, device)

    pipe_inference.scheduler = scheduler_class.from_config(pipe_inference.scheduler.config)
    pipe_inversion.scheduler = scheduler_class.from_config(pipe_inversion.scheduler.config)

    if is_sd(model_type):
        pipe_inference.scheduler.add_noise = lambda init_latents, noise, timestep: init_latents
        pipe_inversion.scheduler.add_noise = lambda init_latents, noise, timestep: init_latents

    return pipe_inversion, pipe_inference
