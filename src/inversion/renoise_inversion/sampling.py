import torch

from src.inversion.renoise_inversion.src.config import RunConfig
from src.inversion.renoise_inversion.src.eunms import Model_Type, Scheduler_Type


def sdxl_invert_to_latent(
    pipe_inversion,
    image,
    prompt_real,
    num_inversion_steps=50,
    num_renoise_steps=1,
    inversion_max_step=1.0,
):
    cfg = RunConfig(
        model_type=Model_Type.SDXL,
        scheduler_type=Scheduler_Type.DDIM,
        num_inversion_steps=num_inversion_steps,
        num_inference_steps=num_inversion_steps,
        num_renoise_steps=num_renoise_steps,
        perform_noise_correction=False,
        guidance_scale=1.0,
        inversion_max_step=inversion_max_step,
        noise_regularization_num_reg_steps=0,
    )
    generator = torch.Generator().manual_seed(420)
    pipe_inversion.cfg = cfg

    res = pipe_inversion(
        prompt=prompt_real,
        num_inversion_steps=cfg.num_inversion_steps,
        num_inference_steps=cfg.num_inference_steps,
        generator=generator,
        image=image,
        guidance_scale=cfg.guidance_scale,
        strength=cfg.inversion_max_step,
        denoising_start=1.0 - cfg.inversion_max_step,
        num_renoise_steps=cfg.num_renoise_steps,
    )

    return res[0][0].clone()


def sdxl_sample_from_latent(
    pipe_forward,
    latent,
    prompt_real,
    num_inference_steps=50,
    guidance_scale=5.0,
    inversion_max_step=1.0,
    output_type="np",
):
    pipe_forward.clean_cache()
    cfg = RunConfig(
        model_type=Model_Type.SDXL,
        scheduler_type=Scheduler_Type.DDIM,
        num_inversion_steps=num_inference_steps,
        num_inference_steps=num_inference_steps,
        num_renoise_steps=1,
        perform_noise_correction=False,
        guidance_scale=1.0,
        inversion_max_step=inversion_max_step,
        noise_regularization_num_reg_steps=0,
    )
    pipe_forward.cfg = cfg
    generator = torch.Generator().manual_seed(420)
    img_out = pipe_forward(
        prompt=[prompt_real],
        image=latent,
        strength=inversion_max_step,
        num_inference_steps=num_inference_steps,
        generator=generator,
        guidance_scale=guidance_scale,
        denoising_start=1.0 - inversion_max_step,
        output_type=output_type,
    )
    return img_out


def sdxl_sample_from_latent_our(
    pipe_forward,
    latent,
    prompt,
    num_inference_steps=50,
    guidance_scale=5.0,
    inversion_max_step=1.0,
    attn_idx_to_patch=None,
    output_type="np",
    run_with_cache=False,
    batch_num=0,
):
    # pipe_forward.clean_cache()
    generator = torch.Generator().manual_seed(420)
    cfg = RunConfig(
        model_type=Model_Type.SDXL,
        scheduler_type=Scheduler_Type.DDIM,
        num_inversion_steps=num_inference_steps,
        num_inference_steps=num_inference_steps,
        num_renoise_steps=1,
        perform_noise_correction=False,
        guidance_scale=guidance_scale,
        inversion_max_step=inversion_max_step,
        noise_regularization_num_reg_steps=0,
    )
    pipe_forward.cfg = cfg

    img_out = pipe_forward(
        prompt=[prompt],
        image=latent,
        strength=inversion_max_step,
        num_inference_steps=num_inference_steps,
        generator=generator,
        guidance_scale=guidance_scale,
        denoising_start=1.0 - inversion_max_step,
        run_with_cache=run_with_cache,
        attn_idx_to_patch=attn_idx_to_patch,
        batch_num=batch_num,
        output_type=output_type,
    )
    return img_out.images[0]
