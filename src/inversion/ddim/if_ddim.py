from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL
import torch

from diffusers import DDIMScheduler, IFImg2ImgPipeline
from diffusers.pipelines.deepfloyd_if.pipeline_if_img2img import (IFPipelineOutput)

from src.inversion.ddim.helpers import _backward_ddim


class IFDDIMPipeline(IFImg2ImgPipeline):
    def fix_scheduler(self):
        self.scheduler = DDIMScheduler.from_config(self.scheduler.config)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        image: Union[
            torch.FloatTensor,
            PIL.Image.Image,
            np.ndarray,
            List[torch.FloatTensor],
            List[PIL.Image.Image],
            List[np.ndarray],
        ] = None,
        strength: float = 0.7,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 10.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        clean_caption: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        pause_at_t: Optional[int] = None,
    ):
        self.scheduler = DDIMScheduler.from_config(self.scheduler.config)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        self.check_inputs(
            prompt, image, batch_size, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        device = self._execution_device

        do_classifier_free_guidance = False

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            do_classifier_free_guidance,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            clean_caption=clean_caption,
        )

        if timesteps is not None:
            self.scheduler.set_timesteps(timesteps=timesteps, device=device)
            timesteps = self.scheduler.timesteps
            num_inference_steps = len(timesteps)
        else:
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps

        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength)

        # 4. Preprocess image
        image = self.preprocess_image(image)
        image = image.to(device=device, dtype=prompt_embeds.dtype)

        noise_timestep = timesteps[0:1]
        noise_timestep = noise_timestep.repeat(batch_size * num_images_per_prompt)

        intermediate_images = self.prepare_intermediate_images(
            image, noise_timestep, batch_size, num_images_per_prompt, prompt_embeds.dtype, device, generator
        )
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        if hasattr(self, "text_encoder_offload_hook") and self.text_encoder_offload_hook is not None:
            self.text_encoder_offload_hook.offload()



        # # 6. Prepare latent variables
        # latents = self.prepare_latents(
        #     image,
        #     None,
        #     batch_size,
        #     num_images_per_prompt,
        #     prompt_embeds.dtype,
        #     device,
        #     generator,
        #     False,
        # )

        # height, width = latents.shape[-2:]
        # height = height * self.vae_scale_factor
        # width = width * self.vae_scale_factor

        # original_size = original_size or (height, width)
        # target_size = target_size or (height, width)

        # # 8. Prepare added time ids & embeddings
        # negative_original_size = original_size
        # negative_target_size = target_size
        # negative_crops_coords_top_left = (0, 0)
        # if self.text_encoder_2 is None:
        #     text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        # else:
        #     text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        # add_text_embeds = pooled_prompt_embeds

        # add_time_ids, add_neg_time_ids = self._get_add_time_ids(
        #     original_size,
        #     crops_coords_top_left,
        #     target_size,
        #     aesthetic_score,
        #     negative_aesthetic_score,
        #     negative_original_size,
        #     negative_crops_coords_top_left,
        #     negative_target_size,
        #     dtype=prompt_embeds.dtype,
        #     text_encoder_projection_dim=text_encoder_projection_dim,
        # )

        # add_time_ids = add_time_ids.repeat(batch_size * num_images_per_prompt, 1)

        prompt_embeds = prompt_embeds.to(device)
        # add_text_embeds = add_text_embeds.to(device)
        # add_time_ids = add_time_ids.to(device)

        # added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
        # prev_timestep = None

        print(intermediate_images.shape)

        for idx, t in enumerate(reversed(self.scheduler.timesteps)):
            if idx == pause_at_t:
                break
            if idx == 0:
                latents = self.scheduler.scale_model_input(intermediate_images, t)

            noise_pred = self.unet(
                latents,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]

            alpha_prod_t = self.scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = (
                self.scheduler.alphas_cumprod[prev_timestep]
                if prev_timestep is not None
                else self.scheduler.final_alpha_cumprod
            )
            prev_timestep = t

            latents = _backward_ddim(
                x_tm1=latents,
                alpha_t=alpha_prod_t,
                alpha_tm1=alpha_prod_t_prev,
                eps_xt=noise_pred,
            )

        image = latents
        return IFPipelineOutput(images=image)
