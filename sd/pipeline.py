from typing import Optional
import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler
from clip import CLIP

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8


def generate(
    prompt: str,
    uncond_prompt: str,  # negative prompt
    input_image: Optional[torch.Tensor] = None,
    strength: float = 0.8,  # how much attention to pay to the input image
    do_cfg: bool = True,  # model to output one with the prompt and one without the prompt
    cfg_scale: float = 7.5,  # how much to weight the positive prompt against the negative prompt
    sampler_name: str = "ddqm",
    n_inference_steps: int = 50,
    models={},
    seed=None,
    device=None,
    idle_device=None,
    tokenizer=None,
) -> torch.Tensor:

    with torch.no_grad():

        if not (0 < strength <= 1):
            raise ValueError(f"Strength must be between 0 and 1, got {strength}")

        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        generator = torch.Generator(device=device)

        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        clip = models["clip"]
        clip.to(device)

        if do_cfg:
            # Convert the prompt into tokens using the tokenizer
            cond_tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids

            # batch_size x sequence_length
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)

            # batch_size x sequence_length -> batch_size x sequence_length x dim
            cond_context = clip(cond_tokens)

            uncond_tokens = tokenizer.batch_encode_plus(
                [uncond_prompt], padding="max_length", max_length=77
            ).input_ids

            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)

            # batch_size x sequence_length -> batch_size x sequence_length x dim
            uncond_context = clip(uncond_tokens)

            # 2, sequence_length, dim -> 2 x 77 x 768
            context = torch.cat([cond_context, uncond_context])
        else:
            # Convert it into a list of tokens
            tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids

            tokens = torch.tensor(tokens, dtype=torch.long, device=device)

            # (1, 77, 768)
            context = clip(tokens)

        to_idle(clip)

        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_steps(n_inference_steps)
        else:
            raise ValueError(f"Sampler {sampler_name} not supported")

        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        if input_image:
            encoder = models["encoder"]
            encoder.to(device)

            input_image_tensor = input_image.resize(WIDTH, HEIGHT)
            input_image_tensor = np.array(input_image_tensor)

            # (Height, Width, Channels)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32)

            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))

            # (Height, Width, Channels) -> (Batch_size, Channels, Height, Width)
            input_image_tensor = input_image_tensor.unsqueeze(0)

            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            encoder_noise = torch.randn(
                latents_shape, device=device, generator=generator
            )

            # run the image through the encoder of the VAE
            latents = encoder(input_image_tensor, encoder_noise)

            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(encoder)
        else:
            # If we are doing text-to-image, start with random noise
            latents = torch.randn(latents_shape, device=device, generator=generator)

        diffusion = models["diffusion"]
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)

        for i, timestep in enumerate(timesteps):
            # (1, 320)
            time_embedding = get_time_embedding(timestep).to(device)

            # batch_size x 4 x LATENTS_HEIGHT, LATENTS_WIDTH
            model_input = latents

            if do_cfg:
                # batch_size x 4 x LATENTS_HEIGHT, LATENTS_WIDTH -> 2 * batch_size x 4 x LATENTS_HEIGHT, LATENTS_WIDTH
                model_input = model_input.repeat(2, 1, 1, 1)

            # model_output is the predicted noise by the UNET

            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2, dim=0)

                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            # Remove noise predicted by the UNET
            latents = sampler.step(timestep, latents, model_output)

        to_idle(diffusion)

        decoder = models["decoder"]
        decoder.to(device)

        images = decoder(latents)
        to_idle(decoder)

        images = rescale(images, (-1, 1), (0, 255), clamp=True)

        # batch_size x channels x height x width -> batch_size x height x width x channels

        images = images.permute(0, 2, 3, 1)

        images = images.to("cpu", torch.uint8).numpy()

        return images[0]


def rescale(x, old_range, new_range, clamp=False):

    old_min, old_max = old_range
    new_min, new_max = new_range

    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min

    if clamp:
        x = x.clamp(new_min, new_max)

    return x


def get_time_embedding(timestep: int) -> torch.Tensor:
    # (160,)
    freqs = torch.pow(1000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)

    # (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]

    # (1, 320)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
