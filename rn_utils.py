import torch
from diffusers import FluxPipeline
import gc
from ComfyUI_RN_OminiControl.src.generate import generate, seed_everything
from ComfyUI_RN_OminiControl.src.condition import Condition

from diffusers.pipelines.flux.pipeline_flux import (
    FluxPipelineOutput,
)

g_width = 512
g_height = 512

def get_device_and_dtype():
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16
    return torch.device("cpu"), torch.float32

def release_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

def encode_condition(flux_dir, image, condition_type='subject'):
    device, dtype = get_device_and_dtype()
    pipeline = FluxPipeline.from_pretrained(
        flux_dir,
        text_encoder=None,
        text_encoder_2=None,
        tokenizer=None,
        tokenizer_2=None,
        transformer=None,
        torch_dtype=dtype,
    ).to(device)

    condition = Condition(condition_type, image)
    tokens, ids, type_id = condition.encode(pipeline)

    del condition
    del pipeline
    release_gpu()

    return (tokens, ids, type_id)

def decode_latents(flux_dir, latents):
    device, dtype = get_device_and_dtype()
    pipeline = FluxPipeline.from_pretrained(
        flux_dir,
        text_encoder=None,
        text_encoder_2=None,
        tokenizer=None,
        tokenizer_2=None,
        transformer=None,
        torch_dtype=dtype,
    ).to(device)

    latents = pipeline._unpack_latents(latents, g_height, g_width, pipeline.vae_scale_factor)
    latents = (
        latents / pipeline.vae.config.scaling_factor
    ) + pipeline.vae.config.shift_factor
    image = pipeline.vae.decode(latents, return_dict=False)[0]
    image = pipeline.image_processor.postprocess(image, output_type="pil")

    return FluxPipelineOutput(images=image)
