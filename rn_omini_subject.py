import torch
from PIL import Image
import numpy as np
from diffusers import FluxPipeline, FluxTransformer2DModel
try:
    from ComfyUI_RN_OminiControl.src.generate import generate, seed_everything
    from ComfyUI_RN_OminiControl.src.condition import Condition
except Exception:
    from src.generate import generate, seed_everything
    from src.condition import Condition
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel,T5TokenizerFast
import os
from ComfyUI_RN_OminiControl.rn_utils import *

def run(t_img, prompt, flux_model, seed):

    assert t_img.shape[0] == 1
    
    i = 255. * t_img[0].numpy()
    image = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8)).convert("RGB").resize((g_width, g_height))

    release_gpu()

    flux_dir = resolve_flux_dir(flux_model)
    lora_model = os.path.join(get_models_dir(), 'flux', 'OminiControl', 'omini', 'subject_512.safetensors')

    encoded_condition = encode_condition(flux_dir, image)

    text_encoder = CLIPTextModel.from_pretrained(
        flux_dir, subfolder="text_encoder", torch_dtype=torch.bfloat16
    )
    text_encoder_2 = T5EncoderModel.from_pretrained(
        flux_dir, subfolder="text_encoder_2", torch_dtype=torch.bfloat16
    )
    tokenizer = CLIPTokenizer.from_pretrained(flux_dir, subfolder="tokenizer")
    tokenizer_2 = T5TokenizerFast.from_pretrained(flux_dir, subfolder="tokenizer_2")

    device, dtype = get_device_and_dtype()
    pipeline = FluxPipeline.from_pretrained(
        flux_dir,
        text_encoder=text_encoder.to(dtype),
        text_encoder_2=text_encoder_2.to(dtype),
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        transformer=None,
        vae=None,
        torch_dtype=dtype,
    ).to(device)

    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
            prompt=prompt, prompt_2=None, max_sequence_length=256
        )

    del text_encoder
    del text_encoder_2
    del tokenizer
    del tokenizer_2
    del pipeline

    release_gpu()

    if isinstance(flux_dir, str) and flux_dir.endswith('.gguf'):
        raise ValueError('gguf 格式的 FLUX 模型当前不支持直接加载。请使用 diffusers 目录或 HuggingFace 仓库名称。')
    pipeline = FluxPipeline.from_pretrained(
        flux_dir,
        text_encoder=None,
        text_encoder_2=None,
        tokenizer=None,
        tokenizer_2=None,
        vae=None,
        torch_dtype=dtype,
    )

    pipeline.to(device)

    pipeline.load_lora_weights(
        lora_model,
        adapter_name="subject",
    )

    condition = Condition("subject", image)

    seed_everything(int(seed) % (2 ^ 16))

    result_latents = generate(
    # result_img = generate(
        pipeline,
        encoded_condition = encoded_condition,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        text_ids=text_ids,
        conditions=[condition],
        output_type="latent",
        return_dict=False,
        num_inference_steps=8,
        height=g_height,
        width=g_width,
    )

    del pipeline

    release_gpu()

    result_img = decode_latents(flux_dir, result_latents[0]).images[0]

    return torch.from_numpy(np.array(result_img).astype(np.float32) / 255.0).unsqueeze(0)
