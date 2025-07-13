import torch
import torch.nn.functional as F
import random
from torchvision import transforms
from diffusers import DDIMScheduler, TextToVideoSDPipeline
from utils.lora import extract_lora_child_module
from utils.video_pipeline import export_to_video, unet_and_text_g_c
import copy
import math
import gc

already_printed_trainables = False

def sample_noise(latents: torch.Tensor, noise_strength: float, use_offset_noise: bool=False) -> torch.Tensor:
    b, c, f, *_ = latents.shape
    device = "cuda" if torch.cuda.is_available() else "cpu"
    noise_latents = torch.randn_like(latents, device=device)
    if use_offset_noise:
        offset_noise = torch.randn(b, c, f, 1, 1, device=device)
        noise_latents = noise_latents + noise_strength * offset_noise
    return noise_latents

def enforce_zero_terminal_snr(betas):
    alphas = 1 - betas
    alphas_bar = alphas.cumprod(0)
    alphas_bar_sqrt = alphas_bar.sqrt()
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()
    alphas_bar_sqrt -= alphas_bar_sqrt_T
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)
    alphas_bar = alphas_bar_sqrt ** 2
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    alphas = torch.cat([alphas_bar[0:1], alphas])
    return 1 - alphas

def handle_trainable_modules(model, trainable_modules: tuple=None, is_enabled: bool=True, negation: list=None):
    global already_printed_trainables
    unfrozen_params = 0
    if trainable_modules:
        for name, module in model.named_modules():
            for tm in tuple(trainable_modules):
                if tm == 'all':
                    model.requires_grad_(is_enabled)
                    unfrozen_params = len(list(model.parameters()))
                    break
                if tm in name and 'lora' not in name:
                    for m in module.parameters():
                        m.requires_grad_(is_enabled)
                        if is_enabled:
                            unfrozen_params += 1
    if unfrozen_params > 0 and not already_printed_trainables:
        already_printed_trainables = True
        print(f"{unfrozen_params} params have been unfrozen for training.")

def save_pipe(
    path: str,
    global_step: int,
    accelerator,
    unet,
    text_encoder,
    vae,
    output_dir: str,
    lora_manager_spatial,
    lora_manager_temporal,
    unet_target_replace_module: list=None,
    text_target_replace_module: list=None,
    is_checkpoint: bool=False,
    save_pretrained_model: bool=True
):
    if is_checkpoint:
        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
        os.makedirs(save_path, exist_ok=True)
    else:
        save_path = output_dir
    u_dtype, t_dtype, v_dtype = unet.dtype, text_encoder.dtype, vae.dtype
    unet_out = copy.deepcopy(accelerator.unwrap_model(unet.cpu(), keep_fp32_wrapper=False))
    text_encoder_out = copy.deepcopy(accelerator.unwrap_model(text_encoder.cpu(), keep_fp32_wrapper=False))
    pipeline = TextToVideoSDPipeline.from_pretrained(
        path,
        unet=unet_out,
        text_encoder=text_encoder_out,
        vae=vae,
    ).to(torch.float32)
    lora_manager_spatial.save_lora_weights(model=copy.deepcopy(pipeline), save_path=save_path+'/spatial', step=global_step)
    if lora_manager_temporal:
        lora_manager_temporal.save_lora_weights(model=copy.deepcopy(pipeline), save_path=save_path+'/temporal', step=global_step)
    if save_pretrained_model:
        pipeline.save_pretrained(save_path)
    if is_checkpoint:
        unet, text_encoder = accelerator.prepare(unet, text_encoder)
        models_to_cast_back = [(unet, u_dtype), (text_encoder, t_dtype), (vae, v_dtype)]
        [x[0].to(accelerator.device, dtype=x[1]) for x in models_to_cast_back]
    logger.info(f"Saved model at {save_path} on step {global_step}")
    del pipeline, unet_out, text_encoder_out
    torch.cuda.empty_cache()
    gc.collect()

def should_sample(global_step: int, validation_steps: int, validation_data: dict) -> bool:
    return global_step % validation_steps == 0 and validation_data.get('sample_preview', False)