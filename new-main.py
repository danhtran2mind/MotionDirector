import argparse
import math
import random
import logging
from tqdm.auto import tqdm
from accelerate import Accelerator
import torch
import torch.nn.functional as F
from torchvision import transforms
from diffusers import DDIMScheduler, TextToVideoSDPipeline
from diffusers.optimization import get_scheduler
from utils.lora_handler import LoraHandler
from utils.lora import extract_lora_child_module
from utils.video_pipeline import load_primary_models, freeze_models, handle_memory_attention

from utils.train_utils.config import create_logging, accelerate_set_verbose, create_output_folders, load_config, logger
from utils.train_utils.dataset_utils import get_train_dataset, extend_datasets, create_dataloader
from utils.train_utils.model_utils import set_torch_2_attn, cast_to_gpu_and_type
from utils.train_utils.optim_utils import get_optimizer, param_optim, create_optimizer_params
from utils.train_utils.latent_utils import handle_cache_latents, tensor_to_vae_latent
from utils.train_utils.train_utils import sample_noise, enforce_zero_terminal_snr, handle_trainable_modules, save_pipe, should_sample, unet_and_text_g_c

def finetune_unet(
    batch, step, unet, vae, text_encoder, noise_scheduler, cache_latents, use_offset_noise,
    rescale_schedule, offset_noise_strength, random_hflip_img, train_temporal_lora,
    spatial_lora_num, validation_data
):
    unet.train()
    handle_trainable_modules(unet, trainable_modules=("attn1", "attn2"), negation=[])
    if not cache_latents:
        latents = tensor_to_vae_latent(batch["pixel_values"].to(vae.device), vae)
    else:
        latents = batch["latents"].to(unet.device)
    use_offset_noise = use_offset_noise and not rescale_schedule
    noise = sample_noise(latents, offset_noise_strength, use_offset_noise)
    bsz = latents.shape[0]
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
    token_ids = batch['prompt_ids'].to(text_encoder.device)
    encoder_hidden_states = text_encoder(token_ids)[0]
    target = noise if noise_scheduler.config.prediction_type == "epsilon" else noise_scheduler.get_velocity(latents, noise, timesteps)
    mask_spatial_lora = random.uniform(0, 1) < 0.2 and train_temporal_lora
    mask_temporal_lora = not train_temporal_lora
    if mask_spatial_lora:
        loras = extract_lora_child_module(unet, target_replace_module=["Transformer2DModel"])
        for lora_i in loras:
            lora_i.scale = 0.
        loss_spatial = None
    else:
        loras = extract_lora_child_module(unet, target_replace_module=["Transformer2DModel"])
        if spatial_lora_num == 1:
            for lora_i in loras:
                lora_i.scale = 1.
        else:
            for lora_i in loras:
                lora_i.scale = 0.
            for lora_idx in range(0, len(loras), spatial_lora_num):
                loras[lora_idx + step].scale = 1.
        loras = extract_lora_child_module(unet, target_replace_module=["TransformerTemporalModel"])
        if len(loras) > 0:
            for lora_i in loras:
                lora_i.scale = 0.
        ran_idx = torch.randint(0, noisy_latents.shape[2], (1,)).item()
        if random.uniform(0, 1) < random_hflip_img:
            pixel_values_spatial = transforms.functional.hflip(batch["pixel_values"][:, ran_idx, :, :, :]).unsqueeze(1).to(vae.device)
            latents_spatial = tensor_to_vae_latent(pixel_values_spatial, vae)
            noise_spatial = sample_noise(latents_spatial, offset_noise_strength, use_offset_noise)
            noisy_latents_input = noise_scheduler.add_noise(latents_spatial, noise_spatial, timesteps)
            target_spatial = noise_spatial
            model_pred_spatial = unet(noisy_latents_input, timesteps, encoder_hidden_states=encoder_hidden_states).sample
            loss_spatial = F.mse_loss(model_pred_spatial[:, :, 0, :, :].float(), target_spatial[:, :, 0, :, :].float(), reduction="mean")
        else:
            noisy_latents_input = noisy_latents[:, :, ran_idx, :, :].unsqueeze(2)
            target_spatial = target[:, :, ran_idx, :, :]
            model_pred_spatial = unet(noisy_latents_input, timesteps, encoder_hidden_states=encoder_hidden_states).sample
            loss_spatial = F.mse_loss(model_pred_spatial[:, :, 0, :, :].float(), target_spatial.float(), reduction="mean")
    if mask_temporal_lora:
        loras = extract_lora_child_module(unet, target_replace_module=["TransformerTemporalModel"])
        for lora_i in loras:
            lora_i.scale = 0.
        loss_temporal = None
    else:
        loras = extract_lora_child_module(unet, target_replace_module=["TransformerTemporalModel"])
        for lora_i in loras:
            lora_i.scale = 1.
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample
        loss_temporal = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        beta = 1
        alpha = (beta ** 2 + 1) ** 0.5
        ran_idx = torch.randint(0, model_pred.shape[2], (1,)).item()
        model_pred_decent = alpha * model_pred - beta * model_pred[:, :, ran_idx, :, :].unsqueeze(2)
        target_decent = alpha * target - beta * target[:, :, ran_idx, :, :].unsqueeze(2)
        loss_ad_temporal = F.mse_loss(model_pred_decent.float(), target_decent.float(), reduction="mean")
        loss_temporal = loss_temporal + loss_ad_temporal
    return loss_spatial, loss_temporal, latents, noise

def main(
    pretrained_model_path: str,
    output_dir: str,
    train_data: dict,
    validation_data: dict,
    single_spatial_lora: bool=False,
    train_temporal_lora: bool=True,
    random_hflip_img: float=-1,
    extra_train_data: list=[],
    dataset_types: tuple=('json',),
    validation_steps: int=100,
    trainable_modules: tuple=("attn1", "attn2"),
    extra_unet_params: dict=None,
    train_batch_size: int=1,
    max_train_steps: int=500,
    learning_rate: float=5e-5,
    lr_scheduler: str="constant",
    lr_warmup_steps: int=0,
    adam_beta1: float=0.9,
    adam_beta2: float=0.999,
    adam_weight_decay: float=1e-2,
    adam_epsilon: float=1e-08,
    gradient_accumulation_steps: int=1,
    gradient_checkpointing: bool=True,
    text_encoder_gradient_checkpointing: bool=True,
    checkpointing_steps: int=500,
    resume_from_checkpoint: str=None,
    resume_step: int=None,
    mixed_precision: str="fp16",
    use_8bit_adam: bool=True,
    enable_xformers_memory_efficient_attention: bool=True,
    enable_torch_2_attn: bool=False,
    seed: int=None,
    use_offset_noise: bool=False,
    rescale_schedule: bool=False,
    offset_noise_strength: float=0.1,
    extend_dataset: bool=False,
    cache_latents: bool=True,
    cached_latent_dir: str=None,
    use_unet_lora: bool=True,
    unet_lora_modules: tuple=[],
    text_encoder_lora_modules: tuple=[],
    save_pretrained_model: bool=True,
    lora_rank: int=8,
    lora_path: str='',
    lora_unet_dropout: float=0.1,
    logger_type: str='tensorboard',
    **kwargs
):
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=logger_type,
        project_dir=output_dir
    )
    create_logging(logging, logger, accelerator)
    accelerate_set_verbose(accelerator)
    if accelerator.is_main_process:
        output_dir = create_output_folders(output_dir, {'main_args': locals()})
    noise_scheduler, tokenizer, text_encoder, vae, unet = load_primary_models(pretrained_model_path)
    freeze_models([vae, text_encoder, unet])
    handle_memory_attention(enable_xformers_memory_efficient_attention, enable_torch_2_attn, unet)
    if enable_torch_2_attn:
        set_torch_2_attn(unet)
    optimizer_cls = get_optimizer(use_8bit_adam)
    train_datasets = get_train_dataset(dataset_types, train_data, tokenizer)
    if extra_train_data:
        for dataset in extra_train_data:
            d_t, t_d = dataset['dataset_types'], dataset['train_data']
            train_datasets += get_train_dataset(d_t, t_d, tokenizer)
    extend_datasets(train_datasets, ['train_data', 'frames', 'image_dir', 'video_files'], extend_dataset)
    train_dataset = train_datasets[0] if len(train_datasets) == 1 else torch.utils.data.ConcatDataset(train_datasets)
    train_dataloader = create_dataloader(train_dataset, train_batch_size)
    cached_data_loader = handle_cache_latents(
        cache_latents, output_dir, train_dataloader, train_batch_size, vae, unet,
        pretrained_model_path, validation_data.get('noise_prior', 0), cached_latent_dir
    )
    if cached_data_loader:
        train_dataloader = cached_data_loader
    extra_unet_params = extra_unet_params or {}
    lora_manager_temporal = None
    unet_lora_params_temporal, unet_negation_temporal = [], []
    optimizer_temporal, lr_scheduler_temporal = None, None
    if train_temporal_lora:
        lora_manager_temporal = LoraHandler(use_unet_lora=use_unet_lora, unet_replace_modules=["TransformerTemporalModel"])
        unet_lora_params_temporal, unet_negation_temporal = lora_manager_temporal.add_lora_to_model(
            use_unet_lora, unet, lora_manager_temporal.unet_replace_modules, lora_unet_dropout,
            lora_path + '/temporal/lora/', r=lora_rank
        )
        optimizer_temporal = optimizer_cls(
            create_optimizer_params([
                param_optim(unet_lora_params_temporal, use_unet_lora, is_lora=True,
                            extra_params={**{"lr": learning_rate}, **extra_unet_params})
            ], learning_rate),
            lr=learning_rate, betas=(adam_beta1, adam_beta2), weight_decay=adam_weight_decay, eps=adam_epsilon
        )
        lr_scheduler_temporal = get_scheduler(
            lr_scheduler, optimizer=optimizer_temporal,
            num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
            num_training_steps=max_train_steps * gradient_accumulation_steps
        )
    lora_managers_spatial, unet_lora_params_spatial_list, optimizer_spatial_list, lr_scheduler_spatial_list = [], [], [], []
    spatial_lora_num = 1 if single_spatial_lora else train_dataset.__len__()
    for i in range(spatial_lora_num):
        lora_manager_spatial = LoraHandler(use_unet_lora=use_unet_lora, unet_replace_modules=["Transformer2DModel"])
        lora_managers_spatial.append(lora_manager_spatial)
        unet_lora_params_spatial, unet_negation_spatial = lora_manager_spatial.add_lora_to_model(
            use_unet_lora, unet, lora_manager_spatial.unet_replace_modules, lora_unet_dropout,
            lora_path + '/spatial/lora/', r=lora_rank
        )
        unet_lora_params_spatial_list.append(unet_lora_params_spatial)
        optimizer_spatial = optimizer_cls(
            create_optimizer_params([
                param_optim(unet_lora_params_spatial, use_unet_lora, is_lora=True,
                            extra_params={**{"lr": learning_rate}, **extra_unet_params})
            ], learning_rate),
            lr=learning_rate, betas=(adam_beta1, adam_beta2), weight_decay=adam_weight_decay, eps=adam_epsilon
        )
        optimizer_spatial_list.append(optimizer_spatial)
        lr_scheduler_spatial = get_scheduler(
            lr_scheduler, optimizer=optimizer_spatial,
            num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
            num_training_steps=max_train_steps * gradient_accumulation_steps
        )
        lr_scheduler_spatial_list.append(lr_scheduler_spatial)
    unet_negation_all = unet_negation_spatial + unet_negation_temporal
    unet, optimizer_spatial_list, optimizer_temporal, train_dataloader, lr_scheduler_spatial_list, lr_scheduler_temporal, text_encoder = accelerator.prepare(
        unet, optimizer_spatial_list, optimizer_temporal, train_dataloader, lr_scheduler_spatial_list, lr_scheduler_temporal, text_encoder
    )
    unet_and_text_g_c(unet, text_encoder, gradient_checkpointing, text_encoder_gradient_checkpointing)
    vae.enable_slicing()
    weight_dtype = torch.float16 if accelerator.mixed_precision == "fp16" else torch.bfloat16 if accelerator.mixed_precision == "bf16" else torch.float32
    cast_to_gpu_and_type([text_encoder, vae], accelerator, weight_dtype)
    if not use_offset_noise and rescale_schedule:
        noise_scheduler.betas = enforce_zero_terminal_snr(noise_scheduler.betas)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
    if accelerator.is_main_process:
        accelerator.init_trackers("text2video-fine-tune")
    total_batch_size = train