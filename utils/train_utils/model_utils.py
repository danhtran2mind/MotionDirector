from typing import Optional, List
import torch
from diffusers.models.attention_processor import AttnProcessor2_0
from diffusers.models.attention import BasicTransformerBlock

from utils.video_pipeline import load_primary_models, freeze_models, handle_memory_attention

def is_attn(name: str) -> bool:
    return 'attn1' in name or 'attn2' in name

def set_processors(attentions):
    for attn in attentions:
        attn.set_processor(AttnProcessor2_0())

def set_torch_2_attn(unet):
    optim_count = 0
    for name, module in unet.named_modules():
        if is_attn(name):
            if isinstance(module, torch.nn.ModuleList):
                for m in module:
                    if isinstance(m, BasicTransformerBlock):
                        set_processors([m.attn1, m.attn2])
                        optim_count += 1
    if optim_count > 0:
        print(f"{optim_count} Attention layers using Scaled Dot Product Attention.")

def cast_to_gpu_and_type(model_list: List, accelerator, weight_dtype: torch.dtype):
    for model in model_list:
        if model is not None:
            model.to(accelerator.device, dtype=weight_dtype)