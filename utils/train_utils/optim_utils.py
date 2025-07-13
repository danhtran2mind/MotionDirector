from typing import Optional, List
import torch
from diffusers.models.attention_processor import AttnProcessor2_0
from diffusers.models.attention import BasicTransformerBlock
import os
import sys
# Add the directory of the current file to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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

def get_optimizer(use_8bit_adam):
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        return bnb.optim.AdamW8bit
    else:
        return torch.optim.AdamW