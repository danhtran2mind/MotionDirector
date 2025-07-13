import torch
import torch.nn as nn
import copy
from typing import List, Optional, Union
import os


class LoRAParametrization(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = 4, scale: float = 1.0, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scale = scale
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input):
        print(f"LoRA input shape: {input.shape}")
        print(f"LoRA linear weight shape: {self.linear.weight.shape}")
        output = self.linear(input)
        lora_output = self.scale * (input @ self.lora_A.t() @ self.lora_B.t())
        print(f"LoRA output shape: {output.shape}, LoRA adjustment shape: {lora_output.shape}")
        return output + lora_output

    @classmethod
    def from_linear(cls, linear: nn.Linear, rank: int = 4, scale: float = 1.0):
        return cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=rank,
            scale=scale,
            bias=linear.bias is not None,
        )


class LoraHandler:
    def __init__(self, use_unet_lora: bool = False, unet_replace_modules: Optional[List[str]] = None):
        self.use_unet_lora = use_unet_lora
        self.unet_replace_modules = unet_replace_modules or []
        self.lora_modules = []

    def add_lora_to_model(
        self,
        use_unet_lora: bool,
        model: nn.Module,
        target_replace_module: List[str],
        lora_path: str = "",
        dropout: float = 0.0,
        r: int = 4,
    ):
        lora_params = []
        negation = []
        if use_unet_lora:
            for name, module in model.named_modules():
                for target_module in target_replace_module:
                    if target_module in name and isinstance(module, nn.Linear):
                        new_module = LoRAParametrization.from_linear(module, rank=r)
                        parent_name, child_name = name.rsplit(".", 1)
                        parent = model
                        for n in parent_name.split("."):
                            parent = getattr(parent, n)
                        if lora_path:
                            lora_weights_path = os.path.join(lora_path, "pytorch_lora_weights.safetensors")
                            if os.path.exists(lora_weights_path):
                                lora_weights = torch.load(lora_weights_path)
                                lora_a_key = f"{name}.lora_A"
                                lora_b_key = f"{name}.lora_B"
                                if lora_a_key in lora_weights and lora_b_key in lora_weights:
                                    lora_a_shape = lora_weights[lora_a_key].shape
                                    lora_b_shape = lora_weights[lora_b_key].shape
                                    expected_a_shape = (r, module.in_features)
                                    expected_b_shape = (module.out_features, r)
                                    if lora_a_shape == expected_a_shape and lora_b_shape == expected_b_shape:
                                        new_module.lora_A.data = lora_weights[lora_a_key]
                                        new_module.lora_B.data = lora_weights[lora_b_key]
                                        print(f"Loaded LoRA weights for {name}: lora_A={lora_a_shape}, lora_B={lora_b_shape}")
                                    else:
                                        print(f"Shape mismatch for {name}: expected lora_A={expected_a_shape}, lora_B={expected_b_shape}, "
                                              f"got lora_A={lora_a_shape}, lora_B={lora_b_shape}. Reinitializing LoRA.")
                                else:
                                    print(f"LoRA weights for {name} not found in {lora_weights_path}. Reinitializing LoRA.")
                            else:
                                print(f"No LoRA weights found at {lora_weights_path}. Reinitializing LoRA.")
                        setattr(parent, child_name, new_module)
                        lora_params.append(new_module)
                        negation.append(name)
        return lora_params, negation

    def save_lora_weights(self, model: nn.Module, save_path: str, step: int):
        os.makedirs(save_path, exist_ok=True)
        lora_weights = {}
        for name, module in model.named_modules():
            if isinstance(module, LoRAParametrization):
                lora_weights[f"{name}.lora_A"] = module.lora_A.data
                lora_weights[f"{name}.lora_B"] = module.lora_B.data
        torch.save(lora_weights, os.path.join(save_path, f"pytorch_lora_weights.safetensors"))
        print(f"Saved LoRA weights to {save_path}/pytorch_lora_weights.safetensors at step {step}")
