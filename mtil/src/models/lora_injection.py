import torch
import torch.nn as nn

def inject_shared_lora(model, target_modules, rank=8, alpha=16.0, shared_A = True, shared_B = False):
    for name, child in model.named_modules():