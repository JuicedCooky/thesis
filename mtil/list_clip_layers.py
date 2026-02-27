"""
list_clip_layers.py
Lists all layers (modules and parameters) in a CLIP model.

Run from the mtil/ directory:
    python list_clip_layers.py [--model ViT-B/32] [--params] [--linear-only]
"""

import argparse
import sys
import torch
import torch.nn as nn

sys.path.insert(0, ".")
import clip.clip as clip


def list_layers(model: nn.Module, params: bool = False, linear_only: bool = False):
    print("\n=== MODULES ===")
    for name, module in model.named_modules():
        if not name:
            continue
        if linear_only and not isinstance(module, nn.Linear):
            continue
        trainable = any(p.requires_grad for p in module.parameters(recurse=False))
        flag = " [trainable]" if trainable else ""
        print(f"  {name:<70} {type(module).__name__}{flag}")

    if params:
        print("\n=== PARAMETERS ===")
        total = 0
        for name, p in model.named_parameters():
            req = "grad" if p.requires_grad else "frozen"
            print(f"  {name:<70} {str(tuple(p.shape)):<25} {req}")
            total += p.numel()
        print(f"\nTotal parameters: {total:,}")


def main():
    parser = argparse.ArgumentParser(description="List all layers in a CLIP model.")
    parser.add_argument("--model", default="ViT-B/32",
                        choices=["RN50", "RN101", "RN50x4", "RN50x16", "ViT-B/32", "ViT-B/16"],
                        help="CLIP model variant (default: ViT-B/32)")
    parser.add_argument("--params", action="store_true",
                        help="Also list all named parameters with shapes")
    parser.add_argument("--linear-only", action="store_true",
                        help="Show only nn.Linear layers (useful for LoRA targeting)")
    args = parser.parse_args()

    print(f"Loading CLIP model: {args.model}")
    model, _, _ = clip.load(args.model, jit=False)
    model.eval()

    list_layers(model, params=args.params, linear_only=args.linear_only)


if __name__ == "__main__":
    main()
