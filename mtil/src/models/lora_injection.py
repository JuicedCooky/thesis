import torch
import torch.nn as nn
import math


class SharedLoRALayer(nn.Module):
    """
    Replaces a linear base_layer with a LoRA-adapted version.
    shared_A and shared_B are nn.Parameter tensors shared across layers.
    If either is None, a local (non-shared) parameter is created instead.
    The original base_layer weights are frozen.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        alpha: float,
        shared_A: nn.Parameter = None,
        shared_B: nn.Parameter = None,
    ):
        super().__init__()

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # Freeze and store the original layer
        for param in base_layer.parameters():
            param.requires_grad = False
        self.base_layer = base_layer

        self.rank = rank
        self.scaling = alpha / rank

        # A: (rank, in_features)
        if shared_A is not None:
            assert shared_A.shape == (rank, in_features), (
                f"shared_A shape mismatch: expected ({rank}, {in_features}), "
                f"got {tuple(shared_A.shape)}"
            )
            self.lora_A = shared_A
            self._owns_A = False
        else:
            self.lora_A = nn.Parameter(torch.empty(rank, in_features))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self._owns_A = True

        # B: (out_features, rank)
        if shared_B is not None:
            assert shared_B.shape == (out_features, rank), (
                f"shared_B shape mismatch: expected ({out_features}, {rank}), "
                f"got {tuple(shared_B.shape)}"
            )
            self.lora_B = shared_B
            self._owns_B = False
        else:
            self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
            self._owns_B = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out

    def merge(self) -> nn.Linear:
        """
        Merge the LoRA delta into the base weight and return a plain nn.Linear.

        Merged weight: W_merged = W_base + (B @ A) * scaling
          - lora_A: (rank, in_features)
          - lora_B: (out_features, rank)
          - B @ A:  (out_features, in_features)  — same shape as W_base
        """
        base = self.base_layer
        delta = (self.lora_B @ self.lora_A) * self.scaling          # (out, in)
        merged_weight = base.weight.data + delta.to(base.weight.device)

        merged = nn.Linear(
            base.in_features,
            base.out_features,
            bias=base.bias is not None,
            device=base.weight.device,
            dtype=base.weight.dtype,
        )
        merged.weight = nn.Parameter(merged_weight)
        if base.bias is not None:
            merged.bias = nn.Parameter(base.bias.data.clone())

        return merged

    def extra_repr(self) -> str:
        return (
            f"rank={self.rank}, scaling={self.scaling}, "
            f"shared_A={not self._owns_A}, shared_B={not self._owns_B}"
        )


def merge_and_unload_shared_lora(model: nn.Module) -> nn.Module:
    """
    Walk the model and replace every SharedLoRALayer with a merged plain nn.Linear.

    Mirrors peft's model.merge_and_unload():
      - Computes W_merged = W_base + (B @ A) * scaling for each LoRA layer
      - Replaces the SharedLoRALayer in-place with the resulting nn.Linear
      - Returns the same model object (mutation in-place, like peft)

    After this call the model contains no LoRA wrappers and can be saved /
    loaded without any dependency on lora_injection.py.
    """
    # Collect (parent_module, attr_name, lora_layer) to avoid mutating during traversal
    targets = []
    for name, module in model.named_modules():
        if isinstance(module, SharedLoRALayer):
            parts = name.rsplit(".", 1)
            parent = model
            if len(parts) == 2:
                for part in parts[0].split("."):
                    parent = getattr(parent, part)
            targets.append((parent, parts[-1], module))

    for parent, attr, lora_layer in targets:
        merged_linear = lora_layer.merge()
        setattr(parent, attr, merged_linear)

    return model


def inject_shared_lora(model, target_modules, rank=8, alpha=16.0, shared_A=True, shared_B=False):
    """
    Walk model and replace named Linear layers in target_modules with SharedLoRALayer.

    shared_A / shared_B can be:
      - True  → create one shared nn.Parameter for that matrix, reused across all replaced layers
      - False → each replaced layer gets its own local parameter
      - nn.Parameter → use the supplied parameter directly
    """

    # Collect target (parent, attr_name, child) triples first to avoid mutation during iteration
    targets = []
    for name, child in model.named_modules():
        if any(name.endswith(t) or name == t for t in target_modules):
            if isinstance(child, nn.Linear):
                parts = name.rsplit(".", 1)
                parent = model
                if len(parts) == 2:
                    for part in parts[0].split("."):
                        parent = getattr(parent, part)
                attr = parts[-1]
                targets.append((parent, attr, child))

    if not targets:
        return model

    # Build shared parameters once (shape derived from first matching layer)
    first = targets[0][2]
    shared_A_param = None
    shared_B_param = None

    if shared_A is True:
        shared_A_param = nn.Parameter(torch.empty(rank, first.in_features))
        nn.init.kaiming_uniform_(shared_A_param, a=math.sqrt(5))
    elif isinstance(shared_A, nn.Parameter):
        shared_A_param = shared_A

    if shared_B is True:
        shared_B_param = nn.Parameter(torch.zeros(first.out_features, rank))
    elif isinstance(shared_B, nn.Parameter):
        shared_B_param = shared_B

    for parent, attr, child in targets:
        lora_layer = SharedLoRALayer(
            base_layer=child,
            rank=rank,
            alpha=alpha,
            shared_A=shared_A_param,
            shared_B=shared_B_param,
        )
        setattr(parent, attr, lora_layer)

    return model
