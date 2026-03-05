import torch
import torch.nn as nn
import loratorch as lora


def validate_target_modules(model: nn.Module, target_modules: list) -> None:
    """Raise ValueError if any target module is not found as a Linear or MultiheadAttention layer."""
    INJECTABLE = (nn.Linear, nn.MultiheadAttention)
    matched = set()
    for name, module in model.named_modules():
        if isinstance(module, INJECTABLE):
            for t in target_modules:
                if name.endswith(t) or name == t:
                    matched.add(t)

    unmatched = [t for t in target_modules if t not in matched]
    if unmatched:
        available = [name for name, m in model.named_modules() if isinstance(m, INJECTABLE)]
        raise ValueError(
            f"[LoRA] The following target_modules were not found as injectable layers: {unmatched}\n"
            f"Available Linear/MultiheadAttention layers:\n  " + "\n  ".join(available)
        )


def _make_lora_linear(child: nn.Linear, rank: int, alpha: float) -> lora.Linear:
    lora_layer = lora.Linear(
        in_features=child.in_features,
        out_features=child.out_features,
        r=rank,
        lora_alpha=alpha,
        bias=child.bias is not None,
    )
    with torch.no_grad():
        lora_layer.weight.copy_(child.weight)
        if child.bias is not None:
            lora_layer.bias.copy_(child.bias)
    return lora_layer


def _make_lora_mha(
    child: nn.MultiheadAttention, rank: int, alpha: float, enable_lora: list
) -> lora.MultiheadAttention:
    lora_layer = lora.MultiheadAttention(
        embed_dim=child.embed_dim,
        num_heads=child.num_heads,
        r=rank,
        lora_alpha=alpha,
        enable_lora=enable_lora,
        dropout=child.dropout,
        bias=child.in_proj_bias is not None,
        batch_first=getattr(child, "batch_first", False),
    )
    # Copy base weights; strict=False since lora_ params won't be in child's state dict
    lora_layer.load_state_dict(child.state_dict(), strict=False)
    return lora_layer


def inject_lora(
    model: nn.Module,
    target_modules: list,
    rank: int = 8,
    alpha: float = 16.0,
    enable_lora: list = None,
) -> nn.Module:
    """
    Walk model and replace target Linear/MultiheadAttention layers with LoRA-Torch equivalents.

    For attention layers target the full "attn" module (nn.MultiheadAttention), not sub-layers
    like "attn.out_proj", because PyTorch's MHA forward bypasses the out_proj.forward() call.

    After injection, call lora.mark_only_lora_as_trainable(model) to freeze non-LoRA params.

    Args:
        model: The model to inject LoRA into.
        target_modules: Module name suffixes to replace (e.g. ["attn", "mlp.c_fc", "mlp.c_proj"]).
        rank: LoRA rank r.
        alpha: LoRA alpha (scaling = alpha / rank).
        enable_lora: For MultiheadAttention, which projections to adapt. Default: ['q', 'k', 'v', 'o'].
    """
    if enable_lora is None:
        enable_lora = ["q", "k", "v", "o"]

    validate_target_modules(model, target_modules)

    # Collect targets before mutating to avoid iteration issues
    targets = []
    for name, child in model.named_modules():
        if any(name.endswith(t) or name == t for t in target_modules):
            if isinstance(child, (nn.Linear, nn.MultiheadAttention)):
                parts = name.rsplit(".", 1)
                parent = model
                if len(parts) == 2:
                    for part in parts[0].split("."):
                        parent = getattr(parent, part)
                targets.append((parent, parts[-1], child, name))

    print(f"[LoRA] Injecting into {len(targets)} layer(s):")
    for _, _, child, name in targets:
        if isinstance(child, nn.MultiheadAttention):
            print(f"  {name}  (MultiheadAttention, embed_dim={child.embed_dim}, heads={child.num_heads})")
        else:
            print(f"  {name}  (Linear, in={child.in_features}, out={child.out_features})")

    for parent, attr, child, _ in targets:
        if isinstance(child, nn.MultiheadAttention):
            lora_layer = _make_lora_mha(child, rank, alpha, enable_lora)
        else:
            lora_layer = _make_lora_linear(child, rank, alpha)
        setattr(parent, attr, lora_layer)

    return model


def merge_and_unload_lora(model: nn.Module) -> nn.Module:
    """
    Permanently merge LoRA deltas into base weights and replace with plain layers.

    After this call the model has no LoRA wrappers and can be saved/loaded
    without any dependency on loratorch.
    """
    targets = []
    for name, module in model.named_modules():
        if isinstance(module, (lora.Linear, lora.MultiheadAttention)):
            parts = name.rsplit(".", 1)
            parent = model
            if len(parts) == 2:
                for part in parts[0].split("."):
                    parent = getattr(parent, part)
            targets.append((parent, parts[-1], module))

    for parent, attr, module in targets:
        # eval() triggers lora_train(False) which calls add_lora_data(),
        # permanently baking the LoRA delta into the base weight tensors.
        module.eval()

        if isinstance(module, lora.Linear):
            plain = nn.Linear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                device=module.weight.device,
                dtype=module.weight.dtype,
            )
            with torch.no_grad():
                plain.weight.copy_(module.weight)
                if module.bias is not None:
                    plain.bias.copy_(module.bias)

        elif isinstance(module, lora.MultiheadAttention):
            plain = nn.MultiheadAttention(
                embed_dim=module.embed_dim,
                num_heads=module.num_heads,
                dropout=module.dropout,
                bias=module.in_proj_bias is not None,
                batch_first=getattr(module, "batch_first", False),
            )
            # Load the merged state dict, skipping lora_ parameters
            merged_state = {
                k: v for k, v in module.state_dict().items()
                if not k.startswith("lora_")
            }
            plain.load_state_dict(merged_state, strict=False)

        setattr(parent, attr, plain)

    return model
