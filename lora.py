"""LoRA adapters over streamed, frozen base weights.

The base weight arrives from a shard with ``requires_grad=False``; only the
adapters are trainable, and they are small enough to stay pinned in host RAM
with an ordinary CPU optimizer. Nothing is written back to disk, so this path
costs no SSD endurance -- the reason it beats full finetuning on consumer
hardware at every size ``planner.py`` reports.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.0
    targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    # Fraction of decoder layers, counted from the top, that receive adapters.
    # Backward stops descending at the lowest adapted layer, so 0.5 halves the
    # weight re-fetch. 1.0 adapts every layer.
    top_fraction: float = 0.5

    @property
    def scale(self) -> float:
        return self.alpha / self.rank


class LoRALinear(nn.Module):
    """A linear layer whose base weight is streamed and frozen.

    ``weight``/``bias`` stay on meta at construction: the pool materializes
    them from the shard manifest at stage time. ``lora_A``/``lora_B`` are real
    tensors and are the only members with gradients.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: LoRAConfig,
        bias: bool = True,
        dtype: torch.dtype | None = None,
        base_device: str | torch.device = "meta",
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = config.scale
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=base_device, dtype=dtype),
            requires_grad=False,
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, device=base_device, dtype=dtype),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)
        self.lora_A = nn.Parameter(torch.empty(config.rank, in_features, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_features, config.rank, dtype=dtype))
        self.dropout = nn.Dropout(config.dropout) if config.dropout else nn.Identity()
        self.reset_adapters()

    def reset_adapters(self) -> None:
        # B starts at zero so the adapted model initially equals the base model.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: Tensor) -> Tensor:
        base = F.linear(x, self.weight, self.bias)
        low = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return base + low * self.scale

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"rank={self.lora_A.shape[0]}, scale={self.scale}"
        )


def adapt_linear(linear: nn.Linear, config: LoRAConfig) -> LoRALinear:
    """Replace one Linear with a LoRA one, keeping shape, dtype, and bias."""
    replacement = LoRALinear(
        linear.in_features,
        linear.out_features,
        config,
        bias=linear.bias is not None,
        dtype=linear.weight.dtype,
        base_device=linear.weight.device,
    )
    return replacement


def inject(module: nn.Module, config: LoRAConfig) -> int:
    """Swap every targeted Linear inside `module` for a LoRALinear."""
    patterns = [re.compile(rf"(^|\.){re.escape(t)}$") for t in config.targets]
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and any(p.search(name) for p in patterns):
            setattr(module, name, adapt_linear(child, config))
            count += 1
        else:
            count += inject(child, config)
    return count


def adapted_layer_indices(num_layers: int, config: LoRAConfig) -> list[int]:
    """Which decoder layers receive adapters, counted from the top."""
    if not 0.0 < config.top_fraction <= 1.0:
        raise ValueError("top_fraction must be in (0, 1]")
    start = num_layers - max(1, round(num_layers * config.top_fraction))
    return list(range(start, num_layers))


def apply_to_blocks(blocks: nn.ModuleList, config: LoRAConfig) -> tuple[list[int], int]:
    """Inject adapters into the top `top_fraction` of blocks.

    Returns the adapted indices and the adapter count, so a caller can report
    the backward re-fetch saving the cutoff buys.
    """
    indices = adapted_layer_indices(len(blocks), config)
    total = 0
    for index in indices:
        total += inject(blocks[index], config)
    return indices, total


def freeze_base(model: nn.Module) -> tuple[int, int]:
    """Freeze everything except adapters. Returns (trainable, frozen) counts."""
    trainable = frozen = 0
    for name, parameter in model.named_parameters():
        is_adapter = "lora_A" in name or "lora_B" in name
        parameter.requires_grad_(is_adapter)
        if is_adapter:
            trainable += parameter.numel()
        else:
            frozen += parameter.numel()
    return trainable, frozen


def adapter_state(model: nn.Module) -> dict[str, Tensor]:
    """Just the adapters -- what a LoRA checkpoint needs to save."""
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if "lora_A" in name or "lora_B" in name
    }


def backward_fetch_fraction(num_layers: int, config: LoRAConfig) -> float:
    """Share of streamed weights backward must re-read given the cutoff."""
    indices = adapted_layer_indices(num_layers, config)
    return (num_layers - min(indices)) / num_layers if indices else 0.0
