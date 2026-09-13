"""gpt-oss MXFP4 experts, dequantized one expert at a time.

The checkpoint stores each layer's experts packed: ``gate_up_proj_blocks`` is
[128, 5760, 90, 16] uint8 with a matching ``_scales``, 1.63 GiB per layer. Two
facts force the design:

* Upcasting the checkpoint to bf16 would take it from 61 GiB to roughly 230 GiB,
  past the free space on a 512 GB drive.
* Dequantizing one *layer* produces 5.93 GiB of bf16 experts (measured),
  which does not fit beside anything else on an 8 GB card.

So the packed tensors stream verbatim as the block's state and each expert is
dequantized inside the loop, used, and dropped -- 47 MiB live at a time.
``functional_call`` passes the packed tensors through without inspecting them,
so the pool needs no MXFP4 awareness.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def dequantize_expert(
    blocks: Tensor, scales: Tensor, dtype: torch.dtype = torch.bfloat16
) -> Tensor:
    """Dequantize one expert's packed MXFP4 weight to a dense matrix.

    ``blocks`` is [1, rows, groups, 16] uint8 and ``scales`` [1, rows, groups]
    -- the leading expert dimension must be kept, because the converter indexes
    ``transpose(1, 2)`` and would fail or mis-orient without it. It already
    returns [1, groups * 32, rows], the orientation the dense path multiplies
    with, so no further transpose is applied here.
    """
    from transformers.integrations.mxfp4 import convert_moe_packed_tensors

    if blocks.dim() != 4:
        raise ValueError(
            f"Expected [1, rows, groups, 16] packed blocks, got {tuple(blocks.shape)}"
        )
    dense = convert_moe_packed_tensors(blocks, scales, dtype=dtype)
    return dense.squeeze(0)


class _MXFP4Matmul(torch.autograd.Function):
    """x @ dequantize(blocks, scales), re-dequantizing in backward.

    Holding the dense weight for the backward pass would defeat per-expert
    dequantization entirely: autograd needs W to compute dL/dx, so every
    expert touched in a block would stay live at 47 MiB. With 128 experts that
    is ~6 GiB and the card OOMs during recompute even though forward fit.

    Saving the packed tensors instead costs nothing -- they are the block's
    staged state and already resident -- and the dense weight becomes a
    temporary in both directions.
    """

    @staticmethod
    def forward(ctx, x: Tensor, blocks: Tensor, scales: Tensor) -> Tensor:
        weight = dequantize_expert(blocks, scales, x.dtype)
        ctx.save_for_backward(x, blocks, scales)
        return x @ weight

    @staticmethod
    def backward(ctx, grad: Tensor) -> tuple[Tensor | None, None, None]:
        x, blocks, scales = ctx.saved_tensors
        grad_x = None
        if ctx.needs_input_grad[0]:
            weight = dequantize_expert(blocks, scales, x.dtype)
            grad_x = grad @ weight.transpose(-2, -1)
        return grad_x, None, None


def mxfp4_matmul(x: Tensor, blocks: Tensor, scales: Tensor) -> Tensor:
    return _MXFP4Matmul.apply(x, blocks, scales)


class MXFP4Experts(nn.Module):
    """Drop-in for ``GptOssExperts`` whose weights stay packed until used.

    Parameter names match the checkpoint exactly, so the shard manifest needs
    no renaming and the streamed state binds straight through.
    """

    def __init__(self, config, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.num_experts = config.num_local_experts
        self.hidden_size = config.hidden_size
        self.expert_dim = config.intermediate_size
        self.alpha = 1.702
        self.limit = getattr(config, "swiglu_limit", 7.0)
        groups = self.hidden_size // 32

        def packed(rows: int) -> nn.Parameter:
            return nn.Parameter(
                torch.empty(
                    self.num_experts,
                    rows,
                    groups,
                    16,
                    dtype=torch.uint8,
                    device="meta",
                ),
                requires_grad=False,
            )

        def scale(rows: int) -> nn.Parameter:
            return nn.Parameter(
                torch.empty(
                    self.num_experts, rows, groups, dtype=torch.uint8, device="meta"
                ),
                requires_grad=False,
            )

        self.gate_up_proj_blocks = packed(2 * self.expert_dim)
        self.gate_up_proj_scales = scale(2 * self.expert_dim)
        self.gate_up_proj_bias = nn.Parameter(
            torch.empty(
                self.num_experts, 2 * self.expert_dim, dtype=dtype, device="meta"
            ),
            requires_grad=False,
        )
        self.down_proj_blocks = packed(self.hidden_size)
        self.down_proj_scales = scale(self.hidden_size)
        self.down_proj_bias = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size, dtype=dtype, device="meta"),
            requires_grad=False,
        )

    def _apply_gate(self, gate_up: Tensor) -> Tensor:
        gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        gate = gate.clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        glu = gate * torch.sigmoid(gate * self.alpha)
        return (up + 1) * glu

    def forward(
        self,
        hidden_states: Tensor,
        router_indices: Tensor | None = None,
        routing_weights: Tensor | None = None,
    ) -> Tensor:
        next_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = torch.nn.functional.one_hot(
                router_indices, num_classes=self.num_experts
            ).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for entry in hit:
            expert = entry[0]
            if expert == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(mask[expert])
            current = hidden_states[token_idx]
            # Dequantized weights are temporaries in both directions: the
            # autograd Function re-derives them in backward from the packed
            # tensors rather than keeping 47 MiB per expert alive.
            gate_up = (
                mxfp4_matmul(
                    current,
                    self.gate_up_proj_blocks[expert : expert + 1],
                    self.gate_up_proj_scales[expert : expert + 1],
                )
                + self.gate_up_proj_bias[expert]
            )
            gated = self._apply_gate(gate_up)
            out = (
                mxfp4_matmul(
                    gated,
                    self.down_proj_blocks[expert : expert + 1],
                    self.down_proj_scales[expert : expert + 1],
                )
                + self.down_proj_bias[expert]
            )
            out = out * routing_weights[token_idx, top_k_pos, None]
            next_states.index_add_(0, token_idx, out.to(hidden_states.dtype))
        return next_states


def install_mxfp4_experts(model: nn.Module, config, dtype=torch.bfloat16) -> int:
    """Swap every dense GptOssExperts for the packed, per-expert version."""
    from transformers.models.gpt_oss import modeling_gpt_oss as gpt_oss

    replaced = 0
    for layer in model.model.layers:
        if isinstance(layer.mlp.experts, gpt_oss.GptOssExperts):
            layer.mlp.experts = MXFP4Experts(config, dtype)
            replaced += 1
    return replaced
