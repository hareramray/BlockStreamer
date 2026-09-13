"""Per-family adapters mapping real checkpoints onto streamed blocks.

``StreamedModel._execute`` hands the same positional and keyword arguments to
every block, so a vision tower and a decoder layer cannot both be blocks
unless they accept a common signature. Each adapter therefore declares:

* which tensors form which block (and how many ways to sub-split a layer),
* which modules stay resident on GPU rather than streaming,
* the union of keyword arguments every block must tolerate,
* which projections receive LoRA adapters.

Vision towers stay resident: they are small next to the decoder and streaming
them complicates argument plumbing for no meaningful saving.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch
from torch import Tensor, nn

LAYER_RE = re.compile(r"\blayers\.(\d+)\.")
EXPERT_RE = re.compile(r"\bexperts\.(\d+)\.")
CHUNK_RE = re.compile(r"\.chunk(\d+)$")
FUSED_EXPERT_RE = re.compile(
    r"\bexperts\.(gate_up_proj|down_proj|gate_proj|up_proj)(\.weight)?$"
)


def chunk_fused_experts(state: dict[str, Tensor], splits: int) -> dict[str, Tensor]:
    """Slice fused ``[num_experts, ...]`` tensors into per-block chunks.

    Qwen3-VL-MoE stores a whole layer's experts as one tensor -- 3 GiB for
    ``experts.gate_up_proj`` at 128 experts -- which no name-based rule can
    divide. Slicing the expert dimension is the only way such a layer fits an
    8 GB card. The block consuming these chunks must declare matching
    parameter names and concatenate or iterate over them.
    """
    if splits <= 1:
        return state
    out: dict[str, Tensor] = {}
    for name, tensor in state.items():
        if FUSED_EXPERT_RE.search(name) and tensor.dim() >= 2:
            experts = tensor.shape[0]
            if experts % splits:
                raise ValueError(
                    f"{name}: {experts} experts is not divisible by {splits}"
                )
            for index, piece in enumerate(tensor.chunk(splits, dim=0)):
                out[f"{name}.chunk{index}"] = piece.contiguous()
        else:
            out[name] = tensor
    return out


class ModelAdapter(Protocol):
    """The contract a model family must satisfy to be streamed."""

    name: str
    splits: int

    def assign(self, tensor_name: str) -> int | None:
        """Block index for a tensor, or None to keep it resident."""
        ...

    def block_kwargs(self) -> frozenset[str]:
        """Keyword arguments every block must accept, needed or not."""
        ...

    def resident_patterns(self) -> tuple[str, ...]:
        """Regexes for state kept on GPU instead of streamed."""
        ...

    def lora_targets(self, layer_index: int) -> tuple[str, ...]:
        """Projection suffixes to adapt in this layer."""
        ...


@dataclass
class BaseAdapter:
    """Shared assignment logic: layer index, optional expert-aware split."""

    name: str = "base"
    splits: int = 1
    resident: tuple[str, ...] = (
        r"\bvisual\b",
        r"\bvision_tower\b",
        r"\bvision_model\b",
        r"embed_tokens",
        r"\bmodel\.norm\b",
        r"\blm_head\b",
    )
    targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    kwargs: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "attention_mask",
                "position_ids",
                "position_embeddings",
                "past_key_values",
                "cache_position",
                "deepstack_features",
                "image_grid_thw",
                "use_cache",
                "output_attentions",
            }
        )
    )

    def is_resident(self, tensor_name: str) -> bool:
        return any(re.search(pattern, tensor_name) for pattern in self.resident)

    def split_index(self, tensor_name: str) -> int:
        """Spread a layer's experts evenly across its sub-blocks."""
        if self.splits == 1:
            return 0
        chunk = CHUNK_RE.search(tensor_name)
        if chunk is not None:
            # Fused expert tensors are pre-sliced by chunk_fused_experts.
            return int(chunk.group(1)) % self.splits
        match = EXPERT_RE.search(tensor_name)
        if match is None:
            # Attention and router weights ride in the first sub-block.
            return 0
        return int(match.group(1)) % self.splits

    def assign(self, tensor_name: str) -> int | None:
        if self.is_resident(tensor_name):
            return None
        match = LAYER_RE.search(tensor_name)
        if match is None:
            return None
        return int(match.group(1)) * self.splits + self.split_index(tensor_name)

    def block_kwargs(self) -> frozenset[str]:
        return self.kwargs

    def resident_patterns(self) -> tuple[str, ...]:
        return self.resident

    def lora_targets(self, layer_index: int) -> tuple[str, ...]:
        return self.targets


@dataclass
class GptOssAdapter(BaseAdapter):
    """gpt-oss: MoE with MXFP4 experts stored as packed uint8 plus scales.

    The packed blocks are streamed verbatim and dequantized inside the block's
    forward; ``functional_call`` passes tensors through without inspecting
    them, so the pool needs no MXFP4 awareness. Upcasting the checkpoint to
    bf16 would take it from 61 GiB to roughly 240 GiB -- do not.
    """

    name: str = "gpt-oss"
    splits: int = 2
    targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")


@dataclass
class QwenVLAdapter(BaseAdapter):
    """Qwen3-VL / Qwen2.5-VL, dense and MoE.

    Native dynamic resolution means the image-token count varies per sample,
    so activation buffers must be sized per step rather than preallocated.
    DeepStack features cross block boundaries and must travel as explicit
    block kwargs, never as module state, which the streaming contract forbids.
    """

    name: str = "qwen-vl"
    splits: int = 1
    targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")


@dataclass
class Glm4VAdapter(BaseAdapter):
    """GLM-4.5V / GLM-4.6V: 106B-A12B MoE, 128 routed experts, top-8.

    Routed experts are ~92% of the checkpoint, so the expert-group split is
    both the VRAM fix and the natural unit for expert-major execution.
    """

    name: str = "glm-4v"
    splits: int = 2
    targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")


ADAPTERS: dict[str, type[BaseAdapter]] = {
    "gpt_oss": GptOssAdapter,
    "qwen3_vl": QwenVLAdapter,
    "qwen3_vl_moe": QwenVLAdapter,
    "qwen2_5_vl": QwenVLAdapter,
    "glm4v_moe": Glm4VAdapter,
    "glm4v": Glm4VAdapter,
}


def for_config(config: dict[str, Any], **overrides: Any) -> BaseAdapter:
    """Pick an adapter from a HF config's model_type."""
    model_type = config.get("model_type", "")
    text = config.get("text_config", {})
    candidates = [model_type, text.get("model_type", "")]
    for candidate in candidates:
        for key, cls in ADAPTERS.items():
            if candidate.startswith(key):
                return cls(**overrides)
    raise KeyError(f"No adapter registered for model_type {candidates}")


class UniformBlock(nn.Module):
    """Wrap a block so it tolerates the family's whole kwarg union.

    Every streamed block receives the same arguments, so blocks that do not
    use a given keyword must accept and ignore it rather than raising.
    """

    def __init__(self, inner: nn.Module, accepts: frozenset[str]) -> None:
        super().__init__()
        self.inner = inner
        self.accepts = accepts

    def forward(self, hidden: Tensor, *args: Any, **kwargs: Any) -> Any:
        import inspect

        signature = inspect.signature(self.inner.forward)
        allowed = set(signature.parameters)
        takes_var = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if takes_var:
            filtered = kwargs
        else:
            filtered = {k: v for k, v in kwargs.items() if k in allowed}
        return self.inner(hidden, *args, **filtered)


class ExpertMajorMoE(nn.Module):
    """Iterate experts outermost so each expert weight is read exactly once.

    Token-major routing would revisit an expert per token. Gathering the tokens
    routed to expert *e*, computing, then moving to *e+1* makes the read cost
    independent of batch size. Selective expert skipping is deliberately absent:
    at any useful token count essentially every expert is hit anyway, so it adds
    complexity without reducing traffic.
    """

    def __init__(self, experts: nn.ModuleList, top_k: int) -> None:
        super().__init__()
        self.experts = experts
        self.top_k = top_k

    def forward(self, hidden: Tensor, router_logits: Tensor) -> Tensor:
        flat = hidden.reshape(-1, hidden.shape[-1])
        weights, indices = torch.topk(
            torch.softmax(router_logits.float(), dim=-1), self.top_k, dim=-1
        )
        weights = weights / weights.sum(dim=-1, keepdim=True)
        output = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            hits = (indices == expert_id).nonzero(as_tuple=False)
            if hits.numel() == 0:
                continue
            tokens = hits[:, 0]
            scale = weights[tokens, hits[:, 1]].unsqueeze(-1).to(flat.dtype)
            output.index_add_(0, tokens, expert(flat[tokens]) * scale)
        return output.view_as(hidden)


def describe_blocks(
    tensors: dict[str, tuple[str, list[int], int]], adapter: BaseAdapter
) -> dict[str, Any]:
    """Apply an adapter to a real manifest and report the resulting layout."""
    blocks: dict[int, int] = {}
    resident = 0
    for name, (_, _, nbytes) in tensors.items():
        index = adapter.assign(name)
        if index is None:
            resident += nbytes
        else:
            blocks[index] = blocks.get(index, 0) + nbytes
    return {
        "blocks": len(blocks),
        "largest_block_bytes": max(blocks.values(), default=0),
        "streamed_bytes": sum(blocks.values()),
        "resident_bytes": resident,
    }
