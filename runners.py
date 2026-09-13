"""End-to-end drivers that wrap StreamedModel in a real model's scaffolding.

A streamed block list is only the decoder stack. Something must still embed
tokens, build rotary position embeddings, run the vision tower, and apply the
final norm and head. Those parts are small, are touched once per step, and are
frozen under LoRA, so they stay resident on GPU while the decoder streams.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from lora import LoRAConfig, LoRALinear, apply_to_blocks, freeze_base
from shards import ShardManifest, convert_modules
from streamer import StreamedModel


@dataclass
class RunnerStats:
    resident_bytes: int = 0
    streamed_bytes: int = 0
    blocks: int = 0
    adapters: int = 0
    trainable: int = 0
    adapted_layers: list[int] = field(default_factory=list)


def materialize_adapters(module: nn.Module, device: str | torch.device = "cpu") -> None:
    """Give LoRA tensors real storage; base weights stay on meta."""
    for child in module.modules():
        if isinstance(child, LoRALinear):
            child.lora_A.data = torch.empty(
                child.lora_A.shape, dtype=child.lora_A.dtype, device=device
            )
            child.lora_B.data = torch.zeros(
                child.lora_B.shape, dtype=child.lora_B.dtype, device=device
            )
            child.reset_adapters()


def export_decoder_shards(
    checkpoint_dir: str | Path,
    out_dir: str | Path,
    num_layers: int,
    prefix: str,
    dtype: torch.dtype | None = None,
) -> ShardManifest:
    """Write one shard per decoder layer, reading the checkpoint lazily.

    Tensors are pulled a layer at a time from the safetensors files, so peak
    host memory is one layer rather than one checkpoint.
    """
    from safetensors import safe_open

    checkpoint_dir = Path(checkpoint_dir)
    files = sorted(checkpoint_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors under {checkpoint_dir}")

    index: dict[str, Path] = {}
    for path in files:
        with safe_open(path, framework="pt") as handle:
            for name in handle.keys():  # noqa: SIM118
                index[name] = path

    def layer_state(layer: int) -> dict[str, Tensor]:
        wanted = f"{prefix}.{layer}."
        state: dict[str, Tensor] = {}
        by_file: dict[Path, list[str]] = {}
        for name, path in index.items():
            if name.startswith(wanted):
                by_file.setdefault(path, []).append(name)
        for path, names in by_file.items():
            with safe_open(path, framework="pt") as handle:
                for name in names:
                    tensor = handle.get_tensor(name)
                    if dtype is not None and tensor.is_floating_point():
                        tensor = tensor.to(dtype)
                    state[name[len(wanted) :]] = tensor
        return state

    def blocks():
        for layer in range(num_layers):
            state = layer_state(layer)
            if not state:
                raise KeyError(f"No tensors found for layer {layer} under {prefix!r}")
            yield layer, state

    return convert_modules(blocks(), out_dir)


def load_resident(
    module: nn.Module,
    checkpoint_dir: str | Path,
    prefix: str,
    device: str | torch.device,
    dtype: torch.dtype | None = None,
) -> int:
    """Materialize one resident submodule from the checkpoint. Returns bytes."""
    from safetensors import safe_open

    checkpoint_dir = Path(checkpoint_dir)
    wanted = dict(module.named_parameters())
    wanted.update(dict(module.named_buffers()))
    loaded: dict[str, Tensor] = {}
    for path in sorted(checkpoint_dir.glob("*.safetensors")):
        with safe_open(path, framework="pt") as handle:
            for name in handle.keys():  # noqa: SIM118
                if not name.startswith(prefix):
                    continue
                short = name[len(prefix) :]
                if short in wanted:
                    tensor = handle.get_tensor(name)
                    if dtype is not None and tensor.is_floating_point():
                        tensor = tensor.to(dtype)
                    loaded[short] = tensor.to(device)
    missing = set(wanted) - set(loaded)
    if missing:
        raise KeyError(
            f"Resident module missing {len(missing)} tensors: {sorted(missing)[:3]}"
        )
    module.load_state_dict(loaded, assign=True)
    return sum(t.numel() * t.element_size() for t in loaded.values())


class Qwen3VLRunner(nn.Module):
    """Qwen3-VL with a streamed decoder and a resident head, vision, and embed."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        shard_dir: str | Path,
        lora: LoRAConfig,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        prefetch_ahead: int = 2,
        with_vision: bool = False,
    ) -> None:
        super().__init__()
        import warnings

        warnings.filterwarnings("ignore")
        from transformers import AutoConfig, AutoModelForImageTextToText

        self.device_ = torch.device(device)
        self.dtype = dtype
        self.stats = RunnerStats()
        config = AutoConfig.from_pretrained(checkpoint_dir)
        with torch.device("meta"):
            skeleton = AutoModelForImageTextToText.from_config(config, dtype=dtype)

        language = skeleton.model.language_model
        self.num_layers = len(language.layers)
        text_prefix = "model.language_model.layers"

        shard_dir = Path(shard_dir)
        if not (shard_dir / "manifest.json").exists():
            export_decoder_shards(
                checkpoint_dir, shard_dir, self.num_layers, text_prefix, dtype
            )
        manifest = ShardManifest.load(shard_dir)
        self.stats.streamed_bytes = manifest.total_bytes
        self.stats.blocks = len(manifest.blocks)

        blocks = nn.ModuleList(list(language.layers))
        indices, adapters = apply_to_blocks(blocks, lora)
        materialize_adapters(blocks)
        trainable, _ = freeze_base(blocks)
        self.stats.adapters = adapters
        self.stats.adapted_layers = indices
        self.stats.trainable = trainable

        self.decoder = StreamedModel(
            blocks, device=device, prefetch_ahead=prefetch_ahead, manifest=manifest
        )

        # Resident, frozen: embeddings, rotary, final norm, head, optional vision.
        self.embed_tokens = language.embed_tokens
        self.rotary_emb = language.rotary_emb
        self.norm = language.norm
        self.lm_head = skeleton.lm_head
        bytes_resident = 0
        bytes_resident += load_resident(
            self.embed_tokens,
            checkpoint_dir,
            "model.language_model.embed_tokens.",
            self.device_,
            dtype,
        )
        bytes_resident += load_resident(
            self.norm, checkpoint_dir, "model.language_model.norm.", self.device_, dtype
        )
        bytes_resident += load_resident(
            self.lm_head, checkpoint_dir, "lm_head.", self.device_, dtype
        )
        self.rotary_emb.to_empty(device=self.device_)
        self.rotary_emb = type(self.rotary_emb)(config.text_config).to(self.device_)
        self.visual = None
        if with_vision:
            self.visual = skeleton.model.visual
            bytes_resident += load_resident(
                self.visual, checkpoint_dir, "model.visual.", self.device_, dtype
            )
        self.stats.resident_bytes = bytes_resident
        for module in (self.embed_tokens, self.norm, self.lm_head):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def trainable_parameters(self) -> list[Tensor]:
        return [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad and not parameter.is_meta
        ]

    def forward(self, input_ids: Tensor, **kwargs: Any) -> Tensor:
        hidden = self.embed_tokens(input_ids)
        batch, length = input_ids.shape
        position_ids = (
            torch.arange(length, device=self.device_)
            .unsqueeze(0)
            .expand(3, batch, length)
        )
        position_embeddings = self.rotary_emb(hidden, position_ids)
        hidden = self.decoder(
            hidden,
            position_embeddings=position_embeddings,
            attention_mask=None,
            position_ids=position_ids,
            **kwargs,
        )
        return self.lm_head(self.norm(hidden))

    def close(self) -> None:
        self.decoder.pool.close()


class GptOssRunner(nn.Module):
    """gpt-oss with a streamed MXFP4 decoder and a resident embed/head.

    Expert weights stay packed on disk and in VRAM; each expert is dequantized
    only for the tokens routed to it. LoRA goes on the attention projections,
    which the checkpoint leaves in bf16 -- adapting packed MXFP4 experts would
    mean dequantizing them to train, defeating the point.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        shard_dir: str | Path,
        lora: LoRAConfig,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        prefetch_ahead: int = 2,
    ) -> None:
        super().__init__()
        import warnings

        warnings.filterwarnings("ignore")
        from transformers import AutoConfig, AutoModelForCausalLM

        from gptoss import install_mxfp4_experts

        self.device_ = torch.device(device)
        self.dtype = dtype
        self.stats = RunnerStats()
        config = AutoConfig.from_pretrained(checkpoint_dir)
        with torch.device("meta"):
            skeleton = AutoModelForCausalLM.from_config(config, dtype=dtype)
        install_mxfp4_experts(skeleton, config, dtype)

        self.num_layers = len(skeleton.model.layers)
        shard_dir = Path(shard_dir)
        if not (shard_dir / "manifest.json").exists():
            # dtype=None: packed uint8 expert blocks must not be cast.
            export_decoder_shards(
                checkpoint_dir, shard_dir, self.num_layers, "model.layers", None
            )
        manifest = ShardManifest.load(shard_dir)
        self.stats.streamed_bytes = manifest.total_bytes
        self.stats.blocks = len(manifest.blocks)

        blocks = nn.ModuleList(list(skeleton.model.layers))
        indices, adapters = apply_to_blocks(blocks, lora)
        materialize_adapters(blocks)
        trainable, _ = freeze_base(blocks)
        self.stats.adapters = adapters
        self.stats.adapted_layers = indices
        self.stats.trainable = trainable

        self.decoder = StreamedModel(
            blocks, device=device, prefetch_ahead=prefetch_ahead, manifest=manifest
        )
        self.embed_tokens = skeleton.model.embed_tokens
        self.norm = skeleton.model.norm
        self.lm_head = skeleton.lm_head
        resident = 0
        resident += load_resident(
            self.embed_tokens,
            checkpoint_dir,
            "model.embed_tokens.",
            self.device_,
            dtype,
        )
        resident += load_resident(
            self.norm, checkpoint_dir, "model.norm.", self.device_, dtype
        )
        resident += load_resident(
            self.lm_head, checkpoint_dir, "lm_head.", self.device_, dtype
        )
        self.rotary_emb = type(skeleton.model.rotary_emb)(config).to(self.device_)
        self.stats.resident_bytes = resident
        for module in (self.embed_tokens, self.norm, self.lm_head):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def trainable_parameters(self) -> list[Tensor]:
        return [p for p in self.parameters() if p.requires_grad and not p.is_meta]

    def forward(self, input_ids: Tensor, **kwargs: Any) -> Tensor:
        hidden = self.embed_tokens(input_ids)
        length = input_ids.shape[1]
        position_ids = torch.arange(length, device=self.device_).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden, position_ids)
        hidden = self.decoder(
            hidden,
            position_embeddings=position_embeddings,
            attention_mask=None,
            position_ids=position_ids,
            **kwargs,
        )
        return self.lm_head(self.norm(hidden))

    def close(self) -> None:
        self.decoder.pool.close()
