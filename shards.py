"""Flat per-block shard files and the manifest describing them.

A streamed block's state is stored as one contiguous file so staging it is a
single sequential read. Tensor identity, dtype, and shape live in a JSON
manifest beside the shards, which lets blocks be constructed on ``meta`` and
materialized only when a slot stages them.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

MANIFEST_NAME = "manifest.json"

DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float64": torch.float64,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
}
DTYPE_NAMES = {value: key for key, value in DTYPES.items()}


def dtype_name(dtype: torch.dtype) -> str:
    try:
        return DTYPE_NAMES[dtype]
    except KeyError:
        raise ValueError(f"Unsupported shard dtype {dtype}") from None


@dataclass(frozen=True)
class TensorSpec:
    """One tensor's placement inside a block shard."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    offset: int
    nbytes: int

    @property
    def torch_dtype(self) -> torch.dtype:
        return DTYPES[self.dtype]

    def view(self, buffer: Tensor) -> Tensor:
        """Interpret this tensor's bytes inside a flat uint8 staging buffer."""
        window = buffer[self.offset : self.offset + self.nbytes]
        return window.view(self.torch_dtype).view(self.shape)


@dataclass(frozen=True)
class BlockSpec:
    index: int
    filename: str
    nbytes: int
    tensors: tuple[TensorSpec, ...]

    def state_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.tensors)


@dataclass
class ShardManifest:
    """Everything needed to stage any block without opening the checkpoint."""

    root: Path
    blocks: tuple[BlockSpec, ...]
    resident: tuple[TensorSpec, ...] = ()
    resident_filename: str = "resident.bin"

    @property
    def largest_block_bytes(self) -> int:
        return max((block.nbytes for block in self.blocks), default=0)

    @property
    def total_bytes(self) -> int:
        return sum(block.nbytes for block in self.blocks)

    def path_for(self, block: BlockSpec) -> Path:
        return self.root / block.filename

    def save(self) -> None:
        payload = {
            "blocks": [asdict(block) for block in self.blocks],
            "resident": [asdict(spec) for spec in self.resident],
            "resident_filename": self.resident_filename,
        }
        (self.root / MANIFEST_NAME).write_text(json.dumps(payload, indent=1))

    @classmethod
    def load(cls, root: str | Path) -> ShardManifest:
        root = Path(root)
        payload = json.loads((root / MANIFEST_NAME).read_text())
        blocks = tuple(
            BlockSpec(
                index=item["index"],
                filename=item["filename"],
                nbytes=item["nbytes"],
                tensors=tuple(
                    TensorSpec(**dict(t, shape=tuple(t["shape"])))
                    for t in item["tensors"]
                ),
            )
            for item in payload["blocks"]
        )
        resident = tuple(
            TensorSpec(**dict(t, shape=tuple(t["shape"])))
            for t in payload.get("resident", ())
        )
        return cls(
            root=root,
            blocks=blocks,
            resident=resident,
            resident_filename=payload.get("resident_filename", "resident.bin"),
        )

    def read_block(self, block: BlockSpec, out: Tensor) -> None:
        """Read one block's bytes into a flat uint8 buffer (host-side)."""
        if out.numel() < block.nbytes:
            raise ValueError(
                f"Staging buffer holds {out.numel()} bytes; "
                f"block {block.index} needs {block.nbytes}"
            )
        with open(self.path_for(block), "rb", buffering=0) as handle:
            view = out[: block.nbytes].numpy()
            got = handle.readinto(memoryview(view))
        if got != block.nbytes:
            raise OSError(
                f"Short read on block {block.index}: {got} of {block.nbytes} bytes"
            )

    def materialize(self, block: BlockSpec, buffer: Tensor) -> dict[str, Tensor]:
        return {spec.name: spec.view(buffer) for spec in block.tensors}


def plan_block(index: int, named: list[tuple[str, Tensor]]) -> tuple[BlockSpec, int]:
    """Lay tensors out back-to-back, 8-byte aligned, and return the spec."""
    specs: list[TensorSpec] = []
    offset = 0
    for name, tensor in named:
        nbytes = tensor.numel() * tensor.element_size()
        specs.append(
            TensorSpec(
                name=name,
                dtype=dtype_name(tensor.dtype),
                shape=tuple(tensor.shape),
                offset=offset,
                nbytes=nbytes,
            )
        )
        offset += nbytes
        offset += -offset % 8
    return BlockSpec(index, f"block_{index:05d}.bin", offset, tuple(specs)), offset


def write_block(path: Path, spec: BlockSpec, named: dict[str, Tensor]) -> None:
    """Write one contiguous shard. Padding between tensors is zeroed."""
    buffer = torch.zeros(spec.nbytes, dtype=torch.uint8)
    for tensor_spec in spec.tensors:
        source = named[tensor_spec.name].detach().contiguous()
        flat = source.view(torch.uint8).reshape(-1)
        window = buffer[tensor_spec.offset : tensor_spec.offset + tensor_spec.nbytes]
        window.copy_(flat)
    path.write_bytes(buffer.numpy().tobytes())


def convert_modules(
    blocks: Iterator[tuple[int, dict[str, Tensor]]],
    out_dir: str | Path,
) -> ShardManifest:
    """Write shards from an iterator of (index, state) pairs.

    The iterator is consumed lazily so the caller can yield one block at a time
    and never hold the whole checkpoint in host memory.
    """
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    specs: list[BlockSpec] = []
    for index, state in blocks:
        named = sorted(state.items())
        spec, _ = plan_block(index, named)
        write_block(root / spec.filename, spec, state)
        specs.append(spec)
    manifest = ShardManifest(root=root, blocks=tuple(specs))
    manifest.save()
    return manifest


def convert_safetensors(
    checkpoint_dir: str | Path,
    out_dir: str | Path,
    assign: Callable[[str], int | None],
    dtype: torch.dtype | None = None,
) -> ShardManifest:
    """Stream a HF safetensors checkpoint into per-block shards.

    ``assign`` maps a tensor name to its block index, or None to mark it
    resident (embeddings, vision towers, final norms). Tensors are read one at
    a time, so peak host memory is one block, not one checkpoint.
    """
    from safetensors import safe_open

    checkpoint_dir = Path(checkpoint_dir)
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    files = sorted(checkpoint_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors files under {checkpoint_dir}")

    location: dict[int, list[tuple[str, Path]]] = {}
    resident_names: list[tuple[str, Path]] = []
    for path in files:
        with safe_open(path, framework="pt") as handle:
            for name in handle.keys():  # noqa: SIM118 - safetensors handle
                index = assign(name)
                if index is None:
                    resident_names.append((name, path))
                else:
                    location.setdefault(index, []).append((name, path))

    def load(name: str, path: Path) -> Tensor:
        with safe_open(path, framework="pt") as handle:
            tensor = handle.get_tensor(name)
        if dtype is not None and tensor.is_floating_point():
            tensor = tensor.to(dtype)
        return tensor

    specs: list[BlockSpec] = []
    for index in sorted(location):
        state = {name: load(name, path) for name, path in location[index]}
        spec, _ = plan_block(index, sorted(state.items()))
        write_block(root / spec.filename, spec, state)
        specs.append(spec)
        del state

    resident_state = {name: load(name, path) for name, path in resident_names}
    resident_specs: tuple[TensorSpec, ...] = ()
    if resident_state:
        resident_spec, _ = plan_block(-1, sorted(resident_state.items()))
        write_block(root / "resident.bin", resident_spec, resident_state)
        resident_specs = resident_spec.tensors

    manifest = ShardManifest(root=root, blocks=tuple(specs), resident=resident_specs)
    manifest.save()
    return manifest


def layer_assigner(
    pattern: str = r"\blayers\.(\d+)\.",
    splits: int = 1,
    split_key: Callable[[str], int] | None = None,
) -> Callable[[str], int | None]:
    """Assign tensors to blocks by layer index, optionally sub-splitting.

    ``splits`` divides each layer into that many blocks, which is what keeps
    a 4.36 GiB MoE layer inside an 8 GB card. ``split_key`` decides which
    sub-block a tensor belongs to; it defaults to hashing the expert index so
    experts of one layer spread evenly across its sub-blocks.
    """
    import re

    regex = re.compile(pattern)
    expert = re.compile(r"\bexperts\.(\d+)\.")

    def default_key(name: str) -> int:
        match = expert.search(name)
        return int(match.group(1)) if match else 0

    key = split_key or default_key

    def assign(name: str) -> int | None:
        match = regex.search(name)
        if match is None:
            return None
        layer = int(match.group(1))
        return layer * splits + (key(name) % splits if splits > 1 else 0)

    return assign


def build_meta_state(manifest: ShardManifest) -> list[dict[str, Any]]:
    """Describe every block's state without allocating storage."""
    return [
        {
            spec.name: torch.empty(spec.shape, dtype=spec.torch_dtype, device="meta")
            for spec in block.tensors
        }
        for block in manifest.blocks
    ]
