"""Per-block Adam whose state lives on disk, applied inside backward.

An ordinary optimizer needs every gradient and every moment resident at
``step()`` time, which is what caps the RAM path near 1B parameters. Here a
block's gradients are consumed the moment they exist: read that block's moments
from disk, update on GPU, write back, discard the gradient. Host memory then
holds one block's state rather than the model's.

Writes dominate the cost -- 12 B/param every step at fp32 -- so consult
``planner.py`` before using this on a consumer SSD. It is usually the wrong
trade against LoRA, which writes nothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor

from shards import BlockSpec, ShardManifest, TensorSpec, dtype_name, plan_block


@dataclass
class AdamConfig:
    lr: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    master_dtype: torch.dtype = torch.float32
    moment_dtype: torch.dtype = torch.float32

    @property
    def bytes_per_param(self) -> int:
        sizes = (self.master_dtype, self.moment_dtype, self.moment_dtype)
        return sum(torch.empty((), dtype=d).element_size() for d in sizes)


@dataclass
class BlockState:
    """One block's optimizer state, laid out as three parallel shards."""

    master: BlockSpec
    exp_avg: BlockSpec
    exp_avg_sq: BlockSpec
    step: int = 0


@dataclass
class FusedDiskAdam:
    """Adam with master weights and moments stored per block on disk."""

    manifest: ShardManifest
    state_dir: Path
    config: AdamConfig = field(default_factory=AdamConfig)
    trainable: frozenset[str] | None = None
    device: torch.device = field(default_factory=lambda: torch.device("cuda"))
    states: dict[int, BlockState] = field(default_factory=dict)
    write_bytes: int = 0
    read_bytes: int = 0

    def __post_init__(self) -> None:
        self.state_dir = Path(self.state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def _names(self, block: BlockSpec) -> list[TensorSpec]:
        if self.trainable is None:
            return list(block.tensors)
        return [spec for spec in block.tensors if spec.name in self.trainable]

    def _plan(self, block: BlockSpec, dtype: torch.dtype, tag: str) -> BlockSpec:
        named = [
            (
                spec.name,
                torch.empty(spec.shape, dtype=dtype, device="meta"),
            )
            for spec in self._names(block)
        ]
        spec, _ = plan_block(block.index, named)
        return BlockSpec(
            block.index, f"{tag}_{block.index:05d}.bin", spec.nbytes, spec.tensors
        )

    def initialize(self, block: BlockSpec, weights: dict[str, Tensor]) -> BlockState:
        """Create the three state shards for one block, from its live weights."""
        cfg = self.config
        state = BlockState(
            master=self._plan(block, cfg.master_dtype, "master"),
            exp_avg=self._plan(block, cfg.moment_dtype, "m"),
            exp_avg_sq=self._plan(block, cfg.moment_dtype, "v"),
        )
        specs = self._names(block)
        master = {s.name: weights[s.name].detach().to(cfg.master_dtype) for s in specs}
        zeros = {s.name: torch.zeros(s.shape, dtype=cfg.moment_dtype) for s in specs}
        self._write(state.master, master)
        self._write(state.exp_avg, zeros)
        self._write(state.exp_avg_sq, zeros)
        self.states[block.index] = state
        return state

    def _write(self, spec: BlockSpec, tensors: dict[str, Tensor]) -> None:
        """One contiguous write per shard: scattered writes multiply SSD wear."""
        buffer = torch.zeros(spec.nbytes, dtype=torch.uint8)
        for tensor_spec in spec.tensors:
            flat = tensors[tensor_spec.name].detach().cpu().contiguous()
            window = buffer[
                tensor_spec.offset : tensor_spec.offset + tensor_spec.nbytes
            ]
            window.copy_(flat.view(torch.uint8).reshape(-1))
        (self.state_dir / spec.filename).write_bytes(buffer.numpy().tobytes())
        self.write_bytes += spec.nbytes

    def _read(self, spec: BlockSpec) -> dict[str, Tensor]:
        buffer = torch.empty(spec.nbytes, dtype=torch.uint8)
        with open(self.state_dir / spec.filename, "rb", buffering=0) as handle:
            handle.readinto(memoryview(buffer.numpy()))
        self.read_bytes += spec.nbytes
        return {s.name: s.view(buffer).to(self.device) for s in spec.tensors}

    def step_block(
        self,
        block: BlockSpec,
        grads: dict[str, Tensor],
        weights: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Apply Adam to one block and return its updated compute-dtype weights.

        `grads` is consumed here and never accumulated across blocks, which is
        what keeps host memory bounded by one block rather than the model.
        """
        state = self.states.get(block.index)
        if state is None:
            state = self.initialize(block, weights)
        state.step += 1
        cfg = self.config
        beta1, beta2 = cfg.betas
        bias1 = 1.0 - beta1**state.step
        bias2 = 1.0 - beta2**state.step

        master = self._read(state.master)
        exp_avg = self._read(state.exp_avg)
        exp_avg_sq = self._read(state.exp_avg_sq)

        updated: dict[str, Tensor] = {}
        for spec in self._names(block):
            name = spec.name
            grad = grads.get(name)
            if grad is None:
                continue
            grad = grad.to(cfg.master_dtype)
            param = master[name]
            if cfg.weight_decay:
                grad = grad.add(param, alpha=cfg.weight_decay)
            m = exp_avg[name].mul_(beta1).add_(grad, alpha=1 - beta1)
            v = exp_avg_sq[name].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            denom = (v / bias2).sqrt_().add_(cfg.eps)
            param.addcdiv_(m / bias1, denom, value=-cfg.lr)
            master[name] = param
            exp_avg[name] = m
            exp_avg_sq[name] = v
            updated[name] = param.to(weights[name].dtype)

        self._write(state.master, master)
        self._write(state.exp_avg, exp_avg)
        self._write(state.exp_avg_sq, exp_avg_sq)
        self.write_block_weights(block, updated)
        return updated

    def write_block_weights(self, block: BlockSpec, updated: dict[str, Tensor]) -> None:
        """Rewrite the weight shard in place so the next forward reads it."""
        if not updated:
            return
        path = self.manifest.path_for(block)
        buffer = torch.empty(block.nbytes, dtype=torch.uint8)
        with open(path, "rb", buffering=0) as handle:
            handle.readinto(memoryview(buffer.numpy()))
        for spec in block.tensors:
            if spec.name not in updated:
                continue
            flat = updated[spec.name].detach().cpu().contiguous()
            window = buffer[spec.offset : spec.offset + spec.nbytes]
            window.copy_(flat.view(torch.uint8).reshape(-1))
        path.write_bytes(buffer.numpy().tobytes())
        self.write_bytes += block.nbytes

    def projected_wear(self, steps: int) -> dict[str, float]:
        """Bytes this optimizer will write over `steps`, for the TBW gate."""
        per_step = 0
        for block in self.manifest.blocks:
            specs = self._names(block)
            params = sum(math.prod(s.shape) for s in specs)
            per_step += params * self.config.bytes_per_param + block.nbytes
        return {
            "bytes_per_step": float(per_step),
            "total_bytes": float(per_step * steps),
            "terabytes": per_step * steps / 1e12,
        }


def state_dtypes_from_name(name: str) -> tuple[torch.dtype, torch.dtype]:
    """Map a preset name to (master, moment) dtypes.

    "fp32" is 12 B/param; "8bit" halves write traffic to 6 B/param at the cost
    of moment precision. Both are reported by ``planner.py``.
    """
    presets = {
        "fp32": (torch.float32, torch.float32),
        "mixed": (torch.float32, torch.bfloat16),
        "8bit": (torch.float32, torch.int8),
        "bf16": (torch.bfloat16, torch.bfloat16),
    }
    if name not in presets:
        raise ValueError(f"Unknown optimizer state preset {name!r}")
    master, moment = presets[name]
    dtype_name(master)
    return master, moment
