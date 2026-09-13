"""Micro-batch looping inside one block visit, with offloaded activations.

``StreamedModel._execute`` invokes its callback once per block. Looping N
micro-batches inside that callback amortizes each block's disk read N times,
which is the largest throughput lever available when weights come from NVMe:
the read cost is paid per step, not per sample.

It is also what makes a fused optimizer step correct. All accumulation for a
block finishes inside its single visit, so the update can be applied before the
block retires -- there is no second backward pass that would need the block
back.

Saved activations move to pinned host memory, since O(depth x micro-batches)
tensors will not fit in 8 GB of VRAM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from memory_pool import tensor_bytes


@dataclass
class MicroBatchConfig:
    count: int = 1
    offload_activations: bool = True
    # Set by the caller when a fused disk optimizer should consume gradients
    # per block instead of returning them through autograd.
    fused_optimizer: Any = None

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ValueError("micro-batch count must be at least 1")


@dataclass
class Telemetry:
    activation_bytes: int = 0
    peak_activation_bytes: int = 0
    grad_bytes: int = 0
    offloaded: int = 0
    history: list[int] = field(default_factory=list)


class ActivationStore:
    """Saved block inputs, optionally held in pinned host memory."""

    def __init__(self, offload: bool, telemetry: Telemetry) -> None:
        self.offload = offload
        self.telemetry = telemetry
        self.saved: dict[tuple[int, int], Tensor] = {}

    def put(self, block: int, micro: int, tensor: Tensor) -> None:
        if self.offload and tensor.is_cuda:
            host = torch.empty(
                tensor.shape, dtype=tensor.dtype, device="cpu", pin_memory=True
            )
            host.copy_(tensor, non_blocking=True)
            stored = host
            self.telemetry.offloaded += 1
        else:
            stored = tensor
        self.saved[(block, micro)] = stored
        self.telemetry.activation_bytes += tensor_bytes(tensor)
        self.telemetry.peak_activation_bytes = max(
            self.telemetry.peak_activation_bytes,
            sum(tensor_bytes(t) for t in self.saved.values()),
        )

    def get(self, block: int, micro: int, device: torch.device) -> Tensor:
        tensor = self.saved[(block, micro)]
        return tensor.to(device, non_blocking=True) if not tensor.is_cuda else tensor

    def clear(self) -> None:
        self.saved.clear()


def split_microbatches(hidden: Tensor, count: int) -> list[Tensor]:
    """Split along the batch dimension, rejecting an uneven split."""
    if count == 1:
        return [hidden]
    if hidden.shape[0] % count:
        raise ValueError(
            f"Batch {hidden.shape[0]} is not divisible by {count} micro-batches"
        )
    return list(hidden.chunk(count, dim=0))


class _MicroBatchAutograd(torch.autograd.Function):
    """Block-major schedule: every micro-batch crosses a block while resident."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        model: Any,
        config: MicroBatchConfig,
        telemetry: Telemetry,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        hidden: Tensor,
        *adapters: Tensor,
    ) -> Tensor:
        from streamer import hidden_tensor

        ctx.set_materialize_grads(False)
        ctx.model = model
        ctx.config = config
        ctx.telemetry = telemetry
        ctx.args = args
        ctx.kwargs = kwargs
        ctx.configuration_version = model._configuration_version
        ctx.autocast = torch.is_autocast_enabled("cuda")
        ctx.autocast_dtype = torch.get_autocast_dtype("cuda")

        store = ActivationStore(config.offload_activations, telemetry)
        ctx.store = store
        current = split_microbatches(hidden, config.count)
        ctx.rng = []

        def run(index: int, state: dict[str, Tensor]) -> None:
            states = []
            for micro, value in enumerate(current):
                store.put(index, micro, value)
                states.append(
                    (torch.get_rng_state(), torch.cuda.get_rng_state(model.device))
                )
                result = model._call_block(index, state, value, args, kwargs)
                current[micro] = hidden_tensor(result)
            ctx.rng.append(states)

        with torch.no_grad():
            model._execute(list(range(len(model.blocks))), run, (hidden, args, kwargs))

        ctx.save_for_backward(*adapters)
        ctx.adapter_count = len(adapters)
        ctx.needs_hidden_grad = hidden.requires_grad
        telemetry.history.append(telemetry.peak_activation_bytes)
        return torch.cat(current, dim=0) if config.count > 1 else current[0]

    @staticmethod
    def backward(ctx: Any, grad_out: Tensor | None) -> tuple[Any, ...]:  # type: ignore[override]
        from streamer import hidden_tensor

        model = ctx.model
        config: MicroBatchConfig = ctx.config
        telemetry: Telemetry = ctx.telemetry
        store: ActivationStore = ctx.store
        if model._configuration_version != ctx.configuration_version:
            raise RuntimeError("Do not call to() between forward and backward")
        adapters = ctx.saved_tensors
        master_index = {id(t): i for i, t in enumerate(adapters)}
        grads: list[Tensor | None] = [None] * len(adapters)

        incoming = (
            list(grad_out.chunk(config.count, dim=0))
            if grad_out is not None and config.count > 1
            else [grad_out]
        )
        hidden_grads: list[Tensor | None] = [None] * config.count

        def run(index: int, state: dict[str, Tensor]) -> None:
            block = model.blocks[index]
            names = [
                name
                for name, parameter in block.named_parameters()
                if parameter.requires_grad and not parameter.is_meta
            ]
            parameters = dict(block.named_parameters())
            accumulated: dict[str, Tensor] = {}
            for micro in range(config.count):
                if incoming[micro] is None:
                    continue
                with (
                    torch.enable_grad(),
                    torch.random.fork_rng(devices=[model.device.index]),
                ):
                    cpu_rng, cuda_rng = ctx.rng[index][micro]
                    torch.set_rng_state(cpu_rng)
                    torch.cuda.set_rng_state(cuda_rng, model.device)
                    saved = store.get(index, micro, model.device)
                    value = saved.detach().requires_grad_(True)
                    targets = [value] + [state[name] for name in names]
                    with torch.autocast(
                        "cuda",
                        enabled=ctx.autocast,
                        dtype=ctx.autocast_dtype,
                        cache_enabled=False,
                    ):
                        result = model._call_block(
                            index, state, value, ctx.args, ctx.kwargs
                        )
                    out = hidden_tensor(result)
                    produced = torch.autograd.grad(
                        [out], targets, [incoming[micro]], allow_unused=True
                    )
                incoming[micro] = produced[0]
                for name, grad in zip(names, produced[1:], strict=True):
                    if grad is None:
                        continue
                    previous = accumulated.get(name)
                    accumulated[name] = grad if previous is None else previous + grad

            if config.fused_optimizer is not None:
                # Consume gradients now; nothing accumulates across blocks.
                spec = model.manifest.blocks[index]
                config.fused_optimizer.step_block(spec, accumulated, state)
                return
            for name, grad in accumulated.items():
                parameter = parameters[name]
                host = torch.empty_like(parameter, device="cpu", pin_memory=True)
                host.copy_(grad, non_blocking=True)
                grad.record_stream(model.compute_stream)
                telemetry.grad_bytes += tensor_bytes(grad)
                slot = master_index[id(parameter)]
                grads[slot] = host if grads[slot] is None else grads[slot] + host

        model._execute(
            list(reversed(range(len(model.blocks)))),
            run,
            (ctx.args, ctx.kwargs),
            grad=True,
        )
        store.clear()
        for micro in range(config.count):
            hidden_grads[micro] = incoming[micro]
        hidden_grad = None
        if ctx.needs_hidden_grad and all(g is not None for g in hidden_grads):
            hidden_grad = (
                torch.cat(hidden_grads, dim=0) if config.count > 1 else hidden_grads[0]
            )
        return (None, None, None, None, None, hidden_grad, *grads)


def micro_forward(
    model: Any,
    hidden: Tensor,
    config: MicroBatchConfig,
    telemetry: Telemetry | None = None,
    *args: Any,
    **kwargs: Any,
) -> tuple[Tensor, Telemetry]:
    """Run a training step with N micro-batches amortizing each block read."""
    telemetry = telemetry or Telemetry()
    adapters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and not parameter.is_meta
    ]
    output = _MicroBatchAutograd.apply(
        model, config, telemetry, args, kwargs, hidden, *adapters
    )
    return output, telemetry
