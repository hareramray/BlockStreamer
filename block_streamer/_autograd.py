"""Recompute one block at a time; return CPU parameter gradients to autograd."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable
from torch.utils._pytree import TreeSpec, tree_flatten, tree_unflatten

from memory_pool import tensor_bytes

if TYPE_CHECKING:
    from streamer import StreamedModel


@dataclass
class TensorTree:
    """Store structural metadata without retaining live tensor references."""

    leaves: list[Any]
    positions: list[int]
    spec: TreeSpec

    @classmethod
    def split(cls, value: Any) -> tuple[TensorTree, list[Tensor]]:
        leaves, spec = tree_flatten(value)
        positions = [i for i, leaf in enumerate(leaves) if isinstance(leaf, Tensor)]
        tensors = [leaves[i] for i in positions]
        for i in positions:
            leaves[i] = None
        return cls(leaves, positions, spec), tensors

    def merge(self, tensors: list[Tensor] | tuple[Tensor, ...]) -> Any:
        leaves = self.leaves.copy()
        for index, tensor in zip(self.positions, tensors, strict=True):
            leaves[index] = tensor
        return tree_unflatten(leaves, self.spec)


@dataclass
class Invocation:
    inputs: TensorTree
    input_count: int
    output: TensorTree | None = None


class _StreamAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        model: StreamedModel,
        call: Invocation,
        *tensors: Tensor,
    ) -> tuple[Tensor, ...]:
        from streamer import hidden_tensor

        ctx.set_materialize_grads(False)
        ctx.model = model
        ctx.call = call
        ctx.configuration_version = model._configuration_version
        ctx.training_modes = tuple(module.training for module in model.modules())
        ctx.input_count = call.input_count
        ctx.master_count = len(tensors) - call.input_count
        ctx.input_grad_flags = [tensor.requires_grad for tensor in tensors]
        ctx.autocast = torch.is_autocast_enabled("cuda")
        ctx.autocast_dtype = torch.get_autocast_dtype("cuda")
        ctx.rng = []
        hidden, args, kwargs = call.inputs.merge(tensors[: call.input_count])
        saved_inputs: list[Tensor] = []
        result: Any = hidden

        def run(index: int, state: dict[str, Tensor]) -> None:
            nonlocal result
            current = hidden_tensor(result)
            saved_inputs.append(current)
            ctx.rng.append(
                (torch.get_rng_state(), torch.cuda.get_rng_state(model.device))
            )
            result = model._call_block(index, state, current, args, kwargs)

        model._execute(list(range(len(model.blocks))), run, (hidden, args, kwargs))
        # Saving CPU master tensors also detects ordinary optimizer/in-place updates
        # between forward and backward, before stale weights can be re-fetched.
        ctx.save_for_backward(*tensors, *saved_inputs)
        call.output, outputs = TensorTree.split(result)
        ctx.mark_non_differentiable(
            *[
                out
                for out in outputs
                if not (out.is_floating_point() or out.is_complex())
            ]
        )
        return tuple(outputs)

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, *output_grads: Tensor | None) -> tuple[Any, ...]:
        from streamer import hidden_tensor

        model: StreamedModel = ctx.model
        if model._configuration_version != ctx.configuration_version:
            raise RuntimeError("Do not call to() between training forward and backward")
        if tuple(module.training for module in model.modules()) != ctx.training_modes:
            raise RuntimeError("Do not change train/eval mode before backward")
        saved = ctx.saved_tensors
        count = ctx.input_count
        masters = saved[count : count + ctx.master_count]
        block_inputs = saved[count + ctx.master_count :]
        master_indices = {id(tensor): index for index, tensor in enumerate(masters)}
        input_leaves = [
            tensor.detach().requires_grad_(ctx.input_grad_flags[index])
            for index, tensor in enumerate(saved[:count])
        ]
        _, args, kwargs = ctx.call.inputs.merge(input_leaves)
        extra_indices = [
            index for index in range(1, count) if input_leaves[index].requires_grad
        ]
        extra_grads: list[Tensor | None] = [None] * count
        cpu_contributions: list[tuple[int, Tensor]] = []
        incoming: Tensor | None = None

        def run(index: int, state: dict[str, Tensor]) -> None:
            nonlocal incoming
            with (
                torch.enable_grad(),
                torch.random.fork_rng(devices=[model.device.index]),
            ):
                torch.set_rng_state(ctx.rng[index][0])
                torch.cuda.set_rng_state(ctx.rng[index][1], model.device)
                hidden = block_inputs[index].detach().requires_grad_(True)
                names = [
                    name
                    for name, parameter in model.blocks[index].named_parameters()
                    if parameter.requires_grad
                ]
                targets = (
                    [hidden]
                    + [input_leaves[i] for i in extra_indices]
                    + [state[name] for name in names]
                )
                with torch.autocast(
                    "cuda",
                    enabled=ctx.autocast,
                    dtype=ctx.autocast_dtype,
                    # An outer autocast scope must not cache converted weights
                    # after this block retires, defeating parameter residency.
                    cache_enabled=False,
                ):
                    result = model._call_block(index, state, hidden, args, kwargs)
                if index == len(model.blocks) - 1:
                    _, outputs = TensorTree.split(result)
                    pairs = [
                        (out, grad)
                        for out, grad in zip(outputs, output_grads, strict=True)
                        if out.requires_grad and grad is not None
                    ]
                else:
                    out = hidden_tensor(result)
                    pairs = (
                        [(out, incoming)]
                        if out.requires_grad and incoming is not None
                        else []
                    )
                if pairs:
                    gradients = torch.autograd.grad(
                        [pair[0] for pair in pairs],
                        targets,
                        [pair[1] for pair in pairs],
                        allow_unused=True,
                    )
                else:
                    gradients = (None,) * len(targets)
            incoming = gradients[0]
            for input_index, gradient in zip(
                extra_indices, gradients[1:], strict=False
            ):
                if gradient is not None:
                    previous = extra_grads[input_index]
                    extra_grads[input_index] = (
                        gradient if previous is None else previous + gradient
                    )
            parameters = dict(model.blocks[index].named_parameters())
            for name, gradient in zip(
                names, gradients[1 + len(extra_indices) :], strict=True
            ):
                if gradient is None:
                    continue
                parameter = parameters[name]
                cpu = torch.empty_like(parameter, device="cpu", pin_memory=True)
                # D2H runs on compute after the gradient kernels. Slot retirement's
                # completion event also proves these pinned destinations are readable.
                cpu.copy_(gradient, non_blocking=True)
                gradient.record_stream(model.compute_stream)
                model.gradient_transfer_bytes += tensor_bytes(gradient)
                cpu_contributions.append((master_indices[id(parameter)], cpu))

        model._execute(
            list(reversed(range(len(model.blocks)))),
            run,
            (block_inputs, input_leaves, output_grads),
            grad=True,
        )
        # _execute has retired all completion events, including the D2H copies.
        master_grads: list[Tensor | None] = [None] * len(masters)
        for index, gradient in cpu_contributions:
            if master_grads[index] is None:
                master_grads[index] = gradient
            else:
                master_grads[index].add_(gradient)
        extra_grads[0] = incoming if ctx.input_grad_flags[0] else None
        return (None, None, *extra_grads, *master_grads)


def training_forward(
    model: StreamedModel,
    hidden: Tensor,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    if not model.blocks:
        return hidden
    if not hidden.is_floating_point():
        raise TypeError("Training requires floating-point hidden states")
    tree, inputs = TensorTree.split((hidden, args, kwargs))
    call = Invocation(tree, len(inputs))
    # Include buffers in saved state so registered-state changes are version checked.
    masters = list(model.parameters()) + list(model.buffers())
    outputs = _StreamAutograd.apply(model, call, *inputs, *masters)
    assert call.output is not None
    return call.output.merge(outputs)
