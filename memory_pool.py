"""Pinned canonical state and bounded, event-retired CUDA allocations."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor, nn


def tensor_bytes(tensor: Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def module_state(module: nn.Module) -> dict[str, Tensor]:
    """Include nonpersistent buffers; functional_call needs those too."""
    return dict(list(module.named_parameters()) + list(module.named_buffers()))


def pin_modules(
    blocks: nn.ModuleList, dtype: torch.dtype | None, skip_meta: bool = False
) -> None:
    """Preserve Parameter identity and shared objects when creating CPU storage."""
    if dtype is not None and not dtype.is_floating_point:
        raise TypeError("dtype must be a floating-point torch.dtype")
    storage_owners: dict[tuple[torch.device, int], int] = {}
    tensors = list(blocks.parameters()) + list(blocks.buffers())
    if skip_meta:
        # Disk-backed blocks are constructed on meta: their master weights live
        # in shard files, so there is nothing to pin. Only adapters are real.
        tensors = [tensor for tensor in tensors if not tensor.is_meta]
    for tensor in tensors:
        if tensor.is_meta or tensor.layout != torch.strided:
            raise ValueError("Only materialized, dense strided state is supported")
        if tensor.numel():
            key = (tensor.device, tensor.untyped_storage().data_ptr())
            if key in storage_owners and storage_owners[key] != id(tensor):
                raise ValueError(
                    "Distinct state tensors sharing storage are unsupported; "
                    "tie weights by sharing the same Parameter object instead"
                )
            storage_owners[key] = id(tensor)
    seen: set[int] = set()
    for tensor in tensors:
        if id(tensor) in seen:
            continue
        seen.add(id(tensor))
        if tensor.is_meta or tensor.layout != torch.strided:
            raise ValueError("Only materialized, dense strided state is supported")
        target_dtype = dtype if tensor.is_floating_point() else None
        cpu = tensor.detach().to(device="cpu", dtype=target_dtype).pin_memory()
        tensor.data = cpu
        if isinstance(tensor, nn.Parameter) and tensor.grad is not None:
            tensor.grad = tensor.grad.to(device="cpu", dtype=cpu.dtype).pin_memory()


@dataclass
class Resident:
    state: dict[str, Tensor]
    ready: torch.cuda.Event
    start: torch.cuda.Event
    nbytes: int
    parameter_bytes: int


class BufferPool:
    """Each live slot owns one block, regardless of shape or dtype.

    Allocations use PyTorch's caching allocator rather than shape-specific slabs.
    A slot is freed only after its compute completion event has completed. Cached
    (reserved) allocator memory is not live parameter residency.
    """

    def __init__(
        self,
        device: torch.device,
        capacity: int,
        transfer_stream: torch.cuda.Stream,
        compute_stream: torch.cuda.Stream,
        budget_bytes: int | None = None,
        observer: Callable[[int], None] | None = None,
    ) -> None:
        self.device = device
        self.capacity = capacity
        self.transfer_stream = transfer_stream
        self.compute_stream = compute_stream
        self.budget_bytes = budget_bytes
        self.observer = observer
        self.live: dict[int, Resident] = {}
        self.reset_stats()

    def reset_stats(self) -> None:
        if self.live:
            raise RuntimeError("Cannot reset statistics while blocks are resident")
        self.transfer_bytes = 0
        self.transfer_ms = 0.0
        self.stall_ms = 0.0
        self.eviction_wait_ms = 0.0
        self.peak_resident_blocks = 0
        self.peak_state_bytes = 0
        self.peak_parameter_bytes = 0
        self.history: deque[int] = deque(maxlen=4096)
        self.wait_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

    def check(self) -> None:
        count = len(self.live)
        assert count <= self.capacity, "GPU block residency exceeded capacity"
        self.history.append(count)
        self.peak_resident_blocks = max(self.peak_resident_blocks, count)
        self.peak_state_bytes = max(
            self.peak_state_bytes, sum(item.nbytes for item in self.live.values())
        )
        self.peak_parameter_bytes = max(
            self.peak_parameter_bytes,
            sum(item.parameter_bytes for item in self.live.values()),
        )
        if self.observer is not None:
            self.observer(count)

    def stage(self, index: int, module: nn.Module, grad: bool = False) -> None:
        if index in self.live:
            return
        if len(self.live) >= self.capacity:
            raise RuntimeError("No free GPU block slot; retire completed compute first")
        source = module_state(module)
        for name, tensor in source.items():
            if tensor.device.type != "cpu" or not tensor.is_pinned():
                raise RuntimeError(
                    f"Block {index} tensor {name!r} is not pinned CPU memory"
                )
        nbytes = sum(tensor_bytes(tensor) for tensor in source.values())
        if self.budget_bytes is not None and nbytes > self.budget_bytes:
            raise ValueError(
                f"Block {index} requires {nbytes} bytes, exceeding per-slot buffer "
                f"budget {self.budget_bytes} bytes"
            )
        parameters = dict(module.named_parameters())
        with torch.cuda.stream(self.transfer_stream):
            start = torch.cuda.Event(enable_timing=True)
            ready = torch.cuda.Event(enable_timing=True)
            start.record(self.transfer_stream)
            state = {
                name: tensor.detach().to(self.device, non_blocking=True)
                for name, tensor in source.items()
            }
            if grad:
                for name, parameter in parameters.items():
                    state[name].requires_grad_(parameter.requires_grad)
            ready.record(self.transfer_stream)
        self.live[index] = Resident(
            state,
            ready,
            start,
            nbytes,
            sum(tensor_bytes(tensor) for tensor in parameters.values()),
        )
        self.transfer_bytes += nbytes
        self.check()

    def acquire(self, index: int) -> dict[str, Tensor]:
        item = self.live[index]
        before = torch.cuda.Event(enable_timing=True)
        after = torch.cuda.Event(enable_timing=True)
        before.record(self.compute_stream)
        # Compute must not read weights until their H2D copies have finished.
        self.compute_stream.wait_event(item.ready)
        after.record(self.compute_stream)
        self.wait_events.append((before, after))
        for tensor in item.state.values():
            # CRITICAL allocator guard: transfer-created storage remains protected
            # while compute consumes it, even if Python drops its final reference.
            tensor.record_stream(self.compute_stream)
        return item.state

    def evict(self, index: int, done: torch.cuda.Event) -> None:
        start = time.perf_counter()
        # Host retirement proves compute is complete BEFORE allocating a new slot;
        # merely enqueueing a wait would permit too many live parameter blocks.
        done.synchronize()
        self.eviction_wait_ms += (time.perf_counter() - start) * 1000
        item = self.live.pop(index)
        self.transfer_ms += item.start.elapsed_time(item.ready)
        item.state.clear()
        self.check()

    def drain(self) -> None:
        # On exceptions, both pending copies and consumers must finish before free.
        self.transfer_stream.synchronize()
        self.compute_stream.synchronize()
        self.live.clear()
        self.check()
        for before, after in self.wait_events:
            self.stall_ms += before.elapsed_time(after)
        self.wait_events.clear()
