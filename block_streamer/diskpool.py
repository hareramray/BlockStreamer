"""NVMe-backed block staging: disk -> pinned ring -> GPU.

The pinned ring is the only large pinned allocation in the system; master
weights live on disk, not in host RAM. Reads run on a background pool so a
block's disk latency overlaps the previous block's compute, and slots retire
lazily from a completion queue so several reads stay in flight without ever
exceeding the residency bound.
"""

from __future__ import annotations

import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from memory_pool import BufferPool, Resident, tensor_bytes

from .shards import BlockSpec, ShardManifest


class PinnedRing:
    """A fixed set of reusable pinned staging buffers."""

    def __init__(self, slots: int, nbytes: int) -> None:
        if slots <= 0 or nbytes <= 0:
            raise ValueError("PinnedRing needs a positive slot count and size")
        self.nbytes = nbytes
        self.buffers = [
            torch.empty(nbytes, dtype=torch.uint8).pin_memory() for _ in range(slots)
        ]
        self.free: deque[int] = deque(range(slots))

    def acquire(self) -> int:
        if not self.free:
            raise RuntimeError("No free pinned staging buffer")
        return self.free.popleft()

    def release(self, slot: int) -> None:
        self.free.append(slot)

    @property
    def bytes_allocated(self) -> int:
        return self.nbytes * len(self.buffers)


@dataclass
class DiskResident(Resident):
    """A staged block plus the ring slot and read future backing it."""

    slot: int = -1
    future: Future[None] | None = None
    blob: Tensor | None = None
    module: nn.Module | None = None
    grad: bool = False


class DiskBufferPool(BufferPool):
    """Stage blocks from shard files instead of pinned master weights.

    Tensors named in the manifest come from disk. Any other tensor the module
    registers -- LoRA adapters, for instance -- comes from its own pinned CPU
    storage, so a block can mix frozen streamed weights with trainable
    resident ones.
    """

    def __init__(
        self,
        device: torch.device,
        capacity: int,
        transfer_stream: torch.cuda.Stream,
        compute_stream: torch.cuda.Stream,
        manifest: ShardManifest,
        budget_bytes: int | None = None,
        observer=None,
        readers: int = 2,
    ) -> None:
        super().__init__(
            device, capacity, transfer_stream, compute_stream, budget_bytes, observer
        )
        self.manifest = manifest
        self.ring = PinnedRing(capacity, manifest.largest_block_bytes)
        self.readers = ThreadPoolExecutor(
            max_workers=readers, thread_name_prefix="shard-read"
        )
        self.pending: deque[tuple[int, DiskResident, torch.cuda.Event]] = deque()
        self.read_seconds = 0.0
        self.read_bytes = 0
        self._by_index = {block.index: block for block in manifest.blocks}

    # -- residency accounting -------------------------------------------------

    def check(self) -> None:
        """Bound live + not-yet-retired blocks, which is the real VRAM held."""
        count = len(self.live) + len(self.pending)
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

    def _reclaim(self, needed: int = 1) -> None:
        """Retire finished blocks until `needed` slots are genuinely free."""
        while (
            self.pending and len(self.live) + len(self.pending) + needed > self.capacity
        ):
            _, item, done = self.pending.popleft()
            start = time.perf_counter()
            done.synchronize()
            self.eviction_wait_ms += (time.perf_counter() - start) * 1000
            self._retire(item)

    def _retire(self, item: DiskResident) -> None:
        self.transfer_ms += item.start.elapsed_time(item.ready)
        item.state.clear()
        item.blob = None
        if item.slot >= 0:
            self.ring.release(item.slot)
            item.slot = -1

    # -- staging --------------------------------------------------------------

    def _read_and_copy(
        self, block: BlockSpec, buffer: Tensor, blob: Tensor, ready: torch.cuda.Event
    ) -> None:
        started = time.perf_counter()
        self.manifest.read_block(block, buffer)
        self.read_seconds += time.perf_counter() - started
        self.read_bytes += block.nbytes
        torch.cuda.set_device(self.device)
        with torch.cuda.stream(self.transfer_stream):
            blob[: block.nbytes].copy_(buffer[: block.nbytes], non_blocking=True)
            ready.record(self.transfer_stream)

    def stage(self, index: int, module: nn.Module, grad: bool = False) -> None:
        if index in self.live:
            return
        self._reclaim(1)
        if len(self.live) + len(self.pending) >= self.capacity:
            raise RuntimeError("No free GPU block slot; retire completed compute first")
        block = self._by_index[index]
        if self.budget_bytes is not None and block.nbytes > self.budget_bytes:
            raise ValueError(
                f"Block {index} requires {block.nbytes} bytes, exceeding per-slot "
                f"buffer budget {self.budget_bytes} bytes"
            )
        slot = self.ring.acquire()
        buffer = self.ring.buffers[slot]
        with torch.cuda.stream(self.transfer_stream):
            start = torch.cuda.Event(enable_timing=True)
            ready = torch.cuda.Event(enable_timing=True)
            start.record(self.transfer_stream)
            blob = torch.empty(block.nbytes, dtype=torch.uint8, device=self.device)
        future = self.readers.submit(self._read_and_copy, block, buffer, blob, ready)

        resident = DiskResident(
            state={},
            ready=ready,
            start=start,
            nbytes=block.nbytes,
            parameter_bytes=block.nbytes,
            slot=slot,
            future=future,
            blob=blob,
            module=module,
            grad=grad,
        )
        self.live[index] = resident
        self.transfer_bytes += block.nbytes
        self.check()

    def acquire(self, index: int) -> dict[str, Tensor]:
        item = self.live[index]
        assert isinstance(item, DiskResident)
        if item.future is not None:
            # Blocks only until the reader issued its copy; the read itself
            # overlapped the previous block's compute.
            start = time.perf_counter()
            item.future.result()
            self.stall_ms += (time.perf_counter() - start) * 1000
            item.future = None
        if not item.state:
            block = self._by_index[index]
            assert item.blob is not None
            state = self.manifest.materialize(block, item.blob)
            if item.module is not None:
                state = merge_resident(state, item.module, self.device, item.grad)
            item.state = state
            item.parameter_bytes = sum(
                tensor_bytes(tensor) for tensor in state.values()
            )
        before = torch.cuda.Event(enable_timing=True)
        after = torch.cuda.Event(enable_timing=True)
        before.record(self.compute_stream)
        self.compute_stream.wait_event(item.ready)
        after.record(self.compute_stream)
        self.wait_events.append((before, after))
        item.blob.record_stream(self.compute_stream)
        for tensor in item.state.values():
            tensor.record_stream(self.compute_stream)
        return item.state

    def evict(self, index: int, done: torch.cuda.Event) -> None:
        """Queue retirement instead of host-syncing inline (deep prefetch)."""
        item = self.live.pop(index)
        assert isinstance(item, DiskResident)
        self.pending.append((index, item, done))
        self.check()

    def drain(self) -> None:
        for _, item, done in self.pending:
            done.synchronize()
            self._retire(item)
        self.pending.clear()
        self.transfer_stream.synchronize()
        self.compute_stream.synchronize()
        for item in self.live.values():
            if isinstance(item, DiskResident):
                if item.future is not None:
                    item.future.result()
                self._retire(item)
        self.live.clear()
        self.check()
        for before, after in self.wait_events:
            self.stall_ms += before.elapsed_time(after)
        self.wait_events.clear()

    def close(self) -> None:
        self.readers.shutdown(wait=True)

    @property
    def read_gbps(self) -> float:
        return self.read_bytes / self.read_seconds / 1e9 if self.read_seconds else 0.0


def merge_resident(
    state: dict[str, Tensor],
    module: nn.Module,
    device: torch.device,
    grad: bool,
) -> dict[str, Tensor]:
    """Add a module's own pinned tensors (LoRA adapters) to a staged state."""
    merged = dict(state)
    for name, parameter in module.named_parameters():
        if name in merged:
            if grad:
                merged[name] = merged[name].requires_grad_(parameter.requires_grad)
            continue
        if parameter.is_meta:
            raise RuntimeError(
                f"Tensor {name!r} is neither in the shard manifest nor materialized"
            )
        staged = parameter.detach().to(device, non_blocking=True)
        merged[name] = (
            staged.requires_grad_(parameter.requires_grad) if grad else staged
        )
    for name, buffer in module.named_buffers():
        if name not in merged and not buffer.is_meta:
            merged[name] = buffer.detach().to(device, non_blocking=True)
    return merged


def parameter_payload(module: nn.Module) -> int:
    return sum(tensor_bytes(p) for p in module.parameters() if not p.is_meta)
