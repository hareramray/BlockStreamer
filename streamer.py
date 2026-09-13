"""Blockwise functional execution with bounded CUDA parameter residency."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Self

import torch
from torch import Tensor, nn
from torch.func import functional_call
from torch.utils._pytree import tree_map

from memory_pool import BufferPool, pin_modules


def hidden_tensor(result: Any) -> Tensor:
    hidden = result[0] if isinstance(result, tuple) else result
    if not isinstance(hidden, Tensor):
        raise TypeError(
            "A block must return a Tensor or a tuple beginning with a Tensor"
        )
    return hidden


class StreamedModel(nn.Module):
    """Sequential block streaming. Intermediate tuple[0] feeds the next block.

    Additional positional/keyword arguments are supplied unchanged to every block;
    the final block's entire result is returned. Inputs must already be on device.
    Construction takes ownership of the supplied modules and pins their CPU state.
    """

    def __init__(
        self,
        blocks: nn.Sequential | nn.ModuleList | Sequence[nn.Module],
        device: str | torch.device = "cuda:0",
        prefetch_ahead: int = 1,
        dtype: torch.dtype | None = None,
        buffer_budget_bytes: int | None = None,
        residency_observer: Callable[[int], None] | None = None,
        manifest: Any = None,
    ) -> None:
        super().__init__()
        if isinstance(prefetch_ahead, bool) or not isinstance(prefetch_ahead, int):
            raise TypeError("prefetch_ahead must be an integer")
        if prefetch_ahead < 0:
            raise ValueError("prefetch_ahead must be nonnegative")
        if buffer_budget_bytes is not None and buffer_budget_bytes <= 0:
            raise ValueError("buffer_budget_bytes must be positive")
        self.device = self._cuda_device(device)
        self.blocks = nn.ModuleList(list(blocks))
        self.prefetch_ahead = prefetch_ahead
        self.dtype = dtype
        self.buffer_budget_bytes = buffer_budget_bytes
        self.residency_observer = residency_observer
        self._busy = False
        self._configuration_version = 0
        self.gradient_transfer_bytes = 0
        self.manifest = manifest
        if manifest is None:
            pin_modules(self.blocks, dtype)
        else:
            pin_modules(self.blocks, dtype, skip_meta=True)
        self._make_engine()
        self.eval()

    @staticmethod
    def _cuda_device(device: str | torch.device) -> torch.device:
        result = torch.device(device)
        if result.type != "cuda":
            raise ValueError("StreamedModel requires a CUDA execution device")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable; install a CUDA-enabled PyTorch build"
            )
        index = torch.cuda.current_device() if result.index is None else result.index
        if not 0 <= index < torch.cuda.device_count():
            raise ValueError(f"CUDA device index {index} is unavailable")
        return torch.device("cuda", index)

    def _make_engine(self) -> None:
        # Borrow the construction stream. A new compute stream per wrapper would
        # create a persistent cuBLAS workspace per instance in PyTorch's handle pool.
        self.compute_stream = torch.cuda.current_stream(device=self.device)
        self.transfer_stream = torch.cuda.Stream(device=self.device)
        if self.manifest is None:
            self.pool = BufferPool(
                self.device,
                self.prefetch_ahead + 1,
                self.transfer_stream,
                self.compute_stream,
                self.buffer_budget_bytes,
                self.residency_observer,
            )
        else:
            from diskpool import DiskBufferPool

            self.pool = DiskBufferPool(
                self.device,
                self.prefetch_ahead + 1,
                self.transfer_stream,
                self.compute_stream,
                self.manifest,
                self.buffer_budget_bytes,
                self.residency_observer,
            )

    def _execute(
        self,
        order: list[int],
        callback: Callable[[int, dict[str, Tensor]], None],
        inputs: Any,
        grad: bool = False,
    ) -> None:
        if self._busy:
            raise RuntimeError("Concurrent or reentrant execution is unsupported")
        self._busy = True
        caller = torch.cuda.current_stream(self.device)
        # Inputs may have been produced on the caller stream just before forward.
        self.compute_stream.wait_stream(caller)

        def protect(value: Any) -> Any:
            if isinstance(value, Tensor) and value.is_cuda:
                if value.device != self.device:
                    raise ValueError(
                        f"Input is on {value.device}, expected {self.device}"
                    )
                value.record_stream(self.compute_stream)
            return value

        try:
            tree_map(protect, inputs)
            for index in order[: self.pool.capacity]:
                self.pool.stage(index, self.blocks[index], grad)
            for offset, index in enumerate(order):
                with torch.cuda.stream(self.compute_stream):
                    callback(index, self.pool.acquire(index))
                    done = torch.cuda.Event()
                    done.record(self.compute_stream)
                self.pool.evict(index, done)
                next_offset = offset + self.pool.capacity
                if next_offset < len(order):
                    upcoming = order[next_offset]
                    self.pool.stage(upcoming, self.blocks[upcoming], grad)
        finally:
            self.pool.drain()
            # Returning tensors to caller must not race their production on compute.
            caller.wait_stream(self.compute_stream)
            self._busy = False

    def _call_block(
        self,
        index: int,
        state: dict[str, Tensor],
        hidden: Tensor,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        versions = {name: value._version for name, value in state.items()}
        identities = {name: id(value) for name, value in state.items()}
        result = functional_call(
            self.blocks[index],
            state,
            (hidden, *args),
            kwargs,
            strict=True,
        )
        if any(
            value._version != versions[name] or id(value) != identities[name]
            for name, value in state.items()
        ):
            raise RuntimeError(
                "Blocks must not mutate parameters or registered buffers"
            )
        hidden_tensor(result)
        # A block may return a view of a weight. Clone outputs so eviction really
        # releases the staged state and callers cannot retain the pool's storage.
        return tree_map(
            lambda value: value.clone() if isinstance(value, Tensor) else value, result
        )

    def forward(self, hidden_states: Tensor, *args: Any, **kwargs: Any) -> Any:
        if hidden_states.device != self.device:
            raise ValueError(f"hidden_states must be on {self.device}")
        if self.training and torch.is_grad_enabled():
            from block_streamer._autograd import training_forward

            return training_forward(self, hidden_states, args, kwargs)
        result: Any = hidden_states

        def run(index: int, state: dict[str, Tensor]) -> None:
            nonlocal result
            result = self._call_block(index, state, hidden_tensor(result), args, kwargs)

        with torch.inference_mode(False), torch.no_grad():
            self._execute(
                list(range(len(self.blocks))), run, (hidden_states, args, kwargs)
            )
        return result

    def to(self, *args: Any, **kwargs: Any) -> Self:
        """Change execution device/dtype while retaining pinned CPU master state."""
        if self._busy:
            raise RuntimeError("Cannot move the model during execution")
        device, dtype, _, memory_format = torch._C._nn._parse_to(*args, **kwargs)
        if memory_format is not None:
            raise ValueError("memory_format conversion is unsupported")
        target = self.device if device is None else self._cuda_device(device)
        self.close()
        self._configuration_version += 1
        pin_modules(self.blocks, dtype)
        self.device = target
        if dtype is not None:
            self.dtype = dtype
        self._make_engine()
        return self

    def _apply(self, fn: Callable[[Tensor], Tensor], recurse: bool = True) -> Self:
        raise RuntimeError(
            "Use StreamedModel.to(device=..., dtype=...) to preserve pinned CPU state; "
            "inherited cpu/cuda/half/bfloat16 and parent-module conversions are unsupported"
        )

    def stats(self) -> dict[str, Any]:
        """Cumulative since construction/reset; byte counts are tensor payloads."""
        return {
            "transfer_bytes": self.pool.transfer_bytes,
            "gradient_transfer_bytes": self.gradient_transfer_bytes,
            "transfer_ms": self.pool.transfer_ms,
            "stall_ms": self.pool.stall_ms,
            "eviction_wait_ms": self.pool.eviction_wait_ms,
            "resident_blocks": len(self.pool.live),
            "peak_resident_blocks": self.pool.peak_resident_blocks,
            "peak_state_bytes": self.pool.peak_state_bytes,
            "peak_parameter_bytes": self.pool.peak_parameter_bytes,
            "residency_history": tuple(self.pool.history),
        }

    def reset_stats(self) -> None:
        self.pool.reset_stats()
        self.gradient_transfer_bytes = 0

    def close(self) -> None:
        if self._busy:
            raise RuntimeError("Cannot close the model during execution")
        self.pool.drain()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
