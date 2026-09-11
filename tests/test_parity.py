"""CUDA tests deliberately exercise heterogeneous state and allocator reuse."""

from __future__ import annotations

import copy
from typing import Any

import pytest
import torch
from torch import Tensor, nn

from block_streamer import StreamedModel

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


class Block(nn.Module):
    def __init__(self, width: int, expansion: int) -> None:
        super().__init__()
        self.up = nn.Linear(width, expansion)
        self.down = nn.Linear(expansion, width)
        self.register_buffer("scale", torch.tensor(0.2))
        self.register_buffer("counter", torch.tensor(1), persistent=False)

    def forward(
        self,
        x: Tensor,
        bias: Tensor | None = None,
        *,
        attention_mask: Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor]:
        x = x + self.scale * self.down(torch.tanh(self.up(x)))
        if bias is not None:
            x = x + bias
        if attention_mask is not None:
            x = x * attention_mask
        return x, x.mean()


def reference(blocks: nn.ModuleList, x: Tensor, *args: Any, **kwargs: Any) -> Any:
    result: Any = x
    for block in blocks:
        result = block(x, *args, **kwargs)
        x = result[0] if isinstance(result, tuple) else result
    return result


@CUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("ahead", [0, 1, 2, 3])
def test_forward_parity(dtype: torch.dtype, ahead: int) -> None:
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("Device does not support bf16")
    torch.manual_seed(13)
    blocks = nn.ModuleList([Block(32, size) for size in [48, 80, 40, 96, 56, 64]])
    baseline = copy.deepcopy(blocks).to("cuda", dtype).eval()
    counts: list[int] = []
    model = StreamedModel(
        blocks, prefetch_ahead=ahead, dtype=dtype, residency_observer=counts.append
    )
    x = torch.randn(4, 8, 32, device="cuda", dtype=dtype)
    bias = torch.randn_like(x) * 0.01
    mask = torch.ones_like(x)
    with torch.no_grad(), model:
        expected = reference(baseline, x, bias, attention_mask=mask)
        actual = model(x, bias, attention_mask=mask)
    # Identical kernels should agree closely; bf16/fp16 allow one rounding unit.
    tolerance = {torch.float32: 1e-5, torch.bfloat16: 8e-3, torch.float16: 1e-3}[dtype]
    for left, right in zip(actual, expected):
        assert torch.allclose(left, right, atol=tolerance, rtol=tolerance)
    assert counts and all(0 <= count <= ahead + 1 for count in counts)
    assert max(counts) == ahead + 1
    assert model.stats()["resident_blocks"] == 0
    assert all(t.is_pinned() for t in list(model.parameters()) + list(model.buffers()))


class HeavyState(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(2048, 2048))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.weight[0, : x.shape[-1]]


@CUDA
def test_bandwidth_starved_and_100_iteration_stress() -> None:
    torch.manual_seed(21)
    blocks = nn.Sequential(*[HeavyState() for _ in range(5)])
    baseline = copy.deepcopy(blocks).cuda()
    model = StreamedModel(blocks, prefetch_ahead=2)
    x = torch.randn(2, 32, device="cuda")
    with torch.no_grad():
        expected = baseline(x)
    for _ in range(100):
        # Vary temporary allocations to exercise caching allocator address reuse.
        noise = torch.empty(1024 * 1024, device="cuda")
        noise.fill_(float("nan"))
        del noise
        assert torch.allclose(model(x), expected, atol=1e-5, rtol=1e-5)
    assert max(model.stats()["residency_history"]) <= 3


@CUDA
def test_errors_and_recovery() -> None:
    model = StreamedModel([nn.Linear(8, 8)], buffer_budget_bytes=1)
    x = torch.randn(2, 8, device="cuda")
    with pytest.raises(ValueError, match="budget"):
        model(x)
    assert model.stats()["resident_blocks"] == 0
    model = StreamedModel([nn.Linear(8, 8)])
    model.blocks[0].weight.data = model.blocks[0].weight.detach().clone()
    assert not model.blocks[0].weight.is_pinned()
    with pytest.raises(RuntimeError, match="not pinned"):
        model(x)
    model.to(torch.float32)
    assert model(x).shape == x.shape


@CUDA
def test_to_eval_empty_and_buffer_mutation() -> None:
    model = StreamedModel(nn.Sequential(nn.Linear(8, 8))).to(dtype=torch.float16).eval()
    assert not model.blocks[0].training
    assert model.blocks[0].weight.is_pinned()
    assert (
        model(torch.randn(2, 8, device="cuda", dtype=torch.float16)).dtype
        == torch.float16
    )
    x = torch.randn(2, 8, device="cuda")
    assert StreamedModel([])(x) is x

    class Mutating(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("count", torch.tensor(0.0))

        def forward(self, x: Tensor) -> Tensor:
            self.count.add_(1)
            return x

    model = StreamedModel([Mutating()])
    with pytest.raises(RuntimeError, match="mutate"):
        model(x)
    assert model.stats()["resident_blocks"] == 0


def test_invalid_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        StreamedModel([], prefetch_ahead=-1)
    with pytest.raises(ValueError, match="CUDA execution"):
        StreamedModel([], device="cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        StreamedModel([])


@CUDA
def test_inference_mode_and_nondefault_caller_stream() -> None:
    blocks = nn.Sequential(nn.Linear(32, 48), nn.ReLU(), nn.Linear(48, 32))
    baseline = copy.deepcopy(blocks).cuda().eval()
    model = StreamedModel(blocks)
    caller = torch.cuda.Stream()
    with torch.inference_mode(), torch.cuda.stream(caller):
        x = torch.randn(64, 32, device="cuda")
        expected = baseline(x)
        actual = model(x)
        # Consume the result immediately on a different stream from model.compute.
        error = (actual - expected).abs().max()
    caller.synchronize()  # The host cannot read this reduction until caller completes.
    assert error.item() < 1e-5


@CUDA
def test_alias_output_and_registered_weight_tying() -> None:
    class Tied(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.left = nn.Linear(8, 8)
            self.right = nn.Linear(8, 8)
            self.right.weight = self.left.weight

        def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
            return self.left(x) + self.right(x), self.left.weight[:2]

    blocks = nn.ModuleList([Tied(), Tied()])
    baseline = copy.deepcopy(blocks).cuda()
    model = StreamedModel(blocks)
    x = torch.randn(2, 8, device="cuda")
    with torch.no_grad():
        expected = reference(baseline, x)
    actual = model(x)
    assert all(torch.allclose(a, b) for a, b in zip(actual, expected, strict=True))
    assert model.stats()["resident_blocks"] == 0
    assert model.blocks[0].left.weight is model.blocks[0].right.weight
    # An escaped weight view must be independent of subsequent staging allocations.
    snapshot = actual[1].clone()
    model(x)
    assert torch.equal(actual[1], snapshot)


@CUDA
def test_reject_storage_aliases_and_unsafe_module_conversion() -> None:
    block = nn.Linear(8, 8)
    block.register_parameter("alias", nn.Parameter(block.weight.view(-1)))
    with pytest.raises(ValueError, match="sharing storage"):
        StreamedModel([block])
    model = StreamedModel([nn.Linear(8, 8)])
    with pytest.raises(RuntimeError, match="preserve pinned"):
        model.cuda()
