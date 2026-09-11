"""Backward and convergence tests, added after inference parity passed on CUDA."""

from __future__ import annotations

import copy

import pytest
import torch
from test_parity import CUDA, Block, reference
from torch import nn

from block_streamer import StreamedModel


@CUDA
@pytest.mark.parametrize("ahead", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_backward_parity(ahead: int, dtype: torch.dtype) -> None:
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("Device does not support bf16")
    torch.manual_seed(7)
    blocks = nn.ModuleList([Block(16, size) for size in [24, 40, 32, 48, 20]])
    baseline = copy.deepcopy(blocks).to("cuda", dtype).train()
    model = StreamedModel(blocks, dtype=dtype, prefetch_ahead=ahead).train()
    x = torch.randn(3, 16, device="cuda", dtype=dtype, requires_grad=True)
    streamed_x = x.detach().clone().requires_grad_(True)
    bias = torch.randn_like(x).requires_grad_(True)
    streamed_bias = bias.detach().clone().requires_grad_(True)
    expected = reference(baseline, x, bias, attention_mask=torch.ones_like(x))
    actual = model(streamed_x, streamed_bias, attention_mask=torch.ones_like(x))
    (expected[0].float().square().mean() + expected[1].float()).backward()
    (actual[0].float().square().mean() + actual[1].float()).backward()
    atol, rtol = {
        torch.float32: (2e-5, 2e-4),
        torch.bfloat16: (2e-3, 3e-2),
        torch.float16: (5e-4, 5e-3),
    }[dtype]
    for expected_p, actual_p in zip(
        baseline.parameters(), model.parameters(), strict=True
    ):
        assert actual_p.grad is not None and expected_p.grad is not None
        assert actual_p.grad.device.type == "cpu"
        assert actual_p.grad.is_pinned()
        assert torch.allclose(
            actual_p.grad, expected_p.grad.cpu(), atol=atol, rtol=rtol
        )
    assert torch.allclose(streamed_x.grad, x.grad, atol=atol, rtol=rtol)
    assert torch.allclose(streamed_bias.grad, bias.grad, atol=atol, rtol=rtol)
    stats = model.stats()
    assert all(count <= ahead + 1 for count in stats["residency_history"])
    assert stats["resident_blocks"] == 0
    assert stats["gradient_transfer_bytes"] > 0


@CUDA
def test_convergence_and_gradient_accumulation() -> None:
    torch.manual_seed(83)
    cpu = nn.Sequential(nn.Linear(8, 24), nn.Tanh(), nn.Linear(24, 8))
    baseline = copy.deepcopy(cpu).cuda()
    model = StreamedModel(cpu).train()
    optimizers = [
        torch.optim.SGD(baseline.parameters(), lr=0.1),
        torch.optim.SGD(model.parameters(), lr=0.1),
    ]
    x = torch.randn(32, 8, device="cuda")
    target = 0.5 * x
    curves: list[list[float]] = [[], []]
    for _ in range(12):
        for index, (network, optimizer) in enumerate(
            zip([baseline, model], optimizers)
        ):
            optimizer.zero_grad(set_to_none=True)
            for _ in range(2):
                loss = (network(x) - target).square().mean() / 2
                loss.backward()
            curves[index].append(loss.item() * 2)
            optimizer.step()
    assert curves[0][-1] < curves[0][0] * 0.9
    assert torch.allclose(
        torch.tensor(curves[0]), torch.tensor(curves[1]), atol=2e-6, rtol=2e-5
    )


@CUDA
def test_dropout_rng_shared_weights_and_unused_parameter() -> None:
    torch.manual_seed(5)
    linear = nn.Linear(16, 16)
    linear.register_parameter("unused", nn.Parameter(torch.randn(3)))
    cpu = nn.Sequential(linear, nn.Dropout(0.25), linear)
    baseline = copy.deepcopy(cpu).cuda().train()
    model = StreamedModel(cpu, prefetch_ahead=1).train()
    x = torch.randn(4, 16, device="cuda")
    torch.manual_seed(46)
    expected = baseline(x)
    expected.sum().backward()
    torch.manual_seed(46)
    actual = model(x)
    rng_after_forward = torch.cuda.get_rng_state().clone()
    actual.sum().backward()
    assert torch.equal(torch.cuda.get_rng_state(), rng_after_forward)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    for left, right in zip(baseline.parameters(), model.parameters(), strict=True):
        if left.grad is None:
            assert right.grad is None
        else:
            assert torch.allclose(left.grad.cpu(), right.grad, atol=2e-5, rtol=2e-5)


@CUDA
def test_reject_change_between_forward_and_backward() -> None:
    model = StreamedModel([nn.Linear(4, 4)]).train()
    output = model(torch.randn(2, 4, device="cuda"))
    with torch.no_grad():
        next(model.parameters()).add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace"):
        output.sum().backward()


@CUDA
def test_autocast_frozen_parameters_and_cpu_adam() -> None:
    torch.manual_seed(24)
    cpu = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 16))
    cpu[0].weight.requires_grad_(False)
    baseline = copy.deepcopy(cpu).cuda()
    model = StreamedModel(cpu).train()
    x = torch.randn(8, 16, device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        baseline(x).float().square().mean().backward()
        model(x).float().square().mean().backward()
    for left, right in zip(baseline.parameters(), model.parameters(), strict=True):
        if left.grad is None:
            assert right.grad is None
        else:
            assert torch.allclose(left.grad.cpu(), right.grad, atol=5e-4, rtol=5e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.step()
    assert all(
        value.device.type == "cpu"
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )
