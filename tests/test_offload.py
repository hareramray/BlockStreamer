"""Tests for NVMe-backed streaming, LoRA, fused optimizer, and micro-batching."""

from __future__ import annotations

import copy

import pytest
import torch
from test_parity import CUDA
from torch import nn

from block_streamer.adapters import (
    BaseAdapter,
    ExpertMajorMoE,
    UniformBlock,
    chunk_fused_experts,
    describe_blocks,
)
from block_streamer.fused_optim import AdamConfig, FusedDiskAdam
from block_streamer.lora import (
    LoRAConfig,
    LoRALinear,
    apply_to_blocks,
    backward_fetch_fraction,
    freeze_base,
)
from block_streamer.microbatch import MicroBatchConfig, micro_forward
from block_streamer.shards import ShardManifest, convert_modules
from streamer import StreamedModel

DIM, HIDDEN, DEPTH = 24, 32, 6


class Block(nn.Module):
    def __init__(self, dim: int = DIM, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.q_proj = nn.Linear(dim, hidden)
        self.o_proj = nn.Linear(hidden, dim)
        self.act = nn.GELU()

    def forward(self, x, *args, **kwargs):
        return x + self.o_proj(self.act(self.q_proj(x)))


def build_shards(tmp_path, blocks: nn.ModuleList) -> ShardManifest:
    convert_modules(((i, dict(b.state_dict())) for i, b in enumerate(blocks)), tmp_path)
    return ShardManifest.load(tmp_path)


def meta_blocks(depth: int = DEPTH, dtype=torch.float32) -> nn.ModuleList:
    with torch.device("meta"):
        return nn.ModuleList([Block() for _ in range(depth)]).to(dtype)


def materialize_adapters(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, LoRALinear):
            child.lora_A.data = torch.empty_like(child.lora_A, device="cpu")
            child.lora_B.data = torch.zeros_like(child.lora_B, device="cpu")
            child.reset_adapters()


def test_shard_round_trip_is_bit_exact(tmp_path) -> None:
    torch.manual_seed(0)
    state = {
        "weight": torch.randn(16, 12, dtype=torch.bfloat16),
        "packed": torch.randint(0, 255, (32,), dtype=torch.uint8),
        "norm": torch.randn(12, dtype=torch.float32),
    }
    convert_modules(iter([(0, state)]), tmp_path)
    manifest = ShardManifest.load(tmp_path)
    buffer = torch.empty(manifest.largest_block_bytes, dtype=torch.uint8)
    block = manifest.blocks[0]
    manifest.read_block(block, buffer)
    restored = manifest.materialize(block, buffer)
    for name, tensor in state.items():
        assert restored[name].dtype == tensor.dtype
        assert torch.equal(restored[name], tensor)


def test_manifest_rejects_undersized_buffer(tmp_path) -> None:
    convert_modules(iter([(0, {"w": torch.randn(64, 64)})]), tmp_path)
    manifest = ShardManifest.load(tmp_path)
    small = torch.empty(8, dtype=torch.uint8)
    with pytest.raises(ValueError, match="Staging buffer"):
        manifest.read_block(manifest.blocks[0], small)


@CUDA
@pytest.mark.parametrize("ahead", [0, 1, 2, 3, 5])
def test_disk_inference_parity_and_residency(tmp_path, ahead: int) -> None:
    torch.manual_seed(1)
    real = nn.ModuleList([Block() for _ in range(DEPTH)])
    manifest = build_shards(tmp_path, real)
    reference = copy.deepcopy(real).to("cuda").eval()
    seen: list[int] = []
    model = StreamedModel(
        meta_blocks(),
        prefetch_ahead=ahead,
        manifest=manifest,
        residency_observer=seen.append,
    )
    x = torch.randn(4, DIM, device="cuda")
    with torch.no_grad():
        actual = model(x)
        expected = x
        for block in reference:
            expected = block(expected)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    assert max(seen) <= ahead + 1
    assert model.stats()["peak_resident_blocks"] <= ahead + 1
    model.pool.close()


@CUDA
def test_deep_prefetch_keeps_residency_bound_under_stress(tmp_path) -> None:
    torch.manual_seed(2)
    manifest = build_shards(tmp_path, nn.ModuleList([Block() for _ in range(DEPTH)]))
    seen: list[int] = []
    model = StreamedModel(
        meta_blocks(),
        prefetch_ahead=4,
        manifest=manifest,
        residency_observer=seen.append,
    )
    x = torch.randn(4, DIM, device="cuda")
    with torch.no_grad():
        for _ in range(100):
            model(x)
    assert max(seen) <= 5
    assert model.pool.read_bytes > 0
    model.pool.close()


@CUDA
def test_pinned_ring_is_the_only_large_pinned_allocation(tmp_path) -> None:
    manifest = build_shards(tmp_path, nn.ModuleList([Block() for _ in range(DEPTH)]))
    model = StreamedModel(meta_blocks(), prefetch_ahead=2, manifest=manifest)
    assert model.pool.ring.bytes_allocated == manifest.largest_block_bytes * 3
    assert all(b.is_pinned() for b in model.pool.ring.buffers)
    for parameter in model.blocks.parameters():
        assert parameter.is_meta
    model.pool.close()


def test_fused_disk_adam_matches_reference(tmp_path) -> None:
    torch.manual_seed(3)
    weight, bias = torch.randn(12, 8), torch.randn(12)
    ref_w = weight.clone().requires_grad_(True)
    ref_b = bias.clone().requires_grad_(True)
    reference = torch.optim.Adam([ref_w, ref_b], lr=1e-3)
    convert_modules(
        iter([(0, {"weight": weight.clone(), "bias": bias.clone()})]), tmp_path / "w"
    )
    manifest = ShardManifest.load(tmp_path / "w")
    fused = FusedDiskAdam(
        manifest, tmp_path / "opt", AdamConfig(lr=1e-3), device=torch.device("cpu")
    )
    block = manifest.blocks[0]
    buffer = torch.empty(manifest.largest_block_bytes, dtype=torch.uint8)
    for step in range(20):
        generator = torch.Generator().manual_seed(step)
        gw = torch.randn(12, 8, generator=generator)
        gb = torch.randn(12, generator=generator)
        ref_w.grad, ref_b.grad = gw.clone(), gb.clone()
        reference.step()
        manifest.read_block(block, buffer)
        fused.step_block(
            block, {"weight": gw, "bias": gb}, manifest.materialize(block, buffer)
        )
    manifest.read_block(block, buffer)
    final = manifest.materialize(block, buffer)
    assert torch.allclose(final["weight"], ref_w, atol=1e-5)
    assert torch.allclose(final["bias"], ref_b, atol=1e-5)
    assert fused.write_bytes > 0


def test_fused_optimizer_reports_projected_wear(tmp_path) -> None:
    convert_modules(iter([(0, {"w": torch.randn(1000, 1000)})]), tmp_path / "w")
    manifest = ShardManifest.load(tmp_path / "w")
    fused = FusedDiskAdam(manifest, tmp_path / "opt", AdamConfig())
    wear = fused.projected_wear(1000)
    # 12 B/param of state plus the rewritten weight shard.
    assert wear["bytes_per_step"] >= 1000 * 1000 * 12
    assert wear["total_bytes"] == wear["bytes_per_step"] * 1000


def test_lora_starts_as_identity_and_freezes_base() -> None:
    config = LoRAConfig(rank=4, alpha=8, targets=("q_proj",))
    layer = LoRALinear(8, 8, config, base_device="cpu")
    x = torch.randn(3, 8)
    with torch.no_grad():
        torch.nn.init.eye_(layer.weight)
        layer.bias.zero_()
        assert torch.allclose(layer(x), x, atol=1e-6)
    blocks = meta_blocks(4)
    indices, count = apply_to_blocks(
        blocks, LoRAConfig(top_fraction=0.5, targets=("q_proj", "o_proj"))
    )
    assert indices == [2, 3]
    assert count == 4
    materialize_adapters(blocks)
    trainable, frozen = freeze_base(blocks)
    assert trainable > 0 and frozen > 0
    assert backward_fetch_fraction(4, LoRAConfig(top_fraction=0.5)) == 0.5


@CUDA
def test_lora_gradients_only_on_adapters(tmp_path) -> None:
    torch.manual_seed(4)
    manifest = build_shards(tmp_path, nn.ModuleList([Block() for _ in range(DEPTH)]))
    blocks = meta_blocks()
    apply_to_blocks(
        blocks, LoRAConfig(rank=4, targets=("q_proj", "o_proj"), top_fraction=1.0)
    )
    materialize_adapters(blocks)
    freeze_base(blocks)
    model = StreamedModel(blocks, prefetch_ahead=1, manifest=manifest).train()
    x = torch.randn(4, DIM, device="cuda")
    model(x).square().mean().backward()
    adapters = 0
    for name, parameter in model.named_parameters():
        if "lora" in name:
            assert parameter.grad is not None
            adapters += 1
        else:
            assert parameter.grad is None
    assert adapters > 0
    model.pool.close()


@CUDA
def test_lora_loss_decreases(tmp_path) -> None:
    torch.manual_seed(6)
    manifest = build_shards(tmp_path, nn.ModuleList([Block() for _ in range(DEPTH)]))
    blocks = meta_blocks()
    apply_to_blocks(
        blocks, LoRAConfig(rank=4, targets=("q_proj", "o_proj"), top_fraction=1.0)
    )
    materialize_adapters(blocks)
    freeze_base(blocks)
    model = StreamedModel(blocks, prefetch_ahead=1, manifest=manifest).train()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=5e-3
    )
    x = torch.randn(8, DIM, device="cuda")
    target = torch.zeros_like(x)
    losses = []
    for _ in range(12):
        optimizer.zero_grad(set_to_none=True)
        loss = (model(x) - target).square().mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.9
    model.pool.close()


@CUDA
@pytest.mark.parametrize("count", [1, 2, 4])
def test_microbatch_gradients_match_and_amortize_reads(tmp_path, count: int) -> None:
    torch.manual_seed(7)
    manifest = build_shards(tmp_path, nn.ModuleList([Block() for _ in range(DEPTH)]))

    def make():
        blocks = meta_blocks()
        apply_to_blocks(
            blocks, LoRAConfig(rank=4, targets=("q_proj", "o_proj"), top_fraction=1.0)
        )
        torch.manual_seed(77)
        materialize_adapters(blocks)
        for child in blocks.modules():
            if isinstance(child, LoRALinear):
                child.lora_B.data = torch.randn_like(child.lora_B) * 0.05
        freeze_base(blocks)
        return StreamedModel(blocks, prefetch_ahead=1, manifest=manifest).train()

    x = torch.randn(8, DIM, device="cuda")
    grads = {}
    reads = {}
    for n in (1, count):
        model = make()
        out, telemetry = micro_forward(model, x, MicroBatchConfig(count=n))
        out.square().mean().backward()
        grads[n] = torch.cat(
            [
                p.grad.flatten()
                for _, p in sorted(model.named_parameters())
                if p.grad is not None
            ]
        )
        reads[n] = model.pool.transfer_bytes
        if n > 1:
            assert telemetry.offloaded == DEPTH * n
        model.pool.close()
    assert torch.allclose(grads[1], grads[count], atol=1e-5, rtol=1e-4)
    # Each block is read once per step no matter how many micro-batches cross it.
    assert reads[1] == reads[count]


def test_uniform_block_ignores_unknown_kwargs() -> None:
    class Narrow(nn.Module):
        def forward(self, x, attention_mask=None):
            return x if attention_mask is None else x + 1

    wrapped = UniformBlock(Narrow(), frozenset({"attention_mask", "image_grid_thw"}))
    x = torch.zeros(2)
    assert torch.equal(wrapped(x, image_grid_thw=[1, 2, 3], position_ids=None), x)


def test_chunk_fused_experts_splits_expert_dimension() -> None:
    state = {
        "mlp.experts.gate_up_proj": torch.randn(8, 4, 6),
        "self_attn.q_proj.weight": torch.randn(4, 4),
    }
    chunked = chunk_fused_experts(state, splits=4)
    assert "self_attn.q_proj.weight" in chunked
    pieces = [chunked[f"mlp.experts.gate_up_proj.chunk{i}"] for i in range(4)]
    assert all(p.shape == (2, 4, 6) for p in pieces)
    assert torch.equal(torch.cat(pieces, dim=0), state["mlp.experts.gate_up_proj"])
    with pytest.raises(ValueError, match="not divisible"):
        chunk_fused_experts(state, splits=3)


def test_adapter_assignment_splits_experts_across_blocks() -> None:
    adapter = BaseAdapter(splits=4)
    assert adapter.assign("model.embed_tokens.weight") is None
    assert adapter.assign("visual.blocks.0.attn.qkv.weight") is None
    assert adapter.assign("model.layers.3.self_attn.q_proj.weight") == 12
    assigned = {
        adapter.assign(f"model.layers.3.mlp.experts.{e}.down_proj.weight")
        for e in range(8)
    }
    assert assigned == {12, 13, 14, 15}
    tensors = {
        "model.layers.0.self_attn.q_proj.weight": ("BF16", [4, 4], 32),
        "model.layers.0.mlp.experts.0.down_proj.weight": ("BF16", [4, 4], 32),
        "model.embed_tokens.weight": ("BF16", [8, 4], 64),
    }
    described = describe_blocks(tensors, adapter)
    assert described["resident_bytes"] == 64
    assert described["streamed_bytes"] == 64


def test_expert_major_moe_reads_each_expert_once() -> None:
    torch.manual_seed(8)
    reads: list[int] = []

    class Counting(nn.Module):
        def __init__(self, index: int) -> None:
            super().__init__()
            self.index = index
            self.linear = nn.Linear(4, 4)

        def forward(self, x):
            reads.append(self.index)
            return self.linear(x)

    experts = nn.ModuleList([Counting(i) for i in range(4)])
    moe = ExpertMajorMoE(experts, top_k=2)
    hidden = torch.randn(6, 4)
    logits = torch.randn(6, 4)
    out = moe(hidden, logits)
    assert out.shape == hidden.shape
    # Expert-major: no expert is visited twice regardless of token count.
    assert len(reads) == len(set(reads))


@CUDA
def test_mxfp4_matmul_gradient_and_memory() -> None:
    """The dense weight must not survive into backward.

    Autograd needs W for dL/dx, so a plain ``x @ dequantize(...)`` keeps every
    expert's 47 MiB weight alive for the whole block. Re-deriving it from the
    packed tensors is what lets a 128-expert layer back-propagate in 8 GB.
    """
    from block_streamer.gptoss import dequantize_expert, mxfp4_matmul

    torch.manual_seed(0)
    blocks = torch.randint(0, 255, (1, 64, 4, 16), dtype=torch.uint8, device="cuda")
    scales = torch.randint(120, 134, (1, 64, 4), dtype=torch.uint8, device="cuda")
    weight = dequantize_expert(blocks, scales, torch.float32)
    x = torch.randn(8, 128, device="cuda", requires_grad=True)
    reference_x = x.detach().clone().requires_grad_(True)

    actual = mxfp4_matmul(x, blocks, scales)
    expected = reference_x @ weight
    assert torch.equal(actual, expected)

    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    assert torch.equal(x.grad, reference_x.grad)

    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    chained = [mxfp4_matmul(x, blocks, scales) for _ in range(64)]
    peak = torch.cuda.max_memory_allocated() - base
    sum(y.sum() for y in chained).backward()
    dense_bytes = weight.numel() * weight.element_size()
    # Retaining one weight per call would cost 64x this; allow generous slack.
    assert peak < dense_bytes * 16


def test_dequantize_expert_requires_expert_dimension() -> None:
    from block_streamer.gptoss import dequantize_expert

    blocks = torch.randint(0, 255, (64, 4, 16), dtype=torch.uint8)
    scales = torch.randint(120, 134, (64, 4), dtype=torch.uint8)
    with pytest.raises(ValueError, match="packed blocks"):
        dequantize_expert(blocks, scales)
    out = dequantize_expert(blocks.unsqueeze(0), scales.unsqueeze(0), torch.float32)
    # [1, rows, groups, 16] -> [groups * 32, rows], the dense orientation.
    assert out.shape == (128, 64)
