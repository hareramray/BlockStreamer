"""Measure the transfer/compute crossover and export an actual CUDA timeline."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import statistics
import time
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from block_streamer import StreamedModel
from memory_pool import module_state, tensor_bytes


class BenchmarkBlock(nn.Module):
    def __init__(self, width: int, expansion: int) -> None:
        super().__init__()
        self.up = nn.Linear(width, expansion, bias=False)
        self.down = nn.Linear(expansion, width, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return x + 0.1 * self.down(torch.relu(self.up(x)))


def timings(
    forward: Callable[[], Any],
    device: torch.device,
    warmup: int,
    runs: int,
) -> dict[str, float]:
    for _ in range(warmup):
        forward()
    samples = []
    for _ in range(runs):
        # Exclude previously queued work from this forward's wall-clock interval.
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        forward()
        # Include all asynchronous compute/copies before stopping the timer.
        torch.cuda.synchronize(device)
        samples.append((time.perf_counter() - start) * 1000)
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "variance_ms2": statistics.pvariance(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def measure_peak(forward: Callable[[], Any], device: torch.device) -> tuple[int, int]:
    # A clean device interval is necessary for reset_peak_memory_stats to be valid.
    torch.cuda.synchronize(device)
    start = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    output = forward()
    # Keep the returned activation alive through peak sampling.
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    del output
    return start, peak


def transfer_only(model: StreamedModel) -> float:
    """Isolated H2D CUDA-event time; one block at a time, no compute."""
    elapsed = 0.0
    for block in model.blocks:
        with torch.cuda.stream(model.transfer_stream):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(model.transfer_stream)
            sources = module_state(block)
            assert all(t.is_pinned() for t in sources.values())
            copies = [t.to(model.device, non_blocking=True) for t in sources.values()]
            end.record(model.transfer_stream)
        # Copies must finish before references are released or elapsed time is read.
        end.synchronize()
        elapsed += start.elapsed_time(end)
        del copies
    return elapsed


def compute_only_cuda(blocks: nn.Sequential, x: Tensor) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = blocks(x)
    end.record()
    # GPU event durations are only readable after the end event completes.
    end.synchronize()
    del output
    return start.elapsed_time(end)


def benchmark_case(
    width: int,
    tokens: int,
    config: argparse.Namespace,
    trace: bool,
) -> dict[str, Any]:
    device = torch.device(config.device)
    dtype = getattr(torch, config.dtype)
    torch.manual_seed(12)
    cpu = (
        nn.Sequential(
            *[
                BenchmarkBlock(width, width * (2 + index % 2))
                for index in range(config.blocks)
            ]
        )
        .to(dtype=dtype)
        .eval()
    )
    x = torch.randn(tokens, width, device=device, dtype=dtype)
    baseline = copy.deepcopy(cpu).to(device)
    parameters = sum(tensor_bytes(p) for p in baseline.parameters())
    baseline_time = timings(
        lambda model=baseline: model(x), device, config.warmup, config.runs
    )
    gpu_compute = statistics.median(
        [compute_only_cuda(baseline, x) for _ in range(config.runs)]
    )
    baseline_start, baseline_peak = measure_peak(
        lambda model=baseline: model(x), device
    )
    # Weights were already allocated at reset. The incremental allocation measures
    # activations + temporary workspaces, not a guessed split of streamed peaks.
    activation_workspace_peak = baseline_peak - baseline_start
    del baseline
    gc.collect()
    torch.cuda.empty_cache()

    with StreamedModel(
        cpu, device=device, dtype=dtype, prefetch_ahead=config.prefetch_ahead
    ) as streamed:
        for _ in range(config.warmup):
            transfer_only(streamed)
        transfer_ms = statistics.median(
            [transfer_only(streamed) for _ in range(config.runs)]
        )
        streamed_time = timings(lambda: streamed(x), device, config.warmup, config.runs)
        streamed.reset_stats()
        streamed_start, streamed_peak = measure_peak(lambda: streamed(x), device)
        stream_stats = streamed.stats()
        if trace:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=True,
            ) as profiler:
                streamed(x)
                # Complete GPU work before the profiler closes its collection window.
                torch.cuda.synchronize(device)
            profiler.export_chrome_trace(str(config.output / "trace.json"))
            trace_data = json.loads((config.output / "trace.json").read_text())
            categories = {event.get("cat", "") for event in trace_data["traceEvents"]}
            config.trace_gpu_activity = bool(categories & {"kernel", "gpu_memcpy"})
            if not config.trace_gpu_activity:
                print(
                    "Profiler trace has no GPU activities: check CUPTI/driver support. "
                    "This CPU trace cannot verify overlap.",
                    flush=True,
                )

    overhead = streamed_time["median_ms"] - baseline_time["median_ms"]
    return {
        "width": width,
        "tokens": tokens,
        "blocks": config.blocks,
        "baseline": baseline_time,
        "streamed": streamed_time,
        "compute_only_gpu_ms": gpu_compute,
        "transfer_only_ms": transfer_ms,
        "overlap_efficiency": 1 - overhead / transfer_ms if transfer_ms else None,
        "regime": "compute-bound" if gpu_compute >= transfer_ms else "bandwidth-bound",
        "baseline_peak_allocated_bytes": baseline_peak,
        "baseline_parameter_bytes": parameters,
        "baseline_activation_workspace_increment_bytes": activation_workspace_peak,
        "input_and_existing_allocations_bytes": baseline_start - parameters,
        "streamed_peak_allocated_bytes": streamed_peak,
        "streamed_peak_increment_bytes": streamed_peak - streamed_start,
        "streamed_peak_parameter_payload_bytes": stream_stats["peak_parameter_bytes"],
        "streamed_peak_state_payload_bytes": stream_stats["peak_state_bytes"],
        "streamed_stats": stream_stats,
        "effective_h2d_gb_s": parameters / (transfer_ms * 1e6) if transfer_ms else None,
    }


def write_chart(rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for tokens in sorted({row["tokens"] for row in rows}):
        subset = [row for row in rows if row["tokens"] == tokens]
        sizes = [
            row["baseline_parameter_bytes"] / row["blocks"] / 2**20 for row in subset
        ]
        axes[0].plot(
            sizes,
            [row["compute_only_gpu_ms"] / row["blocks"] for row in subset],
            "o-",
            label=f"Compute, {tokens} tokens",
        )
    first = [row for row in rows if row["tokens"] == rows[0]["tokens"]]
    axes[0].plot(
        [row["baseline_parameter_bytes"] / row["blocks"] / 2**20 for row in first],
        [row["transfer_only_ms"] / row["blocks"] for row in first],
        "s--",
        color="black",
        label="Measured H2D (first token count)",
    )
    axes[0].set(
        xlabel="Mean parameter MiB / block",
        ylabel="GPU ms / block",
        title="Crossover: compute vs. measured H2D",
    )
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    for index, tokens in enumerate(sorted({row["tokens"] for row in rows})):
        subset = [row for row in rows if row["tokens"] == tokens]
        widths = [row["width"] for row in subset]
        axes[1].plot(
            widths,
            [row["baseline_peak_allocated_bytes"] / 2**20 for row in subset],
            "o-",
            color=f"C{index}",
            label=f"GPU, {tokens} tokens",
        )
        axes[1].plot(
            widths,
            [row["streamed_peak_allocated_bytes"] / 2**20 for row in subset],
            "s--",
            color=f"C{index}",
            label=f"Streamed, {tokens} tokens",
        )
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(
        sorted({row["width"] for row in rows}),
        labels=sorted({row["width"] for row in rows}),
    )
    axes[1].set(
        xlabel="Block width",
        ylabel="Peak allocated MiB",
        title="Total peak VRAM (parameters + activations)",
    )
    for axis in axes:
        axis.legend(fontsize=8)
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "roofline.png", dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16"
    )
    parser.add_argument("--widths", nargs="+", type=int, default=[256, 512, 1024])
    parser.add_argument("--tokens", nargs="+", type=int, default=[32, 512, 2048])
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--prefetch-ahead", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--no-trace", action="store_true")
    config = parser.parse_args()
    config.trace_gpu_activity = None
    if min(config.widths + config.tokens + [config.blocks, config.runs]) <= 0:
        parser.error("widths, tokens, blocks, and runs must be positive")
    if config.warmup < 0 or config.prefetch_ahead < 0:
        parser.error("warmup and prefetch-ahead must be nonnegative")
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable; the benchmark requires an NVIDIA GPU")
    torch.cuda.set_device(config.device)
    config.output.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.inference_mode():
        for tokens in config.tokens:
            for width in config.widths:
                row = benchmark_case(
                    width, tokens, config, trace=not rows and not config.no_trace
                )
                rows.append(row)
                print(
                    f"width={width} tokens={tokens}: {row['regime']}; "
                    f"compute={row['compute_only_gpu_ms']:.3f} ms, "
                    f"H2D={row['transfer_only_ms']:.3f} ms",
                    flush=True,
                )
    crossovers = []
    for tokens in config.tokens:
        subset = sorted(
            (row for row in rows if row["tokens"] == tokens),
            key=lambda row: row["width"],
        )
        transitions = [
            (left["width"], right["width"])
            for left, right in pairwise(subset)
            if left["regime"] != right["regime"]
        ]
        crossovers.append({"tokens": tokens, "width_brackets": transitions})
        print(
            f"Crossover at {tokens} tokens: "
            f"{transitions if transitions else 'not observed in sampled widths'}"
        )
    report = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dtype": config.dtype,
        "prefetch_ahead": config.prefetch_ahead,
        "runs": config.runs,
        "trace_gpu_activity": config.trace_gpu_activity,
        "assumed_pcie_gen4_x16_gb_s": 31.5,
        "memory_note": "Activation/workspace peak is independently measured on the "
        "baseline. Independent maxima cannot exactly decompose the "
        "streamed total peak. Payload bytes exclude allocator rounding.",
        "crossovers": crossovers,
        "cases": rows,
    }
    (config.output / "benchmark.json").write_text(json.dumps(report, indent=2))
    write_chart(rows, config.output)


if __name__ == "__main__":
    main()
