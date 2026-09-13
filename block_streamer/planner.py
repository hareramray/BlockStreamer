"""Capacity, throughput, and endurance planning for tiered offload training.

Sizing comes from safetensors shard headers, never from a checkpoint index's
``metadata.total_size``: several published indexes report a parameter count
there rather than a byte count, which understates BF16 checkpoints by 2x.
"""

from __future__ import annotations

import json
import re
import struct
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

DTYPE_BYTES = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "F64": 8,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "BOOL": 1,
}

LAYER_RE = re.compile(r"\blayers\.(\d+)\.")
VISION_RE = re.compile(r"\b(visual|vision_tower|vision_model)\b")
GIB = 1024**3


class Verdict(StrEnum):
    RAM = "RAM"
    LORA_DISK = "LORA_DISK"
    FULL_FT_DISK = "FULL_FT_DISK"
    INFEASIBLE = "INFEASIBLE"


@dataclass(frozen=True)
class Hardware:
    """Measured properties of one machine. Bandwidths are bytes per second."""

    vram_bytes: int = 8 * GIB
    vram_reserved_bytes: int = 2 * GIB
    ram_bytes: int = int(15.7 * GIB)
    pinned_budget_bytes: int = 8 * GIB
    disk_free_bytes: int = int(330.9 * GIB)
    # Measured medians over three full LoRA runs on this machine:
    #
    #   model           streamed   GB/step  median s  effective
    #   Qwen3-VL-8B     12.9 GiB     20.8      14.2   1.47 GB/s
    #   gpt-oss-120b    58.6 GiB     94.4      82.3   1.15 GB/s
    #   Qwen3-VL-32B    58.1 GiB     93.6      94.1   1.00 GB/s
    #
    # At least three effects are tangled here and three points cannot separate
    # them: page cache (the 12.9 GiB set partly fits 15.7 GB of RAM, the 58 GiB
    # ones do not), per-block overhead (Qwen3-VL-32B does 96 block visits per
    # step against gpt-oss's 54, so its smaller blocks cost more per byte), and
    # MXFP4 dequantization compute in gpt-oss. Fitting a tidy formula to three
    # points would be a story, not a model.
    #
    # So this is deliberately the conservative end of the measured range, which
    # over-estimates step time for small models rather than under-estimating it
    # for large ones. Predictions against the runs above: 20.8 s vs 14.2
    # measured, 94.4 vs 82.3, 93.6 vs 94.1.
    read_bps: float = 1.0e9
    write_bps: float = 0.9e9
    endurance_bytes: float = 300e12

    def effective_read_bps(self, streamed_bytes: int) -> float:
        """Conservative constant rate; see read_bps for why it is not fitted."""
        return self.read_bps


DEFAULT_HARDWARE = Hardware()


@dataclass
class Tensor:
    name: str
    dtype: str
    shape: list[int]
    nbytes: int

    @property
    def numel(self) -> int:
        count = 1
        for dim in self.shape:
            count *= dim
        return count


@dataclass
class Profile:
    """Per-block byte sizes derived from a real checkpoint manifest."""

    name: str
    tensors: list[Tensor]
    layer_bytes: dict[int, int] = field(default_factory=dict)
    resident_bytes: int = 0
    vision_bytes: int = 0

    def __post_init__(self) -> None:
        for tensor in self.tensors:
            match = LAYER_RE.search(tensor.name)
            if VISION_RE.search(tensor.name):
                self.vision_bytes += tensor.nbytes
            elif match is not None:
                index = int(match.group(1))
                self.layer_bytes[index] = self.layer_bytes.get(index, 0) + tensor.nbytes
            else:
                self.resident_bytes += tensor.nbytes

    @property
    def total_bytes(self) -> int:
        return sum(tensor.nbytes for tensor in self.tensors)

    @property
    def total_params(self) -> int:
        return sum(tensor.numel for tensor in self.tensors)

    @property
    def streamed_bytes(self) -> int:
        return sum(self.layer_bytes.values())

    @property
    def largest_block_bytes(self) -> int:
        return max(self.layer_bytes.values(), default=0)

    @property
    def num_layers(self) -> int:
        return len(self.layer_bytes)

    def group_bytes(self, pattern: str) -> int:
        regex = re.compile(pattern)
        return sum(t.nbytes for t in self.tensors if regex.search(t.name))


@dataclass
class Plan:
    profile: Profile
    hardware: Hardware
    mode: str
    verdict: Verdict
    reasons: list[str]
    disk_bytes: int
    read_per_step: int
    write_per_step: int
    seconds_per_step: float
    steps_to_tbw: float
    vram_for_slots: int
    prefetch_ahead: int

    def row(self) -> str:
        tbw = "-" if self.write_per_step == 0 else f"{self.steps_to_tbw:,.0f}"
        return (
            f"{self.profile.name:22}{self.profile.total_params / 1e9:>8.1f}B"
            f"{self.profile.total_bytes / GIB:>9.0f}G"
            f"{self.disk_bytes / GIB:>10.0f}G"
            f"{self.seconds_per_step:>9.0f}s"
            f"{tbw:>13}  {self.verdict}"
        )


def read_header(url: str, timeout: int = 60) -> dict[str, Any]:
    """Read one safetensors header via two ranged requests."""
    request = urllib.request.Request(url, headers={"Range": "bytes=0-7"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        length = struct.unpack("<Q", response.read())[0]
    request = urllib.request.Request(url, headers={"Range": f"bytes=8-{7 + length}"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def iter_shards(repo: str, cache: Path) -> Iterator[str]:
    index_path = cache / "index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
    else:
        url = f"https://huggingface.co/{repo}/resolve/main/model.safetensors.index.json"
        index = json.loads(urllib.request.urlopen(url, timeout=60).read())
        cache.mkdir(parents=True, exist_ok=True)
        index_path.write_text(json.dumps(index))
    yield from sorted(set(index["weight_map"].values()))


def fetch_manifest(name: str, repo: str, cache_root: Path) -> list[Tensor]:
    """Collect every tensor's dtype/shape/bytes, caching the result on disk."""
    cache = cache_root / name
    manifest_path = cache / "tensors.json"
    if manifest_path.exists():
        raw = json.loads(manifest_path.read_text())
        return [Tensor(k, v[0], v[1], v[2]) for k, v in raw.items()]
    tensors: dict[str, list[Any]] = {}
    for shard in iter_shards(repo, cache):
        header = read_header(f"https://huggingface.co/{repo}/resolve/main/{shard}")
        for key, value in header.items():
            if key == "__metadata__":
                continue
            numel = 1
            for dim in value["shape"]:
                numel *= dim
            nbytes = numel * DTYPE_BYTES.get(value["dtype"], 2)
            tensors[key] = [value["dtype"], value["shape"], nbytes]
    cache.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(tensors))
    return [Tensor(k, v[0], v[1], v[2]) for k, v in tensors.items()]


def plan(
    profile: Profile,
    hardware: Hardware = DEFAULT_HARDWARE,
    mode: str = "lora",
    prefetch_ahead: int = 1,
    lora_fraction: float = 0.5,
    optimizer_bytes_per_param: int = 12,
    min_steps_to_tbw: float = 5000,
) -> Plan:
    """Decide whether `mode` is feasible and report its measured-rate costs.

    ``mode`` is "lora" (frozen base, no writes) or "full" (fused optimizer step,
    which reads and rewrites optimizer state for every parameter every step).
    """
    reasons: list[str] = []
    params = profile.total_params
    weight_bytes = profile.total_bytes

    if mode == "lora":
        disk_bytes = weight_bytes
        # Resident state (embeddings, head, vision) is loaded once at startup
        # and stays on GPU, so it is not part of any step's read.
        read_per_step = int(profile.streamed_bytes * (1 + lora_fraction))
        write_per_step = 0
    else:
        disk_bytes = weight_bytes + params * optimizer_bytes_per_param
        read_per_step = weight_bytes * 2 + params * optimizer_bytes_per_param
        write_per_step = params * optimizer_bytes_per_param

    seconds = read_per_step / hardware.effective_read_bps(profile.streamed_bytes)
    if write_per_step:
        seconds += write_per_step / hardware.write_bps
    steps_to_tbw = (
        float("inf")
        if write_per_step == 0
        else hardware.endurance_bytes / write_per_step
    )

    # Vision towers stay resident on GPU; embed_tokens gathers on CPU so only
    # the selected rows transfer; lm_head streams as the final block.
    slots = prefetch_ahead + 1
    vram_available = hardware.vram_bytes - hardware.vram_reserved_bytes
    vram_resident = profile.vision_bytes
    split = 1
    while (
        vram_resident + profile.largest_block_bytes / split * slots > vram_available
        and split < 64
    ):
        split *= 2
    vram_for_slots = int(profile.largest_block_bytes / split * slots)

    verdict = Verdict.LORA_DISK if mode == "lora" else Verdict.FULL_FT_DISK
    if split > 1:
        reasons.append(
            f"sub-split each layer into {split} blocks "
            f"({profile.largest_block_bytes / GIB:.2f} -> "
            f"{profile.largest_block_bytes / split / GIB:.2f} GiB) to fit "
            f"{slots} slots beside {vram_resident / GIB:.2f} GiB resident"
        )

    if weight_bytes <= hardware.pinned_budget_bytes and mode == "lora":
        verdict = Verdict.RAM
        reasons.append(
            f"weights {weight_bytes / GIB:.1f} GiB fit the "
            f"{hardware.pinned_budget_bytes / GIB:.0f} GiB pinned budget; "
            "the existing RAM path already works"
        )

    if disk_bytes > hardware.disk_free_bytes:
        verdict = Verdict.INFEASIBLE
        reasons.append(
            f"needs {disk_bytes / GIB:.0f} GiB but only "
            f"{hardware.disk_free_bytes / GIB:.0f} GiB free; "
            "store weights in FP8/MXFP4 or pick a smaller model"
        )

    if vram_resident + vram_for_slots > vram_available:
        verdict = Verdict.INFEASIBLE
        reasons.append(
            f"even at a {split}-way split, {vram_resident / GIB:.2f} GiB "
            f"resident + {slots} slots = "
            f"{(vram_resident + vram_for_slots) / GIB:.2f} GiB exceeds the "
            f"{vram_available / GIB:.0f} GiB VRAM budget"
        )

    if mode == "full" and steps_to_tbw < min_steps_to_tbw:
        verdict = Verdict.INFEASIBLE
        reasons.append(
            f"writes {write_per_step / GIB:.0f} GiB/step, exhausting rated "
            f"endurance in {steps_to_tbw:,.0f} steps (< {min_steps_to_tbw:,.0f}); "
            "use LoRA instead"
        )

    return Plan(
        profile=profile,
        hardware=hardware,
        mode=mode,
        verdict=verdict,
        reasons=reasons,
        disk_bytes=disk_bytes,
        read_per_step=read_per_step,
        write_per_step=write_per_step,
        seconds_per_step=seconds,
        steps_to_tbw=steps_to_tbw,
        vram_for_slots=vram_for_slots,
        prefetch_ahead=prefetch_ahead,
    )


TARGETS = {
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "Qwen3-VL-8B": "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen3-VL-32B": "Qwen/Qwen3-VL-32B-Instruct",
    "Qwen3-VL-235B-A22B": "Qwen/Qwen3-VL-235B-A22B-Instruct",
    "GLM-4.6V": "zai-org/GLM-4.6V",
}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("configs"))
    parser.add_argument("--models", nargs="*", default=list(TARGETS))
    parser.add_argument("--prefetch-ahead", type=int, default=1)
    args = parser.parse_args()

    hardware = Hardware()
    print(
        f"{'model':22}{'params':>9}{'ckpt':>9}{'disk need':>10}"
        f"{'s/step':>9}{'steps->TBW':>13}  verdict"
    )
    profiles: dict[str, Profile] = {}
    for name in args.models:
        tensors = fetch_manifest(name, TARGETS[name], args.cache)
        profile = Profile(name, tensors)
        profiles[name] = profile
        for mode in ("lora", "full"):
            result = plan(profile, hardware, mode, args.prefetch_ahead)
            label = f"{name} [{mode}]"
            print(f"{label:22}", end="")
            print(result.row()[22:])
            for reason in result.reasons:
                print(f"{'':22}  - {reason}")

    print("\nper-model structure")
    for name, profile in profiles.items():
        print(
            f"  {name:22} layers={profile.num_layers:>3} "
            f"largest_block={profile.largest_block_bytes / GIB:6.2f} GiB "
            f"resident={profile.resident_bytes / GIB:6.2f} GiB "
            f"vision={profile.vision_bytes / GIB:5.2f} GiB"
        )


if __name__ == "__main__":
    main()
