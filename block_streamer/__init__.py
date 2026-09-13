"""Public BlockStreamer API.

``streamer`` and ``memory_pool`` remain top-level modules for compatibility with
0.1.0. Everything added since lives under this package rather than claiming more
top-level import names, several of which belong to unrelated projects on PyPI.

Submodules, imported explicitly:

``block_streamer.shards``       flat per-block shard files and their manifest
``block_streamer.diskpool``     disk -> pinned ring -> GPU staging
``block_streamer.planner``      capacity/throughput/endurance verdicts
``block_streamer.lora``         adapters over streamed frozen base weights
``block_streamer.microbatch``   micro-batch looping, activation offload
``block_streamer.fused_optim``  per-block Adam with disk-resident state
``block_streamer.adapters``     per-family block assignment, expert-major MoE
``block_streamer.gptoss``       MXFP4 experts dequantized one at a time
``block_streamer.runners``      end-to-end drivers (needs transformers)

``planner``, ``runners``, and ``gptoss`` additionally require ``transformers``
and ``safetensors``; install with ``pip install "block-streamer[models]"``.
"""

from streamer import StreamedModel

from .lora import LoRAConfig, LoRALinear
from .shards import ShardManifest

__all__ = ["LoRAConfig", "LoRALinear", "ShardManifest", "StreamedModel"]
