# BlockStreamer

Licensed under the [Apache License 2.0](https://github.com/hareramray/BlockStreamer/blob/main/LICENSE).

Execute ordered PyTorch blocks with master weights in pinned CPU memory **or on
NVMe**, asynchronous CUDA prefetch, and at most `prefetch_ahead + 1` resident
blocks. Includes inference, first-order training with backward re-fetch, LoRA
finetuning of models far larger than host RAM, a fused per-block optimizer,
correctness tests, and a measured transfer/compute crossover benchmark.

Three models have been LoRA-finetuned end to end on a single 8 GB laptop GPU:
Qwen3-VL-8B, Qwen3-VL-32B, and gpt-oss-120b. See
[NVMe offload and LoRA finetuning](#nvme-offload-and-lora-finetuning).

## Prior art and binding decision

Accelerate's `cpu_offload_with_hook` leaves a module on the execution device until
its offload hook runs; BlockStreamer instead schedules an explicit bounded window
of individual blocks and transfer-ready events.
[Accelerate documentation](https://huggingface.co/docs/accelerate/en/package_reference/big_modeling#accelerate.cpu_offload_with_hook)
ZeRO-Infinity is a broader training system that spans GPU, CPU, and NVMe memory
across many devices. BlockStreamer now spans the same three tiers, but for one GPU
and an ordered block list, with no sharding, collectives, or partitioning: the unit
of movement is a block, not a shard, and the residency bound is a block count the
caller sets.
[ZeRO-Infinity paper](https://arxiv.org/abs/2104.07857)
FSDP `CPUOffload` moves parameters and gradients to CPU within FSDP's sharding and
distributed training machinery; this library has no distributed collectives or
parameter sharding.
[FSDP documentation](https://docs.pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.CPUOffload)
BlockStreamer's smaller scope makes the residency policy explicit, at the cost of
less general model execution and substantial Python/event overhead for tiny blocks.

**Binding mechanism: `torch.func.functional_call`.** Each block executes with a
dictionary of its staged GPU parameters and buffers, with strict key checking and
PyTorch's tied-weight handling. Registered parameters stay on pinned CPU storage,
or on `meta` when a shard manifest supplies them from disk.
Construction and `.to()` replace CPU storage to pin/convert it while preserving
`Parameter` identity; `.data` is never used to bind staged GPU weights for compute.
[Functional-call documentation](https://docs.pytorch.org/docs/stable/generated/torch.func.functional_call.html)

The tradeoff is functional state discipline: blocks must not mutate registered
parameters or buffers, retain staged tensors, or use custom kernels that cache
parameter pointers outside module lookups. Detected registered-state mutation
raises an error. Python attributes, global state, hooks, and opaque external caches
cannot all be audited automatically; callers must keep those free of side effects.
Training BatchNorm's running-state updates and mutable KV caches are unsupported.
Ordinary dropout is supported through RNG replay during recomputation.

## Install and run

Requires Python 3.11+, PyTorch 2.4+, and a CUDA GPU. Install a CUDA-enabled PyTorch
wheel suitable for your GPU first. For the published package:

```bash
python -m pip install block-streamer
```

`torch` is the only hard dependency. The streaming core, LoRA, and the fused
optimizer need nothing else. `block_streamer.planner`, `block_streamer.runners`, and `block_streamer.gptoss` read real
Hugging Face checkpoints and additionally need `transformers` and `safetensors`;
they import these lazily, so the rest of the package works without them:

```bash
python -m pip install "block-streamer[models]"
```

For development, clone the repository and install from its root:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

The development environment here uses `.venv` with the preinstalled CUDA PyTorch
available through system site packages. On this Windows workspace:

```powershell
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe benchmark.py --widths 128 256 512 1024 --tokens 32 2048 8192 --runs 10 --warmup 4
```

```python
import torch
from torch import nn
from block_streamer import StreamedModel

blocks = nn.Sequential(nn.Linear(256, 512), nn.GELU(), nn.Linear(512, 256))
x = torch.randn(16, 256, device="cuda:0", dtype=torch.bfloat16)

with StreamedModel(
    blocks, device="cuda:0", prefetch_ahead=1, dtype=torch.bfloat16
) as streamed:
    out = streamed(x)
    print(streamed.stats())
```

Construction takes ownership of the supplied `Sequential`, `ModuleList`, or list
of modules, pins their parameters **and buffers** on CPU, and defaults to `.eval()`.
Make a deepcopy first if another model must retain independent state. Evaluation
forwards disable autograd, including when the caller did not use `no_grad()`.
Call `.train()` to enable the training path. `.eval()` propagates to every block.

Inputs must already be on the execution device and have a suitable dtype. Additional
positional and keyword arguments are passed unchanged to **every** block:

```python
out = streamed(x, attention_mask=mask, position_ids=positions)
```

Every block must accept the supplied arguments. A block may return a tensor or a
tuple whose first element is the next hidden state. Intermediate auxiliary tuple
elements are discarded; the final block's entire result is returned. Tensor leaves
in ordinary tuple/list/dict output trees are cloned so a returned parameter view
cannot keep staged weights resident. There is no automatic per-layer KV-cache
selection or collection; use adapter blocks to express that policy. Read-only
attention masks, positions, rotary tensors, and cache values can pass through.

`streamed.to(device="cuda:0", dtype=torch.float16)` changes execution placement and
repins CPU state. Perform conversion before constructing an optimizer or a pending
training graph. CPU execution and inherited `.cuda()`, `.cpu()`, `.half()`, and
parent-module conversions are rejected; use this wrapper's `.to()` instead.
`close()` and context exit drain outstanding work; the wrapper remains reusable.

## Streams, residency, and memory budget

The engine uses two streams: the compute stream active at construction and a
separate transfer stream. Borrowing the compute stream avoids creating persistent
cuBLAS workspaces for every new wrapper. A call made on another CUDA stream is
bridged to the compute stream, with input storage protected by `record_stream()`.

The scheduler initially stages up to `prefetch_ahead + 1` blocks. Compute waits on
each block's H2D-ready event. **Every transfer-created tensor consumed by compute
calls `record_stream(compute_stream)` in `BufferPool.acquire()`**, protecting it
from premature caching-allocator reuse.
[PyTorch allocator documentation](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html)
A compute-completion event is host-synchronized before eviction and replacement
staging. This conservative retirement rule enforces the actual live-block limit,
including in-flight transfers, and introduces host scheduling overhead. The forward
returns after its work completes; H2D and compute can still overlap within it.

The GPU pool uses PyTorch's caching allocator with event-retired slots. Allocations
are sized per block, so heterogeneous shapes, integer buffers, and mixed state
dtypes require no homogeneous-buffer fallback. `prefetch_ahead=0` is supported;
`1` provides two slots. Parameter-free blocks still count as logical slots.
CUDA **reserved** cache memory may exceed live tensor payload and is not the
residency invariant. For heterogeneous models the byte bound is the largest live
window's summed state size, not a fixed number of bytes per block.

Set `buffer_budget_bytes=...` for a per-slot parameter-plus-buffer payload limit.
An oversized block raises a clear error before its copy. This is not a total-VRAM
budget: activations, CUDA library workspaces, gradients, and allocator rounding
are additional. Every H2D source is checked for pinned CPU storage immediately
before copying; an unpinned source fails loudly. Meta/sparse state and distinct
tensor objects sharing backing storage are rejected. Weight tying by sharing the
same `Parameter` object is supported within and across blocks.

`stats()` reports cumulative H2D bytes/time, D2H gradient bytes, GPU transfer-wait
time (`stall_ms`), host eviction wait time, current/peak resident blocks, and peak
parameter/state payload bytes. GPU stall time measures event waits on the compute
stream; it excludes host launch delays. Host eviction wait can include compute
and transfer waiting, so these two times must not be added as independent costs.
`residency_history` retains the last 4,096 transitions; an optional
`residency_observer` receives every transition. `reset_stats()` resets all counters.

## Training design and usage

The inference suite passed on CUDA before the training implementation was added.
The training design makes these choices:

1. **Re-fetch weights during backward.** Keeping forward weights would require
   O(all blocks) parameter VRAM. Instead, save block inputs on GPU, discard staged
   weights, and re-fetch/recompute one block at a time during backward. A training
   step transfers each block's state twice and performs an extra forward compute.
   Backward stops at the lowest block holding a trainable parameter, so adapting
   only the top layers halves the re-fetch rather than merely discarding the
   gradients. Saved activations are O(depth) on GPU by default;
   `block_streamer.microbatch` offloads them to pinned host memory when depth makes that
   impossible.
2. **Offload gradients after each block.** Compute that block's gradients on GPU,
   copy them into pinned CPU tensors on the compute stream, and make CPU reads wait
   for the completion event. Return CPU parameter gradients through autograd so
   normal `.grad` accumulation works. Peak GPU memory also includes one block's
   gradients and recomputation graph, plus saved activations and input gradients.
   Shared parameters' CPU gradient contributions are summed after copies finish.
3. **CPU optimizer and state, by default.** An ordinary CPU optimizer on
   `model.parameters()` runs its step on CPU and does not stream to GPU. This
   incurs host RAM for every gradient and every moment, which is what caps this
   path near 1B parameters on a 16 GB host. `block_streamer.fused_optim.FusedDiskAdam` lifts
   that cap by consuming each block's gradients during backward; see
   [NVMe offload and LoRA finetuning](#nvme-offload-and-lora-finetuning).
   No automatic fp32 master-weight conversion or integrated loss scaling is
   provided.
4. **Custom `torch.autograd.Function`.** It owns the forward/backward schedule,
   saves version-checked master state and block inputs, then visits blocks in
   reverse with the same bounded prefetch window. Block N-1 is fetched ahead of
   its recompute/backward while block N executes. This avoids relying on module
   hook order and explicitly returns gradients for tensor arguments and weights.

```python
streamed = StreamedModel(blocks, dtype=torch.float32).train()
optimizer = torch.optim.Adam(streamed.parameters(), lr=1e-3)
x = torch.randn(16, 256, device="cuda")
target = torch.zeros_like(x)

for _ in range(10):
    optimizer.zero_grad(set_to_none=True)
    loss = (streamed(x) - target).float().square().mean()
    loss.backward()
    optimizer.step()  # CPU parameters, CPU gradients, CPU optimizer state
```

Recomputation restores PyTorch CPU/CUDA RNG states and CUDA autocast settings.
Backward does not advance the caller's RNG. First-order gradients, frozen/unused
parameters, tied weights, extra tensor arguments, and final tuple-output losses
are supported. Higher-order differentiation, `torch.compile`, CUDA graph capture,
distributed execution, and concurrent/reentrant calls are unsupported. Modules
must not modify block inputs in place, change execution behavior between forward
and backward, or retain GPU weights in external state. Normal in-place changes to
saved tensors are version-checked; edits through `.data` bypass PyTorch's protection.

## NVMe offload and LoRA finetuning

Keeping master weights in pinned host RAM caps the model near 1B parameters on a
16 GB machine: weights, gradients, and Adam state together cost about 8 bytes per
parameter. Moving only the **weights** to disk barely helps, because gradients and
optimizer state still sit in RAM (about 6 B/param, so ~1.3x). Two things actually
raise the ceiling:

* **LoRA over a streamed, frozen base.** The base is read-only, so there is no
  optimizer state and nothing is written back. Disk holds 2 B/param and SSD
  endurance is a non-issue.
* **A fused per-block optimizer** (`block_streamer.fused_optim`) that reads a block's moments,
  updates on GPU, writes back, and discards the gradient before the next block.
  Host memory then holds one block rather than the model. It writes 12 B/param
  every step, which exhausts a 300 TBW consumer SSD in a few hundred steps for
  any model above ~8B, so `block_streamer.planner` gates it behind an explicit check.

Per-parameter cost and the resulting ceiling, for bf16 compute with Adam on a
16 GB host and ~330 GB of free disk:

| mode | RAM B/param | disk B/param | writes/step | max params |
| --- | ---: | ---: | ---: | ---: |
| all pinned RAM (the original path) | 8 | 0 | 0 | ~1.0B |
| weights to disk only | 6 | 2 | 0 | ~1.3B |
| full finetune, fused step, fp32 Adam | ~0 | 14 | 12 | ~21B |
| full finetune, fused step, 8-bit Adam | ~0 | 8 | 6 | ~37B |
| **LoRA, frozen base** | ~0 | 2 | 0 | **~150B** |

The last row is why LoRA is the default. Writing nothing per step removes both the
endurance limit and two thirds of the per-step traffic.

Start with `block_streamer.planner`, which reads a checkpoint's real safetensors headers and
reports capacity, throughput, and endurance before anything is downloaded:

```bash
python -m block_streamer.planner --models gpt-oss-120b Qwen3-VL-32B
```

It computes sizes from tensor shapes, never from an index's `metadata.total_size`
-- several published indexes report a parameter count there rather than a byte
count, which understates a BF16 checkpoint by 2x. Verdicts are `RAM` (the existing
pinned path already fits, so none of this is needed), `LORA_DISK`, `FULL_FT_DISK`,
or `INFEASIBLE`, with a reason for each rejection: too large for free disk, too
large for VRAM even after sub-splitting layers, or too few steps before the drive's
rated write endurance is exhausted.

### End to end

```python
import torch
from block_streamer.lora import LoRAConfig
from block_streamer.runners import Qwen3VLRunner

# Converts the checkpoint to per-block shards on first use, then streams them.
runner = Qwen3VLRunner(
    "models/Qwen3-VL-8B",
    "models/shards-qwen8b",
    LoRAConfig(rank=8, alpha=16, top_fraction=0.5),
    prefetch_ahead=2,
    dtype=torch.bfloat16,
).train()

optimizer = torch.optim.AdamW(runner.trainable_parameters(), lr=2e-4)
logits = runner(input_ids)
loss = torch.nn.functional.cross_entropy(
    logits[:, :-1].reshape(-1, logits.shape[-1]).float(), input_ids[:, 1:].reshape(-1)
)
loss.backward()
optimizer.step()
```

`.train()` matters: `StreamedModel` constructs in `eval()`, and the inference path
runs under `no_grad`, so a forward taken in eval mode produces a loss with no
`grad_fn`. `top_fraction=0.5` puts adapters on the top half of the decoder, which
halves the backward re-fetch because the walk stops at the lowest trainable block.

`train_qwen8b.py`, `train_qwen32b.py`, and `train_gptoss.py` are the exact scripts
used for the runs below.

### Measured results

Three models LoRA-finetuned end to end on one RTX 5050 Laptop (8 GB VRAM,
15.7 GB RAM, consumer NVMe), bf16, rank 8, adapters on the top half of layers,
sequence length 128:

| model | streamed | blocks | peak VRAM | median s/step | loss |
| --- | ---: | ---: | ---: | ---: | --- |
| Qwen3-VL-8B | 12.94 GiB | 36 x 368 MiB | 3.52 GiB | 14.2 | 13.14 -> 0.33 (60 steps) |
| gpt-oss-120b | 58.61 GiB | 36 x 1.63 GiB | 5.69 GiB | 82.3 | 13.69 -> 0.40 (30 steps) |
| Qwen3-VL-32B | 58.13 GiB | 64 x 0.91 GiB | 5.83 GiB | 94.1 | 13.45 -> 4.83 (30 steps) |

These runs overfit a single fixed token sequence. They demonstrate that gradients
flow correctly through the streaming machinery end to end; they are **not** useful
finetunes on real data. Qwen3-VL-32B was still descending steeply at step 30, with
10M adapter parameters against Qwen3-VL-8B's 3.8M.

Effective throughput was 1.47, 1.15, and 1.00 GB/s respectively. At least three
effects are mixed into that spread and three measurements cannot separate them:
page cache (12.9 GiB partly fits in 15.7 GB of RAM, 58 GiB does not),
per-block overhead (Qwen3-VL-32B makes 96 block visits per step against
gpt-oss's 54, so its smaller blocks cost more per byte), and MXFP4 dequantization
compute in gpt-oss. `block_streamer.planner` therefore assumes a flat, deliberately
conservative 1.0 GB/s rather than fitting a curve to three points; it predicts
21/94/94 s against the 14.2/82.3/94.1 measured, erring toward over-estimating.

### MXFP4 and the backward pass

gpt-oss ships its experts packed: one layer is 1.63 GiB of `uint8` blocks and
scales. Upcasting the checkpoint to bf16 would take it from 61 GiB to roughly
230 GiB, and dequantizing a single *layer* yields 5.93 GiB of dense experts, which
does not fit beside anything else on an 8 GB card. So the packed tensors stream
verbatim and each expert is dequantized inside the routing loop, 47 MiB at a time.

That is sufficient for forward and insufficient for backward. Autograd needs the
dense weight to compute `dL/dx`, so it retains every expert it touched -- about
6 GiB per layer -- and recompute runs out of memory even though forward fit.
`gptoss._MXFP4Matmul` saves only the packed tensors, which are the block's staged
state and therefore free, and re-derives the dense weight in backward. Gradients
are bit-identical to the dense path.

### Components

| module | role |
| --- | --- |
| `block_streamer.planner` | capacity/throughput/endurance verdicts from real checkpoints |
| `block_streamer.shards` | flat per-block shard files and their manifest |
| `block_streamer.diskpool` | disk -> pinned ring -> GPU staging, background readers |
| `block_streamer.lora` | adapters over streamed frozen base weights |
| `block_streamer.microbatch` | micro-batch looping inside one block visit, activation offload |
| `block_streamer.fused_optim` | per-block Adam with disk-resident state |
| `block_streamer.adapters` | per-family block assignment, expert-major MoE |
| `block_streamer.gptoss` | MXFP4 experts dequantized one expert at a time |
| `block_streamer.runners` | end-to-end drivers (Qwen3-VL, gpt-oss) |

Blocks are built on `meta` and materialized only when a slot stages them, so a
model far larger than RAM can be constructed. Pass a manifest to opt in:

```python
from block_streamer import ShardManifest, StreamedModel

manifest = ShardManifest.load("shards-qwen8b")
model = StreamedModel(meta_blocks, prefetch_ahead=2, manifest=manifest)
```

### Limits

Sequence length 128 was used throughout; gpt-oss's 128-token sliding-window
attention coincides with full attention there, and longer sequences need real
sliding-window masks passed as block kwargs, which is not implemented.

The runs are text-only, and stricter than that: `Qwen3VLRunner` takes
`with_vision=False` by default and the scripts leave it there, so the vision tower
was never even loaded. The 2.32 GiB of resident state measured for Qwen3-VL-8B is
embeddings, final norm, and `lm_head` alone. Passing `with_vision=True` loads and
freezes the tower, but nothing here has fed it an image, so the whole image path
is unexercised despite two of the three models being VLMs.

The fused optimizer is correctness-tested against `torch.optim.Adam` at small
scale only, because the planner rejects every full-finetune target on this
hardware on endurance grounds. Micro-batch looping and activation offload
(`block_streamer.microbatch`) are tested for gradient equivalence and byte-identical reads but
were not used in the runs above, which are batch 1.

Only Qwen3-VL and gpt-oss have runners. `block_streamer.adapters` carries a GLM-4.x adapter
validated against a real manifest for block assignment and sizing, but no GLM
model was downloaded or run. Qwen3-VL-MoE stores experts fused in one tensor per
layer, so it needs expert-dimension slicing (`block_streamer.adapters.chunk_fused_experts`)
rather than name-based splitting; that slicing is unit-tested but never exercised
against a real MoE checkpoint, since Qwen3-VL-235B needs 439 GiB and does not fit.

## Measured crossover and memory

![Measured transfer/compute crossover and peak CUDA allocation](https://raw.githubusercontent.com/hareramray/BlockStreamer/main/results/roofline.png)

Local measurement: NVIDIA GeForce RTX 5050 Laptop GPU (8 GB), Windows WDDM,
PyTorch 2.11.0+cu128 / CUDA runtime 12.8, bf16, six heterogeneous MLP blocks,
`prefetch_ahead=1`, four warmups and ten timed runs. These are measurements on this
machine, not predictions for a desktop PCIe Gen4 x16 link. The benchmark records
31.5 GB/s as the nominal Gen4 x16 reference but uses **measured pinned H2D time**
for its conclusions. See [the complete report](https://github.com/hareramray/BlockStreamer/blob/main/results/benchmark.json).

**Headline:** at 32 and 2,048 tokens, the sampled compute/H2D crossover falls
between widths 128 and 256. At 8,192 tokens, compute time exceeds H2D time at every
sampled width; no width crossover was observed. At width 1,024, increasing tokens
from 2,048 to 8,192 changes the measured regime from bandwidth-bound to compute-bound.
Small-block CUDA event intervals include host launch gaps; the small-width
"compute-bound" label means the measured compute path takes longer than H2D,
not that GPU arithmetic units are saturated. This sweep is an empirical roofline
diagnostic, not a hardware FLOP/s roofline or a universal crossover constant.

| Width / tokens | GPU median ms | Streamed median ms | GPU peak MiB | Streamed peak MiB | Activation/workspace increment MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1,024 / 32 | 0.569 | 10.524 | 68.625 | 28.625 | 0.438 |
| 1,024 / 2,048 | 5.915 | 14.883 | 100.125 | 60.125 | 28.000 |
| 1,024 / 8,192 | 24.304 | 29.825 | 196.125 | 156.125 | 112.000 |

For these width-1,024 cases, baseline parameter payload is 60 MiB and streamed
peak parameter payload is 20 MiB. The 8,192-token activation/workspace increment
alone is 112 MiB: parameter streaming does not eliminate activation-dominated VRAM.
This implementation saves memory but **was slower in every sampled case**.

The benchmark reports raw `torch.cuda.max_memory_allocated()` for total peak VRAM,
exact parameter/state tensor payload maxima, and an **independently measured**
activation/workspace increment: peak baseline allocation above the warmed baseline
with weights already resident. Existing input/library allocations are also recorded.
Separate maxima occur at different times; subtracting peak parameter payload from
total peak does not produce an exact activation peak. PyTorch's aggregate allocator
counter also cannot distinguish an activation from a temporary workspace.
Therefore an exact additive parameter-versus-activation split is not claimed.

Latency uses synchronized wall-clock intervals and reports medians, means,
population variance, minima, and maxima. The requested overlap score is:

```text
1 - (streamed_median_ms - all_on_GPU_median_ms) / isolated_H2D_ms
```

It is not clamped: Python, event, cloning, and scheduling overhead can make it
negative; noise can put it above one. The score is 0.454 for width 1,024 / 8,192
tokens in this run. Kernel overlap cannot fully hide transfer when per-block H2D
time exceeds compute time. Increasing token count changes arithmetic intensity;
increasing width alone need not produce a crossover.

## Profiler and validation status

`benchmark.py` exports `results/trace.json`, readable by Chrome tracing or Perfetto,
and checks whether GPU activities are present. On this machine CUPTI initialization
returned `CUPTI_ERROR_INVALID_DEVICE`, so the exported trace contains **CPU events
only** and `trace_gpu_activity` is `false`. GPU timeline overlap is therefore **not
visually verified here**. CUDA-event timing, real GPU parity tests, and memory
measurements did run. Re-run on a compatible CUPTI/driver setup to capture GPU
kernels and copies; `--no-trace` explicitly skips profiler collection.

The final local suite passed **55 tests** on CUDA, with Ruff lint/format checks
also passing. Tests cover inference in fp32/bf16/fp16 at prefetch depths 0–3,
heterogeneous blocks and nonpersistent buffers, extra arguments and tuple results,
every residency transition, a 100-iteration bandwidth-starved allocator stress,
nondefault caller streams, inference mode, output aliases, tied weights, unpinned
and over-budget errors, and exception cleanup. Training tests cover fp32/bf16/fp16
gradient parity at depths 1–3, CPU pinned gradients, extra-input gradients, dropout
RNG replay, unused/shared parameters, accumulation, autocast with frozen weights,
CPU Adam state placement, and a 12-step SGD toy loss curve.

The offload tests add bit-exact shard round-trips across mixed dtypes, disk-backed
inference parity and the residency bound at prefetch depths 0–5, a 100-iteration
deep-prefetch stress, proof that the pinned ring is the only large pinned
allocation and every base parameter stays on `meta`, fused-optimizer parity
against `torch.optim.Adam` over 20 steps, projected-wear accounting, LoRA identity
at initialization, the absence of gradients on any non-adapter parameter, a LoRA
loss curve, micro-batch gradient equivalence with byte-identical reads at N = 1,
2 and 4, expert-major visiting order, fused-expert slicing, and the MXFP4
gradient/memory check described above.

Inference `atol=rtol` is `1e-5` for fp32, `8e-3` for bf16, and `1e-3` for fp16,
allowing roughly one low-precision rounding unit at unit scale. Backward
`(atol, rtol)` is `(2e-5, 2e-4)`, `(2e-3, 3e-2)`, and `(5e-4, 5e-3)` respectively;
backward includes reductions and multiple gradient contributions. The convergence
curves use fp32 with `(2e-6, 2e-5)` tolerances and must decrease by at least 10%.
These checks are evidence for the tested workloads, not proof for arbitrary custom
modules or hardware. The environment emitted a Triton CUDA-toolkit discovery
warning; the tests use PyTorch CUDA operations and do not require Triton compilation.
