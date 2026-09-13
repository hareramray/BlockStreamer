"""Real LoRA finetune of Qwen3-VL-32B with NVMe-streamed frozen base weights."""

import time
import warnings

import torch

warnings.filterwarnings("ignore")
from lora import LoRAConfig, adapter_state
from runners import Qwen3VLRunner

torch.manual_seed(0)
STEPS, SEQ = 30, 128
r = Qwen3VLRunner(
    "models/Qwen3-VL-32B",
    "models/shards-qwen32b",
    LoRAConfig(
        rank=8,
        alpha=16,
        targets=("q_proj", "k_proj", "v_proj", "o_proj"),
        top_fraction=0.5,
    ),
    prefetch_ahead=2,
    dtype=torch.bfloat16,
)
r.train()
params = r.trainable_parameters()
opt = torch.optim.AdamW(params, lr=2e-4)
print(
    f"trainable {sum(p.numel() for p in params) / 1e6:.2f}M across {len(params)} tensors",
    flush=True,
)

g = torch.Generator(device="cuda").manual_seed(7)
ids = torch.randint(0, 151936, (1, SEQ), device="cuda", generator=g)
losses = []
for step in range(STEPS):
    t0 = time.perf_counter()
    opt.zero_grad(set_to_none=True)
    logits = r(ids)
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]).float(), ids[:, 1:].reshape(-1)
    )
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    torch.cuda.synchronize()
    losses.append(loss.item())
    print(
        f"step {step:3d} loss {loss.item():8.4f} gnorm {gn:7.3f} "
        f"{time.perf_counter() - t0:6.2f}s peakVRAM {torch.cuda.max_memory_allocated() / 1024**3:.2f}G",
        flush=True,
    )
torch.save(adapter_state(r), "models/qwen32b_lora.pt")
print(
    f"RESULT first={losses[0]:.4f} last={losses[-1]:.4f} "
    f"min={min(losses):.4f} decreased={losses[-1] < losses[0] * 0.9}",
    flush=True,
)
print(f"read {r.decoder.pool.read_gbps:.2f} GB/s", flush=True)
r.close()
