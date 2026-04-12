# Profiling and Optimization

How to measure performance and fix bottlenecks. Read `principles.md` first for the mental model.

---

## Table of Contents

1. [MFU: The One Metric That Matters](#1-mfu-the-one-metric-that-matters)
2. [FLOPs Estimation](#2-flops-estimation)
3. [Memory Profiling](#3-memory-profiling)
4. [Mixed Precision (AMP)](#4-mixed-precision-amp)
5. [FlashAttention](#5-flashattention)
6. [torch.compile](#6-torchcompile)
7. [Gradient Checkpointing](#7-gradient-checkpointing)
8. [Sequence Packing](#8-sequence-packing)
9. [Throughput Benchmarking](#9-throughput-benchmarking)

---

## 1. MFU: The One Metric That Matters

**Model FLOPs Utilization (MFU)** = actual training throughput / theoretical GPU peak.

This is the single best measure of how well your training setup uses the hardware.

$$\text{MFU} = \frac{\text{tokens/sec} \times \text{FLOPs/token}}{\text{GPU peak FLOPS}}$$

| GPU | bf16 peak TFLOPS | fp8 peak TFLOPS |
|-----|------------------|-----------------|
| A100 80GB | 312 | — |
| H100 80GB | 989 | 1979 |
| B200 | ~2250 | ~4500 |

**Targets:**
- MFU > 50%: good — you're using the GPU well
- MFU 30–50%: acceptable — look for easy wins (FlashAttention, compile)
- MFU < 30%: something is wrong — profile to find the bottleneck

**Rule: always compute MFU before and after any optimization. If MFU didn't improve, the optimization didn't work, regardless of what the wall clock says.**

---

## 2. FLOPs Estimation

For a standard Transformer with $L$ layers, dimension $d$, sequence length $S$, FFN dimension $d_{ff}$, vocabulary $V$, batch size $B$:

**Per-layer FLOPs (forward only):**

| Component | FLOPs | Notes |
|-----------|-------|-------|
| QKV projection | $6BSd^2$ | Three linear layers: Q, K, V |
| Attention logits | $2BS^2d$ | $QK^\top$ matmul |
| Attention over values | $2BS^2d$ | Score × V matmul |
| Output projection | $2BSd^2$ | Linear after attention |
| FFN (standard) | $4BSd \cdot d_{ff}$ | Two linear layers |
| FFN (SwiGLU) | $6BSd \cdot d_{ff}$ | Three linear layers (gate, up, down) |

**Total model FLOPs (forward):**

$$\text{Forward} = L \times (\text{attention} + \text{FFN}) + 2BSdV$$

The last term is the embedding/output projection.

**Training FLOPs ≈ 3 × forward FLOPs** (forward + backward, where backward ≈ 2× forward).

**Chinchilla shortcut:** Total training FLOPs $C \approx 6ND$, where $N$ = parameters, $D$ = tokens.

```python
def estimate_transformer_flops(
    n_params: int,
    n_tokens: int,
    is_training: bool = True,
) -> float:
    """Quick estimate: C ≈ 6ND for training, 2ND for inference."""
    factor = 6 if is_training else 2
    return factor * n_params * n_tokens
```

---

## 3. Memory Profiling

### Quick Debugging

```python
import torch

def print_memory(tag=""):
    """Print current GPU memory usage. Call at key points in the training loop."""
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"[{tag}] allocated={alloc:.2f}GB  peak={peak:.2f}GB  reserved={reserved:.2f}GB")
```

Place `print_memory()` before forward, after forward, after backward, and after optimizer step to see where memory goes.

### torch.profiler (Detailed)

```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_flops=True,
) as prof:
    with record_function("forward"):
        output = model(input_data)
    with record_function("backward"):
        output.sum().backward()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
prof.export_chrome_trace("trace.json")  # open in chrome://tracing
```

**Rule: profile with realistic input shapes.** A profiling run with batch_size=1 and seq_len=32 tells you nothing about production with batch_size=64 and seq_len=4096.

---

## 4. Mixed Precision (AMP)

### bf16 (recommended for A100+)

```python
from torch.amp import autocast

optimizer.zero_grad()
with autocast("cuda", dtype=torch.bfloat16):
    loss = model(x, y)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

No `GradScaler` needed — bf16 has the same exponent range as fp32.

### fp16 (for V100, T4)

```python
from torch.amp import autocast, GradScaler

scaler = GradScaler("cuda")
optimizer.zero_grad()
with autocast("cuda", dtype=torch.float16):
    loss = model(x, y)
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()
```

`GradScaler` prevents gradient underflow in fp16's narrow exponent range.

### FP8 (H100+, experimental)

FP8 training uses `torchao` (torch architecture optimization):

```python
from torchao.float8 import Float8LinearConfig, convert_to_float8_training

config = Float8LinearConfig()
convert_to_float8_training(model, config=config)
# Then train as normal with bf16 autocast — Linear layers internally use fp8
```

FP8 roughly doubles throughput vs bf16 on H100. The model still accumulates in bf16/fp32; only the matmuls use fp8.

---

## 5. FlashAttention

FlashAttention fuses the entire attention operation (QKV matmul → softmax → output) into a single kernel that never materializes the $S \times S$ attention matrix in HBM.

**Impact:**
- Memory: $O(S)$ instead of $O(S^2)$ — enables much longer sequences
- Speed: 2–4× faster on typical sequence lengths

**In PyTorch:** `F.scaled_dot_product_attention` automatically dispatches to FlashAttention when available:

```python
import torch.nn.functional as F

out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=True)
```

**Rule: never implement attention manually with separate matmul + softmax + matmul. Always use `F.scaled_dot_product_attention`, which picks the fastest available kernel (FlashAttention, memory-efficient attention, or math fallback).**

---

## 6. torch.compile

### Basic Usage

```python
compiled_model = torch.compile(model)                     # safe default
compiled_model = torch.compile(model, mode="max-autotune") # maximum throughput
compiled_model = torch.compile(model, dynamic=True)        # variable input shapes
```

### Debugging Graph Breaks

```python
import torch._dynamo

# See what's happening
torch._dynamo.config.verbose = True

# Get a report of all graph breaks
explanation = torch._dynamo.explain(model)(sample_input)
print(explanation)
```

Common graph break causes and fixes:

| Cause | Example | Fix |
|-------|---------|-----|
| Data-dependent control flow | `if x.sum() > 0` | Use `torch.where` or `torch.cond` |
| Python side effects | `print(x)`, `list.append(x)` | Remove from forward path |
| Leaving PyTorch | `x.numpy()` | Stay in tensor operations |
| Unregistered custom ops | C++ extension | Register with `torch.library` |

**Rule: zero graph breaks is the target. Each break splits the graph and prevents cross-break optimization. Run `torch._dynamo.explain()` on your model before claiming "compile didn't help."**

---

## 7. Gradient Checkpointing

```python
from torch.utils.checkpoint import checkpoint

class CheckpointedBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        return checkpoint(self.block, x, use_reentrant=False)
```

**`use_reentrant=False` is required for PyTorch 2.x.** The old default (`True`) has known bugs with autograd and will be removed.

Trade-off: ~60–70% activation memory reduction at ~20–30% speed cost.

**Rule: apply checkpointing at the transformer block granularity, not at individual operations. Too-fine granularity adds overhead without proportional memory savings.**

---

## 8. Sequence Packing

Variable-length sequences padded to the same length waste compute on padding tokens. Sequence packing concatenates multiple short sequences into one long sequence, with attention masks preventing cross-contamination.

```
Without packing (waste):
  [tok tok tok PAD PAD PAD PAD PAD]   ← 62% padding
  [tok tok tok tok tok tok PAD PAD]   ← 25% padding

With packing (efficient):
  [tok tok tok | tok tok tok tok tok]  ← 0% padding, separator between sequences
```

**Impact:** for datasets with high variance in sequence length (common in instruction tuning), packing can improve throughput by 30–60%.

**Rule: if your average sequence length is less than 70% of max length, implement packing. The attention mask must block cross-sequence attention — getting this wrong silently corrupts training.**

---

## 9. Throughput Benchmarking

```python
import time
import torch

def benchmark(model, input_shape, steps=100, warmup=10, device="cuda"):
    """Measure throughput in samples/sec and tokens/sec."""
    model = model.to(device).eval()
    x = torch.randn(*input_shape, device=device)

    for _ in range(warmup):
        with torch.inference_mode():
            model(x)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        with torch.inference_mode():
            model(x)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    batch_size = input_shape[0]
    seq_len = input_shape[1] if len(input_shape) > 1 else 1
    samples_per_sec = batch_size * steps / elapsed
    tokens_per_sec = samples_per_sec * seq_len

    print(f"Throughput: {samples_per_sec:.1f} samples/s, {tokens_per_sec:.0f} tokens/s")
    print(f"Latency:    {elapsed / steps * 1000:.2f} ms/batch")
    return tokens_per_sec
```

**Rule: always warm up before measuring. The first few iterations include CUDA kernel compilation and caching — they are 2–10× slower and will skew your numbers.**
