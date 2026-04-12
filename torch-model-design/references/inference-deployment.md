# Inference and Deployment

How to serve trained models efficiently. Read `principles.md` §6 first for the
prefill vs decode mental model.

---

## Table of Contents

1. [KV Cache](#1-kv-cache)
2. [Quantization](#2-quantization)
3. [Speculative Decoding](#3-speculative-decoding)
4. [Continuous Batching](#4-continuous-batching)
5. [torch.export](#5-torchexport)
6. [Inference Optimization Checklist](#6-inference-optimization-checklist)

---

## 1. KV Cache

### Why It Exists

In autoregressive generation, producing token $t$ requires attending to all previous tokens
$1, \dots, t-1$. Without caching, the Key and Value projections for every past token are
recomputed at every step — total work is $O(L^2)$ over a sequence of length $L$.

With KV cache: store each token's K and V after computing them once. At step $t$, only
compute K/V for the new token and append to the cache. Total work becomes $O(L)$.

**KV cache is not an optimization — it is a correctness requirement for practical
autoregressive inference.** Without it, generating 1000 tokens from a 7B model takes
minutes instead of seconds.

### Memory Cost

Per layer, the KV cache stores:

$$\text{KV cache per layer} = 2 \times B \times L \times d_{kv} \times \text{bytes\_per\_element}$$

where the factor 2 is for K and V, $B$ is batch size, $L$ is sequence length,
and $d_{kv} = n_{kv\_heads} \times d_{head}$.

For a full model:

$$\text{Total KV cache} = n_{layers} \times 2 \times B \times L \times d_{kv} \times \text{bytes}$$

**Example: LLaMA-3 70B, bf16, batch=1, 8K context**
- 80 layers, 8 KV heads (GQA), head_dim=128 → $d_{kv} = 1024$
- Per layer: $2 \times 1 \times 8192 \times 1024 \times 2 = 32$ MB
- Total: $80 \times 32$ MB = **2.56 GB**

At 128K context: **40.96 GB** — more than half of an A100-80GB.

**Rule: KV cache grows linearly with sequence length. For long-context models,
KV cache — not model weights — is the memory bottleneck during inference.
GQA directly reduces KV cache by the group ratio.**

### Implementation Pattern

```python
class CausalLMWithCache(nn.Module):
    def generate(self, prompt_ids, max_new_tokens=256):
        """Basic generation loop with KV cache."""
        past_kv = None
        input_ids = prompt_ids

        generated = []
        for _ in range(max_new_tokens):
            logits, past_kv = self.forward(input_ids, past_kv=past_kv, use_cache=True)

            # Only need logits for the last token
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(next_token)

            # Next iteration: only feed the new token (past context is in cache)
            input_ids = next_token

            if next_token.item() == self.eos_token_id:
                break

        return torch.cat(generated, dim=1)
```

Key detail: after the first step (prefill), `input_ids` has length 1 — only the new token.
All past information lives in `past_kv`. This is what makes decode fast.

---

## 2. Quantization

Quantization reduces the precision of model weights (and optionally activations) to use
less memory and enable faster matrix multiplications.

### Quantization Landscape

| Method | Precision | When to use | Quality impact |
|--------|-----------|-------------|----------------|
| **bf16** (baseline) | 16-bit float | Default for serving | None |
| **INT8 (W8A8)** | 8-bit weights + activations | General-purpose speedup | Minimal (<0.5% degradation) |
| **INT8 (W8A16)** | 8-bit weights, 16-bit activations | Memory reduction, less speedup | Minimal |
| **INT4 (W4A16)** | 4-bit weights, 16-bit activations | Aggressive memory reduction | Noticeable on small models, acceptable on 13B+ |
| **FP8** | 8-bit float | H100+ native support | Minimal, similar to INT8 |
| **GPTQ** | 4-bit, post-training, one-shot | Quick quantization of any model | Good for 7B+, degrades on small models |
| **AWQ** | 4-bit, activation-aware | Better quality than GPTQ | Slightly better than GPTQ, similar speed |

### The Core Trade-off

Quantization makes decode faster because decode is **memory-bandwidth-bound**:

$$\text{Decode speed} \approx \frac{\text{memory bandwidth}}{\text{bytes per parameter}}$$

Going from bf16 (2 bytes) to INT4 (0.5 bytes) means 4× less data to read → up to 4× faster.

Prefill, being compute-bound, benefits less from weight quantization.

### torchao (PyTorch Architecture Optimization)

`torchao` is PyTorch's official quantization library, replacing the older `torch.quantization`.

```python
import torchao

# INT8 weight-only quantization (simplest, safe)
torchao.quantize_(model, torchao.quantization.int8_weight_only())

# INT4 weight-only (more aggressive, needs calibration data)
torchao.quantize_(model, torchao.quantization.int4_weight_only(group_size=128))

# INT8 dynamic quantization (weights + activations)
torchao.quantize_(model, torchao.quantization.int8_dynamic_activation_int8_weight())
```

**Rule: start with INT8 weight-only. It's a single line of code, requires no calibration
data, and gives ~50% memory reduction with negligible quality loss. Only go to INT4
if you need further reduction and can tolerate some degradation.**

### Verifying Quantization Quality

Always measure quality after quantization on a representative evaluation set.

```python
# Quick sanity check: compare logits before and after quantization
with torch.inference_mode():
    logits_original = original_model(sample_input)
    logits_quantized = quantized_model(sample_input)

cosine_sim = F.cosine_similarity(
    logits_original.flatten(), logits_quantized.flatten(), dim=0
)
print(f"Cosine similarity: {cosine_sim:.6f}")  # should be > 0.999 for INT8
```

**Rule: quantization that looks fine on perplexity can fail on specific tasks (math, code).
Always evaluate on your actual downstream task, not just perplexity.**

---

## 3. Speculative Decoding

### The Problem

Autoregressive decode is slow because each token requires a full forward pass through the
model, and each forward pass is memory-bandwidth-bound with low arithmetic intensity.
The GPU is mostly idle, waiting for weight data to stream from HBM.

### The Idea

Use a small, fast **draft model** to generate $k$ candidate tokens cheaply,
then use the large **target model** to verify all $k$ tokens in a single forward pass.

```
Draft model (fast):     generate tokens [a, b, c, d, e]     ← 5 cheap forward passes
Target model (slow):    verify [a, b, c, d, e] in one pass  ← 1 expensive forward pass
Result:                 [a, b, c] accepted, [d, e] rejected  ← net: 3 tokens for ~2 passes
```

### Why It Works

Verification is cheap because the target model processes all $k$ candidates in parallel
(like prefill), which is compute-bound and efficient. The total wall-clock time is
roughly: $k \times t_{\text{draft}} + 1 \times t_{\text{target}}$ instead of
$k \times t_{\text{target}}$.

If the draft model is 10× faster and the acceptance rate is ~70%, you get roughly 2–3×
speedup in end-to-end generation latency.

### When It Helps

| Condition | Speculative decoding helps? |
|-----------|---------------------------|
| Large target model (70B+) | Yes — the draft/target speed ratio is highest |
| Small target model (1–7B) | Rarely — the target is already fast enough |
| High-entropy output (creative writing) | Less — low acceptance rate |
| Low-entropy output (code, translation) | More — draft model is often correct |
| High batch size | Less — target is already better utilized |

**Rule: speculative decoding optimizes latency for single-request generation
from large models. If your bottleneck is throughput (many concurrent requests),
continuous batching (§4) is more impactful.**

### Correctness Guarantee

A key property: speculative decoding produces **exactly the same distribution** as the
target model alone. Rejected tokens are resampled from a corrected distribution, so the
output quality is identical — it's a pure speed optimization with no quality trade-off.

---

## 4. Continuous Batching

### The Problem with Static Batching

In static batching, all requests in a batch must finish before any new request starts.
Short requests wait for long ones, wasting GPU time.

```
Static batching:
  Request A (short):  ████░░░░░░░░  ← done, but waits for B
  Request B (long):   ████████████  ← still running
  Request C (queued): ░░░░░░░░░░░░████████  ← can't start until B finishes
```

### Continuous Batching (PagedAttention / vLLM Pattern)

Requests enter and leave the batch independently. When request A finishes, request C
immediately takes its slot.

```
Continuous batching:
  Request A (short):  ████
  Request C:              ████████
  Request B (long):   ████████████
```

This is the standard serving pattern for LLM inference in production (vLLM, TGI, TensorRT-LLM).

### Key Concepts

**Paged KV cache:** Instead of pre-allocating a contiguous KV cache for max_seq_len per request,
allocate KV memory in pages (blocks) and assign pages on demand. This eliminates wasted memory
from over-allocation.

**Iteration-level scheduling:** At each decode step, the scheduler decides which requests to
run. New requests can join mid-generation; completed requests free their resources immediately.

**Prefill vs decode interleaving:** some systems run prefill (compute-bound) and decode
(bandwidth-bound) in separate scheduling phases to avoid interference.

### When to Build vs Use Off-the-Shelf

| Situation | Recommendation |
|-----------|---------------|
| Standard LLM serving | Use vLLM or TGI — don't reimplement |
| Custom architecture (non-standard attention) | May need custom serving code |
| Research / prototyping | HuggingFace `generate()` is fine |
| Latency-critical production | Evaluate TensorRT-LLM or custom CUDA |

**Rule: continuous batching is a serving infrastructure concern, not a model design concern.
Design your model with standard attention patterns so that existing serving frameworks
(vLLM, TGI) can handle it. Non-standard attention that breaks these frameworks
creates an operational burden that outweighs most accuracy gains.**

---

## 5. torch.export

`torch.export` is PyTorch's path from research model to production artifact.
It replaces the older `torch.jit.trace` / `torch.jit.script` (TorchScript), which is
in maintenance mode and should not be used for new projects.

### What torch.export Does

It captures the full computation graph into an `ExportedProgram` — a self-contained,
serializable, framework-independent representation. This can then be:

- Lowered to hardware-specific backends (TensorRT, CoreML, XNNPACK)
- Deployed on mobile / edge devices via ExecuTorch
- Analyzed for operator coverage and performance

### Basic Usage

```python
import torch

model = MyModel().eval()
example_input = (torch.randn(1, 3, 224, 224),)

# Export — traces the graph, captures all operations
exported = torch.export.export(model, example_input)

# Save to disk
torch.export.save(exported, "model.pt2")

# Load and run (no Python needed at inference time)
loaded = torch.export.load("model.pt2")
output = loaded.module()(torch.randn(1, 3, 224, 224))
```

### Dynamic Shapes

By default, exported models have fixed input shapes. For variable batch size or
sequence length, declare dynamic dimensions:

```python
from torch.export import Dim

batch = Dim("batch", min=1, max=64)
seq_len = Dim("seq_len", min=1, max=4096)

exported = torch.export.export(
    model,
    (torch.randn(1, 128, 512),),
    dynamic_shapes={"x": {0: batch, 1: seq_len}},
)
```

### When to Use torch.export

| Situation | Use torch.export? |
|-----------|-------------------|
| Deploying to mobile / edge | Yes — only path that supports ExecuTorch |
| TensorRT / ONNX conversion | Yes — cleaner graph than tracing |
| Python serving (vLLM, TGI) | No — these frameworks work with eager models directly |
| Research prototyping | No — unnecessary overhead during iteration |

**Rule: use torch.export when you need to leave the Python ecosystem. If you're serving
with Python (which most LLM inference does), torch.compile gives the same speed benefits
without the export step.**

---

## 6. Inference Optimization Checklist

Apply in order. Each step is independent — skip any that don't apply.

```
1. KV cache enabled?
   └─ No  → Implement it. This is not optional for autoregressive models.

2. Using bf16 or quantized weights?
   └─ fp32 → Switch to bf16 (free speedup, no quality loss).

3. Using F.scaled_dot_product_attention?
   └─ No  → Replace manual attention. Gets FlashAttention kernel automatically.

4. torch.compile applied?
   └─ No  → Try torch.compile(model, mode="reduce-overhead"). Check for graph breaks.

5. Quantization appropriate?
   └─ Latency-sensitive or memory-constrained
      → INT8 weight-only first, measure quality, then consider INT4 if needed.

6. Batch size > 1?
   └─ Yes → Check if continuous batching framework (vLLM) is applicable.
   └─ No, and target model is large
      → Evaluate speculative decoding for latency reduction.

7. Deploying outside Python?
   └─ Yes → torch.export to target backend (TensorRT, CoreML, ExecuTorch).
```

**Rule: most inference speedups come from steps 1–3. Quantization and speculative decoding
are refinements. Get the basics right before reaching for advanced techniques.**