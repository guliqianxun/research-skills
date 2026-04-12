# PyTorch Engineering Principles

Engineering principles. After reading this file, every PyTorch optimization decision
you make will have a clear rationale behind it.
Assumes no PyTorch experience, but basic understanding of matrix multiplication and gradient descent.

---

## Table of Contents

1. [Computation Graph: PyTorch's Core Contract](#1-computation-graph-pytorchs-core-contract)
2. [Memory: Always the First Bottleneck](#2-memory-always-the-first-bottleneck)
3. [Numerical Precision: Stability First, Speed Second](#3-numerical-precision-stability-first-speed-second)
4. [Training Stability Rules](#4-training-stability-rules)
5. [Three Dimensions of Parallel Training](#5-three-dimensions-of-parallel-training)
6. [Inference vs Training: Two Different Problems](#6-inference-vs-training-two-different-problems)
7. [Multimodal Engineering Rules](#7-multimodal-engineering-rules)
8. [Arithmetic Intensity and the Roofline Model](#8-arithmetic-intensity-and-the-roofline-model)
9. [Data Loading at Scale](#9-data-loading-at-scale)
10. [Reproducibility](#10-reproducibility)
11. [Loss Function Patterns](#11-loss-function-patterns)

---

## 1. Computation Graph: PyTorch's Core Contract

### Core Concept

PyTorch's eager mode **dynamically builds** a directed acyclic graph (DAG) on every forward pass.
Each tensor operation attaches a `grad_fn` node that records how to compute gradients during backward.

```
x ──[Linear]──> h ──[ReLU]──> y ──[Loss]──> scalar
                                             ↑ loss.backward() 从这里往左走
```

**What this means:**
- The graph only exists while forward is running; it is destroyed by default after backward
- You can write arbitrary Python control flow inside forward — `if`, `for`, `while` all work
- The cost is Python overhead on every forward pass

### Eager Mode vs torch.compile

`torch.compile` converts the dynamic graph into a **static graph** and applies compiler
optimizations (operator fusion, memory reuse, etc.).

**Decision rules (top to bottom):**

```
Does the model have data-dependent control flow? (branching based on tensor values)
  → Yes: use eager. Static graphs cannot express dynamic branches.
  → No: continue

Are you debugging / frequently changing the architecture?
  → Yes: use eager. Compiled error messages are hard to trace.
  → No: continue

Do you have throughput requirements?
  → Yes: compile(mode="max-autotune"). Typically 10–40% speedup.
  → No: eager is sufficient.
```

**Graph breaks are the biggest pitfall.**
When `torch.compile` encounters an operation it cannot compile, it "breaks" the graph —
splitting it into multiple segments compiled separately. More breaks = less optimization.
In the worst case, compiled code is slower than eager (compilation overhead with no benefit).

Common causes of graph breaks:
- Calling `.numpy()`, `print()`, Python's `random` module
- Accessing tensor values (`if x.item() > 0`)
- Using unregistered custom C++ operators

**Rule: get the model running in eager first, then consider compile.**

### detach vs no_grad vs inference_mode

All three "turn off gradients," but they mean different things:

| Usage | Effect | Typical scenario |
|-------|--------|------------------|
| `tensor.detach()` | Disconnects this tensor from the graph | Use an intermediate result as a constant |
| `torch.no_grad()` | No graph construction inside the block, saves memory | Validation eval loop |
| `torch.inference_mode()` | More aggressive than no_grad — disables all tracking | Pure inference, no gradient-related functionality needed |

**Rule: use `inference_mode` for eval loops — it's faster than `no_grad`.**

---

## 2. Memory: Always the First Bottleneck

### Where Training Memory Goes

For a model with $P$ parameters trained with AdamW, memory splits into four parts:

| Component | fp32 bytes | bf16 mixed precision |
|-----------|-----------|---------------------|
| Parameters | $4P$ | $2P$ |
| Gradients | $4P$ | $2P$ (often keep an fp32 copy) |
| Adam first moment $m$ | $4P$ | $4P$ (always fp32) |
| Adam second moment $v$ | $4P$ | $4P$ (always fp32) |
| **Total (excluding activations)** | $16P$ | $12P$–$14P$ |

**Rule: estimate memory by multiplying parameter count by 16 (fp32) or 12 (bf16 mixed precision). This is the floor — activations are extra.**

Example: 1B parameter model → at least 12–16 GB, activations not included.

### Activation Memory: Explodes with Sequence Length

Activations (intermediate results from forward) must be kept until backward finishes.

For a Transformer, the main activation memory sources per layer:

- Attention weight matrix: $B \times H \times L \times L$ (grows quadratically with sequence length $L$)
- FFN intermediate: $B \times L \times d_{ff}$

**Rule: doubling sequence length quadruples attention activation memory. Long-sequence training requires FlashAttention or gradient checkpointing.**

### Gradient Checkpointing

Don't store forward activations — recompute them during backward.

- Activation memory reduced by ~60–70%
- Training speed reduced by ~20–30% (one extra forward pass)
- **Use when:** activations are the memory bottleneck (typically long sequences or large batches)

**Rule: enable mixed precision first. If memory is still insufficient, add gradient checkpointing. Both together is standard for large model training.**

### Decision Order When Running Out of Memory

```
1. Enable bf16 mixed precision  → halves activation + parameter memory, near-zero quality loss
2. Reduce batch size            → linearly reduces activation memory, but hurts throughput
3. Enable gradient checkpointing → large activation memory reduction, costs 20–30% speed
4. FSDP2 (multi-GPU sharding)   → shards parameters/gradients/optimizer state across GPUs
5. Use a smaller model           → last resort
```

Don't skip steps. Many people jump straight to FSDP on first OOM — often bf16 alone is enough.

---

## 3. Numerical Precision: Stability First, Speed Second

### The Four Precisions

| Format | Exponent bits | Mantissa bits | Range | Precision | Typical use |
|--------|--------------|---------------|-------|-----------|-------------|
| fp32 | 8 | 23 | Wide | High | Optimizer state, loss scaling reference |
| bf16 | 8 | 7 | **Same as fp32** | Low | Training forward/backward (A100+) |
| fp16 | 5 | 10 | **Narrow (overflow-prone)** | Medium | Older GPUs (V100, T4) |
| fp8 | 4/5 | 3/2 | Very narrow | Very low | H100+ training acceleration, requires special handling |

**Rule: use bf16, not fp16, on A100 and above.**

Why: bf16 has the same exponent bits as fp32, so it won't overflow and doesn't need loss scaling.
fp16 has only 5 exponent bits — large gradients or activations overflow to `inf`,
requiring `GradScaler` for dynamic rescaling.

### Mixed Precision: The Correct Way (PyTorch 2.x)

```python
# Old path (deprecated — do not use)
# from torch.cuda.amp import autocast, GradScaler

# New path (device-agnostic, PyTorch 2.x standard)
from torch.amp import autocast, GradScaler

# bf16: no GradScaler needed (exponent range is sufficient)
with autocast("cuda", dtype=torch.bfloat16):
    loss = model(x)
loss.backward()
optimizer.step()

# fp16: GradScaler required (prevents gradient underflow)
scaler = GradScaler("cuda")
with autocast("cuda", dtype=torch.float16):
    loss = model(x)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Three Symptoms of Precision Problems

| Symptom | Root cause | Diagnosis |
|---------|-----------|-----------|
| Loss suddenly becomes `nan` | Numerical overflow | Check logit magnitudes, add gradient clipping |
| Loss slowly drifts to `nan` | Gradient accumulation error | Switch to fp32 training and see if it disappears |
| Loss oscillates wildly but no nan | Learning rate too high or insufficient precision | Reduce lr first, then consider precision |

**Rule: when training produces nan, check for overflow or learning rate issues first — don't jump to changing precision.**

Changing precision is a last resort. Most nans come from a learning rate that's too high
or numerically unstable operations in the model (manual softmax, log of zero, etc.).

---

## 4. Training Stability Rules

### Normalization: Pre-LN Is the Modern Standard

Early Transformers (BERT, GPT-1) used Post-LN: attention/FFN first, then normalize.
Modern models (LLaMA, Gemma, GPT-4 architecture) all use Pre-LN: normalize first, then attention/FFN.

```
Post-LN: x → Attention → x + residual → LayerNorm → output
Pre-LN:  x → LayerNorm → Attention → x + residual → output
```

**Why Pre-LN is more stable:**
The residual path has no LayerNorm, so gradients flow directly from loss back to input layers
without being truncated by normalization. Post-LN with many layers (> 12) is prone to vanishing
gradients and requires careful warmup tuning to converge.

**Rule: for models deeper than 12 layers, use Pre-LN + RMSNorm. RMSNorm is ~15% faster than
LayerNorm (no mean subtraction), with equivalent quality.**

### Gradient Clipping: Not Optional

Gradient clipping in large model training is **mandatory**, not a tuning knob.

```python
# Standard practice: global norm clipping, max_norm=1.0 is the industry default
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Why it's mandatory:**
Early in training, parameters are randomly initialized and some batches produce extremely
large gradients. Without clipping, a single bad batch can derail the entire training run.

**Rule: `max_norm=1.0` is a reasonable starting point. If loss oscillates severely, reduce to 0.5.
If gradient norms consistently stay well below 1.0, you can relax it. Monitor `grad_norm` as a
metric — it's the earliest warning signal when something goes wrong.**

### Learning Rate Warmup: Cold Start Protection

Warmup means starting with a very small lr and linearly (or cosine) ramping up to the target lr
over the first portion of training, then decaying.

```
lr
 ↑      /‾‾‾‾‾‾‾‾‾\
 |     /            \
 |    /              \___________
 |   /
 |__/
 0  warmup   stable        decay   → steps
```

**Why warmup is needed:**
Adam's second moment $v$ (variance estimate) is near zero at the start of training,
making the effective learning rate $\text{lr}/\sqrt{v}$ extremely large.
Warmup uses a small lr to give $v$ time to accumulate, preventing oversized early steps.

**Rule:**
- Warmup steps: typically 1–5% of total steps, or a fixed 1000–4000 steps
- Deeper models (> 24 layers) need longer warmup
- Pre-LN needs less warmup than Post-LN, but it's still required

### Diagnosing Training Instability

When training loss is abnormal (oscillating, not decreasing, nan), diagnose in this order:

```
1. Check the grad_norm curve
   → Consistently > 10: gradient explosion. Reduce lr or reduce max_norm.
   → Consistently ≈ 0: vanishing gradients. Check architecture (Post-LN?), activation functions, residual connections.
   → Sudden spike then recovery: normal — clipping is working.

2. Check the loss curve shape
   → Normal at first, then oscillates: lr too high or scheduling issue.
   → Never decreases: data pipeline bug, or model output dimension mismatches loss function.
   → Decreases then rebounds: overfitting, or validation data issue.

3. Check per-layer gradients
   → Use gradient flow plot to locate which layer has abnormal gradients (see architecture.md).

4. Only then consider precision
   → Run 100 steps in fp32. If stable, it's a precision issue. If still unstable, it's architecture or data.
```

---

## 5. Three Dimensions of Parallel Training

Large model training has three parallelism strategies, each splitting work along a different axis.
Understanding their **fundamental differences** matters more than memorizing APIs.

### Intuition for the Three Types

```
Data Parallel (DP):
  Same model replicated on every GPU, each GPU processes different data.
  → Communication: gradient sync after each step (all-reduce)
  → When: model fits on a single GPU

Tensor Parallel (TP):
  Same layer's weight matrix split by columns or rows across N GPUs, each GPU stores one piece.
  → Communication: all-reduce within every layer's forward/backward (frequent!)
  → When: single-layer parameters too large; requires NVLink high bandwidth

Pipeline Parallel (PP):
  Different layers placed on different GPUs, data flows through like an assembly line.
  → Communication: pass activations between GPUs (point-to-point, low volume)
  → When: very deep model with many layers; PCIe is sufficient
```

### Communication Determines Parallelism Choice

**Rule: communication is the tax on parallelism. The higher the tax, the lower the efficiency.**

| Parallelism type | Communication frequency | Volume | Bandwidth requirement |
|-----------------|------------------------|--------|----------------------|
| DP/FSDP2 | Once per step | Parameter-scale | Low (100 GB/s sufficient) |
| TP | Multiple times per layer | Activation-scale | **High (needs NVLink, 400+ GB/s)** |
| PP | Per micro-batch | Single-layer activations | Low (PCIe works) |

**Rule: TP must run between NVLink-connected GPUs (within one machine). PP can cross machines.**

Practical rules:
- Single machine, up to 8 GPUs: FSDP2 is usually sufficient
- Single machine 8 GPUs not enough: add PP (across machines), then consider TP (within machine)
- Very large models (70B+): FSDP2 × TP × PP (3D parallelism)

### The Essence of FSDP2

FSDP (Fully Sharded Data Parallel) is an upgrade of DP:
it shards not just data, but **parameters, gradients, and optimizer state** across GPUs.

- Each GPU stores only $1/N$ of the parameters
- When a layer's parameters are needed, temporarily all-gather them, then release immediately after use
- Result: $N$ GPUs can train an $N\times$ larger model

**FSDP2 (PyTorch 2.4+) vs FSDP1 key differences:**
- FSDP2 is built on DTensor (distributed tensor) — each parameter natively knows its sharding
- FSDP2 uses `fully_shard(module)` instead of wrapping in a class
- FSDP2 composes naturally with TP — the foundation for 2D parallelism

See `distributed-training.md` for implementation details.

### When You Don't Need Parallelism

**Rule: if the experiment fits on a single GPU, don't introduce distributed training.
Distribution makes every bug harder to reproduce.**

Principle:
1. First verify model correctness on a single GPU
2. Then validate distributed logic on a single machine with multiple GPUs (DDP/FSDP2)
3. Only then scale to multiple machines

---

## 6. Inference vs Training: Two Different Problems

Training and inference have **completely different bottlenecks** and opposite optimization directions.
Confusing the two is a common mistake.

### Training Is Compute-Bound

During training, every parameter is read, used to compute gradients, and updated.
Computation far exceeds memory I/O → **compute is the bottleneck**.

Optimization direction: increase compute efficiency
- Increase batch size (better GPU utilization)
- Use FlashAttention (reduces HBM reads, makes attention compute-bound)
- FP8/bf16 (more computation per unit time)
- torch.compile (eliminates kernel launch overhead)

### Autoregressive Inference Is Memory-Bandwidth-Bound

During generation, only one token is produced at a time, with batch size typically 1 (or very small).
Each step reads **all parameters from GPU memory once** with minimal computation — memory I/O is the bottleneck.

**Rule: generation speed ceiling ≈ memory bandwidth / model size in bytes.**

Example: A100 memory bandwidth 2 TB/s, 70B parameter bf16 model = 140 GB.
Theoretical ceiling: 2000 / 140 ≈ 14 tokens/sec (single GPU, no optimization).

Optimization direction: reduce bytes read per step
- Quantization (INT8/INT4): halves or quarters parameter size → direct speedup
- KV cache: avoids recomputing attention for already-processed tokens
- Speculative decoding: draft multiple tokens with a small model, verify in one pass with the large model

### Prefill vs Decode: Same Model, Two Modes

LLM inference has two phases with completely different characteristics:

| Phase | Operation | Bottleneck | Optimization focus |
|-------|-----------|------------|-------------------|
| **Prefill** (processing input) | Process all input tokens in parallel | Compute | FlashAttention, large batch |
| **Decode** (token-by-token generation) | Generate one token at a time | Memory bandwidth | Quantization, KV cache, speculative decoding |

**Rule: don't apply the same optimization strategy to prefill and decode. Production systems
typically decouple the two phases and optimize each separately.**

### The Essence of KV Cache

Generating a new token requires attending to all previous tokens.
Without KV cache, the Key/Value matrices for all history tokens are recomputed every step — $O(L^2)$ total.
With KV cache: store each step's K/V, read them directly next step, only compute the new token's attention — $O(L)$ total.

Cost: KV cache memory grows linearly with sequence length.
For long sequences (128K tokens) + large models, KV cache can exceed model parameter memory.

**Rule: KV cache is a requirement for generation, not an optimization.
Generation without KV cache is an incomplete implementation.**

See `inference-deployment.md` for implementation details.

---

## 7. Multimodal Engineering Rules

### Core Problem: Different Modalities Must Be Unified into One Sequence Space

Language model input is a token sequence where each token is an integer ID.
The multimodal challenge: **images, audio, and video are not tokens but must become token sequences.**

This conversion is called **modality tokenization** — the first core problem in multimodal engineering.

### Rule 1: Each Modality Has Its Own "Vocabulary" and "Tokenizer"

| Modality | Typical tokenization | Characteristics |
|----------|---------------------|-----------------|
| Text | BPE / SentencePiece | Discrete token IDs, fixed vocabulary |
| Image | Patch embedding (ViT-style) | Split image into $N \times N$ patches, linearly project each |
| Audio | Mel-spectrogram + conv | Convert to spectrogram, extract local features with convolution |
| Video | Spatiotemporal patches (3D) | Patch across both spatial and temporal dimensions |

**Rule: after tokenization, all modalities must output vector sequences of the same dimension.
Unified dimension is the prerequisite for the backbone to process multimodal input.**

### Rule 2: Alignment Happens in a Shared Space — Don't Force the Backbone to Do Both

Multimodal alignment (making "a photo of a cat" semantically close to "cat") is a separate task.
It should not be mixed into the backbone's forward pass.

Standard approach:
1. Train or pre-train each modality encoder independently (text encoder, vision encoder)
2. Use a lightweight **projection layer** (linear or small MLP) to map each encoder's output into a shared space
3. The backbone processes only the shared-space token sequence, agnostic to modality origin

**Rule: the projection layer is the alignment boundary. Everything before it is modality-specific;
everything after it is modality-agnostic.**

### Rule 3: Different Modalities Have Different Sampling Rates — Alignment Is an Engineering Challenge

Text is event-driven (one word per step), video is 30fps, audio is 16000Hz.
In the same "1 second," text may have 5 tokens, audio has 16000 samples, video has 30 frames.

Approaches:
- **Downsample to a common rate:** audio typically to 50–100 Hz (mel-spectrogram + stride), video to 1–4 fps
- **Explicit timestamps:** add timestamp embeddings to each token so the model knows physical time
- **Modality delimiters:** use special tokens (`[IMAGE_START]`, `[AUDIO_END]`) to mark modality boundaries

**Rule: don't assume different modalities naturally produce aligned token counts.
Timestamp or position embeddings are required for multimodal temporal models.**

### Rule 4: Multimodal Sequences Create Far More Memory Pressure Than Text

A $224 \times 224$ image with $16 \times 16$ patches produces $14 \times 14 = 196$ image tokens.
A 10-second video (4fps, 196 patches/frame) produces ~7840 tokens.

This is far more than text tokens carrying equivalent information,
with a massive impact on attention's $O(L^2)$ complexity.

**Rule: multimodal models almost certainly need long-sequence optimizations (FlashAttention,
sliding window attention, sequence parallelism). Plan for this when designing the architecture ���
don't wait for the first OOM.**

---

## 8. Arithmetic Intensity and the Roofline Model

### The One Concept That Explains Performance

**Arithmetic intensity** = FLOPs performed / bytes moved from memory.

Every operation on a GPU is either:
- **Compute-bound:** the GPU's compute units are the bottleneck (high arithmetic intensity)
- **Memory-bandwidth-bound:** waiting for data from HBM is the bottleneck (low arithmetic intensity)

The **roofline model** shows which regime you're in:

```
Throughput
(TFLOPS)
    │         ╱‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾  ← compute ceiling (e.g., 989 TFLOPS bf16 on H100)
    │        ╱
    │       ╱
    │      ╱  ← bandwidth ceiling (slope = memory bandwidth)
    │     ╱
    │    ╱
    │   ╱
    │──╱──────────────────────────
    0        Arithmetic intensity (FLOPs/byte)
```

Below the knee: memory-bandwidth-bound. Above the knee: compute-bound.

**The knee point for H100:** ~989 TFLOPS / 3.35 TB/s ≈ **295 FLOPs/byte**.
Operations with intensity below 295 are bandwidth-bound; above 295 are compute-bound.

### What This Means in Practice

| Operation | Arithmetic intensity | Regime | Implication |
|-----------|---------------------|--------|-------------|
| Large matmul ($M, N, K > 1024$) | $O(K)$ — scales with inner dimension | Compute-bound | Larger batch size → more efficient |
| Attention (FlashAttention) | High (fused kernel) | Compute-bound | FlashAttention wins by avoiding HBM writes |
| Attention (naive, with materialization) | Low (writes $L \times L$ matrix to HBM) | Bandwidth-bound | Slow — most time spent writing attention matrix |
| LayerNorm, RMSNorm | Very low (~5 FLOPs/element) | Bandwidth-bound | Fusion with adjacent ops helps; standalone is always slow |
| Element-wise ops (GELU, dropout) | ~1 FLOP/element | Bandwidth-bound | Always fuse with adjacent operations |
| Autoregressive decode ($B=1$) | Very low | Bandwidth-bound | Quantization helps; batch size increase helps |

**Rule: before optimizing anything, estimate its arithmetic intensity. If it's
bandwidth-bound, compute-level optimizations (FP8, larger batch) won't help —
you need to reduce memory traffic (fusion, quantization, caching). If it's
compute-bound, reducing memory traffic won't help — you need more raw FLOPS.**

### Why This Matters for Architecture Design

- **SwiGLU adds a third matmul vs standard FFN.** But matmuls are compute-bound, so the extra
  FLOPS are cheap relative to the bandwidth-bound norms and activations around them.
- **FlashAttention is fast not because it does fewer FLOPs** — it actually does slightly
  *more* (recomputation). It's fast because it avoids writing the $L \times L$ attention
  matrix to HBM, turning a bandwidth-bound operation into a compute-bound one.
- **torch.compile helps primarily by fusing element-wise ops** — turning multiple
  bandwidth-bound kernels into one kernel that reads and writes HBM once instead of N times.

---

## 9. Data Loading at Scale

### When torch DataLoader Isn't Enough

`torch.utils.data.DataLoader` with a local `Dataset` works for datasets that fit on disk.
When your dataset is terabytes on cloud storage, or you're training across hundreds of GPUs,
you need streaming data loading.

### The Problem at Scale

- **Storage:** 15 TB of images on S3/GCS won't fit on local disk
- **Shuffling:** true random access over 15 TB is impractical — you need approximate shuffling
- **Multi-node:** each GPU must read different data without coordination overhead
- **Fault tolerance:** if a node dies mid-epoch, training must continue without replaying all data

### Streaming Data Formats

| Format | Key idea | Used by |
|--------|----------|---------|
| **WebDataset** | Tar archives of samples, streamed sequentially | Common in research, works with any cloud storage |
| **Mosaic StreamingDataset** | Sharded dataset with deterministic shuffling | MosaicML / Databricks |
| **HuggingFace datasets (streaming mode)** | Iterator over Arrow-format shards | HuggingFace ecosystem |
| **TFRecord** | Protocol buffer shards | TensorFlow ecosystem, some PyTorch usage |

**Rule: for datasets > 1 TB or multi-node training, use a streaming format.
The choice between formats is less important than using any streaming format at all.**

### Data Loading Principles

1. **Shard your data.** Split the dataset into many files (1000+ shards). Each worker
   reads different shards. This enables parallelism without coordination.

2. **Shuffle at shard level, then within shards.** True global shuffling is impossible at scale.
   Approximate shuffling (random shard order + shuffle buffer within each shard) is sufficient.

3. **Separate preprocessing from training.** Expensive preprocessing (tokenization, image resizing,
   augmentation) should happen offline and be saved to the sharded format. Don't do heavy
   transforms in the DataLoader at training time.

4. **Profile data loading independently.** Set `num_workers` high enough that the GPU never
   waits for data. If GPU utilization is < 80%, data loading is likely the bottleneck.

**Rule: if your GPU utilization is low, check data loading before blaming the model.
Run `nvidia-smi` during training — if GPU usage is bursty (spikes then drops), the GPU is
starving for data.**

---

## 10. Reproducibility

### Why It Matters

A result that can't be reproduced is not a result. For published research, reviewers
expect bit-exact reproducibility across runs with the same seed.

### The Minimum Reproducibility Checklist

```
1. Set all random seeds:
   torch.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)

2. Enable deterministic algorithms:
   torch.use_deterministic_algorithms(True)
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False

3. Control data ordering:
   Use a seeded sampler for DataLoader.
   In distributed training, use DistributedSampler with a fixed seed.

4. Record environment:
   PyTorch version, CUDA version, GPU model, number of GPUs,
   pip freeze output, git commit hash.

5. Save full training state:
   Model weights, optimizer state, scheduler state, RNG states, step count.
   This enables exact resume after interruption.
```

### Non-Deterministic Operations

Some CUDA operations are non-deterministic by default (atomic operations in scatter/gather,
some cuDNN convolution algorithms). `torch.use_deterministic_algorithms(True)` forces
deterministic alternatives but may be slower.

**Rule: enable deterministic mode during development and evaluation. For large-scale
training runs where a 5–10% speed penalty is unacceptable, disable it but document
the expected variance across seeds (run 3+ seeds and report mean ± std).**

### Distributed Reproducibility

Multi-GPU training introduces additional non-determinism:
- All-reduce order can vary across runs
- Mixed precision reductions are not associative (floating-point addition order matters)
- Data loading order differs if worker counts change

**Rule: small loss differences (< 0.1% relative) between 1-GPU and 8-GPU runs are normal
and expected. Large differences indicate a bug in gradient synchronization or data sampling.**

---

## 11. Loss Function Patterns

### Beyond Cross-Entropy

Standard cross-entropy is the default for classification and next-token prediction.
But modern training pipelines use several other loss patterns:

### Knowledge Distillation Loss

Train a small (student) model to match a large (teacher) model's output distribution.

$$L_{\text{distill}} = \alpha \cdot \text{KL}(p_{\text{teacher}}^{(\tau)} \| p_{\text{student}}^{(\tau)}) + (1 - \alpha) \cdot L_{\text{CE}}(p_{\text{student}}, y)$$

where $\tau$ is temperature (higher = softer distribution) and $\alpha$ balances the two terms.

**When to use:** you have a large model and need a smaller model for deployment.
Distillation consistently outperforms training the small model from scratch.

### Contrastive Loss (CLIP / SimCLR)

Push matching pairs close, push non-matching pairs apart in embedding space.

$$L_{\text{contrastive}} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k \neq i} \exp(\text{sim}(z_i, z_k) / \tau)}$$

**When to use:** multimodal alignment (image-text pairs), representation learning,
retrieval systems. Requires large batch sizes for sufficient negatives.

### DPO (Direct Preference Optimization)

Aligns models with human preferences without a separate reward model.

$$L_{\text{DPO}} = -\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \beta \log \frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)}\right)$$

where $y_w$ is the preferred response, $y_l$ is the dispreferred response,
$\pi_\theta$ is the policy being trained, $\pi_{\text{ref}}$ is the reference policy.

**When to use:** alignment / RLHF training. Simpler than PPO — no reward model needed.
Requires paired preference data.

### Multi-Task Loss Balancing

When training on multiple objectives simultaneously, losses are on different scales.
Naive summation leads to one task dominating.

**Fixed weighting:** $L = \sum_i w_i L_i$ — simple, requires manual tuning.

**Uncertainty weighting (Kendall et al.):** learn $w_i$ as trainable parameters
that capture task uncertainty. Automatic, but adds parameters.

**Rule: start with equal weights. If one task's loss is 100× larger than another,
manually rescale to similar magnitudes before trying learned balancing.**
