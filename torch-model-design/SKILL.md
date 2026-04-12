---
name: torch-model-design
version: 1.0.0
description: >-
  PyTorch model design, computation graph analysis, profiling, and optimization — from
  architecture implementation to production-ready training and inference. Use when the user
  asks about implementing neural architectures in PyTorch, profiling FLOPs/memory/throughput,
  optimizing training speed or memory usage, debugging gradient flow or numerical stability,
  choosing between dynamic and static graphs (eager vs torch.compile), setting up distributed
  training (FSDP2, tensor parallel, pipeline parallel, 2D/3D parallelism), multimodal or
  temporal model architecture, or optimizing inference (quantization, KV cache, speculative
  decoding, continuous batching). Also trigger on: computation graph, autograd, torch.compile,
  FSDP2, DTensor, tensor parallelism, pipeline parallelism, mixed precision, AMP, bf16, fp8,
  gradient checkpointing, FlashAttention, model profiling, OOM debugging, CUDA memory,
  inference optimization, model quantization, torchao, multimodal, cross-attention, vision
  encoder, temporal model.
---

# PyTorch Model Design

Implementation-focused skill for ML model development in PyTorch — architecture design,
profiling, distributed training, and production inference.

---

## How to Use This Skill

Six reference files cover distinct concerns. Read only what the current task needs.

| Reference | When to Read |
|-----------|-------------|
| [references/principles.md](references/principles.md) | **Start here.** Engineering principles: computation graph, memory model, precision, training stability, parallelism, inference vs training, multimodal conventions, arithmetic intensity / roofline model, data loading at scale, reproducibility, loss function patterns |
| [references/architecture.md](references/architecture.md) | Implementing a model: attention variants (MHA/GQA/MQA/sliding window), FFN variants (SwiGLU/MoE), position encoding (RoPE), block assembly, initialization, gradient debugging |
| [references/profiling-optimization.md](references/profiling-optimization.md) | Making it fast: MFU, FLOPs counting, memory analysis, AMP (bf16/fp16/fp8), FlashAttention, torch.compile, gradient checkpointing, sequence packing, benchmarking |
| [references/distributed-training.md](references/distributed-training.md) | Multi-GPU/multi-node: DDP, FSDP2, tensor parallel, pipeline parallel, DeviceMesh, 2D/3D parallelism, context parallel, distributed checkpointing |
| [references/multimodal-temporal.md](references/multimodal-temporal.md) | Multimodal and temporal models: modality tokenization (text/image/audio/video), cross-attention vs interleaved, modality alignment, temporal patterns, causal masking, streaming inference, memory budgeting |
| [references/inference-deployment.md](references/inference-deployment.md) | Production inference: KV cache, quantization (INT8/FP8/GPTQ/AWQ via torchao), speculative decoding, continuous batching, torch.export |

---

## Phase 1: Architecture Design

### Decision: Graph Mode

```
Does the model have data-dependent control flow?
(early exit, variable-length loops, conditional layers)
├─ YES → Eager mode. torch.compile(dynamic=True) may work
│        but test carefully — graph breaks eliminate most benefit.
└─ NO ↓

Research iteration mode (architecture changes frequently)?
├─ YES → Start eager. Compile when architecture stabilizes.
│        Compiled model debugging is significantly harder.
└─ NO ↓

Production / throughput-critical?
└─ YES → torch.compile(mode="max-autotune").
         Invest in eliminating graph breaks first.
```

### Decision: Architecture Selection

The standard mapping (sequences → Transformer, images → CNN/ViT, graphs → GNN) is textbook.
Focus on **non-obvious choices** where defaults are wrong:

| Situation | Choice | Why |
|----------|--------|-----|
| Sequence $> 8$K tokens | SSM (Mamba) or sliding window attention | Quadratic attention dominates at scale; sub-quadratic trades small accuracy loss for 5-10× throughput |
| Multi-head attention, memory-constrained | GQA (Grouped Query Attention) | $k$ KV heads shared by $h/k$ Q heads — LLaMA 2/3, Gemma, Mistral all use this |
| Large capacity without compute cost | Sparse MoE, top-$k$ routing | Linear capacity scaling, sublinear compute — but load balancing is non-trivial |
| Tabular data, $< 10$K samples | XGBoost/LightGBM, not neural | Trees learn feature interactions from splitting; neural nets need much more data |
| Small labeled + large unlabeled | SSL pretrain then fine-tune | Domain-specific SSL beats ImageNet transfer for specialized domains |
| Strict latency ($< 10$ ms) | Distill large → small | Distilled small models consistently beat architectures designed small from scratch |
| Multi-modal inputs | Separate encoders → shared latent space → task head | Shared trunk forces all modalities through the same bottleneck |

Read [references/architecture.md](references/architecture.md) for implementation patterns.

---

## Phase 2: Profiling and Bottleneck Diagnosis

Run top-to-bottom, stop at the first hit:

```
1. GPU utilization < 80%?
   ├─ YES → Data loading bottleneck
   │        Fix: num_workers, pin_memory, prefetch_factor,
   │             WebDataset for streaming, move preprocessing to GPU
   └─ NO ↓

2. Memory OOM or > 90% usage?
   ├─ YES → Activation/gradient memory bottleneck
   │        Fix: gradient checkpointing, bf16 AMP, sequence packing,
   │             FSDP2 (shards params + grads + optimizer state)
   └─ NO ↓

3. MFU < 40%? (MFU = actual throughput / theoretical peak)
   ├─ YES → Kernel efficiency bottleneck
   │        Fix: torch.compile, FlashAttention, operator fusion,
   │             eliminate graph breaks, FP8 on H100+
   └─ NO ↓

4. Multi-GPU, communication slow?
   ├─ YES → Communication bottleneck
   │        Fix: overlap compute/comm (FSDP2 default),
   │             tune NCCL settings, check NVLink vs PCIe topology
   └─ NO → Compute-bound (good). Scale with more GPUs.
```

### Optimization Map

| Problem | Solution | Impact | Reference |
|---------|----------|--------|-----------|
| OOM training | AMP (bf16) | ~50% memory | profiling-optimization.md §AMP |
| OOM large activations | Gradient checkpointing | ~60-70% activation memory, ~30% slower | profiling-optimization.md §Checkpointing |
| Model too large for 1 GPU | FSDP2 | Shards params+grads+optimizer | distributed-training.md §FSDP2 |
| Need more capacity | Tensor parallel | Splits weight matrices across GPUs | distributed-training.md §TP |
| Very deep model, PP needed | Pipeline parallel | Splits layers across GPUs | distributed-training.md §PP |
| Slow training throughput | torch.compile | 10-40% speedup typical | profiling-optimization.md §Compile |
| Slow attention | FlashAttention-3 | 2-4× speedup, O(1) memory | profiling-optimization.md §FlashAttention |
| H100+ training | FP8 mixed precision | ~2× throughput vs bf16 | profiling-optimization.md §FP8 |
| Slow inference | INT8/FP8 quantization | 2-4× speedup, 50-75% memory | inference-deployment.md §Quantization |
| Autoregressive inference | KV cache | Removes $O(L)$ recomputation per step | inference-deployment.md §KVCache |
| Single request latency | Speculative decoding | 2-3× latency reduction | inference-deployment.md §Speculative |
| Multi-modal architecture | Cross-attention + modality tokenization | — | multimodal-temporal.md |

---

## Phase 3: Distributed Training Strategy

For models too large for a single GPU, choose the parallelism strategy:

```
Model fits on 1 GPU?
├─ YES → DDP (data parallel only). FSDP2 not needed.
└─ NO ↓

Model fits with FSDP2 (parameter sharding)?
├─ YES → FSDP2 alone. Covers most cases up to ~70B params on 8×A100.
└─ NO ↓

Need tensor parallel (split individual layers)?
├─ YES → FSDP2 + TP (2D parallelism). Standard for 70B+ models.
│        Requires NVLink — PCIe TP is communication-bound.
└─ NO → Consider Pipeline Parallel for depth-heavy models.

Training extremely long sequences (> 32K)?
└─ YES → Context Parallel (CP) — split sequence across GPUs.
         Ring Attention pattern. Add as 4th dimension.
```

Read [references/distributed-training.md](references/distributed-training.md) for setup code.

---

## Output Format

### Architecture Design
```
## Architecture: [Name]

### Design Decisions
| Decision | Choice | Rationale |
|----------|--------|-----------|
| Graph mode | eager / compiled | [why] |
| Attention variant | MHA / GQA / MLA / Sliding Window | [why] |
| Position encoding | RoPE / ALiBi / NoPE | [why] |
| Normalization | Pre-LN RMSNorm / QK-Norm | [why] |
| FFN | SwiGLU / MoE | [why] |

### Implementation Notes
[Key PyTorch patterns, non-obvious gotchas, custom layers needed]
```

### Profiling Report
```
## Profile: [Model Name]

### Bottleneck: [diagnosis from the 4-step checklist]

| Component | Params | FLOPs/sample | Peak Memory |
|-----------|--------|-------------|-------------|

### MFU: [X]%  (target: >50% for large models)

### Top-3 Optimizations
1. [What] → [expected impact] — [reference section]
2. ...
3. ...
```
