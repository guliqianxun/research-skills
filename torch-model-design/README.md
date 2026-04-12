# torch-model-design

PyTorch model design from zero to production — engineering principles, architecture patterns, profiling, distributed training, multimodal/temporal models, and inference optimization.

## What It Does

When you ask the AI to design, profile, or optimize a PyTorch model, this skill provides:

- **Engineering principles** — mental models for memory, precision, stability, parallelism, and the roofline model, written for someone with no PyTorch experience
- **Architecture patterns** — GQA, SwiGLU, RoPE, MoE, sliding window attention with decision tables for when to use each
- **Profiling workflow** — bottleneck diagnosis (GPU util → memory → MFU → communication), MFU computation, torch.profiler
- **Distributed training** — FSDP2, tensor parallel, pipeline parallel, 2D/3D parallelism with DeviceMesh, all using PyTorch 2.4+ APIs
- **Multimodal and temporal models** — modality tokenization (text/image/audio/video), cross-attention vs interleaved, alignment, streaming inference
- **Inference optimization** — KV cache, quantization via torchao, speculative decoding, continuous batching, torch.export

## Reference Files

| File | Contents |
|------|----------|
| `references/principles.md` | **Start here.** 11 engineering principles: computation graph, memory, precision, training stability, parallelism, inference vs training, multimodal rules, arithmetic intensity, data loading, reproducibility, loss patterns |
| `references/architecture.md` | Attention variants (MHA/GQA/MQA/sliding window), FFN variants (SwiGLU/MoE), RoPE, block assembly, initialization, gradient diagnostics |
| `references/profiling-optimization.md` | MFU, FLOPs estimation, memory profiling, AMP (bf16/fp16/fp8), FlashAttention, torch.compile, gradient checkpointing, sequence packing, benchmarking |
| `references/distributed-training.md` | DDP, FSDP2, tensor parallel, pipeline parallel, DeviceMesh, 2D/3D parallelism, context parallel, distributed checkpointing, common failures |
| `references/multimodal-temporal.md` | Modality tokenization, cross-attention vs interleaved, projection/alignment, temporal patterns, causal masking, streaming inference, memory budgeting |
| `references/inference-deployment.md` | KV cache, quantization (INT8/INT4/FP8/GPTQ/AWQ), speculative decoding, continuous batching, torch.export, optimization checklist |

## Example Prompts

- "Design a multimodal model that takes video + text and generates captions"
- "My training OOMs on 8K sequence length — diagnose and fix"
- "Set up FSDP2 + tensor parallel for a 70B model on 8 GPUs"
- "Quantize this model to INT8 for inference and benchmark the speedup"
- "Why is my MFU only 25%?"

## Install

```bash
# Claude Code
claude mcp add-skill /path/to/research-skills/torch-model-design
```
