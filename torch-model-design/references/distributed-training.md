# Distributed Training

How to scale training across multiple GPUs and machines.
Read `principles.md` §5 first for the mental model of DP/TP/PP.

---

## Table of Contents

1. [Decision Framework](#1-decision-framework)
2. [DDP: The Baseline](#2-ddp-the-baseline)
3. [FSDP2: Sharded Data Parallel](#3-fsdp2-sharded-data-parallel)
4. [Tensor Parallel (TP)](#4-tensor-parallel-tp)
5. [Pipeline Parallel (PP)](#5-pipeline-parallel-pp)
6. [2D and 3D Parallelism](#6-2d-and-3d-parallelism)
7. [Context Parallel (CP)](#7-context-parallel-cp)
8. [Distributed Checkpointing](#8-distributed-checkpointing)
9. [Common Failures](#9-common-failures)

---

## 1. Decision Framework

Start from the simplest option. Only add complexity when forced by a concrete bottleneck.

```
Can the full model + optimizer state + activations fit on 1 GPU?
├─ YES → DDP (data parallel). No sharding needed.
└─ NO ↓

Can it fit with bf16 + gradient checkpointing?
├─ YES → DDP + AMP + checkpointing. Still no sharding.
└─ NO ↓

Can FSDP2 shard it across GPUs in one machine?
├─ YES → FSDP2. Covers most cases up to ~30B params on 8×A100.
└─ NO ↓

Is the bottleneck per-layer size (single layer > GPU memory)?
├─ YES → Add Tensor Parallel (TP) within the machine.
│        FSDP2 + TP = 2D parallelism.
└─ NO ↓

Is the model extremely deep (hundreds of layers)?
├─ YES → Add Pipeline Parallel (PP) across machines.
│        FSDP2 + TP + PP = 3D parallelism.
└─ NO → Revisit — you may have a data loading or communication bottleneck,
         not a model size problem.
```

**Rule: every added parallelism dimension adds communication overhead and debugging complexity. Justify each one with a concrete measurement, not a guess.**

---

## 2. DDP: The Baseline

Distributed Data Parallel replicates the model on each GPU and synchronizes gradients after each backward pass via all-reduce.

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

model = MyModel().to(local_rank)
model = DDP(model, device_ids=[local_rank])

# Training loop is identical to single-GPU — DDP handles gradient sync
```

Launch with `torchrun`:

```bash
# Single machine, 4 GPUs
torchrun --nproc_per_node=4 train.py

# Multi-machine (run on each machine)
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
         --master_addr=10.0.0.1 --master_port=29500 train.py
```

**When DDP is enough:** model + optimizer fits on one GPU. DDP only communicates gradients (same size as parameters) once per step — very efficient.

---

## 3. FSDP2: Sharded Data Parallel

FSDP2 (PyTorch 2.4+) shards parameters, gradients, and optimizer state across GPUs. Each GPU stores only $1/N$ of the model.

### Core API

```python
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard

dist.init_process_group("nccl")
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

model = MyTransformer().cuda()

# Shard each transformer block individually, then the whole model
for block in model.layers:
    fully_shard(block)
fully_shard(model)

# Training loop — same as single-GPU
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()
```

### How FSDP2 Differs from FSDP1

| Aspect | FSDP1 | FSDP2 |
|--------|-------|-------|
| API | Wraps module in `FullyShardedDataParallel(model)` | In-place: `fully_shard(model)` |
| Foundation | Custom flat-parameter implementation | Built on DTensor |
| TP composition | Difficult, requires manual setup | Natural — DTensor handles both |
| Parameter access | Flat parameter buffer, original params not accessible | Original `nn.Parameter` objects preserved |
| Mixed precision | Separate `MixedPrecision` policy object | Uses `torch.amp.autocast` directly |

**Rule: for new projects, use FSDP2. FSDP1 is in maintenance mode.**

### What Happens Under the Hood

1. **Before forward:** `all-gather` to reconstruct the full parameter for the current layer
2. **After forward:** discard the gathered copy (only keep the shard)
3. **During backward:** `all-gather` again for the layer, compute gradients
4. **After backward:** `reduce-scatter` to shard the gradients
5. **Optimizer step:** each GPU updates only its shard

Communication and computation overlap by default — while one layer is doing matmul, the next layer's parameters are being gathered in the background.

---

## 4. Tensor Parallel (TP)

Tensor Parallel splits individual weight matrices across GPUs. Unlike FSDP (which gathers the full parameter before compute), TP does partial computation on each GPU and combines the results.

### When to Use

- A single layer's parameters don't fit on one GPU (rare for < 70B)
- You have NVLink interconnect (**TP over PCIe is almost always a bad idea** — the per-layer all-reduce is too frequent for low-bandwidth links)

### Column and Row Parallel

A linear layer $Y = XW$ can be split two ways:

**Column parallel:** split $W$ by columns across GPUs. Each GPU computes a slice of the output. Requires an all-gather to reconstruct the full output.

**Row parallel:** split $W$ by rows. Each GPU has the full input but produces a partial sum. Requires an all-reduce to combine.

In a Transformer, the standard pattern is:
- QKV projection: column parallel (each GPU gets a subset of heads)
- Output projection: row parallel (reduces back to full dimension)
- FFN first linear: column parallel
- FFN second linear: row parallel

This way, each column→row pair requires only one all-reduce, not two.

### PyTorch API

```python
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)

# Assume tp_mesh is a DeviceMesh for tensor parallelism
parallelize_module(
    model.attention,
    tp_mesh,
    {
        "q_proj": ColwiseParallel(),
        "k_proj": ColwiseParallel(),
        "v_proj": ColwiseParallel(),
        "o_proj": RowwiseParallel(),
    },
)
parallelize_module(
    model.ffn,
    tp_mesh,
    {
        "w1": ColwiseParallel(),   # gate / up projection
        "w3": ColwiseParallel(),   # SwiGLU second gate
        "w2": RowwiseParallel(),   # down projection
    },
)
```

---

## 5. Pipeline Parallel (PP)

Pipeline Parallel assigns different layers to different GPUs. Data flows through them sequentially.

### The Bubble Problem

Naive PP leaves GPUs idle while waiting for activations:

```
GPU 0: [layer 0-3] ████░░░░░░░░████░░░░░░░░
GPU 1: [layer 4-7] ░░░░████░░░░░░░░████░░░░
                    ↑ idle (bubble)
```

**Micro-batching** reduces the bubble by splitting each batch into smaller chunks:

```
GPU 0: ████ ████ ████ ████ ░░░░
GPU 1: ░████ ████ ████ ████░░░░
         ↑ much smaller bubble
```

With $M$ micro-batches and $P$ pipeline stages, bubble fraction $\approx (P-1)/M$.
Rule of thumb: use $M \geq 4P$ to keep bubble overhead under 20%.

### PyTorch API

```python
from torch.distributed.pipelining import SplitPoint, pipeline, ScheduleGPipe

# Define where to split the model
pipe = pipeline(
    model,
    mb_args=(micro_batch,),
    split_spec={
        "layers.8": SplitPoint.BEGINNING,   # split after layer 7
        "layers.16": SplitPoint.BEGINNING,  # split after layer 15
        "layers.24": SplitPoint.BEGINNING,  # split after layer 23
    },
)

# GPipe schedule (simple, good baseline)
schedule = ScheduleGPipe(pipe, n_microbatches=16)
schedule.step(micro_batch)
```

**Rule: PP is the right choice when you need to scale across machines (where interconnect is slow). Use TP within a machine, PP across machines.**

---

## 6. 2D and 3D Parallelism

Real large-model training combines multiple dimensions using `DeviceMesh`.

### DeviceMesh

A `DeviceMesh` represents the logical topology of your GPUs:

```python
from torch.distributed.device_mesh import init_device_mesh

# 2D: 8 GPUs as (2 DP nodes × 4 TP within each node)
mesh_2d = init_device_mesh("cuda", (2, 4), mesh_dim_names=("dp", "tp"))

# 3D: 32 GPUs as (4 DP × 2 PP × 4 TP)
mesh_3d = init_device_mesh("cuda", (4, 2, 4), mesh_dim_names=("dp", "pp", "tp"))
```

### 2D Parallelism (FSDP2 + TP)

The standard setup for 70B+ models on a single machine or small cluster:

```python
mesh_2d = init_device_mesh("cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp"))

# Step 1: apply TP within each node (high-bandwidth NVLink)
tp_mesh = mesh_2d["tp"]
for block in model.layers:
    parallelize_module(block.attention, tp_mesh, {...})  # as shown in §4
    parallelize_module(block.ffn, tp_mesh, {...})

# Step 2: apply FSDP2 across nodes (lower-bandwidth network)
dp_mesh = mesh_2d["dp"]
for block in model.layers:
    fully_shard(block, mesh=dp_mesh)
fully_shard(model, mesh=dp_mesh)
```

**Rule: TP goes on the fast dimension (NVLink within machine), FSDP goes on the slow dimension (network across machines). Swapping them destroys performance.**

---

## 7. Context Parallel (CP)

Context Parallel splits a long sequence across GPUs. Each GPU handles a chunk of the sequence.

This is necessary when the sequence is so long that even FlashAttention's $O(S)$ memory per GPU is too much, or when prefill compute for very long contexts needs to be distributed.

The typical implementation is **Ring Attention**: each GPU holds a chunk of Q and circulates K/V chunks around a ring of GPUs. After one full rotation, every Q chunk has attended to every K/V chunk.

CP is an advanced technique — use it only when:
- Sequence length > 32K tokens
- Memory or compute for a single-GPU attention is the bottleneck
- You've already applied FlashAttention and gradient checkpointing

---

## 8. Distributed Checkpointing

Standard `torch.save(model.state_dict())` doesn't work with sharded models — each GPU only has a fragment.

```python
import torch.distributed.checkpoint as dcp

# Save (each GPU writes its shard)
dcp.save(
    state_dict={"model": model.state_dict(), "optimizer": optimizer.state_dict()},
    storage_writer=dcp.FileSystemWriter("checkpoints/step_1000"),
)

# Load (can load on a different number of GPUs — resharding is automatic)
state_dict = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
dcp.load(
    state_dict=state_dict,
    storage_reader=dcp.FileSystemReader("checkpoints/step_1000"),
)
model.load_state_dict(state_dict["model"])
optimizer.load_state_dict(state_dict["optimizer"])
```

**Key advantage over `torch.save`:** resharding. You can save with 8 GPUs and load with 16 GPUs (or vice versa). The checkpoint system handles the redistribution.

**Rule: for any model using FSDP2 or TP, use `torch.distributed.checkpoint`, not `torch.save`. Saving to a single file from a sharded model requires gathering everything to one GPU — wasteful and often OOMs.**

---

## 9. Common Failures

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Hang at startup | NCCL can't connect across machines | Check firewall, `MASTER_ADDR`, `MASTER_PORT` |
| Hang during training | One GPU crashed, others wait for it | Add `NCCL_TIMEOUT` env var, check GPU logs |
| Loss diverges with more GPUs | Effective batch size increased without adjusting lr | Scale lr with $\sqrt{N}$ or use linear scaling with warmup |
| OOM only on rank 0 | Logging / metrics accumulated on rank 0 | Distribute or offload logging |
| Slow multi-node training | PCIe or ethernet bottleneck | Profile NCCL with `NCCL_DEBUG=INFO`, check if TP is crossing machines |
| Different loss on 1 GPU vs 8 GPUs | Non-deterministic reductions in mixed precision | Expected — small differences are normal. Large differences indicate a bug |

**Rule: always verify that your distributed training produces the same loss curve (within noise) as single-GPU training on a small model first. If they diverge, you have a bug in gradient synchronization, learning rate scaling, or data sampling.**
