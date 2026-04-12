# Architecture Patterns

Reference implementations for common neural architecture components.
For the *why* behind these choices, read `principles.md`.
For graph modes and torch.compile, see `principles.md` §1.

---

## Table of Contents

1. [Attention Variants](#1-attention-variants)
2. [Feed-Forward Variants](#2-feed-forward-variants)
3. [Position Encoding](#3-position-encoding)
4. [Transformer Block Assembly](#4-transformer-block-assembly)
5. [Initialization](#5-initialization)
6. [Gradient Flow Diagnostics](#6-gradient-flow-diagnostics)

---

## 1. Attention Variants

### Which Attention to Use

| Variant | KV heads | When to use | Used by |
|---------|----------|-------------|---------|
| **MHA** (Multi-Head) | $h$ (all heads) | Small models, no inference pressure | BERT, GPT-2, ViT |
| **GQA** (Grouped Query) | $h/k$ (groups of $k$ share KV) | Default for modern LLMs — best quality/memory trade-off | LLaMA 2/3, Gemma, Mistral |
| **MQA** (Multi-Query) | 1 | Extreme inference efficiency, slight quality loss | PaLM, Falcon |
| **Sliding Window** | $h$ or $h/k$, but attention limited to window $w$ | Long sequences where full attention is too expensive | Mistral, Gemma 2 |

**Rule: GQA is the default for new projects. MHA only if you have no inference latency
concerns. MQA only if you've verified GQA isn't fast enough.**

### GQA Implementation

Core idea: $n_{kv}$ KV heads are shared by $n_{heads} / n_{kv}$ query heads each.
At compute time, expand KV heads with `repeat_interleave` before the attention matmul.

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

    def forward(self, x, mask=None, past_kv=None):
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        # Expand KV heads to match Q head count
        k = k.repeat_interleave(self.n_groups, dim=1)
        v = v.repeat_interleave(self.n_groups, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = out.transpose(1, 2).reshape(B, L, -1)
        return self.o_proj(out), (k[:, ::self.n_groups], v[:, ::self.n_groups])
```

### Sliding Window Attention

Each token attends only to the nearest $w$ tokens instead of the full sequence.
Complexity drops from $O(L^2)$ to $O(L \times w)$. Information beyond window $w$
propagates through stacking layers (layer $l$ has effective receptive field $l \times w$).

Typical pattern: alternate sliding-window layers with a few full-attention layers
(e.g., full attention every 4th layer). This gives global context without global cost.

---

## 2. Feed-Forward Variants

### Which FFN to Use

| Variant | Param count (relative) | Quality | Used by |
|---------|----------------------|---------|---------|
| **Standard (ReLU/GELU)** | 1× ($2 \times d \times d_{ff}$) | Baseline | BERT, GPT-2, ViT |
| **SwiGLU** | 1.5× ($3 \times d \times d_{ff}$, but $d_{ff}$ is reduced to compensate) | Better | LLaMA, PaLM, Gemma |
| **MoE (Sparse)** | Many experts, only top-$k$ active per token | Highest capacity per FLOP | Mixtral, Switch Transformer |

**Rule: SwiGLU is the default for new dense models. MoE when you need more
capacity without proportional compute increase — but adds routing complexity.**

### SwiGLU

Three linear layers instead of two. The gate projection controls information flow.

```python
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        d_ff = d_ff or int(d_model * 8 / 3)
        d_ff = ((d_ff + 63) // 64) * 64    # round to 64 for GPU efficiency
        self.w1 = nn.Linear(d_model, d_ff, bias=False)   # gate
        self.w3 = nn.Linear(d_model, d_ff, bias=False)   # up
        self.w2 = nn.Linear(d_ff, d_model, bias=False)   # down

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

### Sparse MoE (Concept)

Replace the single FFN with $E$ expert FFNs. A router picks top-$k$ experts per token.

```
Router(x) → weights for top-k experts
Output = sum_i weight_i * Expert_i(x)     (only k out of E experts run)
```

Key engineering concerns:
- **Load balancing:** if all tokens route to the same expert, other experts waste memory.
  Add an auxiliary loss that penalizes uneven routing.
- **All-to-all communication:** in distributed training, tokens on GPU A may need expert
  on GPU B. This requires an all-to-all exchange — the main bottleneck for MoE.
- **Memory:** all $E$ experts must be in memory even though only $k$ run per token.

---

## 3. Position Encoding

### Which Position Encoding to Use

| Method | Type | Context extrapolation | Used by |
|--------|------|----------------------|---------|
| **RoPE** | Relative, rotary | Good (with NTK/YaRN scaling) | LLaMA, Gemma, Mistral, Qwen |
| **ALiBi** | Relative, additive bias | Natural extrapolation | BLOOM, MPT |
| **Learned absolute** | Absolute | Poor beyond training length | GPT-2, BERT |

**Rule: RoPE is the default. It supports relative positioning, composes well with
GQA, and can be extended to very long contexts via frequency scaling.**

### RoPE

Encodes relative position by rotating Q and K vectors in 2D subspaces.
Token distance determines the angle between them after rotation.

```python
def precompute_freqs(dim, max_seq_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64

def apply_rotary_emb(xq, xk, freqs):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs = freqs[:xq_.shape[-2]]
    xq_out = torch.view_as_real(xq_ * freqs).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)
```

**RoPE for long contexts:** the base frequency $\theta$ controls the maximum effective
context length. To extend beyond training length, scale $\theta$ upward (NTK-aware RoPE)
or apply a piecewise frequency adjustment (YaRN). Both allow extending from 4K to 128K+
without retraining.

---

## 4. Transformer Block Assembly

Putting attention + FFN + normalization together. The modern standard:

```
Pre-LN block:  x → RMSNorm → Attention → + residual → RMSNorm → FFN → + residual → output
```

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, d_ff):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads)
        self.norm2 = nn.RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x, mask=None, past_kv=None):
        h, new_kv = self.attn(self.norm1(x), mask=mask, past_kv=past_kv)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x, new_kv
```

**Key conventions:**
- Pre-LN (normalize before sublayer), not Post-LN — see `principles.md` §4
- RMSNorm, not LayerNorm — ~15% faster, equivalent quality
- No bias in linear layers (LLaMA convention) — marginal param savings, no quality impact
- `F.scaled_dot_product_attention` inside the attention module — gets FlashAttention automatically

---

## 5. Initialization

Wrong initialization can make training 10× slower or prevent convergence entirely.

| Component | Init | Why |
|-----------|------|-----|
| Q, K, V projections | Xavier uniform | Preserves variance through attention |
| FFN intermediate | Kaiming (He) | Matches GELU/SiLU non-linearity |
| Output projections (o_proj, w2) | $\mathcal{N}(0,\, 0.02 / \sqrt{2L})$ | Prevents residual stream from growing with depth $L$ |
| Embeddings | $\mathcal{N}(0,\, 0.02)$ | Empirical standard for language models |

```python
def init_weights(model, n_layers):
    for name, p in model.named_parameters():
        if p.dim() < 2:
            continue
        if "o_proj" in name or "w2" in name:
            nn.init.normal_(p, mean=0.0, std=0.02 / (2 * n_layers) ** 0.5)
        else:
            nn.init.normal_(p, mean=0.0, std=0.02)
```

**Rule: the $1/\sqrt{2L}$ scaling on output projections is the most important initialization
detail for deep transformers. Without it, activations grow linearly with depth and
training becomes unstable past ~24 layers.**

---

## 6. Gradient Flow Diagnostics

Use this utility after `loss.backward()` to spot vanishing or exploding gradients.

```python
def plot_gradient_flow(model):
    import matplotlib.pyplot as plt
    ave_grads, max_grads, layers = [], [], []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            layers.append(name)
            ave_grads.append(param.grad.abs().mean().item())
            max_grads.append(param.grad.abs().max().item())
    plt.figure(figsize=(max(12, len(layers) * 0.3), 6))
    plt.bar(range(len(max_grads)), max_grads, alpha=0.3, color="c", label="max")
    plt.bar(range(len(ave_grads)), ave_grads, alpha=0.3, color="b", label="mean")
    plt.xticks(range(len(layers)), layers, rotation=90, fontsize=7)
    plt.yscale("log")
    plt.ylabel("Gradient magnitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig("gradient_flow.png", dpi=150)
    plt.close()
```

| Symptom | Diagnosis | Fix |
|---------|-----------|-----|
| Mean gradient $< 10^{-7}$ in early layers | Vanishing gradients | Pre-LN, residual connections, GELU/SiLU |
| Max gradient $> 10^3$ | Exploding gradients | Gradient clipping, reduce lr |
| Gradient exactly 0 | Dead neurons (ReLU) | Check bias init, switch to SiLU/GELU |
| Gradients much larger in later layers | Poor flow | Add skip connections, switch to Pre-LN |

For systematic diagnosis flow, see `principles.md` §4.
