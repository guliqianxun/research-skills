# Multimodal and Temporal Models

Engineering patterns for models that process multiple modalities (text, image, audio, video)
and/or temporal sequences. Read `principles.md` §7 first for the core conventions.

---

## Table of Contents

1. [Modality Tokenization](#1-modality-tokenization)
2. [Cross-Attention vs Interleaved Attention](#2-cross-attention-vs-interleaved-attention)
3. [Modality Alignment and Projection](#3-modality-alignment-and-projection)
4. [Temporal Modeling Patterns](#4-temporal-modeling-patterns)
5. [Causal Masking for Generation](#5-causal-masking-for-generation)
6. [Streaming Inference](#6-streaming-inference)
7. [Memory Budget for Multimodal](#7-memory-budget-for-multimodal)

---

## 1. Modality Tokenization

Every modality must be converted into a sequence of vectors with the same dimension $d$
before the backbone can process it. This conversion is called **modality tokenization**.

The design goal: produce a token sequence that is **information-dense** (no wasted tokens)
and **compatible** with the backbone's position encoding and attention pattern.

### Text

Standard and well-understood. A tokenizer (BPE, SentencePiece) maps strings to integer IDs,
then an embedding table maps IDs to vectors.

```python
# text_ids: (B, L_text) integer tensor
text_tokens = self.text_embed(text_ids)  # (B, L_text, d)
```

Token count is proportional to word count. Typical: 1 word ≈ 1.3 tokens.

### Image

Split the image into non-overlapping patches, flatten each patch, project linearly.
This is the ViT (Vision Transformer) pattern.

```python
class PatchEmbed(nn.Module):
    """Convert image to patch token sequence. ViT / SigLIP pattern."""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, d_model=1024):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        # Conv2d with kernel=stride=patch_size is equivalent to patch + linear
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)              # (B, d, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, n_patches, d)
        return x
```

**Token budget:** a 224×224 image with 16×16 patches → 196 tokens.
A 336×336 image → 441 tokens. Higher resolution = quadratically more tokens.

**Rule: image resolution directly controls token count, which controls attention cost.
Choose the minimum resolution that preserves task-relevant detail.**

### Audio

Convert raw waveform to mel-spectrogram, then use a small convolutional encoder
to produce token-rate vectors.

```python
class AudioEncoder(nn.Module):
    """Mel-spectrogram + conv stack → token sequence. Whisper-style."""
    def __init__(self, n_mels=80, d_model=1024):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        self.gelu = nn.GELU()

    def forward(self, mel):
        # mel: (B, n_mels, T_frames)
        x = self.gelu(self.conv1(mel))
        x = self.gelu(self.conv2(x))  # stride=2 halves the time dimension
        return x.transpose(1, 2)       # (B, T_frames/2, d)
```

**Token rate:** raw audio at 16kHz → mel-spectrogram at ~100 frames/sec → after stride-2 conv,
~50 tokens/sec. A 30-second clip ≈ 1500 tokens.

### Video

Video = spatial (image) + temporal (frame sequence). Two tokenization strategies:

**Strategy A: per-frame patch embedding (simple, high token count)**

Process each frame with a ViT-style patch embed, concatenate across frames.
Token count = n_frames × patches_per_frame. A 10s video at 4fps with 196 patches/frame = 7840 tokens.

**Strategy B: 3D patch embedding (efficient, lower token count)**

Treat video as a 3D volume (T, H, W) and extract spatiotemporal patches.

```python
class VideoPatchEmbed(nn.Module):
    """3D patch embedding for video. Compresses temporal and spatial dimensions jointly."""
    def __init__(self, temporal_patch=2, spatial_patch=16, in_channels=3, d_model=1024):
        super().__init__()
        self.proj = nn.Conv3d(
            in_channels, d_model,
            kernel_size=(temporal_patch, spatial_patch, spatial_patch),
            stride=(temporal_patch, spatial_patch, spatial_patch),
        )

    def forward(self, x):
        # x: (B, C, T, H, W)
        x = self.proj(x)                          # (B, d, T/tp, H/sp, W/sp)
        x = x.flatten(2).transpose(1, 2)          # (B, num_tokens, d)
        return x
```

With temporal_patch=2: token count is halved compared to per-frame approach.

**Rule: for video, always use temporal compression (3D patches or temporal stride).
Without it, token counts explode and attention becomes the bottleneck within seconds of video.**

---

## 2. Cross-Attention vs Interleaved Attention

Once each modality is tokenized, the backbone needs to combine them. Two main patterns exist,
with different trade-offs.

### Pattern A: Cross-Attention (Flamingo, Gemini 1.0)

The backbone processes one modality (usually text) as the primary sequence.
Other modalities are injected via **cross-attention layers** inserted between self-attention layers.

```
Text tokens:   [t1  t2  t3  t4  t5 ]
                 ↓ self-attn  ↓
               [t1' t2' t3' t4' t5']
                 ↓ cross-attn(Q=text, KV=image) ↓
               [t1" t2" t3" t4" t5"]
                 ↓ FFN  ↓
               [...]

Image tokens:  [i1  i2  ... i196]  ← frozen encoder output, used as KV
```

```python
class CrossAttentionBlock(nn.Module):
    """Text attends to another modality. Text is Q, other modality is KV."""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.norm_q = nn.RMSNorm(d_model)
        self.norm_kv = nn.RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

    def forward(self, x, context):
        # x: primary modality (B, L, d) — text
        # context: other modality (B, S, d) — image/audio
        q = self.q_proj(self.norm_q(x))
        k = self.k_proj(self.norm_kv(context))
        v = self.v_proj(self.norm_kv(context))

        B, L, _ = q.shape
        S = k.shape[1]
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, L, -1)
        return x + self.o_proj(out)
```

**Advantages:**
- Vision encoder can be frozen — only cross-attention layers are trained
- Primary modality's self-attention complexity is unchanged ($O(L_\text{text}^2)$)
- Clean separation: easy to add/remove modalities

**Disadvantages:**
- Modalities don't interact bidirectionally (text sees image, but image doesn't see text)
- Requires choosing which layers get cross-attention (every layer? every 4th layer?)

### Pattern B: Interleaved Sequence (LLaVA, GPT-4V style)

Concatenate all modality tokens into a single flat sequence, separated by special tokens.
The backbone's self-attention processes everything uniformly.

```
Input sequence: [BOS] t1 t2 [IMG_START] i1 i2 ... i196 [IMG_END] t3 t4 [EOS]
                 ↓ standard self-attention over the whole sequence ↓
```

```python
def build_multimodal_sequence(text_tokens, image_tokens, special_tokens):
    """Concatenate modality tokens with separator tokens."""
    img_start = special_tokens["IMG_START"].unsqueeze(0)  # (1, d)
    img_end = special_tokens["IMG_END"].unsqueeze(0)

    # text_tokens: (B, L_text, d), image_tokens: (B, L_img, d)
    seq = torch.cat([
        text_tokens[:, :insert_pos],
        img_start.expand(B, -1, -1),
        image_tokens,
        img_end.expand(B, -1, -1),
        text_tokens[:, insert_pos:],
    ], dim=1)
    return seq
```

**Advantages:**
- All modalities attend to each other bidirectionally
- No architectural changes to the backbone — just a longer sequence
- Simple implementation

**Disadvantages:**
- Sequence length = sum of all modality token counts → attention cost explodes
- All modality encoders must produce the same dimension $d$
- Position encoding must handle mixed modalities coherently

### When to Use Which

| Situation | Recommended | Why |
|-----------|------------|-----|
| Frozen vision encoder + trainable LLM | Cross-attention | Only trains the cross-attention layers; efficient |
| End-to-end training, all modalities equal | Interleaved | Bidirectional interaction from the start |
| Many modalities (text + image + audio + video) | Cross-attention | Adding modalities doesn't change backbone |
| Low token budget, short sequences | Interleaved | Simpler, no extra layers needed |
| Very long context (video + long text) | Cross-attention | Avoids $O((L_\text{text} + L_\text{video})^2)$ |

---

## 3. Modality Alignment and Projection

Each modality encoder lives in its own embedding space. Before tokens reach the backbone,
they must be **projected** into a shared space where "a photo of a cat" and the word "cat"
are nearby.

### Projection Layer: The Alignment Boundary

```python
class ModalityProjector(nn.Module):
    """Projects modality encoder output into backbone's embedding space.
    This is the alignment boundary — everything before it is modality-specific,
    everything after it is modality-agnostic."""
    def __init__(self, encoder_dim, backbone_dim, n_hidden=1):
        super().__init__()
        if n_hidden == 0:
            self.proj = nn.Linear(encoder_dim, backbone_dim)
        else:
            layers = [nn.Linear(encoder_dim, backbone_dim), nn.GELU()]
            for _ in range(n_hidden - 1):
                layers += [nn.Linear(backbone_dim, backbone_dim), nn.GELU()]
            layers.append(nn.Linear(backbone_dim, backbone_dim))
            self.proj = nn.Sequential(*layers)

    def forward(self, x):
        return self.proj(x)
```

**Design choices:**
- **Linear projection (n_hidden=0):** simplest, used in LLaVA v1. Fast to train, limited expressiveness.
- **2-layer MLP (n_hidden=1):** used in LLaVA v1.5+. Better alignment, marginal cost increase.
- **Deeper MLP or small transformer:** diminishing returns in practice.

**Rule: a 2-layer MLP projector is the default. Go simpler only if training data is very limited;
go deeper only if you have evidence it helps on your specific task.**

### Pre-training for Alignment

Projectors are typically pre-trained with a **contrastive objective** (CLIP-style) or a
**captioning objective** before end-to-end fine-tuning:

**Contrastive (CLIP):** push matching (image, text) pairs close, push non-matching pairs apart.
Good for retrieval and zero-shot tasks, but doesn't learn fine-grained spatial understanding.

**Captioning (generative):** the LLM generates a caption conditioned on image tokens.
Forces the projector to preserve spatial and semantic detail.

**Two-stage training (LLaVA recipe):**
1. **Stage 1 — alignment pre-training:** freeze encoder + backbone, only train projector
   on image-caption pairs. Cheap (a few hours on 8 GPUs).
2. **Stage 2 — instruction tuning:** unfreeze backbone (optionally encoder too),
   train on instruction-following data. More expensive but teaches the model to reason.

**Rule: always freeze the backbone in stage 1. Training backbone + projector together from
scratch will destroy the LLM's pre-trained knowledge before alignment has converged.**

### Normalization Across Modalities

Different encoders produce outputs with different value ranges.
If image encoder outputs have norm ~50 and text embeddings have norm ~1,
the backbone's attention will be dominated by image tokens.

**Fix:** normalize each modality's tokens independently before mixing.

```python
# Apply per-modality normalization before feeding to backbone
text_tokens = self.text_norm(text_tokens)    # RMSNorm or LayerNorm
image_tokens = self.image_norm(image_tokens)
```

**Rule: never assume two different encoders produce outputs on the same scale.
Always add a normalization layer after each projector.**

---

## 4. Temporal Modeling Patterns

Temporal data has an ordering constraint that spatial data does not: the future depends on
the past, but not vice versa. This constraint shapes every design decision.

### The Fundamental Split: Causal vs Bidirectional

| Mode | Can attend to | Used for | Examples |
|------|--------------|----------|----------|
| **Causal** (autoregressive) | Past and present only | Generation, prediction | GPT, LLaMA, audio synthesis |
| **Bidirectional** | Everything | Understanding, classification | BERT, ViT, Whisper encoder |
| **Prefix + causal** | Prefix sees all; generation sees past only | Conditioned generation | T5, multimodal LLMs |

**Rule: if the model needs to generate output token by token, the generation part must be
causal. If it only needs to understand/classify, bidirectional is strictly better.**

Most multimodal models use **prefix + causal**: the input (image/audio/prompt) is processed
bidirectionally, then the output is generated causally.

### Position Encoding for Temporal Data

For temporal sequences, position encoding carries **physical meaning** — it encodes *when*
something happened, not just *where it is in the sequence*.

**RoPE (Rotary Position Embedding):**
Standard for modern LLMs. Encodes relative position via rotation in complex space.
Naturally supports variable-length sequences and has been extended to very long contexts
(via frequency scaling: NTK-aware RoPE, YaRN).

**Temporal timestamps:**
For multimodal temporal data (e.g., video frames at irregular intervals, interleaved
text-and-image), raw position indices are meaningless — frame 5 and frame 6 might be
0.25 seconds apart (video) or 30 seconds apart (surveillance).

Solution: **continuous-time position encoding.**

```python
class ContinuousTimeEmbed(nn.Module):
    """Encode absolute timestamps as position embeddings.
    Useful when tokens arrive at irregular intervals across modalities."""
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(1, d_model)

    def forward(self, timestamps):
        # timestamps: (B, L) float tensor, in seconds
        return self.linear(timestamps.unsqueeze(-1))  # (B, L, d)
```

Add to token embeddings: `tokens = tokens + self.time_embed(timestamps)`.

**Rule: if different modality tokens correspond to different physical times,
use timestamp-based position encoding, not integer position indices.
Integer indices encode order; timestamps encode *when*.**

### Variable-Rate Fusion

Different modalities produce tokens at different rates for the same time window:

| Modality | Typical token rate | 10 seconds = |
|----------|-------------------|-------------|
| Text | event-driven | 5–50 tokens |
| Audio | ~50 tokens/sec | ~500 tokens |
| Video (4fps, 196 patches/frame) | ~784 tokens/sec | ~7840 tokens |
| Sensor (IMU, 100Hz) | ~100 tokens/sec | ~1000 tokens |

The backbone sees a single interleaved sequence, but modalities are wildly unbalanced.

**Strategies:**

1. **Downsample heavy modalities.** Video: use 1–2 fps instead of 30. Audio: increase stride
   in the convolutional encoder. Goal: bring all modalities to roughly the same token rate.

2. **Perceiver-style bottleneck.** Use a small set of learned query tokens to cross-attend
   into the high-rate modality, compressing it to a fixed token budget regardless of duration.

3. **Hierarchical encoding.** Process each modality at its native rate with a modality-specific
   encoder, then produce a summary at a lower rate for the backbone.

**Rule: the backbone's context window is a fixed budget. Allocate it across modalities based
on information density, not raw signal rate. A 10-second audio clip doesn't need 10× more
tokens than 10 words of text.**

---

## 5. Causal Masking for Generation

### Standard Causal Mask

For autoregressive generation, token $i$ can only attend to tokens $j \leq i$.
This is a lower-triangular boolean mask.

```python
# PyTorch's SDPA has a built-in causal mode — always prefer this over manual masks
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

Using `is_causal=True` is both faster (FlashAttention uses an optimized kernel path)
and safer (no risk of off-by-one in a hand-written mask) than constructing an explicit mask tensor.

### Prefix-Causal Mask (Multimodal Generation)

In multimodal generation, the input prefix (image tokens, prompt) should attend bidirectionally
to itself, but the generated tokens should only attend causally.

```
Attention pattern:
              prefix (bidirectional)     generation (causal)
              [img1 img2 ... prompt]     [gen1 gen2 gen3 ...]
  img1          ✓    ✓    ✓    ✓           ✗    ✗    ✗
  img2          ✓    ✓    ✓    ✓           ✗    ✗    ✗
  prompt        ✓    ✓    ✓    ✓           ✗    ✗    ✗
  gen1          ✓    ✓    ✓    ✓           ✓    ✗    ✗
  gen2          ✓    ✓    ✓    ✓           ✓    ✓    ✗
  gen3          ✓    ✓    ✓    ✓           ✓    ✓    ✓
```

```python
def make_prefix_causal_mask(prefix_len: int, total_len: int, device="cuda"):
    """Prefix tokens attend to all prefix tokens; generated tokens attend causally."""
    mask = torch.ones(total_len, total_len, dtype=torch.bool, device=device).tril()
    mask[:prefix_len, :prefix_len] = True  # prefix is fully bidirectional
    return mask
```

### Sequence Packing with Causal Masking

When packing multiple sequences into one batch entry (§8 of profiling-optimization.md),
each sequence must be masked independently — token 1 of sequence B must not attend to
the last token of sequence A.

```
Packed:    [seq_A tok1 tok2 tok3 | seq_B tok1 tok2]
Correct:    A attends to A only;  B attends to B only
Wrong:      B.tok1 attends to A.tok3 (information leak across sequences)
```

The standard solution is a **block-diagonal causal mask**, or equivalently, using
`document_id` arrays with FlashAttention's variable-length API.

**Rule: when implementing sequence packing, always verify that cross-sequence attention
is blocked. Test by checking that two independent sequences produce identical outputs
whether packed together or processed separately.**

---

## 6. Streaming Inference

Streaming inference processes input incrementally as it arrives, rather than waiting for
the complete input. Critical for real-time applications: live transcription, video
understanding, conversational agents.

### The Core Constraint

In streaming mode, the model sees input up to time $t$ and must produce output
**without access to future input**. This means:

1. The encoder must be causal or chunked (no full-sequence bidirectional attention)
2. KV cache must be maintained across chunks
3. Latency = time to process one chunk, not the full sequence

### Chunked Processing Pattern

Split the input stream into fixed-size chunks. Process each chunk, update the KV cache,
emit output.

```python
class StreamingModel:
    def __init__(self, model):
        self.model = model
        self.kv_cache = None

    def process_chunk(self, chunk_tokens):
        """Process one chunk and return output. Maintains state across calls."""
        output, self.kv_cache = self.model(
            chunk_tokens,
            past_kv=self.kv_cache,
            use_cache=True,
        )
        return output

    def reset(self):
        """Call at the start of a new stream."""
        self.kv_cache = None
```

### Chunk Size Trade-off

| Chunk size | Latency | Throughput | Quality |
|-----------|---------|------------|---------|
| Small (100ms) | Low | Low (more overhead per token) | May miss long-range context |
| Large (2s) | High | High (better GPU utilization) | Better context, but delayed output |

**Rule: chunk size is a latency-throughput-quality trade-off. Start with 500ms–1s chunks,
then tune based on your application's latency requirement.**

---

## 7. Memory Budget for Multimodal

Multimodal models are memory-heavy because they carry multiple encoders, a projector,
and a backbone. Estimate memory before building.

### Quick Budget Template

For a multimodal model with a frozen vision encoder, trainable projector, and trainable LLM:

| Component | Parameters | Training memory (bf16 + AdamW) |
|-----------|-----------|-------------------------------|
| Vision encoder (frozen) | $P_v$ | $2P_v$ (weights only, no gradients/optimizer) |
| Projector (trainable) | $P_p$ | $12P_p$ (weights + grads + Adam state) |
| LLM backbone (trainable) | $P_l$ | $12P_l$ |
| Activations | — | Depends on sequence length |
| KV cache (inference) | — | $2 \times n\_layers \times 2 \times B \times L \times d_{kv} \times \text{bytes}$ |

Example: SigLIP-400M (frozen) + 2-layer MLP projector (10M) + LLaMA-7B (trainable)

- Vision encoder: 400M × 2 bytes = 0.8 GB
- Projector: 10M × 12 bytes = 0.12 GB
- LLM: 7B × 12 bytes = 84 GB
- **Total (no activations):** ~85 GB → needs at least 2× A100-80GB with FSDP2

**Rule: the LLM backbone dominates memory in nearly all multimodal architectures.
Freezing the vision encoder saves relatively little memory — it's already the small part.
To reduce memory, target the LLM: quantize it, shard it, or use a smaller one.**
