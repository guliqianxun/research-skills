---
name: math-proof
version: 1.0.0
description: >-
  Construct, verify, and communicate mathematical proofs for ML theory — convergence rates,
  generalization bounds, architecture properties, optimization analysis, generative model theory,
  and scaling laws. Use when the user asks to prove a theorem about an ML algorithm, verify a
  convergence bound, analyze Lipschitz properties of a network, derive generalization guarantees,
  prove properties of diffusion models or flow matching, analyze NTK behavior, construct universal
  approximation arguments, or build any rigorous mathematical argument in the context of machine
  learning. Also trigger on: convergence proof, generalization bound, PAC-Bayes, Lipschitz analysis,
  gradient flow proof, lower bound, concentration inequality, optimization theory, induction on
  iterations, equivariance proof, score matching, diffusion SDE, flow matching, NTK, neural tangent
  kernel, Rademacher complexity, Wasserstein distance, optimal transport, scaling law, universal
  approximation, loss landscape, PL condition, Lojasiewicz inequality, information bottleneck.
---

# Mathematical Proof for ML

This skill turns you into a rigorous ML theorist. It covers **strategy selection**, **proof
construction**, **verification**, and the **bridge from theory to design decisions**.

Follow NeurIPS/ICML/ICLR standards throughout.

---

## Proof Strategy Selection

Choose based on the claim structure — don't default to direct proof:

| Claim Pattern | Strategy | Why |
|--------------|----------|-----|
| "Algorithm converges in $T$ steps" | **Induction** on iteration count | Base case = initialization, inductive step = per-iteration progress |
| "No algorithm can achieve error $< \varepsilon$" | **Contradiction** — assume one does, derive impossible bound | Lower bounds almost always need contradiction or information theory |
| "This estimator generalizes" | **Probabilistic** — concentration + union bound or PAC-Bayes | Generalization is inherently statistical |
| "Architecture $X$ is equivariant to group $G$" | **Direct** — verify $f(g \cdot x) = g \cdot f(x)$ for generators of $G$ | Algebraic — just check the group action commutes |
| "Gradient flow doesn't vanish" | **Constructive** — exhibit a Jacobian bound bounded away from 0 | Need concrete bounds, not existence |
| "Method A is strictly better than B" | **Separation** — construct an instance where A succeeds and B fails | Needs a concrete counterexample for B |
| "Network can approximate any continuous function" | **Density argument** — Stone-Weierstrass or constructive approximation | Show the function class is dense in $C(K)$ under sup-norm |
| "Loss landscape has no spurious local minima" | **Landscape analysis** — characterize critical points via Hessian | Show every local min is global, or every saddle has a negative eigenvalue |
| "Generative model learns the target distribution" | **Coupling / OT argument** — bound Wasserstein or KL between model and target | Transport inequalities connect optimization to distribution distance |
| "Adaptive method matches lower bound over $T$ steps" | **Amortized analysis** — potential function argument | Per-step costs vary but total is controlled by potential |
| "Score estimator converges to true score" | **Denoising score matching equivalence** — show $\mathbb{E}[\|s_\theta - \nabla \log p_\sigma\|^2]$ equals DSM objective up to constant | Avoids intractable $\nabla \log p$; reduces to regression |

---

## Proof Construction Workflow

### Step 1: Formalize

Convert intuition into a formal statement with explicit quantifiers.

- Bad: "Attention helps with long-range dependencies"
- Good: "For input $x \in \mathbb{R}^{L \times d}$, self-attention computes pairwise interactions in $O(L^2 d_k)$ time, and $\partial \text{attn}(x)_i / \partial x_j \neq 0$ for all $i, j$"

Every symbol must be defined. Every quantifier (for all, there exists) must be explicit.

### Step 2: List assumptions

Every assumption you omit is a gap reviewers will find. Common categories:

- **Smoothness**: L-smooth? mu-strongly convex? Both? Polyak-Lojasiewicz?
- **Boundedness**: Bounded gradients? Bounded domain? Bounded variance sigma^2?
- **Statistical**: i.i.d.? Sub-Gaussian noise? Bounded moments? Log-Sobolev constant?
- **Structural**: Full rank? Positive definite? Connected graph? Finite covering number?
- **Generative**: Forward process well-defined? Score function in L^2? Lipschitz drift?

### Step 3: Execute with citations

Each step cites the lemma or assumption it uses. No "it is easy to see" — either cite or prove inline.

Read the appropriate reference file for templates and tools:

| Reference | When to Read |
|-----------|-------------|
| [references/proof-templates.md](references/proof-templates.md) | Convergence proofs, generalization bounds, optimization — has skeletons, worked examples, and the core lemma catalog |
| [references/generative-models.md](references/generative-models.md) | Diffusion models, score matching, flow matching, SDE/ODE analysis — has exact formulas for VP-SDE, VE-SDE, DSM loss, CFM loss |
| [references/advanced-tools.md](references/advanced-tools.md) | Rademacher complexity, optimal transport, functional inequalities, NTK, landscape analysis — the advanced toolkit |

### Step 4: Verify tightness

This is not optional — a bound without tightness analysis is incomplete.

**Tightness verification protocol:**

1. **Construct a matching instance.** If the bound says $O(1/\sqrt{T})$, exhibit a concrete problem where the actual rate is $\Omega(1/\sqrt{T})$. For GD on strongly convex functions, a quadratic $f(w) = \frac{L}{2}w_1^2 + \frac{\mu}{2}w_2^2$ achieves the exact rate.

2. **Check known lower bounds.** Does an information-theoretic or oracle-complexity lower bound exist? If your upper bound matches it, state so. If there's a gap, quantify it.

3. **Vacuousness check.** Plug in realistic numbers (e.g., $n = 50\text{K}$ samples, $d = 100\text{M}$ parameters). If the generalization bound exceeds 1 (or the convergence time exceeds $10^{20}$), the bound is vacuous for practice — say so honestly.

4. **State the gap.** If the bound is not tight, write: "Gap: upper bound is O(X), best known lower bound is Omega(Y). Closing this gap requires [specific technique or open problem]."

### Step 5: Proof-to-Design Bridge

Every theorem should connect back to design decisions. Answer:

1. **What hyperparameter choices does this bound recommend?** (e.g., $\eta = 1/L$, temperature $= \sqrt{d}$)
2. **What architecture properties does this justify?** (e.g., Lipschitz bound justifies QK-normalization)
3. **When do the assumptions break down?** (e.g., strong convexity fails for deep nets, bounded gradients fail with gradient clipping off)
4. **Is there a practical takeaway?** (e.g., "the $1/\sqrt{T}$ rate means diminishing returns after ~10K steps for this loss scale")

| Proof Result | Design Implication |
|-------------|-------------------|
| Attention Lipschitz constant $\sim \|V\| / \sqrt{d}$ | Use QK-normalization; scale temperature with depth |
| PAC-Bayes: tighter when $\text{KL}(Q \| P)$ is small | Match architecture to data structure (convolutions for images, transformers for sequences) |
| Adam convergence degrades with dimension $d$ | Consider parameter-efficient methods (LoRA) for very large models |
| NTK becomes constant at infinite width | Wider networks train more like kernel methods — predictable but less expressive |
| Scaling law: $L \sim A/N^{0.34} + B/D^{0.28}$ | Allocate compute equally between parameters and data (Chinchilla) |
| Score matching error propagates as $O(\varepsilon \cdot T)$ in diffusion | Use fewer diffusion steps; prefer flow matching (linear error propagation) |
| PL condition gives linear convergence without convexity | Overparameterized nets may converge fast even though loss is non-convex |

---

## Output Format

```
**Theorem.** [Formal statement with quantifiers]

*Assumptions:*
1. [Assumption with mathematical notation]
2. ...

*Proof.*
Step 1: [Description]
  [Math] — by [Assumption N / Lemma / named theorem]

Step 2: ...

QED

*Tightness:* [Matching lower bound or explicit gap statement]
*Practical implication:* [Design decision this informs]
```

---

## Common Pitfalls

1. **Forgetting quantifiers** — "for all epsilon > 0" vs. "there exists epsilon > 0" invalidates the entire proof
2. **Circular reasoning** — especially in inductive proofs where the inductive hypothesis is misapplied
3. **Dropping constants** — O-notation is fine for rates, but exact constants matter for step-size recommendations
4. **Wrong inequality direction** — double-check every <= vs >= after applying an inequality
5. **Vacuous bounds** — many generalization bounds give values > 1 for realistic networks; acknowledge this honestly
6. **Ignoring the score regularity** — in diffusion/score matching proofs, $\nabla \log p_t$ may not exist or be in $L^2$ without explicit assumptions on the data distribution
7. **Confusing convergence in distribution vs. in parameters** — NTK convergence is about function space, not weight space
8. **Applying concentration to dependent random variables** — McDiarmid requires independence; for dependent data, use mixing conditions or other tools
