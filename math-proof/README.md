# math-proof

Construct, verify, and communicate mathematical proofs for ML theory — convergence rates, generalization bounds, architecture properties, optimization analysis, generative model theory, and scaling laws.

## What It Does

When you ask the AI to prove a theorem about an ML algorithm, this skill provides:

- **Proof strategy selection** — 11 strategies matched to claim patterns (induction for convergence, contradiction for lower bounds, coupling for generative models, etc.)
- **Step-by-step proof construction** — formalize, list assumptions, execute with citations, verify tightness, bridge to design decisions
- **Reference formulas** — exact formulations from foundational papers (Song et al. 2021, Lipman et al. 2022, Hoffmann et al. 2022) so the AI doesn't have to recall them from memory

## Reference Files

| File | Contents |
|------|----------|
| `references/proof-templates.md` | 6 worked proof templates (GD convergence, PAC-Bayes, attention Lipschitz, Adam/AMSGrad, PL condition, Chinchilla scaling law) + core lemma catalog (Young's, Jensen's, Hoeffding, Bernstein, McDiarmid, etc.) |
| `references/generative-models.md` | Diffusion SDE framework (VP/VE/SubVP), denoising score matching equivalence, flow matching CFM theorem, Gaussian and OT conditional paths, ELBO decomposition |
| `references/advanced-tools.md` | Rademacher complexity + Dudley's integral, Wasserstein distances + Kantorovich-Rubinstein duality, Log-Sobolev / Talagrand T2 / Poincaré inequalities, Neural Tangent Kernel, Fisher information, Fano's inequality, landscape analysis |

## Example Prompts

- "Prove that GD on a strongly convex L-smooth function converges at rate $(1 - \mu/L)^t$"
- "Derive the flow matching CFM loss equivalence from Lipman et al. 2022"
- "What's the Lipschitz constant of softmax attention in Q?"
- "Show that the PL condition gives linear convergence without convexity"

## Install

```bash
# Claude Code
claude mcp add-skill /path/to/research-skills/math-proof
```
