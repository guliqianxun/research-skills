# Proof Templates and Core Lemma Catalog

Worked examples and standard tools for ML theory proofs.
Follow NeurIPS/ICML/ICLR standards.

---

## Table of Contents

1. [Proof Templates](#proof-templates)
   - Template 1: Convergence of Gradient Descent (strongly convex)
   - Template 2: Generalization via PAC-Bayes
   - Template 3: Attention Mechanism Lipschitz Analysis
   - Template 4: Adam Convergence (Simplified)
   - Template 5: SGD on Non-Convex Loss (PL Condition)
   - Template 6: Scaling Law Derivation
2. [Core Lemma Catalog](#core-lemma-catalog)
   - Foundational Inequalities
   - Concentration Inequalities
   - Optimization Theory

---

## Proof Templates

### Template 1: Convergence of Gradient Descent

**Setting:** Minimize $f(w)$ where $f$ is $L$-smooth and $\mu$-strongly convex.

**Claim:** GD with step size $\eta = 1/L$ converges as:

$$f(w_t) - f(w^*) \leq \left(1 - \frac{\mu}{L}\right)^t \left[f(w_0) - f(w^*)\right]$$

**Assumptions:**
1. $f$ is $L$-smooth: $\|\nabla f(x) - \nabla f(y)\| \leq L\|x - y\|$ for all $x, y$
2. $f$ is $\mu$-strongly convex: $f(y) \geq f(x) + \langle \nabla f(x), y - x \rangle + \frac{\mu}{2}\|y - x\|^2$
3. $\eta = 1/L$

**Proof:**

Step 1 (Descent lemma from $L$-smoothness):

$$f(w_{t+1}) \leq f(w_t) + \langle \nabla f(w_t), w_{t+1} - w_t \rangle + \frac{L}{2}\|w_{t+1} - w_t\|^2$$

— by $L$-smoothness (quadratic upper bound)

Step 2 (Substitute GD update $w_{t+1} = w_t - \eta \nabla f(w_t)$, $\eta = 1/L$):

$$f(w_{t+1}) \leq f(w_t) - \frac{1}{2L}\|\nabla f(w_t)\|^2$$

— algebra after substitution

Step 3 (Use strong convexity gradient lower bound):

$$\|\nabla f(w_t)\|^2 \geq 2\mu \left[f(w_t) - f(w^*)\right]$$

— consequence of $\mu$-strong convexity

Step 4 (Combine Steps 2 and 3):

$$f(w_{t+1}) - f(w^*) \leq \left(1 - \frac{\mu}{L}\right) \left[f(w_t) - f(w^*)\right]$$

Step 5 (Unroll recursion):

$$f(w_t) - f(w^*) \leq \left(1 - \frac{\mu}{L}\right)^t \left[f(w_0) - f(w^*)\right]$$

— apply Step 4 inductively

QED

**Tightness:** The quadratic $f(w) = \frac{L}{2}w_1^2 + \frac{\mu}{2}w_2^2$ achieves this rate exactly.
The condition number $\kappa = L/\mu$ gives complexity $O(\kappa \log(1/\varepsilon))$.

**Practical implication:** For Adam/SGD in deep learning, $\mu$ is effectively 0 (non-convex),
so this bound doesn't apply directly. But the step-size intuition $\eta \sim 1/L$ is a useful
upper bound on the learning rate for stable training.

---

### Template 2: Generalization Bound via PAC-Bayes

**Setting:** Classifier $h$ drawn from posterior $Q$ over hypothesis class $\mathcal{H}$,
prior $P$ chosen before seeing data, $n$ i.i.d. samples.

**Claim (McAllester's bound):**
With probability $\geq 1 - \delta$ over $S \sim D^n$:

$$\mathbb{E}_{h \sim Q}[L_D(h)] \leq \mathbb{E}_{h \sim Q}[L_S(h)] + \sqrt{\frac{\text{KL}(Q \| P) + \ln(n/\delta)}{2n}}$$

**Key steps:**
1. Start from the change-of-measure inequality (Donsker-Varadhan):
   For any measurable $f$ and distributions $P, Q$:

$$\mathbb{E}_Q[f] \leq \text{KL}(Q \| P) + \log \mathbb{E}_P[\exp(f)]$$

2. Apply to the moment generating function of the excess loss:
   $f(h) = \lambda \cdot n \cdot (L_D(h) - L_S(h))$

3. Use Hoeffding's lemma to bound $\mathbb{E}_P[\exp(\lambda(L_D - L_S))]$ via the
   bounded difference (losses in $[0, 1]$)

4. Apply Markov's inequality to convert expectation bound to high-probability bound

5. Optimize $\lambda = \sqrt{2n / (\text{KL}(Q \| P) + \ln(n/\delta))}$ to minimize RHS

**Honest caveat:** For modern networks ($>$10M parameters), $\text{KL}(Q \| P)$ is typically so
large that this bound exceeds 1 — it is vacuous numerically. Non-vacuous PAC-Bayes bounds
(Dziugaite & Roy, 2017) exist but require careful posterior optimization via SGD on the
bound itself. The theoretical value is qualitative: smaller KL means better generalization.

**Why this matters for design:** The KL term is the complexity measure.
- Architectures with good inductive biases (convolutions for images, transformers for sequences)
  lead to posteriors $Q$ close to simple priors $P$
- Flat minima in the loss landscape correspond to low KL posteriors
- This justifies sharpness-aware minimization (SAM) theoretically

---

### Template 3: Attention Mechanism Lipschitz Analysis

**Setting:** Self-attention with $Q, K, V \in \mathbb{R}^{n \times d}$.

$$\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V$$

**Claim:** Softmax attention is Lipschitz in $V$ with constant 1, and in $Q$ with constant
proportional to $\|V\| / \sqrt{d}$.

**Proof sketch:**

Step 1 (Softmax Jacobian):
For $\sigma = \text{softmax}(z)$, the Jacobian is:

$$\frac{\partial \sigma_i}{\partial z_j} = \sigma_i (\delta_{ij} - \sigma_j)$$

Operator norm: $\|J_\sigma\|_{\text{op}} \leq 1/2$ (tight for two-class case)

Step 2 (Lipschitz in $V$):

$$\|\text{Attn}(Q,K,V) - \text{Attn}(Q,K,V')\| \leq \left\|\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)\right\|_{\text{op}} \|V - V'\| = \|V - V'\|$$

(softmax rows sum to 1, so row-stochastic matrix has op-norm $\leq 1$)

Step 3 (Lipschitz in $Q$):
By chain rule: the attention score matrix $S = QK^\top / \sqrt{d}$ changes as
$\partial S / \partial Q = K^\top / \sqrt{d}$, and softmax is $\frac{1}{2}$-Lipschitz in its input.
Full Lipschitz constant in $Q$:

$$L_Q = \frac{\|K\| \cdot \|V\|}{2\sqrt{d}}$$

**Practical consequence:**
- Temperature scaling (the $1/\sqrt{d}$ factor) controls the Lipschitz constant in $Q$
- Large $\|V\|$ increases sensitivity to query perturbations
- QK-normalization (used in ViT-22B, Gemma) bounds $\|K\|$ explicitly, stabilizing training
- This analysis justifies gradient clipping thresholds for attention layers

---

### Template 4: Adam Convergence (Simplified, following Reddi et al. 2019)

**Theorem:** Under bounded gradients ($\|g_t\| \leq G$), bounded second moments ($\mathbb{E}[g_t^2] \leq v$),
and $\beta_1 < \sqrt{\beta_2}$, Adam with the AMSGrad fix converges for convex objectives:

$$\frac{1}{T} \sum_{t=1}^{T} \left[f(w_t) - f(w^*)\right] = O\!\left(\frac{d \cdot G}{\sqrt{T} \cdot (1 - \beta_1)}\right)$$

**Key insight:** Original Adam can fail to converge because the effective learning rate
$\eta_t / \sqrt{\hat{v}_t}$ can oscillate. AMSGrad fixes this with:

$$\hat{v}_t = \max(\hat{v}_{t-1},\, v_t) \quad \text{[ensures non-increasing effective LR]}$$

**Proof sketch:**

Step 1 (Potential function):

$$\Phi_t = \sum_i \frac{(w_{t,i} - w^*_i)^2}{2\eta \cdot \hat{v}_{t,i}^{1/2}}$$

Step 2 (Per-step descent):

$$\Phi_{t+1} - \Phi_t \leq -f(w_t) + f(w^*) + \text{(terms from momentum and adaptive LR change)}$$

— requires $\hat{v}_t$ non-decreasing (guaranteed by AMSGrad fix)

Step 3 (Telescope):

$$\sum_{t=1}^{T} \left[f(w_t) - f(w^*)\right] \leq \Phi_1 - \Phi_{T+1} + \text{(bounded correction terms)}$$

The correction terms are $O(d \cdot G \cdot \sqrt{T} / (1 - \beta_1))$ using bounded gradients

Step 4 (Divide by $T$):
Time-averaged regret $= O(d \cdot G / (\sqrt{T} \cdot (1 - \beta_1)))$

**Practical implications:**
- The $d$ factor: Adam's convergence degrades with model dimension
- The $(1 - \beta_1)$ factor: higher momentum ($\beta_1 \to 1$) slows *proven* convergence,
  though often helps in practice — a persistent theory-practice gap
- The $1/\sqrt{T}$ rate matches SGD; Adam's advantage is in constants, not rate

---

### Template 5: SGD on Non-Convex Loss with PL Condition

**Setting:** Minimize $f(w)$ (possibly non-convex) satisfying the Polyak-Łojasiewicz (PL) condition.

**PL condition:**

$$\frac{1}{2}\|\nabla f(w)\|^2 \geq \mu \left[f(w) - f(w^*)\right] \quad \forall\, w$$

**Claim:** GD with step size $\eta = 1/L$ converges linearly:

$$f(w_t) - f(w^*) \leq \left(1 - \frac{2\mu}{L}\right)^t \left[f(w_0) - f(w^*)\right]$$

**Key point:** PL does NOT require convexity. It holds for:
- Overparameterized linear networks (exactly)
- Overparameterized ReLU networks near initialization (approximately, Jacot et al.)
- Any strongly convex function (PL is weaker than strong convexity)

**Proof:**

Step 1 (Same descent lemma as Template 1):

$$f(w_{t+1}) \leq f(w_t) - \frac{1}{2L}\|\nabla f(w_t)\|^2$$

Step 2 (Apply PL condition directly):

$$f(w_t) - f(w^*) \leq \frac{1}{2\mu}\|\nabla f(w_t)\|^2 \quad \text{[rearrange PL]}$$

Step 3 (Combine):

$$f(w_{t+1}) - f(w^*) \leq \left(1 - \frac{2\mu}{L}\right)\left[f(w_t) - f(w^*)\right]$$

QED

**Tightness:** PL is strictly weaker than strong convexity. Functions satisfying
PL can have non-convex loss landscapes but still converge linearly to a global minimum.
This is the theoretical foundation for why overparameterized networks train "well".

---

### Template 6: Scaling Law Derivation (Chinchilla)

**Empirical law (Hoffmann et al. 2022 — Chinchilla):**

$$L(N, D) = \frac{A}{N^\alpha} + \frac{B}{D^\beta} + L_0$$

where (fitted values):
$A = 406.4$, $\alpha = 0.34$, $B = 410.7$, $\beta = 0.28$, $L_0 = 1.69$ (irreducible entropy)

**Optimal allocation under compute budget $C \approx 6ND$:**

Minimize $L(N, D)$ subject to $ND = C/6$.
Setting $\partial L / \partial N = \partial L / \partial D$ (equal marginal returns):

$$\frac{\alpha A}{N^{\alpha+1}} = \frac{\beta B}{D^{\beta+1}}$$

This gives:

$$N_{\text{opt}}(C) \propto C^{\beta/(\alpha+\beta)} \sim C^{0.45}, \qquad D_{\text{opt}}(C) \propto C^{\alpha/(\alpha+\beta)} \sim C^{0.55}$$

**Key result:** Both $N$ and $D$ should scale roughly as $\sqrt{C}$ (approximately equal scaling),
in contrast to the earlier Kaplan et al. (2020) recommendation to scale $N$ much faster than $D$.

**Derivation approach:**
1. Fit the power-law form to cross-entropy loss at many $(N, D, C)$ points
2. Use constrained optimization (Lagrange multipliers) on the analytical form
3. The constraint $C = 6ND$ comes from the empirical observation that each token
   requires ~6 FLOPs per forward+backward pass

**Honest caveat:** This is an *empirical* power law, not a theorem. The fit is good
across 3–4 orders of magnitude in compute but:
- May not extrapolate beyond current data (emergent abilities may change the curve)
- The $6ND$ estimate is approximate (varies by architecture)
- $L_0 = 1.69$ is dataset-specific (Gopher training data)
- $\alpha$ and $\beta$ values vary across different fitting methodologies

---

## Core Lemma Catalog

### Foundational Inequalities

| Inequality | Statement | Use in ML Proofs |
|-----------|-----------|-----------------|
| **Young's** | $ab \leq \frac{\varepsilon a^2}{2} + \frac{b^2}{2\varepsilon}$ for $\varepsilon > 0$ | Decoupling products of norms in gradient bounds |
| **AM-GM** | $\frac{a+b}{2} \geq \sqrt{ab}$ | Bounding mixed terms in convergence proofs |
| **Jensen's** | $\varphi(\mathbb{E}[X]) \leq \mathbb{E}[\varphi(X)]$ for convex $\varphi$ | Bounding expectations of losses, justifying ELBO |
| **Cauchy-Schwarz** | $|\langle u, v \rangle| \leq \|u\| \cdot \|v\|$ | Bounding inner products, gradient correlations |
| **Pinsker's** | $\text{TV}(P, Q)^2 \leq \frac{1}{2}\text{KL}(P \| Q)$ | Connecting KL divergence to total variation |

### Concentration Inequalities

**Hoeffding:** For bounded $X_i \in [a_i, b_i]$:

$$P(\bar{X} - \mathbb{E}[\bar{X}] \geq t) \leq \exp\!\left(-\frac{2n^2 t^2}{\sum_i (b_i - a_i)^2}\right)$$

*When to use:* Bounded losses, need deviation of empirical mean from true mean.
*Weakness:* Ignores variance; Bernstein is tighter when variance is small.

**Bernstein:** For bounded $X_i$ with variance $\sigma^2$ and $|X_i| \leq M$:

$$P(\bar{X} - \mu \geq t) \leq \exp\!\left(-\frac{nt^2 / 2}{\sigma^2 + Mt/3}\right)$$

*When to use:* When you know the variance is small (e.g., well-trained model with low loss).

**McDiarmid:** If changing one $x_i$ changes $f$ by at most $c_i$ (bounded differences):

$$P(f - \mathbb{E}[f] \geq t) \leq \exp\!\left(-\frac{2t^2}{\sum_i c_i^2}\right)$$

*When to use:* Generalization bounds via algorithmic stability. Requires independence.

**Matrix Bernstein:** For random symmetric matrices $S_i$ with $\|S_i\| \leq R$:

$$P\!\left(\left\|\sum_i S_i\right\| \geq t\right) \leq d \cdot \exp\!\left(-\frac{t^2 / 2}{\sigma^2 + Rt/3}\right)$$

where $\sigma^2 = \left\|\sum_i \mathbb{E}[S_i^2]\right\|$.
*When to use:* Random attention matrices, random feature approximations,
spectral bounds on random weight matrices.

**Sub-Gaussian:** $X$ is $\sigma^2$-sub-Gaussian if $\mathbb{E}[\exp(\lambda X)] \leq \exp(\lambda^2 \sigma^2 / 2)$.
Tail bound: $P(X \geq t) \leq \exp(-t^2 / (2\sigma^2))$.
*When to use:* General-purpose when the distribution has Gaussian-like tails.

### Optimization Theory

**Descent lemma ($L$-smooth $f$):**

$$f(y) \leq f(x) + \langle \nabla f(x), y - x \rangle + \frac{L}{2}\|y - x\|^2$$

*Foundation of every GD convergence proof.*

**Co-coercivity (convex $L$-smooth $f$):**

$$\langle \nabla f(x) - \nabla f(y),\, x - y \rangle \geq \frac{1}{L}\|\nabla f(x) - \nabla f(y)\|^2$$

*Tighter bound than descent lemma when both endpoints have small gradient.*

**Polyak-Łojasiewicz (PL) condition:**

$$\frac{1}{2}\|\nabla f(x)\|^2 \geq \mu \left[f(x) - f(x^*)\right]$$

*Convergence for non-convex functions satisfying PL (e.g., overparameterized nets, linear nets).
Weaker than strong convexity — does not require uniqueness of minimizer.*

**Łojasiewicz inequality (general form):**

$$\|\nabla f(x)\| \geq c \cdot |f(x) - f(x^*)|^\theta \quad \text{for } \theta \in [1/2,\, 1)$$

*When $\theta = 1/2$, this is the PL condition. The exponent $\theta$ controls convergence speed:
$\theta$ close to 1 gives sublinear convergence; $\theta = 1/2$ gives linear convergence.*
