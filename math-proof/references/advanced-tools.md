# Advanced Mathematical Tools

Tools for deeper ML theory: complexity measures, optimal transport, functional inequalities,
kernel methods, and information theory. These go beyond the basics in proof-templates.md.

---

## Table of Contents

1. [Rademacher Complexity](#rademacher-complexity)
2. [Optimal Transport and Wasserstein Distances](#optimal-transport-and-wasserstein-distances)
3. [Functional Inequalities](#functional-inequalities)
4. [Neural Tangent Kernel (NTK)](#neural-tangent-kernel)
5. [Information-Theoretic Tools](#information-theoretic-tools)
6. [Landscape Analysis](#landscape-analysis)

---

## Rademacher Complexity

### Definition

For a function class $\mathcal{F}$ and sample $S = \{z_1, \dots, z_m\}$:

$$\text{Rad}_S(\mathcal{F}) = \frac{1}{m}\, \mathbb{E}_\sigma\!\left[\sup_{f \in \mathcal{F}} \left|\sum_{i=1}^m \sigma_i f(z_i)\right|\right]$$

where $\sigma_i$ are i.i.d. Rademacher random variables ($P(\sigma_i = +1) = P(\sigma_i = -1) = 1/2$).

**Expected Rademacher complexity:** $\text{Rad}_m(\mathcal{F}) = \mathbb{E}_S[\text{Rad}_S(\mathcal{F})]$

### Generalization Bound

**Theorem:** For any $\delta > 0$, with probability $\geq 1 - \delta$ over $S \sim D^m$:

$$\sup_{f \in \mathcal{F}} |L_D(f) - L_S(f)| \leq 2\,\text{Rad}_m(\mathcal{F}) + \sqrt{\frac{2\ln(2/\delta)}{m}}$$

If the loss is $L$-Lipschitz in predictions, then $\text{Rad}_m(\text{loss class}) \leq L \cdot \text{Rad}_m(\mathcal{F})$
by the contraction principle (Ledoux-Talagrand).

### Dudley's Entropy Integral

Bounds Rademacher complexity via the covering number $\mathcal{N}(\mathcal{F}, \varepsilon, \|\cdot\|_2)$:

$$\text{Rad}_S(\mathcal{F}) \leq \frac{12}{\sqrt{m}} \int_0^\infty \sqrt{\log \mathcal{N}(\mathcal{F}, \varepsilon, \|\cdot\|_2)}\, d\varepsilon$$

where $\mathcal{N}(\mathcal{F}, \varepsilon, \|\cdot\|)$ is the minimum number of $\varepsilon$-balls needed to cover $\mathcal{F}$.

**When to use:** When you can estimate the covering number of the hypothesis class.
For parametric classes with bounded weights, $\mathcal{N}$ grows polynomially in $1/\varepsilon$,
giving $\text{Rad} = O(1/\sqrt{m})$.

### Key Examples

**Linear functions** $\{x \mapsto \langle w, x \rangle : \|w\| \leq B,\, \|x\| \leq R\}$:

$$\text{Rad}_m(\mathcal{F}) = \frac{BR}{\sqrt{m}}$$

**Neural networks** (1-hidden-layer, $B$-bounded weights, $n$ units):

$$\text{Rad}_m(\mathcal{F}) \leq \frac{B^2 R \sqrt{2\log(2d)}}{\sqrt{m}}$$

— grows with weight norm, not number of parameters (Bartlett 1998)

**Deep networks** (depth $L$, spectral norm bound $s_i$ per layer):

$$\text{Rad}_m(\mathcal{F}) \leq \frac{R \cdot \prod_{i=1}^L s_i \cdot \sqrt{2L\log(2d)}}{\sqrt{m}}$$

— product of spectral norms controls complexity (Neyshabur et al. 2015)

**Practical note:** For modern transformers, the spectral norm product can be enormous,
making these bounds vacuous. Tighter bounds use data-dependent measures
(e.g., compression, noise stability) rather than worst-case covering numbers.

---

## Optimal Transport and Wasserstein Distances

### Wasserstein-$p$ Distance

For probability measures $\mu, \nu$ on a metric space $(X, d)$:

$$W_p(\mu, \nu) = \left(\inf_{\gamma \in \Gamma(\mu,\nu)} \mathbb{E}_{(x,y) \sim \gamma}\!\left[d(x, y)^p\right]\right)^{1/p}$$

where $\Gamma(\mu, \nu)$ is the set of all couplings (joint distributions with marginals $\mu$ and $\nu$).

### Kantorovich-Rubinstein Duality ($W_1$)

$$W_1(\mu, \nu) = \sup_{\|f\|_{\text{Lip}} \leq 1} \left(\mathbb{E}_\mu[f] - \mathbb{E}_\nu[f]\right)$$

where $\|f\|_{\text{Lip}} = \sup_{x \neq y} |f(x) - f(y)| / d(x, y)$.

**Why this matters for ML:**
- WGAN (Arjovsky et al. 2017) uses this dual form as the discriminator objective
- The Lipschitz constraint on $f$ is enforced via gradient penalty or spectral normalization
- $W_1$ metrizes weak convergence on compact spaces — it detects mode collapse that KL/JS miss

### Wasserstein-2 and Brenier's Theorem

**Brenier's theorem:** If $\mu$ is absolutely continuous, the optimal transport map $T^*$ for $W_2$
is the gradient of a convex function:

$$T^* = \nabla \varphi \quad \text{for some convex } \varphi: \mathbb{R}^d \to \mathbb{R}$$

**Monge-Ampère equation:** The optimal $\varphi$ satisfies:

$$\det(\nabla^2 \varphi(x)) = \frac{p_\mu(x)}{p_\nu(\nabla \varphi(x))}$$

**Connection to diffusion models:** The probability flow ODE in score-based models
traces an approximate OT path from noise to data. Flow matching with OT conditional
paths ($x_t = (1-t)x_0 + t x_1$ with OT coupling) explicitly uses Brenier-like maps.

### Sinkhorn Distance (Computational)

Exact OT is $O(n^3 \log n)$ for discrete measures with $n$ support points.
Entropic regularization gives the Sinkhorn distance:

$$W_\varepsilon(\mu, \nu) = \inf_{\gamma \in \Gamma(\mu,\nu)} \left[\mathbb{E}_\gamma[c(x,y)] + \varepsilon \cdot \text{KL}(\gamma \| \mu \otimes \nu)\right]$$

Solvable via alternating matrix scaling (Sinkhorn iterations) in $O(n^2 / \varepsilon^2)$.
As $\varepsilon \to 0$, $W_\varepsilon \to W$ (exact OT).

**Use in practice:** Sinkhorn divergence is differentiable and GPU-friendly,
used in generative modeling (OT-Flow), domain adaptation, and dataset comparison.

---

## Functional Inequalities

These inequalities control how distributions concentrate and how fast sampling
algorithms mix. They form a hierarchy: $\text{LSI} \Rightarrow \text{Talagrand T2} \Rightarrow \text{Poincaré}$.

### Log-Sobolev Inequality (LSI)

A probability measure $\mu$ satisfies LSI with constant $\lambda > 0$ if for all smooth $f$:

$$\text{Ent}_\mu(f^2) \leq \frac{2}{\lambda}\, \mathbb{E}_\mu\!\left[\|\nabla f\|^2\right]$$

where $\text{Ent}_\mu(g) = \mathbb{E}_\mu[g \log g] - \mathbb{E}_\mu[g] \log \mathbb{E}_\mu[g]$ is the entropy functional.

**Equivalent form** (for probability densities $p$ with respect to $\mu$):

$$\text{KL}(p \| \mu) \leq \frac{1}{2\lambda}\, \mathbb{E}_p\!\left[\|\nabla \log(p/\mu)\|^2\right]$$

The right side is the Fisher information $I(p \| \mu) = \mathbb{E}_p[\|\nabla \log(p/\mu)\|^2]$.
So LSI says: $\text{KL}(p \| \mu) \leq I(p \| \mu) / (2\lambda)$.

**Key examples:**
- Standard Gaussian $\mathcal{N}(0, I_d)$: $\lambda = 1$
- $\mathcal{N}(0, \Sigma)$: $\lambda = 1/\|\Sigma\|_{\text{op}}$ (inverse of largest eigenvalue)
- Strongly log-concave $\mu$ (i.e., $-\log \mu$ is $\alpha$-strongly convex): $\lambda = \alpha$

### Talagrand T2 Inequality

If $\mu$ satisfies $\text{LSI}(\lambda)$, then for all probability measures $\nu$:

$$W_2(\mu, \nu)^2 \leq \frac{2}{\lambda}\, \text{KL}(\nu \| \mu)$$

**Why this matters:**
- Connects optimal transport ($W_2$) to information divergence (KL)
- Bounds how far a distribution can be from the reference in transport distance
  given a KL budget — fundamental for analyzing score-based samplers
- For Langevin dynamics sampling from $\mu$, T2 controls convergence rate:
  $W_2(p_t, \mu)$ decays exponentially with rate $\lambda$

### Poincaré Inequality

Weaker than LSI. $\mu$ satisfies $\text{Poincaré}(\lambda)$ if for all smooth $f$:

$$\text{Var}_\mu(f) \leq \frac{1}{\lambda}\, \mathbb{E}_\mu\!\left[\|\nabla f\|^2\right]$$

**Hierarchy:** $\text{LSI}(\lambda) \Rightarrow \text{T2}(\lambda) \Rightarrow \text{Poincaré}(\lambda)$.
The converses are false in general.

### Applications to Diffusion/Sampling

**Langevin dynamics convergence:** If the target $\mu$ satisfies $\text{LSI}(\lambda)$:

$$\text{KL}(p_t \| \mu) \leq e^{-2\lambda t}\, \text{KL}(p_0 \| \mu)$$

— exponential convergence in KL, with rate determined by the LSI constant.

**Score matching connection:** The Fisher information $I(p \| \mu) = \mathbb{E}_p[\|\nabla \log(p/\mu)\|^2]$
is exactly the quantity minimized by score matching. LSI says that driving Fisher information
to zero also drives KL to zero, with a quantitative rate controlled by $\lambda$.

**Discrete-time Langevin (ULA):** With step size $\eta$ and $\text{LSI}(\lambda)$ target:

$$\text{KL}(p_T \| \mu) \leq e^{-2\lambda\eta T}\, \text{KL}(p_0 \| \mu) + O\!\left(\frac{dL^2\eta}{\lambda}\right)$$

The bias term $O(dL^2\eta/\lambda)$ comes from discretization and never vanishes.

---

## Neural Tangent Kernel (NTK)

### Definition

For a neural network $f(x; \theta)$ with parameters $\theta \in \mathbb{R}^p$, the NTK is:

$$K(x, x'; \theta) = \langle \nabla_\theta f(x; \theta),\, \nabla_\theta f(x'; \theta) \rangle = \sum_{i=1}^p \frac{\partial f}{\partial \theta_i}(x) \cdot \frac{\partial f}{\partial \theta_i}(x')$$

This is a $p$-dimensional inner product of Jacobians — it measures how similarly
the network's output changes at $x$ and $x'$ when parameters are perturbed.

### Infinite-Width Limit (Jacot et al. 2018)

**Theorem:** For fully-connected networks with standard initialization (NTK parameterization),
as width $\to \infty$:

1. $K(x, x'; \theta_0)$ converges in probability to a deterministic kernel $K_\infty(x, x')$
2. $K(x, x'; \theta_t)$ remains approximately equal to $K_\infty$ throughout training
3. Training dynamics become linear: $f(x; \theta_t)$ evolves as kernel regression with $K_\infty$

**Linearized dynamics:** Under gradient flow (continuous-time GD) on MSE loss:

$$\frac{df(x; \theta_t)}{dt} = -\eta\, K_\infty(x, X_{\text{train}}) \left(f(X_{\text{train}}; \theta_t) - Y_{\text{train}}\right)$$

**Solution:**

$$f(x; \theta_t) = f(x; \theta_0) + K_\infty(x, X_{\text{train}})\, K_\infty(X_{\text{train}}, X_{\text{train}})^{-1} \left(I - e^{-\eta K_\infty(X_{\text{train}}, X_{\text{train}}) t}\right) (Y_{\text{train}} - f(X_{\text{train}}; \theta_0))$$

As $t \to \infty$: $f$ converges to the kernel regression solution with $K_\infty$.

### Convergence Guarantee

**Theorem (Du et al. 2019, simplified):** For a two-layer ReLU network with $m$ hidden units,
if $m \geq \Omega(n^4 / \lambda_{\min}^2)$ where $\lambda_{\min}$ is the smallest eigenvalue of $K_\infty$
on training data, then GD with $\eta = O(\lambda_{\min} / n^2)$ converges:

$$\|f(X_{\text{train}}; \theta_t) - Y_{\text{train}}\|^2 \leq \left(1 - \frac{\eta \lambda_{\min}}{2}\right)^t \|Y_{\text{train}} - f(X_{\text{train}}; \theta_0)\|^2$$

**Key requirement:** $\lambda_{\min} > 0$ (the NTK Gram matrix on training data is positive definite).
This holds when data points are distinct and the activation function is non-polynomial.

### Limitations and the Feature Learning Debate

**What NTK captures:** Global convergence to zero training loss for sufficiently wide networks.
The proof works because wide networks stay close to initialization (lazy training regime).

**What NTK misses:**
- Feature learning: NTK regime = fixed features (random at init). Real networks learn features.
- Depth benefits: NTK theory treats depth as irrelevant (only the kernel matters).
  In practice, deeper networks learn hierarchical representations that NTK cannot explain.
- Generalization: NTK predicts kernel regression generalization, but real networks
  often generalize better than their NTK approximation (Wei et al. 2019).

**Mean-field regime** (alternative scaling): parameters move $O(1)$ from initialization,
features evolve, and the network escapes the kernel regime. This is where the
"neural networks are more than kernels" story lives, but rigorous theory is harder.

**Practical takeaway:** NTK is a useful proof tool for convergence guarantees, but design
decisions should not assume networks operate in the NTK regime. Use NTK theory to
understand *when* training succeeds, not *why* networks generalize.

---

## Information-Theoretic Tools

### Mutual Information

$$I(X; Y) = \text{KL}(P_{X,Y} \| P_X \otimes P_Y) = \mathbb{E}\!\left[\log \frac{p(x, y)}{p(x)\, p(y)}\right]$$

**Chain rule:** $I(X; Y, Z) = I(X; Y) + I(X; Z \mid Y)$

**Data Processing Inequality (DPI):** For any Markov chain $X \to Y \to Z$:

$$I(X; Z) \leq I(X; Y)$$

**Application to deep networks:** For a network with layers $X \to h_1 \to h_2 \to \cdots \to Y$,
DPI implies $I(X; h_k)$ is non-increasing in $k$. The information bottleneck hypothesis
(Tishby et al. 2015) posits that good representations compress $X$ (low $I(X; h_k)$)
while retaining information about $Y$ (high $I(Y; h_k)$).

**Caveat:** For deterministic networks with continuous inputs, $I(X; h_k) = \infty$
(or is undefined) unless noise is added. The information bottleneck theory for
deterministic networks requires careful handling via geometric measures or binning.

### Fisher Information

**Fisher information matrix** for a parametric model $p(x; \theta)$:

$$F(\theta)_{ij} = \mathbb{E}_{x \sim p(x;\theta)}\!\left[\frac{\partial \log p}{\partial \theta_i} \frac{\partial \log p}{\partial \theta_j}\right] = -\mathbb{E}_{x \sim p(x;\theta)}\!\left[\frac{\partial^2 \log p}{\partial \theta_i \partial \theta_j}\right]$$

**Connection to score matching:** The Fisher information of the data distribution
at noise level $t$ is exactly $\mathbb{E}[\|\nabla_x \log p_t(x)\|^2]$ — the quantity estimated
by the score network.

**Cramér-Rao bound:** For any unbiased estimator $\hat{\theta}$ of $\theta$:

$$\text{Cov}(\hat{\theta}) \succeq F(\theta)^{-1} \quad \text{(in the PSD sense)}$$

**Natural gradient:** The natural gradient update is:

$$\theta_{t+1} = \theta_t - \eta\, F(\theta_t)^{-1} \nabla L(\theta_t)$$

This is invariant to reparameterization of the model. Adam approximates
the natural gradient when the loss is close to the negative log-likelihood.

### Fano's Inequality (for Lower Bounds)

For any estimator $\hat{\theta}$ of a discrete parameter $\theta \in \{1, \dots, M\}$:

$$P(\hat{\theta} \neq \theta) \geq 1 - \frac{I(\theta; X) + \log 2}{\log M}$$

**Use in ML:** Constructing minimax lower bounds. To show no algorithm can achieve
error $< \varepsilon$, construct $M$ hypotheses that are mutually hard to distinguish
(low mutual information) but require different outputs (separated by $\varepsilon$).

### Matrix Chernoff Inequality

For independent random PSD matrices $X_i$ with $\|X_i\| \leq R$ and $\mathbb{E}[\sum X_i] = \mu I$:

$$P\!\left(\lambda_{\min}\!\left(\sum X_i\right) \leq (1-\delta)\mu\right) \leq d \cdot \exp\!\left(-\frac{\mu \delta^2}{2R}\right)$$

**Use in ML:** Random feature approximations (show that random features approximate
the kernel matrix), attention matrix concentration (random queries concentrate
around expected attention pattern), sketching guarantees.

---

## Landscape Analysis

### Critical Point Classification

For a loss function $L(\theta)$, a critical point satisfies $\nabla L = 0$. Classify via the Hessian:

- **Local minimum:** $\nabla^2 L \succeq 0$ (all eigenvalues $\geq 0$)
- **Local maximum:** $\nabla^2 L \preceq 0$
- **Saddle point:** $\nabla^2 L$ has both positive and negative eigenvalues
- **Strict saddle:** $\nabla^2 L$ has at least one strictly negative eigenvalue

### No Spurious Local Minima Results

**Linear networks** (Bhojanapalli et al. 2016, Kawaguchi 2016):
For $L(W_1, \dots, W_L) = \|W_L \cdots W_1 X - Y\|^2$:
- Every local minimum is a global minimum
- Every saddle point is a strict saddle

**Matrix factorization** (Ge et al. 2017):
For $L(U, V) = \|UV^\top - M\|^2$ with $M \succeq 0$:
- Every local minimum satisfies $UV^\top = M$ (global optimum)
- All saddles are strict

**Overparameterized networks** (partial results):
- Wide two-layer ReLU networks: no spurious local minima in a neighborhood
  of the NTK solution (Du et al. 2018)
- General deep networks: landscape results are much weaker; most guarantees
  require width $\gg$ depth or special initialization

### Strict Saddle Property and Escape

**Definition:** A function is strict-saddle if at every critical point, either:
1. It is a local minimum, or
2. The Hessian has a strictly negative eigenvalue ($\lambda_{\min}(\nabla^2 L) < -\gamma$)

**Theorem (Ge et al. 2015, Lee et al. 2016):** For strict-saddle functions:
- GD with random initialization avoids saddle points almost surely
- Perturbed GD escapes saddles in $O(d / \gamma)$ steps
- SGD (with sufficient noise) escapes even faster: $O(\log(d) / \gamma)$ steps

**Practical implication:** The "saddle point problem" in deep learning is largely
resolved for strict-saddle losses — gradient methods escape saddles efficiently.
The real challenge is navigating among many global minima with different
generalization properties (flat vs. sharp minima debate).
