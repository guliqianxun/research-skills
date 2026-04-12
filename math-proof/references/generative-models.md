# Generative Model Proof Reference

Mathematical foundations for diffusion models, score matching, and flow matching.
Exact formulations from Song et al. (2021) and Lipman et al. (2022).

---

## Table of Contents

1. [Score-Based Diffusion Models (SDE Framework)](#score-based-diffusion-models)
2. [Denoising Score Matching](#denoising-score-matching)
3. [Flow Matching](#flow-matching)
4. [Theoretical Guarantees](#theoretical-guarantees)

---

## Score-Based Diffusion Models (SDE Framework)

### Forward SDE (Song et al. 2021)

The forward process corrupts data $x_0 \sim p_{\text{data}}$ via an Itô SDE:

$$dx = f(x, t)\, dt + g(t)\, dW$$

where:
- $f(x, t)$: drift coefficient (vector-valued)
- $g(t)$: diffusion coefficient (scalar)
- $W$: standard Wiener process

**Three standard forward processes:**

**VP-SDE** (Variance Preserving, DDPM corresponds to this):

$$dx = -\tfrac{1}{2}\beta(t)\, x\, dt + \sqrt{\beta(t)}\, dW$$

where $\beta(t)$ is a noise schedule (e.g., linear from $\beta_{\min}$ to $\beta_{\max}$).

Marginal: $p_t(x \mid x_0) = \mathcal{N}(x;\, \alpha(t) x_0,\, \sigma(t)^2 I)$ where:

$$\alpha(t) = \exp\!\left(-\tfrac{1}{2}\int_0^t \beta(s)\, ds\right), \qquad \sigma(t)^2 = 1 - \exp\!\left(-\int_0^t \beta(s)\, ds\right)$$

**VE-SDE** (Variance Exploding, NCSN corresponds to this):

$$dx = \sqrt{\frac{d[\sigma(t)^2]}{dt}}\, dW$$

where $\sigma(t)$ increases from $\sigma_{\min}$ to $\sigma_{\max}$.

Marginal: $p_t(x \mid x_0) = \mathcal{N}(x;\, x_0,\, \sigma(t)^2 I)$

**SubVP-SDE** (better likelihood than VP):

$$dx = -\tfrac{1}{2}\beta(t)\, x\, dt + \sqrt{\beta(t)\left(1 - e^{-2\int_0^t \beta(s)\,ds}\right)}\, dW$$

### Reverse SDE

By Anderson (1982), the time-reversal of the forward SDE is:

$$dx = \left[f(x, t) - g(t)^2 \nabla_x \log p_t(x)\right] dt + g(t)\, d\bar{W}$$

where $d\bar{W}$ is a reverse-time Wiener process, and $\nabla_x \log p_t(x)$ is the **score function**.

**Key insight:** The score function is the only unknown. If we can estimate it,
we can run the reverse process and generate samples.

### Probability Flow ODE

Every SDE has an equivalent deterministic ODE with the same marginals:

$$\frac{dx}{dt} = f(x, t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x)$$

This ODE gives:
- Exact likelihood computation (via change-of-variables / instantaneous change of variables)
- Deterministic sampling (faster, controllable)
- Interpolation between data and noise

---

## Denoising Score Matching

### The Core Equivalence (Vincent 2011, extended by Song et al.)

**Problem:** The score matching objective

$$J_{\text{SM}}(\theta) = \frac{1}{2}\, \mathbb{E}_t \mathbb{E}_{x \sim p_t}\!\left[\|s_\theta(x, t) - \nabla_x \log p_t(x)\|^2\right]$$

requires $\nabla_x \log p_t(x)$, which is intractable for complex distributions.

**Denoising Score Matching (DSM) objective:**

$$J_{\text{DSM}}(\theta) = \frac{1}{2}\, \mathbb{E}_t \mathbb{E}_{x_0 \sim p_{\text{data}}} \mathbb{E}_{x_t \mid x_0}\!\left[\|s_\theta(x_t, t) - \nabla_{x_t} \log p_t(x_t \mid x_0)\|^2\right]$$

**Theorem (Vincent 2011):** $J_{\text{SM}}(\theta) = J_{\text{DSM}}(\theta) + C$, where $C$ is a constant
independent of $\theta$.

**Proof sketch:**
Expand $\|s_\theta - \nabla \log p_t\|^2$ and $\|s_\theta - \nabla \log p(x_t \mid x_0)\|^2$.
The cross term $\mathbb{E}[\langle s_\theta,\, \nabla \log p_t - \nabla \log p(x_t \mid x_0) \rangle]$ vanishes
by the identity:

$$\mathbb{E}_{x_t}[\nabla_{x_t} \log p_t(x_t)] = \mathbb{E}_{x_0, x_t}[\nabla_{x_t} \log p_t(x_t \mid x_0)]$$

This follows from: $p_t(x_t) = \int p(x_t \mid x_0)\, p_{\text{data}}(x_0)\, dx_0$.

**Consequence:** We only need to regress against $\nabla \log p(x_t \mid x_0)$, which is known
analytically. For Gaussian corruption $p(x_t \mid x_0) = \mathcal{N}(x_t;\, \mu_t,\, \sigma_t^2 I)$:

$$\nabla_{x_t} \log p(x_t \mid x_0) = -\frac{x_t - \mu_t(x_0)}{\sigma_t^2} = -\frac{\varepsilon}{\sigma_t}$$

where $\varepsilon \sim \mathcal{N}(0, I)$ is the noise added. This gives the familiar $\varepsilon$-prediction
loss used in DDPM:

$$L_{\text{DDPM}} = \mathbb{E}_{t,\, x_0,\, \varepsilon}\!\left[\|\varepsilon_\theta(\alpha_t x_0 + \sigma_t \varepsilon,\, t) - \varepsilon\|^2\right]$$

---

## Flow Matching

### Continuous Normalizing Flows (CNF)

A CNF defines a time-dependent diffeomorphism via an ODE:

$$\frac{d\phi_t}{dt} = v_t(\phi_t(x)), \qquad \phi_0(x) = x \;\text{(identity at } t=0\text{)}$$

The induced probability path: $p_t = (\phi_t)_\# p_0$ (pushforward of source).

### Flow Matching Loss (Lipman et al. 2022)

**Goal:** Learn $v_\theta$ such that $\phi_1$ pushes $p_0$ (noise) to $p_1$ (data).

**Flow Matching (FM) loss:**

$$L_{\text{FM}}(\theta) = \mathbb{E}_{t \sim U[0,1]} \mathbb{E}_{x \sim p_t}\!\left[\|v_\theta(x, t) - u_t(x)\|^2\right]$$

where $u_t(x)$ is the true marginal velocity field — intractable for general $p_t$.

**Conditional Flow Matching (CFM) loss:**

$$L_{\text{CFM}}(\theta) = \mathbb{E}_{t \sim U[0,1]} \mathbb{E}_{x_1 \sim p_{\text{data}}} \mathbb{E}_{x \sim p_t(x \mid x_1)}\!\left[\|v_\theta(x, t) - u_t(x \mid x_1)\|^2\right]$$

**Theorem (Lipman et al. 2022):** $\nabla_\theta L_{\text{FM}} = \nabla_\theta L_{\text{CFM}}$

**Proof sketch:**
Expand $L_{\text{FM}}$: the cross term between $v_\theta$ and $u_t(x)$ factors as

$$\mathbb{E}_x[\langle v_\theta(x),\, u_t(x) \rangle] = \mathbb{E}_{x_1} \mathbb{E}_{x \mid x_1}[\langle v_\theta(x),\, u_t(x \mid x_1) \rangle]$$

using the identity $u_t(x) = \mathbb{E}_{x_1 \mid x}[u_t(x \mid x_1)]$ (mixture of conditional velocities).
Therefore the two losses have identical gradients — minimize one to minimize the other.

**Consequence:** We can train on the tractable conditional loss $L_{\text{CFM}}$, which only requires
knowing $u_t(x \mid x_1)$ — a simple vector field from $x$ to $x_1$.

### Gaussian Conditional Paths

The simplest choice: $p_t(x \mid x_1) = \mathcal{N}(x;\, \mu_t(x_1),\, \sigma_t(x_1)^2 I)$

Linear (affine) interpolation:

$$\mu_t(x_1) = t \cdot x_1, \qquad \sigma_t(x_1) = 1 - (1 - \sigma_{\min}) t$$

Conditional velocity (derivative of conditional mean + noise contraction):

$$u_t(x \mid x_1) = \frac{x_1 - (1 - \sigma_{\min}) x}{1 - (1 - \sigma_{\min}) t}$$

As $\sigma_{\min} \to 0$:

$$u_t(x \mid x_1) = \frac{x_1 - x}{1 - t} \quad \text{[straight-line velocity toward } x_1\text{]}$$

### Optimal Transport (OT) Conditional Paths

Match each noise sample $x_0$ to its nearest data sample $x_1$ via OT plan $\pi^*(x_0, x_1)$.
Then the conditional path is the straight line:

$$x_t = (1-t) x_0 + t\, x_1, \qquad u_t(x \mid x_1, x_0) = x_1 - x_0 \quad \text{[constant velocity]}$$

**Advantage of OT paths:**
- Straight-line trajectories → fewer NFE (neural function evaluations) at inference
- Theoretically: OT paths minimize kinetic energy $\int_0^1 \mathbb{E}[\|v_t\|^2]\, dt$
- Practically: FLUX (Black Forest Labs) and Stable Diffusion 3 use this approach

---

## Theoretical Guarantees

### Score Estimation Error Propagation

**Theorem (informal, Song et al. 2021):** If the score network satisfies

$$\mathbb{E}\!\left[\|s_\theta(x, t) - \nabla \log p_t(x)\|^2\right] \leq \varepsilon^2$$

then the distribution $\hat{p}_T$ generated by running the reverse SDE satisfies:

$$W_2(\hat{p}_T,\, p_{\text{data}}) = O(\varepsilon \cdot T) \quad \text{(Wasserstein-2 distance)}$$

**Implication:** Error accumulates linearly in the number of diffusion steps $T$.
This motivates:
1. Fewer diffusion steps (DDIM, DPM-Solver)
2. Flow matching (single-step or few-step via straight trajectories)

### ELBO for Diffusion Models

The variational lower bound for diffusion models decomposes as:

$$-\log p_\theta(x_0) \leq \sum_{t=1}^{T} L_t$$

where each term:

$$L_t = \text{KL}\!\left(q(x_{t-1} \mid x_t, x_0) \;\|\; p_\theta(x_{t-1} \mid x_t)\right)$$

For Gaussian reverse process, $L_t$ reduces to:

$$L_t = \mathbb{E}\!\left[\frac{1}{2\sigma_t^2}\|x_0 - \hat{x}_\theta(x_t, t)\|^2\right] + \text{const}$$

This is equivalent to the $\varepsilon$-prediction loss up to a constant rescaling — the
connection between ELBO optimization and denoising score matching.
