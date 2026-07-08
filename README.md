# Major-Minor Mean Field Game: Major Agent Detection

## Table of Contents

- [Project Goal](#project-goal)
- [Market Model](#market-model)
  - [State Dynamics](#state-dynamics)
  - [Optimal Control (MFG Equilibrium)](#optimal-control-mfg-equilibrium)
  - [Riccati ODEs](#riccati-odes-solved-backward-from-terminal-conditions)
- [Single Major Agent Detection](#single-major-agent-detection)
  - [Likelihood Scores](#likelihood-scores)
  - [Softmax Relaxation](#softmax-relaxation-primary-method)
  - [Gap-Argmax Baseline](#gap-argmax-baseline)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Dependencies](#dependencies)

---

## Project Goal

Many large-scale systems — financial markets, power grids, opinion dynamics — exhibit hierarchical interactions where a small number of influential agents significantly affect the behavior of a large population. Mean field game (MFG) models with major and minor agents provide a framework for capturing such asymmetries. However, existing formulations typically assume the identity of the major agent is known a priori, whereas in practice the major agent may only be indirectly identifiable.

This project develops **statistical tools to infer the existence and identity of a major agent from observed trajectory data**. Concretely, we use likelihood-based tests to identify the major agent in finite-population linear–quadratic–Gaussian (LQG) MFGs, where one major agent and a large but finite number of minor agents interact through the population's empirical mean field. Theoretical results are complemented by numerical studies on synthetic data.

---

## Market Model

The model follows Chang, Firoozi & Benatia (2025) and describes an interbank market with one major bank $\mathcal{A}^0$ and $N$ minor banks $\mathcal{A}^1, \ldots, \mathcal{A}^N$. All state variables are log-monetary reserves.

### State Dynamics

**Major bank** — log-reserve $x^0_t$:

$$dx^0_t = a_0\left(\bar{x}_t - x^0_t\right)dt + u^0_t dt + \sigma_0 dW^0_t$$

The major bank mean-reverts to the average minor-bank reserve $\bar{x}_t$ at rate $a_0 = aG$, where $G$ is its relative market size and $F = 1 - G$ is the collective size of the minor banks.

**Representative minor bank** — log-reserve $x^i_t$:

$$dx^i_t = a\left(F\bar{x}_t + Gx^0_t - x^i_t\right)dt + u^i_t dt + \sigma dW^i_t$$

Each minor bank mean-reverts to the **market state** $m_t = F\bar{x}_t + G x^0_t$, a weighted average of the mean field and the major bank's reserve.

**Mean-field equation** (limiting dynamics of $\bar{x}_t$):

$$d\bar{x}_t = \left(a + q - \phi_t\right)\left[(F-1)\bar{x}_t + Gx^0_t\right]dt$$

where $\phi_t$ is the solution of the minor bank's Riccati ODE (see below).

### Optimal Control (MFG Equilibrium)

The equilibrium strategies are derived via convex/variational analysis. Each bank optimally borrows or lends with the central bank at the following rates:

**Major bank:**

$$u^{0,*}_t = \left(q_0 - \phi^0_t\right)\left(\bar{x}_t - x^0_t\right)$$

**Minor bank:**

$$u^{i,*}_t = \left(q - \phi_t\right)\left(F\bar{x}_t + Gx^0_t - x^i_t\right)$$

### Riccati ODEs (solved backward from terminal conditions)

**Minor bank** — $\phi_T = -c$:

$$\dot{\phi}_t = 2(a+q)\phi_t - \phi_t^2 + \varepsilon - q^2$$

**Major bank** — $\phi^0_T = -c_0$:

$$\dot{\phi}^0_t = 2\left[(a_0 + q_0) + G(a + q - \phi_t)\right]\phi^0_t - (\phi^0_t)^2 + \varepsilon_0 - q_0^2$$

These ODEs are solved numerically via backward Euler in [`single_major_template/mfg.py`](single_major_template/mfg.py). The effective mean-reversion rates after optimal control are $a + q - \phi_t$ (minor) and $a_0 + q_0 - \phi^0_t$ (major).

---

## Single Major Agent Detection

Given observed trajectories $X \in \mathbb{R}^{(N+1)\times(T+1)}$ of all agents (with unknown identity permutation), we want to recover which agent is the major one.

The method is implemented in [`single_major_template/detect_major.py`](single_major_template/detect_major.py).

### Likelihood Scores

For each agent $i$ and each time step, the one-step transition log-likelihoods are:

- **Major score** $\ell^{\text{maj}}_i(t)$: agent $i$ follows the major SDE — Gaussian transition with mean $x_i(t) + (a_0 + q_0 - \phi^0_t)(\bar{x}_t - x_i(t)) \Delta t$ and noise $\sigma_0\sqrt{\Delta t}$.
- **Minor score** $\ell^{\text{min}}_i(t)$: agent $i$ follows the minor SDE — Gaussian transition with mean $x_i(t) + (a + q - \phi_t)(m_t - x_i(t)) \Delta t$ and noise $\sigma\sqrt{\Delta t}$, where $m_t = F\bar{x}_t + G x^0_t$ is the market state.

### Softmax Relaxation (primary method)

Because the major agent's trajectory $x^0_t$ enters the market state $m_t$, the minor scores are coupled to the unknown identity. We introduce a soft assignment $w = \text{softmax}(\theta) \in \Delta^N$ (one weight per agent) and construct a **soft market state**:

$$\hat{x}^0_t(w) = \sum_i w_i  x_i(t), \qquad \hat{m}_t(w) = F\bar{x}_t + G\hat{x}^0_t(w)$$

The objective is then maximized by gradient ascent on $\theta$:

$$J(w) = \sum_i w_i \cdot L^{\text{maj}}_i + (1 - w_i) \cdot L^{\text{min}}_i(w)$$

where $L^{\text{maj}}_i = \sum_t \ell^{\text{maj}}_i(t)$ and $L^{\text{min}}_i(w) = \sum_t \ell^{\text{min}}_i(t; w)$.

Optimization is performed with Adam. At convergence, the agent with the highest $w_i$ is predicted as the major agent. A **leave-one-out** correction removes agent $i$'s own contribution from $\hat{x}^0_t$ when computing its minor score.

Two observation regimes are supported:

| Regime | Mean field $\bar{x}_t$ | Notes |
|--------|------------------------|-------|
| Tier 2 | **Observed** | $L^{\text{maj}}$ is $w$-free (precomputed); only $L^{\text{min}}$ is $w$-coupled |
| Tier 3 | **Unobserved** (estimated from $X$) | Both scores are $w$-coupled; heavier computation |

### Gap-Argmax Baseline

A closed-form baseline scores each agent $i$ by the log-likelihood ratio of being major vs. minor, using $\bar{x}_t$ for the major reference and $(F+G)\bar{x}_t$ as a $w$-free market proxy:

$$\text{gap}_i = L^{\text{maj}}_i - L^{\text{min}}_i, \qquad \hat{i} = \arg\max_i  \text{gap}_i$$

---

## Repository Structure

```
major_minor_MFG/
└── single_major_template/
    ├── mfg.py            # MFG_config, MFG class: Riccati ODEs, simulation, log-likelihood
    └── detect_major.py   # detect_major_relaxed, detect_major_gap, make_example
```

---

## Quick Start

```python
from single_major_template.mfg import MFG, MFG_config
from single_major_template.detect_major import make_example, detect_major_relaxed

cfg = MFG_config(
    T=1.0, Ndt=200,
    a_0=0.3, sigma_0=0.2, c_0=0.4, epslon_0=0.5, q_0=0.6,
    a=0.5,   sigma=0.3,   c=0.5,   epslon=0.5,   q=0.7,
    G=0.4,
)
mfg = MFG(cfg)
mfg.solve_ODE()

X, x_bar_obs, true_idx = make_example(mfg, N=20, seed=0)
prob, history = detect_major_relaxed(mfg, X, x_bar_obs, n_steps=1000, verbose=True)
pred = int(prob.argmax())
print(f"predicted={pred}  true={true_idx}  correct={pred == true_idx}")
```

Run the built-in demo (includes 50-seed accuracy sweep):

```bash
python -m single_major_template.detect_major
```

---

## Dependencies

- Python 3.8+
- `numpy`
- `torch`
- `matplotlib`
