"""
Detect which of N+1 agents is the major agent in a finite-population LQG-MFG,
using the SOFTMAX RELAXATION of the enumeration likelihood test.

Regime: TIER 2 -- the empirical mean field  xbar_t  is OBSERVED, but the market
state  m_t = F*xbar_t + G*x0_t  is NOT directly observed because it still depends
on the (unknown) major trajectory x0_t. So a genuine w-coupling survives through
the soft major  x0_hat_t(w) = sum_i w_i x_i(t), and the relaxation does real work.

The relaxation:
    w = softmax(theta) in the simplex (one entry per agent = P(agent i is major))
    maximize  J(w) = sum_i sum_t [ w_i * ell_major_i(t) + (1-w_i) * ell_minor_i(t) ]
    by unconstrained gradient ascent on theta (autodiff through softmax).

ell_major_i(t): log N( x_i(t+1) ; revert-to-XBAR with major rate, sigma0^2 dt )   -- w-free
ell_minor_i(t): log N( x_i(t+1) ; revert-to-MARKET(w) with minor rate, sigma^2 dt) -- w-coupled
"""

import numpy as np
import torch
import torch.distributions as dist
import matplotlib.pyplot as plt
from mfg import MFG, MFG_config


# ----------------------------------------------------------------------------
# Core: Tier-2 relaxation (observed mean field)
# ----------------------------------------------------------------------------
def detect_major_relaxed(mfg:MFG, X, x_bar_obs=None,
                         n_steps=1000, lr=0.05,
                         leave_one_out=True, temp_anneal=False,
                         init_theta=None, verbose=False,
                         true_major_idx=None):
    """
    mfg         : MFG instance with solve_ODE() already called.
    X           : (N+1, Ndt+1) observed agent trajectories. Rows = agents.
                  ORDER MUST BE LABEL-AGNOSTIC (permute before calling).
    x_bar_obs   : (Ndt+1,) observed empirical mean field, or None.
                  If None, the mean field is estimated inside the loop as
                  x_bar_hat(w) = (1/N) * sum_i (1 - w_i) * x_i(t),
                  making Lmaj w-coupled (no precomputation possible).
    true_major_idx : DEBUG ONLY -- index of the actual major agent in X (known
                  because this is synthetic data from make_example). Not used
                  by the estimator itself; only used, when verbose=True, to
                  save a 'debug_step_NNNN.png' plot every 10 steps overlaying
                  the true major trajectory and true (or data-only-estimated)
                  mean field against the model's current soft estimates.
    returns     : (major_prob (N+1,) tensor, history list of J values)
    """
    dt = mfg.dt
    X  = torch.as_tensor(np.asarray(X), dtype=torch.float64)
    Np1, _ = X.shape
    N = Np1 - 1

    # known, deterministic Riccati coefficients (length Ndt: one per transition)
    phi   = torch.as_tensor(mfg.phi[:-1],   dtype=torch.float64)   # (Ndt,)
    phi_0 = torch.as_tensor(mfg.phi_0[:-1], dtype=torch.float64)   # (Ndt,)

    sig_M = mfg.sigma_0 * dt**0.5     # major noise std per step
    sig_m = mfg.sigma   * dt**0.5     # minor noise std per step
    F, G  = mfg.F, mfg.G
    a, q, a0, q0 = mfg.a, mfg.q, mfg.a_0, mfg.q_0

    xt   = X[:, :-1]    # (N+1, Ndt) state at t
    xtp1 = X[:, 1:]     # (N+1, Ndt) state at t+1

    observed_xbar = x_bar_obs is not None

    if observed_xbar:
        # observed mean field, aligned to the transition grid (use t, not t+1)
        xbar_fixed = torch.as_tensor(np.asarray(x_bar_obs), dtype=torch.float64)[:-1]  # (Ndt,)

        # --- MAJOR score is w-FREE when xbar is observed: precompute once ---
        # agent i (as major) reverts to the OBSERVED mean field xbar_t.
        mu_major = xt + (a0 + (q0 - phi_0)) * (xbar_fixed.unsqueeze(0) - xt) * dt  # (N+1,Ndt)
        ll_major = dist.Normal(mu_major, sig_M).log_prob(xtp1)                      # (N+1,Ndt)
        Lmaj_fixed = ll_major.sum(dim=1)                                             # (N+1,)

    # --- optimize theta -----------------------------------------------------
    theta = (torch.zeros(Np1, dtype=torch.float64) if init_theta is None
             else torch.as_tensor(init_theta, dtype=torch.float64).clone())
    theta.requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=lr)
    history = []

    for step in range(n_steps):
        tau = 1.0 if not temp_anneal else max(0.3, 1.0 - step / n_steps)
        w = torch.softmax(theta / tau, dim=0)          # (N+1,)
        wt = w.unsqueeze(1)                            # (N+1,1)

        # soft major trajectory  x0_hat_t(w) = sum_i w_i x_i(t)   -> (Ndt,)
        x0_hat = (wt * xt).sum(dim=0)

        if observed_xbar:
            xbar = xbar_fixed
            Lmaj = Lmaj_fixed
        else:
            # soft mean field: weighted average over the N "minor" contributions
            # x_bar_hat(w) = (1/N) * sum_i (1 - w_i) * x_i(t)    -> (Ndt,)
            # sum_i (1-w_i) = N since sum_i w_i = 1, so dividing by N normalises.
            xbar = ((1 - w).unsqueeze(1) * xt).sum(dim=0) / N     # (Ndt,)

            # Lmaj is now w-coupled: recompute each step
            mu_major = xt + (a0 + (q0 - phi_0)) * (xbar.unsqueeze(0) - xt) * dt  # (N+1,Ndt)
            ll_major = dist.Normal(mu_major, sig_M).log_prob(xtp1)                # (N+1,Ndt)
            Lmaj = ll_major.sum(dim=1)                                             # (N+1,)

        # market state m_t(w) = F*xbar_t + G*x0_hat_t(w)          -> (Ndt,)
        market = F * xbar + G * x0_hat

        # MINOR score (w-coupled): agent i reverts to market(w) with minor rate.
        # Optional leave-one-out: when agent i is treated as minor, it should not
        # contribute to the soft major used in its OWN market reference.
        if leave_one_out:
            # remove i's own weighted contribution from x0_hat, per row
            x0_hat_loo = (x0_hat.unsqueeze(0) - wt * xt)        # (N+1,Ndt)
            market_i = F * xbar.unsqueeze(0) + G * x0_hat_loo    # (N+1,Ndt)
        else:
            market_i = market.unsqueeze(0)                       # (1,Ndt) broadcast

        mu_minor = xt + (a + (q - phi)) * (market_i - xt) * dt   # (N+1,Ndt)
        ll_minor = dist.Normal(mu_minor, sig_m).log_prob(xtp1)  # (N+1,Ndt)
        Lmin = ll_minor.sum(dim=1)                              # (N+1,)

        J = (w * Lmaj + (1 - w) * Lmin).sum()
        loss = -J
        opt.zero_grad(); loss.backward(); opt.step()
        history.append(J.item())

        if verbose and (step % 10 == 0 or step == n_steps - 1):
            am = torch.softmax(theta, dim=0).argmax().item()
            print(f"  step {step:4d}  J={J.item():.4e}  argmax={am}")

            if true_major_idx is not None:
                with torch.no_grad():
                    w_now = w.detach()
                    x0_hat_full = (w_now.unsqueeze(1) * X).sum(dim=0)                     # (Ndt+1,)
                    xbar_hat_full = ((1 - w_now).unsqueeze(1) * X).sum(dim=0) / N          # (Ndt+1,)
                    true_major = X[true_major_idx]
                    # "true" mean field: use x_bar_obs if given (Tier 2), else fall back
                    # to the empirical average of the REAL minors (data-only, works in
                    # Tier 3 too since it only needs true_major_idx, not x_bar_obs).
                    true_xbar = (np.asarray(x_bar_obs) if x_bar_obs is not None
                                 else ((X.sum(dim=0) - true_major) / N).numpy())

                t_axis = np.arange(X.shape[1])
                fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

                ax = axes[0]
                ax.plot(t_axis, true_xbar, 'o-', label='True mean field (xbar)', linewidth=2, markersize=3)
                ax.plot(t_axis, xbar_hat_full.numpy(), 's--', label='Estimated mean field (xbar_hat)', linewidth=2, markersize=3)
                ax.set_ylabel('State value')
                ax.set_title(f'Mean field: true vs estimated  (step {step})')
                ax.legend(); ax.grid(True, alpha=0.3)

                ax = axes[1]
                ax.plot(t_axis, true_major.numpy(), 'o-', label=f'True major (agent {true_major_idx})', linewidth=2, markersize=3)
                ax.plot(t_axis, x0_hat_full.numpy(), 's--', label='Estimated soft major (x0_hat)', linewidth=2, markersize=3)
                ax.set_xlabel('Time step'); ax.set_ylabel('State value')
                ax.set_title(f'Major agent: true vs estimated  (argmax={am}, max(w)={w_now.max().item():.3f})')
                ax.legend(); ax.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(f'debug_step_{step:04d}.png', dpi=150)
                plt.close(fig)

    with torch.no_grad():
        major_prob = torch.softmax(theta, dim=0)
    return major_prob, history


# ----------------------------------------------------------------------------
# Baseline: fixed-score gap argmax (enumeration's per-agent likelihood ratio)
# ----------------------------------------------------------------------------
def detect_major_gap(mfg, X, x_bar_obs):
    """
    Closed-form baseline. Scores each agent as major-vs-minor using the OBSERVED
    mean field for the major reference and F*xbar+G*xbar as a w-free market proxy.
    Returns (major_prob = softmax(gap), gap (N+1,), argmax index).
    """
    dt = mfg.dt
    X  = torch.as_tensor(np.asarray(X), dtype=torch.float64)
    Np1, _ = X.shape
    phi   = torch.as_tensor(mfg.phi[:-1],   dtype=torch.float64)
    phi_0 = torch.as_tensor(mfg.phi_0[:-1], dtype=torch.float64)
    sig_M = mfg.sigma_0 * dt**0.5
    sig_m = mfg.sigma   * dt**0.5
    F, G  = mfg.F, mfg.G
    a, q, a0, q0 = mfg.a, mfg.q, mfg.a_0, mfg.q_0

    xt, xtp1 = X[:, :-1], X[:, 1:]
    xbar = torch.as_tensor(np.asarray(x_bar_obs), dtype=torch.float64)[:-1]

    mu_major = xt + (a0 + (q0 - phi_0)) * (xbar.unsqueeze(0) - xt) * dt
    Lmaj = dist.Normal(mu_major, sig_M).log_prob(xtp1).sum(dim=1)

    market = (F + G) * xbar                       # w-free proxy
    mu_minor = xt + (a + (q - phi)) * (market.unsqueeze(0) - xt) * dt
    Lmin = dist.Normal(mu_minor, sig_m).log_prob(xtp1).sum(dim=1)

    gap = Lmaj - Lmin
    return torch.softmax(gap, dim=0), gap, int(gap.argmax().item())


# ----------------------------------------------------------------------------
# Helper: simulate one system, stack, permute (kill positional label leakage)
# ----------------------------------------------------------------------------
def make_example(mfg, N, seed=None):
    if seed is not None:
        np.random.seed(seed)
    x_bar, x_major, x_minor = mfg.simulate(N=N, N_sim=1, do_plot=False)
    # stack major as row 0, then minors
    X_ordered = np.vstack([x_major[0:1, :], x_minor[0, :, :]])   # (N+1, Ndt+1)
    perm = np.random.permutation(N + 1)
    X = X_ordered[perm]
    true_idx = int(np.where(perm == 0)[0][0])                    # where major landed
    return X, x_bar[0], true_idx                                # x_bar[0]: (Ndt+1,)


# ----------------------------------------------------------------------------
# Demo / sanity check
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = MFG_config(
        T=1.0, Ndt=200,
        a_0=0.3, sigma_0=0.2, c_0=0.4, epslon_0=0.5, q_0=0.6,
        a=0.5,   sigma=0.3,   c=0.5,   epslon=0.5,   q=0.7,
        G=0.4,
    )
    mfg = MFG(cfg)
    mfg.solve_ODE()

    N = 20
    X, x_bar_obs, true_idx = make_example(mfg, N, seed=0)

    print("=== Relaxation (Tier 2, observed mean field) ===")
    prob, hist = detect_major_relaxed(mfg, X, x_bar_obs, verbose=True, true_major_idx=true_idx)
    pred = int(prob.argmax().item())
    print(f"predicted={pred}  true={true_idx}  correct={pred==true_idx}")
    print("top-3 prob:", np.round(np.sort(prob.numpy())[::-1][:3], 3))

    print("\n=== Relaxation (Tier 3, mean field unobserved -- estimated from X) ===")
    prob3, hist3 = detect_major_relaxed(mfg, X, x_bar_obs=None, verbose=True, true_major_idx=true_idx)
    pred3 = int(prob3.argmax().item())
    print(f"predicted={pred3}  true={true_idx}  correct={pred3==true_idx}")
    print("top-3 prob:", np.round(np.sort(prob3.numpy())[::-1][:3], 3))

    print("\n=== Gap-argmax baseline ===")
    pb, gap, pred_b = detect_major_gap(mfg, X, x_bar_obs)
    print(f"predicted={pred_b}  true={true_idx}  correct={pred_b==true_idx}")

    # quick accuracy over seeds
    print("\n=== Accuracy over 50 seeds ===")
    nc_relax = nc_relax3 = nc_gap = 0
    for s in range(50):
        Xs, xbs, ti = make_example(mfg, N, seed=100 + s)
        p,  _ = detect_major_relaxed(mfg, Xs, xbs,        n_steps=400, verbose=False)
        p3, _ = detect_major_relaxed(mfg, Xs, x_bar_obs=None, n_steps=400, verbose=False)
        _, _, pg = detect_major_gap(mfg, Xs, xbs)
        nc_relax  += (int(p.argmax())  == ti)
        nc_relax3 += (int(p3.argmax()) == ti)
        nc_gap    += (pg == ti)
    print(f"relaxation (observed xbar): {nc_relax}/50   "
          f"relaxation (estimated xbar): {nc_relax3}/50   "
          f"gap-baseline: {nc_gap}/50")
