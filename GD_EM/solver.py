"""
Detect which of N+1 agents is the major agent in a finite-population LQG-MFG,
using the SOFTMAX RELAXATION of the enumeration likelihood test -- and, on top
of that, estimate the unknown coupling weight G in the market state
m_t = (1-G)*xbar_t + G*x0_t (F is fixed to 1-G, matching MFG_config).

Regime: TIER 2 -- the empirical mean field  xbar_t  is OBSERVED, but the market
state  m_t  is NOT directly observed because it depends on the (unknown) major
trajectory x0_t AND the unknown mixing weight G. So a genuine w-coupling survives
through the soft major  x0_hat_t(w) = sum_i w_i x_i(t), and the relaxation does
real work.

phi_0(t) -- the MAJOR bank's Riccati coefficient -- solves (mfg.py, eq 4.2):
    dphi0/dt = 2*((a0+q0) + G*(a+q-phi_t))*phi0 - phi0^2 + eps0 - q0^2
which itself depends on G. So the moment G is unknown, phi_0(t) is not a fixed
"given, known" coefficient -- it must be RE-SOLVED at the current G estimate,
and doing so honestly makes ell_major genuinely G-dependent (through a nonlinear
ODE, no closed form). phi(t) -- the MINOR bank's Riccati coefficient -- has no G
in its own ODE, so it stays G-free and is solved once, in plain numpy, up front.

This is solved with an EM-like alternating scheme over (w, G):

    E-step: fix G, run gradient ascent on theta (w = softmax(theta)) to
            (approximately) maximize  J(w | G) = sum_i sum_t
                [ w_i * ell_major_i(t) + (1-w_i) * ell_minor_i(t; G) ]
            by unconstrained gradient ascent on theta (autodiff through softmax).
            phi_0(G) is re-solved once at the start of the E-step and held as a
            detached constant -- theta's gradient never needs to see G's graph.

    M-step: fix w (detached, i.e. responsibilities frozen), update G by
            GRADIENT ascent (no closed form -- see above). Every M-step substep
            re-solves phi_0(G) with a torch-differentiable backward-Euler
            recursion, so autograd carries the full total derivative of J
            w.r.t. G, including the indirect "G reshapes the major bank's own
            optimal control law" channel, not just the direct market-mixing one.

ell_major_i(t): log N( x_i(t+1) ; revert-to-XBAR with major rate, sigma0^2 dt )     -- w-free (if xbar observed), G-coupled via phi_0(G)
ell_minor_i(t): log N( x_i(t+1) ; revert-to-MARKET(w,G) with minor rate, sigma^2 dt) -- w- and G-coupled
"""

import numpy as np
import torch
import torch.distributions as dist
from mfg import MFG, MFG_config
import matplotlib.pyplot as plt

def _solve_phi0_torch(G, phi_full, mfg):
    """
    Differentiable backward-Euler solve of the major Riccati ODE (mirrors
    MFG.solve_ODE()'s phi_0 recursion), keeping G attached to the autograd graph.

    phi_full : (Ndt+1,) tensor, the fixed (G-independent) minor Riccati
               coefficients already solved once by mfg.solve_ODE().
    returns  : (Ndt+1,) tensor, phi_0(t) as a function of G.
    """
    dt, Ndt = mfg.dt, mfg.Ndt
    a0, q0, a, q, eps0, c0 = mfg.a_0, mfg.q_0, mfg.a, mfg.q, mfg.epslon_0, mfg.c_0

    phi0 = [None] * (Ndt + 1)
    phi0[Ndt] = torch.as_tensor(-c0, dtype=torch.float64)
    for kk in range(Ndt - 1, -1, -1):
        prev, phi_t = phi0[kk + 1], phi_full[kk + 1]
        dphi0 = 2 * ((a0 + q0) + G * (a + q - phi_t)) * prev - prev**2 + eps0 - q0**2
        phi0[kk] = prev - dt * dphi0
    return torch.stack(phi0)


def detect_major_G(mfg:MFG, X,true_major_idx:int,
                         n_em_iters=50, n_inner_E_steps=20, n_inner_M_steps=20,
                         lr_E=0.05, lr_G=0.05, lam_entropy=0.0,
                         leave_one_out=True, temp_anneal=False,
                         init_theta=None, init_G=None, verbose=False,
                         fix_phi0=False):
    """
    mfg         : MFG instance with solve_ODE() already called.
    X           : (N+1, Ndt+1) observed agent trajectories. Rows = agents.
                  ORDER MUST BE LABEL-AGNOSTIC (permute before calling).
    n_em_iters      : number of outer EM iterations (E-step + M-step).
    n_inner_E_steps : number of Adam steps on theta per E-step (G held fixed).
    n_inner_M_steps : number of Adam steps on G per M-step (w held fixed);
                      phi_0(G) is re-solved, differentiably, every substep
                      (default -- see fix_phi0 below).
    fix_phi0    : if False (default), phi_0(G) is re-solved differentiably every E-step
                  (detached snapshot) and every M-step substep (tracked), so
                  ell_major is genuinely G-coupled through the major bank's
                  own Riccati ODE, not just ell_minor through the market mix --
                  the fully self-consistent estimator.
                  If True, phi_0(t) is instead treated as a KNOWN, FIXED
                  coefficient -- taken once from mfg.solve_ODE() (i.e. at
                  mfg's own G) and never re-solved as G is estimated. ell_major
                  is then G-free, exactly like the very first version of this
                  file, and only ell_minor carries the direct G-coupling
                  through the market mix. Gradient ascent on G is still used
                  for the M-step (not the closed-form LS solution), so this is
                  a clean ablation against fix_phi0=False: same optimizer,
                  only difference is whether phi_0 tracks G or not.
    returns     : (major_prob (N+1,) tensor, G (scalar tensor), history list of J values)
    """
    # mfg.solve_ODE()
    dt, Ndt = mfg.dt, mfg.Ndt
    X  = torch.as_tensor(np.asarray(X), dtype=torch.float64)
    Np1, _ = X.shape
    true_major = X[true_major_idx]
    N = Np1 - 1

    # minor Riccati coefficient: G-independent, solved once, frozen forever
    phi_full = torch.as_tensor(mfg.phi, dtype=torch.float64)   # (Ndt+1,)
    phi = phi_full[:-1]                                        # (Ndt,) aligned to transitions

    if fix_phi0:
        # major Riccati coefficient treated as KNOWN/GIVEN: taken once from
        # mfg.solve_ODE() (solved at mfg's own G) and never touched again.
        phi0_fixed = torch.as_tensor(mfg.phi_0[:-1], dtype=torch.float64)  # (Ndt,)

    sig_M = mfg.sigma_0 * dt**0.5     # major noise std per step
    sig_m = mfg.sigma   * dt**0.5     # minor noise std per step

    a, q = mfg.a, mfg.q
    a0, q0 = mfg.a_0, mfg.q_0
    k = (a + (q - phi)) * dt          # (Ndt,) minor drift coefficient on (market - x), G-independent

    xt   = X[:, :-1]    # (N+1, Ndt) state at t
    xtp1 = X[:, 1:]     # (N+1, Ndt) state at t+1

    def Lmaj_of(phi0_slice, xbar):
        # ell_major_i(t): reverts to xbar with the major rate, phi_0(G)-coupled
        mu_major = xt + (a0 + (q0 - phi0_slice)) * (xbar.unsqueeze(0) - xt) * dt  # (N+1,Ndt)
        return dist.Normal(mu_major, sig_M).log_prob(xtp1).sum(dim=1)              # (N+1,)

    def Lmin_of(w, wt, xbar, x0_hat, G_):
        # ell_minor_i(t): reverts to market(w, G) with the minor rate
        if leave_one_out:
            x0_hat_i = x0_hat.unsqueeze(0) - wt * xt            # (N+1,Ndt)
        else:
            x0_hat_i = x0_hat.unsqueeze(0).expand_as(xt)        # (N+1,Ndt)
        market_i = (1 - G_) * xbar.unsqueeze(0) + G_ * x0_hat_i  # (N+1,Ndt)
        mu_minor = xt + k * (market_i - xt)                      # (N+1,Ndt)
        return dist.Normal(mu_minor, sig_m).log_prob(xtp1).sum(dim=1)  # (N+1,)

    # --- theta: E-step parameter --------------------------------------------
    theta = (torch.zeros(Np1, dtype=torch.float64) if init_theta is None
            else torch.as_tensor(init_theta, dtype=torch.float64).clone())
    theta.requires_grad_(True)
    opt_theta = torch.optim.Adam([theta], lr=lr_E)

    # --- G: M-step parameter ---------------------------------------------
    # Reparametrize G = sigmoid(g_raw) so it's confined to (0,1) no matter how
    # large a gradient step Adam takes on g_raw. This matters: G is an ODE
    # coefficient inside _solve_phi0_torch's explicit-Euler recursion, and an
    # unconstrained G can wander into a regime where that recursion is
    # numerically unstable (diverges), producing garbage phi_0 / -inf
    # log-likelihoods / exploding gradients -- a runaway that unconstrained G
    # has no way to recover from.
    def _logit(p, eps=1e-6):
        p = min(max(float(p), eps), 1 - eps)
        return np.log(p / (1 - p))

    g_raw_init = np.random.uniform(-1, 1) if init_G is None else _logit(init_G)
    g_raw = torch.as_tensor(g_raw_init, dtype=torch.float64)
    g_raw.requires_grad_(True)
    opt_G = torch.optim.Adam([g_raw], lr=lr_G)

    history_E = []      # E-step J trace (one point per theta gradient step)
    history_M = []    # M-step J trace (one point per G gradient step)
    recent_G = []     # rolling window of G, for the early-stop check below
    major_snapshots = []   # (step, x0_hat_full) every 20 steps, for the trajectory-dynamics plot
    n_steps = n_em_iters * n_inner_E_steps
    step = 0

    for em_iter in range(n_em_iters+1):
        tau = 1.0 if not temp_anneal else max(0.3, 1.0 - step / n_steps)

        # ============================= E-step ================================
        # fix G (and its induced phi_0), take gradient-ascent steps on theta to
        # maximize J(w | G). theta's gradient never needs G's graph, so phi_0
        # is solved once here as a detached constant for the whole E-step.
        with torch.no_grad():
            G_frozen = torch.sigmoid(g_raw)
            phi0_frozen = phi0_fixed if fix_phi0 else _solve_phi0_torch(G_frozen, phi_full, mfg)[:-1]   # (Ndt,)

        for _ in range(n_inner_E_steps):
            w = torch.softmax(theta / tau, dim=0)          # (N+1,)
            wt = w.unsqueeze(1)                            # (N+1,1)
            x0_hat = (wt * xt).sum(dim=0)                  # soft major traj -> (Ndt,)

            xbar = ((1 - w).unsqueeze(1) * xt).sum(dim=0) / N     # (Ndt,)

            Lmaj = Lmaj_of(phi0_frozen, xbar)
            Lmin = Lmin_of(w, wt, xbar, x0_hat, G_frozen)

            J = (w * Lmaj + (1 - w) * Lmin).sum()
            H_w = -(w * torch.log(w + 1e-12)).sum()
            loss = -(J - lam_entropy * H_w)
            opt_theta.zero_grad(); loss.backward(); opt_theta.step()
            history_E.append(J.item())
            

        # ============================= M-step ================================
        # freeze responsibilities w, update G by gradient ascent. phi_0(G) is
        # re-solved DIFFERENTIABLY every substep, so the backward pass carries
        # the full total derivative of J w.r.t. G (direct, through the market
        # mix, AND indirect, through how G reshapes the major bank's own
        # Riccati coefficient) -- there is no closed form for this any more.
        with torch.no_grad():
            w_frozen = torch.softmax(theta, dim=0)
            # print("frozen w:",np.sort(w_frozen.detach().numpy())[::-1][:3])
            wt_frozen = w_frozen.unsqueeze(1)
            x0_hat_frozen = (wt_frozen * xt).sum(dim=0)
            xbar_frozen = ((1 - w_frozen).unsqueeze(1) * xt).sum(dim=0) / N

        for _ in range(n_inner_M_steps):
            G = torch.sigmoid(g_raw)                            # bounded to (0,1)
            phi0 = phi0_fixed if fix_phi0 else _solve_phi0_torch(G, phi_full, mfg)[:-1]  # (Ndt,)

            Lmaj = Lmaj_of(phi0, xbar_frozen)
            Lmin = Lmin_of(w_frozen, wt_frozen, xbar_frozen, x0_hat_frozen, G)

            J = (w_frozen * Lmaj + (1 - w_frozen) * Lmin).sum()
            loss = -J
            opt_G.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([g_raw], max_norm=5.0)
            opt_G.step()
            history_M.append(J.item())
        
        # --- early stop: w has essentially committed to one agent AND G has
        # stopped moving -- no point burning more EM iterations past this.
        recent_G.append(torch.sigmoid(g_raw).item())
        if len(recent_G) > 5:
            recent_G.pop(0)
        if w_frozen.max().item() >= 0.99 and len(recent_G) == 5 and (max(recent_G) - min(recent_G)) < 1e-3:
            if verbose:
                print(f"  early stop at EM iter {em_iter}: max(w)={w_frozen.max().item():.4f}  G={recent_G[-1]:.4f}")
            break

        if verbose:
            am = torch.softmax(theta, dim=0).argmax().item()
            print(f"  EM iter {em_iter:3d}  J={history_E[-1]:.4e}  G={torch.sigmoid(g_raw).item():.4f}  argmax={am}")
        if step % 10 == 0:
            with torch.no_grad():
                x0_hat_full = (w.unsqueeze(1) * X).sum(dim=0)                     # (Ndt+1,)
            major_snapshots.append((step, x0_hat_full.numpy()))
        step += 1
    with torch.no_grad():
        major_prob = torch.softmax(theta, dim=0)

    if verbose:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        axes[0].plot(history_E)
        axes[0].set_xlabel('E-step gradient step (cumulative)'); axes[0].set_ylabel('J')
        axes[0].set_title('E-step loss (theta ascent, G fixed)')
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(history_M, color='tab:orange')
        axes[1].set_xlabel('M-step gradient step (cumulative)'); axes[1].set_ylabel('J')
        axes[1].set_title('M-step loss (G ascent, w fixed)')
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('loss_detect_major_G.png', dpi=150)
        plt.close(fig)
        print("  saved loss plot to loss_detect_major_G.png")

        # --- one combined plot: how x0_hat evolved toward the true major -----
        if major_snapshots:
            with torch.no_grad():
                x0_hat_terminal = (major_prob.unsqueeze(1) * X).sum(dim=0).numpy()

            t_axis = np.arange(X.shape[1])
            fig, ax = plt.subplots(figsize=(11, 6))

            cmap = plt.cm.viridis
            n_snap = len(major_snapshots)
            for i, (snap_step, x0_hat_snap) in enumerate(major_snapshots):
                color = cmap(i / max(n_snap - 1, 1))
                ax.plot(t_axis, x0_hat_snap, color=color, alpha=0.6, linewidth=1)

            ax.plot(t_axis, true_major.numpy(), color='black', linewidth=2.5,
                     label=f'True major (agent {true_major_idx})')
            ax.plot(t_axis, x0_hat_terminal, color='red', linewidth=2, linestyle='--',
                     label='Terminal estimate (x0_hat)')

            sm = plt.cm.ScalarMappable(cmap=cmap,
                                        norm=plt.Normalize(vmin=major_snapshots[0][0], vmax=major_snapshots[-1][0]))
            sm.set_array([])
            fig.colorbar(sm, ax=ax, label='training step')

            ax.set_xlabel('Time step'); ax.set_ylabel('State value')
            ax.set_title('Major agent estimate: dynamics over training vs true')
            ax.legend(loc='upper left'); ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('major_trajectory_dynamics.png', dpi=150)
            plt.close(fig)
            print("  saved major trajectory dynamics plot to major_trajectory_dynamics.png")

    return major_prob, torch.sigmoid(g_raw).detach(),step


# ----------------------------------------------------------------------------
# Same EM scheme, but the M-step solves for G in CLOSED FORM (not gradient).
# ----------------------------------------------------------------------------
def detect_major_G_closedform(mfg: MFG, X,true_major_idx:int, 
                               n_em_iters=50, n_inner_E_steps=20, lr_E=0.05,
                               leave_one_out=True, temp_anneal=False,
                               init_theta=None, verbose=False, lam_entropy=0.0,
                               recalc_phi0=False):
    """
    Same E-step as detect_major_G (gradient ascent on theta, G held fixed).
    The M-step, however, solves for G in CLOSED FORM instead of by gradient
    ascent -- this is only valid because phi_0 is treated as G-FREE here
    (taken once from mfg.solve_ODE(), never re-solved; equivalent to
    detect_major_G(..., fix_phi0=True)). With phi_0 fixed, ell_major has no
    G-dependence at all, and mu_minor_i(t) = c_i(t) + G*d_i(t) is exactly
    affine in G, so the weighted Gaussian log-likelihood is quadratic in G and
    its maximizer (for the current, frozen w) is the ordinary weighted
    least-squares solution:

        G* = sum_{i,t} (1-w_i) * d_i(t) * e_i(t)  /  sum_{i,t} (1-w_i) * d_i(t)^2

    where c_i(t) = mu_minor_i(t) at G=0, d_i(t) = d(mu_minor_i(t))/dG, and
    e_i(t) = x_i(t+1) - c_i(t). No sigmoid/bounding is needed for G here: unlike
    detect_major_G's fix_phi0=False path, G never feeds a numerically-touchy
    ODE recursion in this function -- it only ever appears in the affine
    market-mixing term, which is smooth and well-defined for any real G.

    mfg, X, x_bar_obs, n_em_iters, n_inner_E_steps, lr_E, leave_one_out,
    temp_anneal, init_theta, verbose : same meaning as in detect_major_G.
    true_major_idx : DEBUG ONLY -- index of the actual major agent in X (known
                  because this is synthetic data from make_example). Not used
                  by the estimator itself (no ground-truth leakage into theta
                  or G); only used, when verbose=True, to overlay the true
                  major trajectory and the true (or data-only-estimated) mean
                  field against the model's own estimates in a saved debug
                  plot ('mean_field_and_major_debug.png').
    lam_entropy : coefficient on an ENTROPY PENALTY added to the E-step's
                  objective: maximize J(w|G) - lam_entropy*H(w), where
                  H(w) = -sum_i w_i*log(w_i). This is the opposite sign from
                  the usual variational-EM entropy BONUS (which guards against
                  w collapsing artificially): here w is instead pulled toward
                  HIGH entropy (flat) by a genuine ridge in the likelihood --
                  ell_minor depends on (G, w) almost only through the product
                  G*max(w), so ell_major can gain "for free" by spreading
                  weight onto agents that look major-like by pure sampling
                  noise, financed by a compensating rise in G. Subtracting
                  entropy (entropy MINIMIZATION, as in Grandvalet-Bengio
                  self-training) directly counteracts that free-lunch
                  direction. Only affects the E-step -- H(w) doesn't depend on
                  G, so the M-step's closed-form solve is unchanged.
    recalc_phi0 : if True, phi_0 is NOT left at mfg.solve_ODE()'s value (which
                  was solved at mfg's own, TRUE G -- information the estimator
                  shouldn't really have if G is genuinely unknown). Instead,
                  at the start of every EM iteration, phi_0 is RE-SOLVED at the
                  CURRENT G estimate (plain non-differentiable backward-Euler,
                  no_grad -- the closed-form M-step still treats phi_0/ell_major
                  as G-free WITHIN that iteration's own G* solve; only the
                  snapshot itself is refreshed between iterations, the same
                  "solve equilibrium at current parameter, then treat it as
                  fixed for this round" pattern as detect_major_G's E-step).
                  This makes ell_major genuinely (iteration-to-iteration)
                  track the evolving G estimate instead of secretly assuming
                  the true one throughout. If False (default), phi_0 is fixed
                  once at mfg's true G for the whole run, as before.
    returns : (major_prob (N+1,) tensor, G (scalar tensor), history list of J values)
    """
    dt = mfg.dt
    X  = torch.as_tensor(np.asarray(X), dtype=torch.float64)
    Np1, _ = X.shape
    N = Np1 - 1
    true_major = X[true_major_idx]

    phi_full = torch.as_tensor(mfg.phi, dtype=torch.float64)   # (Ndt+1,)
    phi = phi_full[:-1]                                        # (Ndt,)
    phi0_fixed = torch.as_tensor(mfg.phi_0[:-1], dtype=torch.float64)  # (Ndt,), G-free

    sig_M = mfg.sigma_0 * dt**0.5
    sig_m = mfg.sigma   * dt**0.5

    a, q = mfg.a, mfg.q
    a0, q0 = mfg.a_0, mfg.q_0
    k = (a + (q - phi)) * dt          # (Ndt,) minor drift coefficient, G-independent

    xt   = X[:, :-1]    # (N+1, Ndt)
    xtp1 = X[:, 1:]     # (N+1, Ndt)

    theta = (torch.zeros(Np1, dtype=torch.float64) if init_theta is None
             else torch.as_tensor(init_theta, dtype=torch.float64).clone())
    theta.requires_grad_(True)
    opt_theta = torch.optim.Adam([theta], lr=lr_E)

    G = torch.as_tensor(np.random.uniform(0, 1), dtype=torch.float64)

    history = []      # E-step J trace (one point per theta gradient step)
    history_M = []    # M-step J trace (one point per EM iteration -- closed-form solve, no inner loop)
    recent_G = []     # rolling window of G, for the early-stop check below
    n_steps = n_em_iters * n_inner_E_steps
    step = 0

    for em_iter in range(n_em_iters):
        tau = 1.0 if not temp_anneal else max(0.3, 1.0 - step / n_steps)

        if recalc_phi0:
            # re-solve phi_0 at the CURRENT G estimate (not mfg's true G),
            # plain/non-differentiable -- treated as fixed for this iteration.
            with torch.no_grad():
                phi0_fixed = _solve_phi0_torch(G, phi_full, mfg)[:-1]
                
        # ============================= E-step ================================
        # fix G, take gradient-ascent steps on theta to maximize J(w | G).
        for _ in range(n_inner_E_steps):
            w = torch.softmax(theta / tau, dim=0)          # (N+1,)
            wt = w.unsqueeze(1)                            # (N+1,1)
            x0_hat = (wt * xt).sum(dim=0)                  # soft major traj -> (Ndt,)

            xbar = ((1 - w).unsqueeze(1) * xt).sum(dim=0) / N     # (Ndt,)
            mu_major = xt + (a0 + (q0 - phi0_fixed)) * (xbar.unsqueeze(0) - xt) * dt
            Lmaj = dist.Normal(mu_major, sig_M).log_prob(xtp1).sum(dim=1)

            if leave_one_out:
                x0_hat_i = x0_hat.unsqueeze(0) - wt * xt            # (N+1,Ndt)
            else:
                x0_hat_i = x0_hat.unsqueeze(0).expand_as(xt)        # (N+1,Ndt)
            market_i = (1 - G) * xbar.unsqueeze(0) + G * x0_hat_i    # (N+1,Ndt)
            mu_minor = xt + k * (market_i - xt)                      # (N+1,Ndt)
            Lmin = dist.Normal(mu_minor, sig_m).log_prob(xtp1).sum(dim=1)  # (N+1,)

            J = (w * Lmaj + (1 - w) * Lmin).sum()
            # entropy MINIMIZATION penalty (opposite sign from the usual ELBO
            # entropy bonus): counteracts the ell_major-driven ridge that
            # otherwise rewards flattening w -- see docstring.
            H_w = -(w * torch.log(w + 1e-12)).sum()
            loss = -(J - lam_entropy * H_w)
            opt_theta.zero_grad(); loss.backward(); opt_theta.step()
            history.append(J.item())
            step += 1

        # ============================= M-step (closed form) ==================
        # freeze responsibilities w, solve for G exactly via weighted least
        # squares -- see docstring for the derivation.
        with torch.no_grad():
            w = torch.softmax(theta, dim=0)
            wt = w.unsqueeze(1)
            x0_hat = (wt * xt).sum(dim=0)

            xbar = ((1 - w).unsqueeze(1) * xt).sum(dim=0) / N

            if leave_one_out:
                x0_hat_i = x0_hat.unsqueeze(0) - wt * xt
            else:
                x0_hat_i = x0_hat.unsqueeze(0).expand_as(xt)

            c_i = xt + k * (xbar.unsqueeze(0) - xt)           # mu_minor at G=0     (N+1,Ndt)
            d_i = k * (x0_hat_i - xbar.unsqueeze(0))          # d(mu_minor)/dG      (N+1,Ndt)
            e_i = xtp1 - c_i                                  # residual at G=0     (N+1,Ndt)

            weights = (1 - w).unsqueeze(1)                    # (N+1,1) minor responsibility

            num = (weights * d_i * e_i).sum()
            den = (weights * d_i * d_i).sum()
            if den.abs() > 1e-12:
                # G is a mixing weight (F=1-G), only physically meaningful in
                # [0,1]. J(G) is concave quadratic, so clamping the unconstrained
                # optimum to [0,1] is the EXACT box-constrained maximizer, not a
                # heuristic: if num/den lies outside [0,1], J is monotonic
                # between it and the interval, so the nearest endpoint is where
                # the constrained maximum is attained.
                G = torch.clamp(num / den, 0.0, 1.0)

            # M-step J, evaluated at the freshly-solved G, for the loss plot
            mu_minor_new = c_i + G * d_i
            Lmin_new = dist.Normal(mu_minor_new, sig_m).log_prob(xtp1).sum(dim=1)
            Lmaj_new = dist.Normal(
                xt + (a0 + (q0 - phi0_fixed)) * (xbar.unsqueeze(0) - xt) * dt, sig_M
            ).log_prob(xtp1).sum(dim=1)
            J_M = (w * Lmaj_new + (1 - w) * Lmin_new).sum()
            history_M.append(J_M.item())

        # --- early stop: w has essentially committed to one agent AND G has
        # stopped moving -- no point burning more EM iterations past this.
        recent_G.append(G.item())
        if len(recent_G) > 5:
            recent_G.pop(0)
        if w.max().item() >= 0.99 and len(recent_G) == 5 and (max(recent_G) - min(recent_G)) < 1e-3:
            if verbose:
                print(f"  early stop at EM iter {em_iter}: max(w)={w.max().item():.4f}  G={recent_G[-1]:.4f}")
            break

        if verbose:
            am = w.argmax().item()   # w here is the M-step's frozen snapshot, already computed above
            print(f"  EM iter {em_iter:3d}  J={history[-1]:.4e}  G={G.item():.4f}  argmax={am}")

            with torch.no_grad():
                x0_hat_full = (w.unsqueeze(1) * X).sum(dim=0)                     # (Ndt+1,)

            t_axis = np.arange(X.shape[1])
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(t_axis, true_major.numpy(), 'o-', label=f'True major (agent {true_major_idx})', linewidth=2, markersize=3)
            ax.plot(t_axis, x0_hat_full.numpy(), 's--', label='Estimated soft major (x0_hat)', linewidth=2, markersize=3)
            ax.set_xlabel('Time step'); ax.set_ylabel('State value')
            ax.set_title(f'Major agent: true vs estimated  (argmax={am}, max(w)={w.max().item():.3f}, G={G.item():.4f})')
            ax.legend(); ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f'debug_iter_{em_iter:03d}.png', dpi=150)
            plt.close(fig)

    with torch.no_grad():
        major_prob = torch.softmax(theta, dim=0)

    if verbose:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        axes[0].plot(history)
        axes[0].set_xlabel('E-step gradient step (cumulative)'); axes[0].set_ylabel('J')
        axes[0].set_title('E-step loss (theta ascent, G fixed)')
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(history_M, color='tab:orange', marker='o', markersize=3)
        axes[1].set_xlabel('EM iteration'); axes[1].set_ylabel('J')
        axes[1].set_title('M-step loss (closed-form G solve, w fixed)')
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('loss_detect_major_G_closedform.png', dpi=150)
        plt.close(fig)
        print("  saved loss plot to loss_detect_major_G_closedform.png")

    return major_prob, G, history


# # ----------------------------------------------------------------------------
# # Baseline: fixed-score gap argmax (enumeration's per-agent likelihood ratio)
# # ----------------------------------------------------------------------------
# def detect_major_gap(mfg, X, x_bar_obs):
#     """
#     Closed-form baseline. Scores each agent as major-vs-minor using the OBSERVED
#     mean field for the major reference and F*xbar+G*xbar as a w-free market proxy.
#     Returns (major_prob = softmax(gap), gap (N+1,), argmax index).
#     """
#     dt = mfg.dt
#     X  = torch.as_tensor(np.asarray(X), dtype=torch.float64)
#     Np1, _ = X.shape
#     phi   = torch.as_tensor(mfg.phi[:-1],   dtype=torch.float64)
#     phi_0 = torch.as_tensor(mfg.phi_0[:-1], dtype=torch.float64)
#     sig_M = mfg.sigma_0 * dt**0.5
#     sig_m = mfg.sigma   * dt**0.5
#     F, G  = mfg.F, mfg.G
#     a, q, a0, q0 = mfg.a, mfg.q, mfg.a_0, mfg.q_0

#     xt, xtp1 = X[:, :-1], X[:, 1:]
#     xbar = torch.as_tensor(np.asarray(x_bar_obs), dtype=torch.float64)[:-1]

#     mu_major = xt + (a0 + (q0 - phi_0)) * (xbar.unsqueeze(0) - xt) * dt
#     Lmaj = dist.Normal(mu_major, sig_M).log_prob(xtp1).sum(dim=1)

#     market = (1 - G) * xbar + G * x0_hat_loo                       # w-free proxy
#     mu_minor = xt + (a + (q - phi)) * (market.unsqueeze(0) - xt) * dt
#     Lmin = dist.Normal(mu_minor, sig_m).log_prob(xtp1).sum(dim=1)

#     gap = Lmaj - Lmin
#     return torch.softmax(gap, dim=0), gap, int(gap.argmax().item())


# ----------------------------------------------------------------------------
# Helper: simulate one system, stack, permute (kill positional label leakage)
# ----------------------------------------------------------------------------
def make_example(mfg: MFG, N: int, seed=None):
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
    T   = 1       # time horizon (Section 6)
    Ndt = 512    # time steps  → dt = 0.001

    # ── Major bank parameters (Fig B.5, Fig B.6) ─────────────────────────────────
    G        = 0.5   # relative market size of major bank  (F = 1-G = 0.5)
    a        = 5     # minor bank mean-reversion rate (Section 6.2 baseline)
    a_0      = a * G # = 2.5  market-clearing condition (eq 2.8): a_0 = a*G
    sigma_0  = 1.0   # major bank reserve volatility
    q_0      = 1.0   # major bank incentive to trade with central bank  (Fig B.5)
    epslon_0 = 10.0  # major bank running penalty on reserve deviation  (Fig B.6)
    c_0      = 0.0   # major bank terminal penalty                       (Fig B.6)

    # ── Minor bank parameters (Fig B.5) ──────────────────────────────────────────
    sigma    = 1.0   # minor bank reserve volatility
    q        = 1.0   # minor bank incentive to trade with central bank
    epslon   = 1.5   # minor bank running penalty  (must satisfy q^2 <= epslon)
    c        = 0.0   # minor bank terminal penalty

    # ── Monte Carlo ───────────────────────────────────────────────────────────────
    N     = 512      # number of minor banks (Section 6)
    N_sim = 300  # Monte Carlo paths     (Section 6)

    cfg = MFG_config(
        T=T, Ndt=Ndt,
        a_0=a_0, sigma_0=sigma_0, c_0=c_0, epslon_0=epslon_0, q_0=q_0,
        a=a,     sigma=sigma,     c=c,     epslon=epslon,       q=q,
        G=G,
    )
    mfg = MFG(cfg)
    mfg.solve_ODE()

    N = 20
    X, x_bar_obs, true_idx = make_example(mfg, N, seed=0)

    print("=== EM relaxation (Tier 2, observed mean field, G unknown) ===")
    prob, G_hat, hist = detect_major_G(mfg, X, x_bar_obs, verbose=True)
    pred = int(prob.argmax().item())
    print(f"predicted={pred}  true={true_idx}  correct={pred==true_idx}  "
          f"G_hat={G_hat.item():.4f}  G_true={cfg.G}")
    print("top-3 prob:", np.round(np.sort(prob.numpy())[::-1][:3], 3))

    print("\n=== EM relaxation (Tier 3, mean field unobserved -- estimated from X) ===")
    prob3, G_hat3, hist3 = detect_major_G(mfg, X, x_bar_obs=None, verbose=True)
    pred3 = int(prob3.argmax().item())
    print(f"predicted={pred3}  true={true_idx}  correct={pred3==true_idx}  "
          f"G_hat={G_hat3.item():.4f}  G_true={cfg.G}")
    print("top-3 prob:", np.round(np.sort(prob3.numpy())[::-1][:3], 3))

    # print("\n=== Gap-argmax baseline ===")
    # pb, gap, pred_b = detect_major_gap(mfg, X, x_bar_obs)
    # print(f"predicted={pred_b}  true={true_idx}  correct={pred_b==true_idx}")

    # # quick accuracy over seeds
    # print("\n=== Accuracy over 50 seeds ===")
    # nc_relax = nc_relax3 = nc_gap = 0
    # for s in range(50):
    #     Xs, xbs, ti = make_example(mfg, N, seed=100 + s)
    #     p,  _ = detect_major_relaxed(mfg, Xs, xbs,        n_steps=400, verbose=False)
    #     p3, _ = detect_major_relaxed(mfg, Xs, x_bar_obs=None, n_steps=400, verbose=False)
    #     _, _, pg = detect_major_gap(mfg, Xs, xbs)
    #     nc_relax  += (int(p.argmax())  == ti)
    #     nc_relax3 += (int(p3.argmax()) == ti)
    #     nc_gap    += (pg == ti)
    # print(f"relaxation (observed xbar): {nc_relax}/50   "
    #       f"relaxation (estimated xbar): {nc_relax3}/50   "
    #       f"gap-baseline: {nc_gap}/50")
