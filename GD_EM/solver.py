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

xbar_drift vs xbar_control -- mirrors the split made in mfg.py's state_transtition:
the physical mean-revert term (eq 2.1/2.3, coefficient a / a_0) reverts to the
ACTUAL population mean, while the optimal-control term (eq 4.1/4.3, coefficient
q-phi / q_0-phi_0) responds to the MEAN-FIELD LIMIT xbar_t (Theorem 4.1's
xbar_t, eq 4.5). xbar_drift is always the (soft, w-weighted) empirical mean of
minor agents. xbar_control is a confidence-weighted BLEND of that same
empirical mean and the self-consistent solution of eq (4.5) itself:
    xbar_control = beta*xbar_mf + (1-beta)*xbar_emp
where beta = self._meanfield_beta in [0,1] ramps up monotonically (see
`_param_confidence` / fit()) as the unknown parameters stabilize -- integrating
eq (4.5) from bad early-EM parameter guesses would inject noise into the
control-term gradient at beta~0, so it fades in gradually rather than a hard
switch (a hard 0/1 ratchet has a real failure mode: if the early-stop
condition and "params just stabilized" coincide in the same EM iteration, the
switch never gets exercised before the loop exits -- a smooth ramp avoids that
by mattering for several iterations on the way to full confidence, not just
at the threshold-crossing instant). See `_xbar_signals` for the selection
logic. TIER-1 (the true mean-field trajectory directly observed) lives in
the separate `XBarObservedEstimator` subclass, not in this base class --
see its docstring.
"""

import numpy as np
import torch
import torch.distributions as dist
from mfg import MFG, MFG_config
import matplotlib.pyplot as plt
from utility import make_example


# ----------------------------------------------------------------------------
# Reparametrizations: unconstrained "raw" tensor <-> physically-constrained
# parameter value. Every estimable parameter (see PARAM_SPECS below) goes
# through one of these so Adam can take unconstrained steps while the value
# fed into the Riccati ODEs / drift terms stays in its valid domain -- same
# trick as the old g_raw = sigmoid^{-1}(G) reparametrization, generalized.
# ----------------------------------------------------------------------------
class _Sigmoid:
    """value in (lo, hi). Used for G (a mixing weight)."""
    def __init__(self, lo=0.0, hi=1.0):
        self.lo, self.hi = lo, hi

    def to_value(self, raw, **_):
        return self.lo + (self.hi - self.lo) * torch.sigmoid(raw)

    def to_raw(self, value, **_):
        p = (float(value) - self.lo) / (self.hi - self.lo)
        p = min(max(p, 1e-6), 1 - 1e-6)
        return np.log(p / (1 - p))

    def sample_init_raw(self):
        return np.random.uniform(-1, 1)


class _Softplus:
    """value > shift. Used for mean-reversion rates a, a_0 (must stay positive)."""
    def __init__(self, shift=1e-3):
        self.shift = shift

    def to_value(self, raw, **_):
        return self.shift + torch.nn.functional.softplus(raw)

    def to_raw(self, value, **_):
        v = max(float(value) - self.shift, 1e-6)
        return np.log(np.expm1(v))

    def sample_init_raw(self):
        return np.random.uniform(-1, 1)


class _Identity:
    """value unconstrained. Used for q, q_0, c, c_0."""
    def to_value(self, raw, **_):
        return raw

    def to_raw(self, value, **_):
        return float(value)

    def sample_init_raw(self):
        return np.random.uniform(-1, 1)


# name -> transform. epslon/epslon_0 are independent positive scalars, same
# treatment as a/a_0 -- deliberately NOT tied to q/q_0 via the q^2 <= epslon
# relationship. That inequality is a real constraint on the true physical
# parameters, but baking it into the parametrization would hand the estimator
# privileged structural knowledge it wouldn't have from data alone; if q and
# epslon are both unknown, the estimator has to find a consistent (and,
# ideally, constraint-respecting) combination on its own.
PARAM_SPECS = {
    'G':        _Sigmoid(0.0, 1.0),
    'a':        _Softplus(),
    'a_0':      _Softplus(),
    'q':        _Identity(),
    'q_0':      _Identity(),
    'epslon':   _Softplus(),
    'epslon_0': _Softplus(),
    'c':        _Identity(),
    'c_0':      _Identity(),
}
PARAM_ORDER = ['G', 'a', 'a_0', 'q', 'q_0', 'epslon', 'epslon_0', 'c', 'c_0']


def _solve_riccati_torch(coef_fn, eps, q, c, Ndt, dt):
    """
    Generic differentiable backward-Euler solve of
        dphi/dt = 2*coef_fn(k, phi[k+1])*phi - phi^2 + eps - q^2
    with terminal condition phi(T) = -c. This is the shared shape of BOTH
    Riccati ODEs in mfg.py (eq 4.2 for the major bank, eq 4.4 for the minor
    bank) -- they differ only in what `coef_fn` is:
        minor: coef_fn(k, prev) = a + q                          (time-invariant)
        major: coef_fn(k, prev) = (a_0+q_0) + G*(a+q-phi_minor[k+1])   (reads the minor solve)
    eps, q, c may be python floats (known) or 0-d torch tensors with
    requires_grad=True (unknown) -- either way this stays differentiable
    w.r.t. whichever inputs are tensors.
    returns (Ndt+1,) tensor phi(t).
    """
    neg_c = -c if torch.is_tensor(c) else torch.as_tensor(-c, dtype=torch.float64)
    phi = [None] * (Ndt + 1)
    phi[Ndt] = neg_c
    for kk in range(Ndt - 1, -1, -1):
        prev = phi[kk + 1]
        coef = coef_fn(kk, prev)
        dphi = 2 * coef * prev - prev**2 + eps - q**2
        phi[kk] = prev - dt * dphi
    return torch.stack(phi)


class MajorAgentEstimator:
    """
    Detects the major agent among N+1 shuffled trajectories (softmax
    relaxation over theta, same as the old detect_major_G) AND jointly
    estimates any subset of {G, a, a_0, q, q_0, epslon, epslon_0, c, c_0}
    declared unknown, by gradient ascent (Adam) on the joint path
    log-likelihood -- generalizes detect_major_G's single-G M-step to N
    unknown "physics" parameters at once.

    Parameters not listed in `unknown` are treated as KNOWN and read once
    from `mfg` (plain floats, never touched again).

    phi(t) (minor Riccati, mfg.py eq 4.4) and phi0(t) (major Riccati, eq 4.2)
    are each solved via the SAME generic recursion (_solve_riccati_torch).
    Whichever one depends on a currently-unknown parameter is re-solved,
    differentiably, every time parameters change (E-step start: detached
    snapshot; M-step substep: tracked). If NONE of an ODE's inputs are
    unknown, it's taken once from mfg.solve_ODE() and reused as a fixed
    constant -- this is exactly detect_major_G's fix_phi0 flag, generalized
    to be decided automatically (per-ODE) from `unknown` instead of set by hand.
    """

    _PHI_MINOR_DEPS = {'a', 'q', 'epslon', 'c'}

    def __init__(self, mfg: MFG, unknown, lr_E=0.05, lr_M=0.05, lam_entropy=0.0,
                 leave_one_out=True, temp_anneal=False, init=None,
                 lr_decay=1.0):
        self.mfg = mfg
        # monotonic confidence ratchet in [0,1] (never decreases): how much of
        # xbar_control comes from the self-consistent eq (4.5) solve vs. the
        # empirical-mean proxy, see _xbar_signals / _param_confidence / fit().
        self._meanfield_beta = 0.0
        self.unknown = list(unknown)
        unknown_set = set(self.unknown)
        for name in self.unknown:
            if name not in PARAM_SPECS:
                raise ValueError(f"unknown parameter {name!r} has no registered transform in PARAM_SPECS")

        self.lr_E, self.lr_M = lr_E, lr_M
        # plain multiplicative decay applied to BOTH opt_theta and opt_M,
        # once per em_iter (not per inner step) -- default 1.0 is a no-op,
        # preserving old behavior. See conversation: near a sharp optimum (as
        # UObservedEstimator's u-term produces once beta~1), a constant lr
        # can overshoot and oscillate around the peak forever instead of
        # settling, which starves params_stable of ever firing. Decaying the
        # step size lets Adam actually converge onto a sharp optimum instead
        # of orbiting it.
        self.lr_decay = lr_decay
        self.lam_entropy = lam_entropy
        self.leave_one_out = leave_one_out
        self.temp_anneal = temp_anneal

        init = init or {}
        self.raw = {}
        for name in PARAM_ORDER:
            if name in unknown_set:
                spec = PARAM_SPECS[name]
                r0 = spec.to_raw(init[name]) if name in init else spec.sample_init_raw()
                self.raw[name] = torch.as_tensor(r0, dtype=torch.float64).requires_grad_(True)

        # phi(t)'s ODE only ever involves {a, q, epslon, c}; phi0(t)'s ODE
        # additionally involves {a_0, q_0, epslon_0, c_0, G} PLUS whatever
        # phi(t) depends on (it reads phi(t) as an input) -- so if phi(t)
        # needs re-solving, phi0(t) does too, regardless of its own params.
        self._minor_unknown = bool(self._PHI_MINOR_DEPS & unknown_set)
        self._major_unknown = self._minor_unknown or bool(
            {'a_0', 'q_0', 'epslon_0', 'c_0', 'G'} & unknown_set
        )

        # histories, populated by fit() -- always recorded (not just under a
        # verbose flag), so diagnostics can be plotted after the fact without
        # re-running the optimization.
        self.history_E, self.history_M = [], []
        self.history_E_grad, self.history_M_grad = [], []
        self.major_snapshots = []

    def _param_values(self, detach: bool):
        """dict of ALL params (known floats + unknown transformed tensors)."""
        vals = {}
        for name in PARAM_ORDER:
            if name in self.raw:
                raw = self.raw[name].detach() if detach else self.raw[name]
                vals[name] = PARAM_SPECS[name].to_value(raw)
            else:
                vals[name] = getattr(self.mfg, name)
        return vals

    def _riccati(self, p, track_grad: bool):
        """Returns (phi_minor, phi_major), each (Ndt,) aligned to transitions [:-1]."""
        dt, Ndt = self.mfg.dt, self.mfg.Ndt
        ctx = torch.enable_grad() if track_grad else torch.no_grad()
        with ctx:
            if self._minor_unknown:
                coef = p['a'] + p['q']
                phi_minor_full = _solve_riccati_torch(
                    lambda k, prev: coef, p['epslon'], p['q'], p['c'], Ndt, dt)
            else:
                phi_minor_full = torch.as_tensor(self.mfg.phi, dtype=torch.float64)

            if self._major_unknown:
                def coef_fn(k, prev):
                    return (p['a_0'] + p['q_0']) + p['G'] * (p['a'] + p['q'] - phi_minor_full[k + 1])
                phi_major_full = _solve_riccati_torch(
                    coef_fn, p['epslon_0'], p['q_0'], p['c_0'], Ndt, dt)
            else:
                phi_major_full = torch.as_tensor(self.mfg.phi_0, dtype=torch.float64)
        return phi_minor_full[:-1], phi_major_full[:-1]

    def _meanfield_xbar(self, p, phi_minor, x0_hat, xbar0, dt):
        """
        Differentiable forward-Euler solve of the mean-field equation
        (mfg.py eq 4.5 / paper eq 3.9):
            d xbar_t = (a+q-phi_t) * ((F-1)*xbar_t + G*x0_t) dt
        seeded at xbar0, driven by the (soft) major trajectory x0_hat.
        phi_minor, x0_hat: (Ndt,) aligned to xt. Returns (Ndt,) xbar_mf,
        also aligned to xt (xbar_mf[0] == xbar0).
        """
        Ndt = x0_hat.shape[0]
        F = 1 - p['G']
        coef = p['a'] + p['q'] - phi_minor                                  # (Ndt,)
        xbar = [None] * Ndt
        xbar[0] = xbar0
        for k in range(Ndt - 1):
            xbar[k + 1] = xbar[k] + coef[k] * ((F - 1) * xbar[k] + p['G'] * x0_hat[k]) * dt
        return torch.stack(xbar)

    @staticmethod
    def _param_confidence(raw, recent_params):
        """
        Continuous confidence-in-parameters score in [0,1], generalizing the
        boolean params_stable check in fit(): for each unknown parameter,
        1.0 once its relative range over the last 5 EM iterations has settled
        below a tight tolerance, 0.0 while it's still moving by more than a
        loose one, linearly interpolated between -- then the MIN over all
        unknown params (only as confident as the least-converged one, same
        "all params must qualify" spirit as the old `all(...)` check).
        `raw` is the dict of raw (unconstrained) tensors being estimated --
        only used to know which names to check; values come from recent_params.
        """
        if not raw:
            return 1.0
        if len(recent_params) < 5:
            return 0.0
        TIGHT, LOOSE = 1e-3, 0.05  # relative-spread thresholds: beta=1 / beta=0
        betas = []
        for name in raw:
            vals = [r[name] for r in recent_params]
            spread = max(vals) - min(vals)
            scale = max(abs(vals[-1]), 1e-8)
            rel = spread / scale
            betas.append(float(np.clip((LOOSE - rel) / (LOOSE - TIGHT), 0.0, 1.0)))
        return min(betas)

    def _xbar_signals(self, p, phi_minor, x0_hat, xbar_emp, dt):
        """
        Returns (xbar_drift, xbar_control) -- see module docstring. Drift is
        always the empirical mean. Control is a confidence-weighted blend
                 beta*xbar_mf + (1-beta)*xbar_emp
        where xbar_mf is eq (4.5) self-consistently solved from the CURRENT
        parameter estimates and beta = self._meanfield_beta (see
        _param_confidence / fit()) -- at beta=0 this is identical to xbar_emp
        (no ODE solve needed, skipped), at beta=1 it's the pure
        self-consistent mean field. Subclasses with a genuinely different
        xbar_control source (e.g. XBarObservedEstimator, UBarObservedEstimator)
        override this method entirely.
        """
        beta = self._meanfield_beta
        if beta <= 0.0:
            return xbar_emp, xbar_emp
        xbar_mf = self._meanfield_xbar(p, phi_minor, x0_hat, xbar_emp[0], dt)
        xbar_control = beta * xbar_mf + (1 - beta) * xbar_emp if beta < 1.0 else xbar_mf
        return xbar_emp, xbar_control

    def _loglik_major(self, p, phi_major, xbar_drift, xbar_control, xt, xtp1, dt, sig_M):
        # eq (2.1)/(3.5) drift term reverts to the actual population mean;
        # eq (4.1) optimal control responds to the mean-field limit xbar_t.
        drift = p['a_0'] * (xbar_drift.unsqueeze(0) - xt)
        ctrl = (p['q_0'] - phi_major) * (xbar_control.unsqueeze(0) - xt)
        mu = xt + (drift + ctrl) * dt
        return dist.Normal(mu, sig_M).log_prob(xtp1).sum(dim=1)

    def _loglik_minor(self, p, phi_minor, xbar_drift, xbar_control, x0_hat, xt, xtp1, wt, dt, sig_m):
        if self.leave_one_out:
            x0_hat_i = x0_hat.unsqueeze(0) - wt * xt
        else:
            x0_hat_i = x0_hat.unsqueeze(0).expand_as(xt)
        # eq (2.3)/(3.7) drift term reverts to the actual market state;
        # eq (4.3) optimal control responds to the mean-field market state.
        # The major-bank component (G*x0_hat_i) is the same in both -- there's
        # only one major bank, so no drift-vs-meanfield distinction applies to it.
        market_drift = (1 - p['G']) * xbar_drift.unsqueeze(0) + p['G'] * x0_hat_i
        market_control = (1 - p['G']) * xbar_control.unsqueeze(0) + p['G'] * x0_hat_i
        drift = p['a'] * (market_drift - xt)
        ctrl = (p['q'] - phi_minor) * (market_control - xt)
        mu = xt + (drift + ctrl) * dt
        return dist.Normal(mu, sig_m).log_prob(xtp1).sum(dim=1)

    def _extra_M_term(self, p, phi_minor, phi_major, xbar_control, x0_hat, dt):
        # Additional term for the major bank's contribution to the log-likelihood.
        # This is a placeholder for any additional terms that might be needed.
        return 0.0

    def fit(self, X, true_major_idx=None, n_em_iters=50, n_inner_E_steps=20,
            n_inner_M_steps=20, init_theta=None, verbose=False):
        """
        X : (N+1, Ndt+1) observed agent trajectories, label-agnostic order.
        returns : (major_prob (N+1,) tensor, fitted params dict[name->float], n_steps)
        """
        dt, Ndt = self.mfg.dt, self.mfg.Ndt
        X = torch.as_tensor(np.asarray(X), dtype=torch.float64)
        Np1, _ = X.shape
        N = Np1 - 1
        true_major = X[true_major_idx] if true_major_idx is not None else None

        sig_M = self.mfg.sigma_0 * dt**0.5
        sig_m = self.mfg.sigma * dt**0.5

        xt, xtp1 = X[:, :-1], X[:, 1:]

        theta = (torch.zeros(Np1, dtype=torch.float64) if init_theta is None
                 else torch.as_tensor(init_theta, dtype=torch.float64).clone())
        theta.requires_grad_(True)
        opt_theta = torch.optim.Adam([theta], lr=self.lr_E)
        opt_M = torch.optim.Adam(list(self.raw.values()), lr=self.lr_M) if self.raw else None
        # plain per-em_iter multiplicative decay, gamma=1.0 is a no-op.
        sched_theta = torch.optim.lr_scheduler.ExponentialLR(opt_theta, gamma=self.lr_decay)
        sched_M = (torch.optim.lr_scheduler.ExponentialLR(opt_M, gamma=self.lr_decay)
                   if opt_M is not None else None)

        recent_params = []
        n_steps = n_em_iters * n_inner_E_steps
        step = 0

        for em_iter in range(n_em_iters + 1):
            tau = 1.0 if not self.temp_anneal else max(0.3, 1.2 - step / n_steps)

            # ============================= E-step ============================
            # freeze all current param estimates (detached), ascend theta only.
            p_frozen = self._param_values(detach=True)
            phi_minor_f, phi_major_f = self._riccati(p_frozen, track_grad=False)

            # xbar_control: refreshed once per EM iter, not per inner E-step --
            # the eq (4.5) solve is an O(Ndt) sequential recursion, and w is
            # already moving every inner step via the live xbar_emp/x0_hat
            # used for the drift term, so a per-em_iter snapshot of the major
            # trajectory is enough to seed it without re-solving 20x per iter.
            with torch.no_grad():
                w_snap = torch.softmax(theta, dim=0)
                x0_hat_snap = (w_snap.unsqueeze(1) * xt).sum(dim=0)
                xbar_emp_snap = ((1 - w_snap).unsqueeze(1) * xt).sum(dim=0) / N
                _, xbar_ctrl_E = self._xbar_signals(p_frozen, phi_minor_f, x0_hat_snap, xbar_emp_snap, dt)

            for _ in range(n_inner_E_steps):
                w = torch.softmax(theta / tau, dim=0)
                wt = w.unsqueeze(1)
                x0_hat = (wt * xt).sum(dim=0)
                xbar_emp = ((1 - w).unsqueeze(1) * xt).sum(dim=0) / N

                Lmaj = self._loglik_major(p_frozen, phi_major_f, xbar_emp, xbar_ctrl_E, xt, xtp1, dt, sig_M)
                Lmin = self._loglik_minor(p_frozen, phi_minor_f, xbar_emp, xbar_ctrl_E, x0_hat, xt, xtp1, wt, dt, sig_m)
                J = (w * Lmaj + (1 - w) * Lmin).sum()
                # normalized to [0,1] by H's own ceiling log(Np1) (max entropy,
                # attained at uniform w) so lam_entropy means the same thing
                # regardless of how many agents (N+1) are in play -- see
                # solver.py history / conversation for the derivation.
                H_w = -(w * torch.log(w + 1e-12)).sum() / np.log(Np1)
                loss = -(J - self.lam_entropy * H_w)
                opt_theta.zero_grad(); loss.backward()
                self.history_E_grad.append(theta.grad.norm().item())
                opt_theta.step()
                self.history_E.append(J.item())

            # ============================= M-step ============================
            # freeze w, jointly ascend every currently-unknown raw parameter.
            with torch.no_grad():
                w_frozen = torch.softmax(theta, dim=0)
                wt_frozen = w_frozen.unsqueeze(1)
                x0_hat_frozen = (wt_frozen * xt).sum(dim=0)
                xbar_emp_frozen = ((1 - w_frozen).unsqueeze(1) * xt).sum(dim=0) / N

            if opt_M is not None:
                for _ in range(n_inner_M_steps):
                    p = self._param_values(detach=False)
                    phi_minor, phi_major = self._riccati(p, track_grad=True)
                    # recomputed every M-substep (tracked) so xbar_control's
                    # dependence on the current a/q/G/phi estimates -- whenever
                    # beta > 0 -- carries proper gradient into the M-step.
                    _, xbar_ctrl_M = self._xbar_signals(p, phi_minor, x0_hat_frozen, xbar_emp_frozen, dt)
                    Lmaj = self._loglik_major(p, phi_major, xbar_emp_frozen, xbar_ctrl_M, xt, xtp1, dt, sig_M)
                    Lmin = self._loglik_minor(p, phi_minor, xbar_emp_frozen, xbar_ctrl_M, x0_hat_frozen, xt, xtp1, wt_frozen, dt, sig_m)
                    J = (w_frozen * Lmaj + (1 - w_frozen) * Lmin).sum()
                    # w-independent, M-step-only extra term -- see
                    # _extra_M_term docstring. No-op in the base class.
                    J = J + self._extra_M_term(p, phi_minor, phi_major, xbar_ctrl_M, x0_hat_frozen, dt)
                    loss = -J
                    opt_M.zero_grad(); loss.backward()
                    gnorm = torch.nn.utils.clip_grad_norm_(list(self.raw.values()), max_norm=5.0)
                    self.history_M_grad.append(gnorm.item())
                    opt_M.step()
                    self.history_M.append(J.item())
            else:
                # nothing unknown -- still record J so history_M stays meaningful
                with torch.no_grad():
                    p = self._param_values(detach=True)
                    phi_minor, phi_major = self._riccati(p, track_grad=False)
                    _, xbar_ctrl_M = self._xbar_signals(p, phi_minor, x0_hat_frozen, xbar_emp_frozen, dt)
                    Lmaj = self._loglik_major(p, phi_major, xbar_emp_frozen, xbar_ctrl_M, xt, xtp1, dt, sig_M)
                    Lmin = self._loglik_minor(p, phi_minor, xbar_emp_frozen, xbar_ctrl_M, x0_hat_frozen, xt, xtp1, wt_frozen, dt, sig_m)
                    self.history_M.append((w_frozen * Lmaj + (1 - w_frozen) * Lmin).sum().item())

            sched_theta.step()
            if sched_M is not None:
                sched_M.step()

            # --- early stop: w committed AND every unknown param stable ---
            with torch.no_grad():
                current_vals = {name: self._param_values(detach=True)[name].item() for name in self.raw}
            recent_params.append(current_vals)
            if len(recent_params) > 5:
                recent_params.pop(0)
            params_stable = (not self.raw) or (
                len(recent_params) == 5 and all(
                    (max(r[name] for r in recent_params) - min(r[name] for r in recent_params)) < 1e-3
                    for name in self.raw
                )
            )
            # monotonic confidence ratchet -- see _xbar_signals / _param_confidence.
            self._meanfield_beta = max(
                self._meanfield_beta, self._param_confidence(self.raw, recent_params)
            )
            if w_frozen.max().item() >= 0.98 and params_stable:
                if verbose:
                    tail = "  ".join(f"{k}={v:.4f}" for k, v in current_vals.items())
                    print(f"  early stop at EM iter {em_iter}: max(w)={w_frozen.max().item():.4f}  {tail} beta={self._meanfield_beta:.3f}")
                break

            if verbose:
                am = torch.softmax(theta, dim=0).argmax().item()
                tail = "  ".join(f"{k}={v:.4f}" for k, v in current_vals.items())
                print(f"  EM iter {em_iter:3d}  J={self.history_E[-1]:.4e}  argmax={am}  {tail} beta={self._meanfield_beta:.3f}")
            if step % 10 == 0:
                with torch.no_grad():
                    x0_hat_full = (w.unsqueeze(1) * X).sum(dim=0)
                self.major_snapshots.append((step, x0_hat_full.numpy()))
            step += 1

        with torch.no_grad():
            major_prob = torch.softmax(theta, dim=0)
            fitted = {name: self._param_values(detach=True)[name].item() for name in self.raw}

        self._X, self._true_major, self._true_major_idx = X, true_major, true_major_idx
        self.major_prob, self.fitted_params, self.n_steps_taken = major_prob, fitted, step
        return major_prob, fitted, step

    def estimated_xbar(self):
        """
        Mean-field trajectories at the fitted solution -- call after fit().
        Returns (xbar_emp, xbar_mf), each a (Ndt,) numpy array aligned to xt
        (i.e. to X[:, :-1]):
          - xbar_emp: the fitted-w-weighted empirical mean of minor agents.
          - xbar_mf:  the ESTIMATED mean field -- eq (4.5) self-consistently
            solved from the fitted parameters and the fitted soft major
            trajectory x0_hat (same construction xbar_control uses inside
            fit(), but at the terminal/fitted params rather than a mid-
            training blend).
        Compare against the TRUE mean field (e.g. utility.make_example's
        second return value -- mfg.simulate's x_bar[0], the actual eq (4.5)
        path driven by the real major trajectory and true parameters) only
        available in simulation studies where ground truth is known.
        """
        if not hasattr(self, '_X'):
            raise RuntimeError("call fit() first")
        dt = self.mfg.dt
        xt = self._X[:, :-1]
        N = self._X.shape[0] - 1
        with torch.no_grad():
            w = self.major_prob
            x0_hat = (w.unsqueeze(1) * xt).sum(dim=0)
            xbar_emp = ((1 - w).unsqueeze(1) * xt).sum(dim=0) / N
            p = self._param_values(detach=True)
            phi_minor, _ = self._riccati(p, track_grad=False)
            xbar_mf = self._meanfield_xbar(p, phi_minor, x0_hat, xbar_emp[0], dt)
        return xbar_emp.numpy(), xbar_mf.numpy()

    # ------------------------------------------------------------------
    # Diagnostics -- separated from fit() so they can be (re)plotted without
    # re-running the optimization.
    # ------------------------------------------------------------------
    def plot_loss(self, path='loss_gd_em.png'):
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        axes[0].plot(self.history_E)
        axes[0].set_xlabel('E-step gradient step (cumulative)'); axes[0].set_ylabel('J')
        axes[0].set_title('E-step loss (theta ascent, params fixed)')
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(self.history_M, color='tab:orange')
        axes[1].set_xlabel('M-step gradient step (cumulative)'); axes[1].set_ylabel('J')
        axes[1].set_title('M-step loss (params ascent, w fixed)')
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(fig)
        print(f"  saved loss plot to {path}")

    def plot_grad_norm(self, path='grad_norm_gd_em.png'):
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        axes[0].plot(self.history_E_grad)
        axes[0].set_xlabel('E-step gradient step (cumulative)'); axes[0].set_ylabel(r'$\|\nabla_\theta J\|$')
        axes[0].set_title('E-step gradient norm (theta)')
        axes[0].set_yscale('log'); axes[0].grid(True, alpha=0.3)
        axes[1].plot(self.history_M_grad, color='tab:orange')
        axes[1].set_xlabel('M-step gradient step (cumulative)'); axes[1].set_ylabel(r'$\|\nabla_{params} J\|$ (pre-clip)')
        axes[1].set_title('M-step gradient norm (params)')
        axes[1].set_yscale('log'); axes[1].grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(fig)
        print(f"  saved gradient norm plot to {path}")

    def plot_trajectory(self, path='major_trajectory_gd_em.png'):
        if not self.major_snapshots:
            return
        with torch.no_grad():
            x0_hat_terminal = (self.major_prob.unsqueeze(1) * self._X).sum(dim=0).numpy()

        t_axis = np.arange(self._X.shape[1])
        fig, ax = plt.subplots(figsize=(11, 6))
        cmap = plt.cm.viridis
        n_snap = len(self.major_snapshots)
        for i, (snap_step, x0_hat_snap) in enumerate(self.major_snapshots):
            ax.plot(t_axis, x0_hat_snap, color=cmap(i / max(n_snap - 1, 1)), alpha=0.6, linewidth=1)

        if self._true_major is not None:
            ax.plot(t_axis, self._true_major.numpy(), color='black', linewidth=2.5,
                     label=f'True major (agent {self._true_major_idx})')
        ax.plot(t_axis, x0_hat_terminal, color='red', linewidth=2, linestyle='--',
                 label='Terminal estimate (x0_hat)')

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(
            vmin=self.major_snapshots[0][0], vmax=self.major_snapshots[-1][0]))
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label='training step')
        ax.set_xlabel('Time step'); ax.set_ylabel('State value')
        ax.set_title('Major agent estimate: dynamics over training vs true')
        ax.legend(loc='upper left'); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(fig)
        print(f"  saved major trajectory dynamics plot to {path}")


class UObservedEstimator(MajorAgentEstimator):
    """
    TIER 2.5 (Case 2 from conversation): the control u_t is directly observed
    for EVERY agent (major and minor), on top of the states X -- e.g. actual
    borrowing/lending actions were logged, not just reserve balances. Adds a
    THIRD, noise-free estimating equation on top of Lmaj/Lmin: the optimal
    control laws (eq 4.1/4.3) are DETERMINISTIC given (a,q,G,phi,phi_0) and
    the market state -- q enters them LINEARLY, unlike phi_minor's own ODE
    where q only appears as q^2 (see conversation: that's exactly why
    TIER-1's x-bar-only info left an exact two-fold (a,q)<->(a+2q,-q)
    ambiguity that this does not).

    Deliberately does NOT touch or subclass around MajorAgentEstimator's
    fit() -- fit() calls self._loglik_minor(...)/self._loglik_major(...) by
    NAME (not MajorAgentEstimator._loglik_minor(...)), so Python's normal
    virtual dispatch means overriding just those two methods here is enough:
    every E-step, M-step, early-stop check, and _meanfield_beta ramp in the
    inherited fit() automatically picks up the u-based term with ZERO changes
    to MajorAgentEstimator. That's the whole "polymorphism" trick -- fit()
    itself is reused completely unmodified.
    """

    def __init__(self, mfg: MFG, unknown, u, u_sigma=1e-3, u_sigma_loose=0.3, **kwargs):
        super().__init__(mfg, unknown, **kwargs)
        # u: (N+1, Ndt+1) observed control for EVERY agent, same row order and
        # full length as the X later passed to fit() -- NOT pre-sliced. Index
        # 0 is always 0/unset (mfg.py's simulate() loop only ever writes
        # u[:, i+1], computed FROM the state at index i), so the value that
        # actually corresponds to xt[:, k] (state at index k) lives at
        # u[:, k+1], not u[:, k] -- sliced as self.u[:, 1:] below, mirroring
        # xtp1 = X[:, 1:]'s alignment, NOT xt = X[:, :-1]'s.
        self.u = torch.as_tensor(np.asarray(u), dtype=torch.float64)
        # eq (4.1)/(4.3) are exact given the TRUE xbar_control -- but when
        # xbar isn't observed, xbar_control is itself only an approximation
        # (xbar_emp early, the self-consistent eq (4.5) solve once beta
        # ramps -- see _xbar_signals), so comparing u against a prediction
        # built on that approximation has a real, a-independent residual
        # floor even at the true params. u_sigma=1e-3 (appropriate once
        # xbar_control is trustworthy) blows that floor up into a dominant,
        # misleading gradient early on -- see conversation. So the EFFECTIVE
        # sigma tracks the same self._meanfield_beta confidence ramp already
        # driving xbar_control: loose (u_sigma_loose) while beta~0, tightening
        # to u_sigma (tight) as beta->1, rather than a second, disconnected
        # hyperparameter to hand-tune.
        self.u_sigma = u_sigma
        self.u_sigma_loose = u_sigma_loose

    def _u_sigma_eff(self):
        beta = self._meanfield_beta
        return self.u_sigma_loose * (1 - beta) + self.u_sigma * beta

    def _loglik_major(self, p, phi_major, xbar_drift, xbar_control, xt, xtp1, dt, sig_M):
        base = super()._loglik_major(p, phi_major, xbar_drift, xbar_control, xt, xtp1, dt, sig_M)
        u_pred = (p['q_0'] - phi_major) * (xbar_control.unsqueeze(0) - xt)   # eq (4.1)
        u_ll = dist.Normal(u_pred, self._u_sigma_eff()).log_prob(self.u[:, 1:]).sum(dim=1)
        return base + u_ll

    def _loglik_minor(self, p, phi_minor, xbar_drift, xbar_control, x0_hat, xt, xtp1, wt, dt, sig_m):
        base = super()._loglik_minor(p, phi_minor, xbar_drift, xbar_control, x0_hat, xt, xtp1, wt, dt, sig_m)
        if self.leave_one_out:
            x0_hat_i = x0_hat.unsqueeze(0) - wt * xt
        else:
            x0_hat_i = x0_hat.unsqueeze(0).expand_as(xt)
        market_control = (1 - p['G']) * xbar_control.unsqueeze(0) + p['G'] * x0_hat_i
        u_pred = (p['q'] - phi_minor) * (market_control - xt)               # eq (4.3)
        u_ll = dist.Normal(u_pred, self._u_sigma_eff()).log_prob(self.u[:, 1:]).sum(dim=1)
        return base + u_ll


class XBarObservedEstimator(MajorAgentEstimator):
    """
    TIER 1: the true mean-field trajectory xbar_t is directly observed (e.g.
    utility.make_example's x_bar[0] return value -- NOT the noisy finite-N
    empirical mean, the actual eq (4.5) limit path). Owns the observed_xbar
    state and the two methods that only make sense when it's set
    (_meanfield_residual, _loglik_meanfield_ode) entirely by itself --
    MajorAgentEstimator has no notion of observed_xbar at all, by design,
    so each observation TIER lives in its own subclass (mirrors
    UObservedEstimator, UBarObservedEstimator).

    Its extra evidence -- eq (4.5)'s own noise-free residual,
    self._loglik_meanfield_ode -- is W-INDEPENDENT: it doesn't discriminate
    between agents, so it can't be expressed as a per-agent quantity the way
    UObservedEstimator's PER-AGENT u-term can (broadcasting a population-
    level residual identically into every agent row would inflate its
    gradient weight by ~N when summed against (w_i, 1-w_i) -- see
    conversation). Instead it plugs into MajorAgentEstimator.fit()'s
    _extra_M_term hook -- see that method's docstring.
    """

    def __init__(self, mfg: MFG, unknown, observed_xbar, ode_sigma=1e-3, **kwargs):
        if observed_xbar is None:
            raise ValueError("XBarObservedEstimator requires observed_xbar (use "
                              "MajorAgentEstimator directly if xbar isn't observed)")
        super().__init__(mfg, unknown, **kwargs)
        # full-length (Ndt+1,), NOT pre-sliced to xt -- _loglik_meanfield_ode
        # needs both endpoints of each step (xbar_true[:-1] and [1:]).
        self.observed_xbar = torch.as_tensor(np.asarray(observed_xbar), dtype=torch.float64)
        # trust level for the eq (4.5) consistency check -- NOT real noise
        # (the equation is exact), just how tightly to weight the residual
        # relative to the genuinely noisy sig_M/sig_m state-transition terms.
        self.ode_sigma = ode_sigma

    def _xbar_signals(self, p, phi_minor, x0_hat, xbar_emp, dt):
        """xbar_control is exactly the observed ground truth (sliced to
        [:-1] to align with xbar_emp/xt) -- no confidence ramp needed since
        it's not a model-implied proxy."""
        return xbar_emp, self.observed_xbar[:-1]

    def _meanfield_residual(self, p, phi_minor, x0_hat, dt):
        """
        Per-step residual of eq (4.5), built directly from self.observed_xbar's
        OWN increments -- a "collocation" formulation, not the "shooting"
        formulation _meanfield_xbar uses (which integrates forward from an
        initial condition and so compounds parameter error over all Ndt
        steps). Each residual here only spans one step, so every timestep is
        an independent, noise-free constraint on (a, q, G) -- eq (4.5) has no
        diffusion term, it's an exact equality at the true parameters, not a
        probabilistic one. phi_minor, x0_hat: (Ndt,) aligned to xt.
        Returns (Ndt,) residuals (actual increment minus predicted increment).
        """
        xbar_true = self.observed_xbar                                     # (Ndt+1,)
        F = 1 - p['G']
        predicted_incr = (p['a'] + p['q'] - phi_minor) * (
            (F - 1) * xbar_true[:-1] + p['G'] * x0_hat
        ) * dt
        actual_incr = xbar_true[1:] - xbar_true[:-1]
        return actual_incr - predicted_incr

    def _loglik_meanfield_ode(self, p, phi_minor, x0_hat, dt):
        """
        log N(0, ode_sigma) density of the eq (4.5) residual -- an extra
        estimating equation on top of Lmaj/Lmin. See _meanfield_residual and
        the class docstring for why this helps (and its limits: an exact
        two-fold (a,q) vs (a+2q,-q) ambiguity survives from this term alone).
        """
        resid = self._meanfield_residual(p, phi_minor, x0_hat, dt)
        return dist.Normal(0.0, self.ode_sigma).log_prob(resid).sum()

    def _extra_M_term(self, p, phi_minor, phi_major, xbar_control, x0_hat, dt):
        # observed_xbar is guaranteed set by __init__ -- unconditional call,
        # nothing to gate. w is frozen throughout the M-step, so this can't
        # "cheat" by moving w to compensate for wrong params; it can only
        # push a/q/G toward satisfying the exact eq (4.5) constraint.
        return self._loglik_meanfield_ode(p, phi_minor, x0_hat, dt)


class UBarObservedEstimator(MajorAgentEstimator):
    """
    Weaker cousin of UObservedEstimator: instead of the individual control
    u_t^i for EVERY agent, only the AGGREGATE control of the minor-bank
    population, u_bar_t (paper eq 3.4: lim_{N->inf} (1/N) sum_i u_t^i), is
    observed -- e.g. a reported net order-flow number, not per-bank
    transaction logs. u_bar_t excludes the major bank entirely (the major
    has its own separate control u_t^0, eq 3.5); it is a single (Ndt+1,)
    series, not one row per agent.

    (1) Major-agent DETECTION is untouched by this: u_bar_t is already
    averaged over all minors, so it carries no per-agent discriminating
    signal -- whichever minor happens to be "responsible" for the observed
    trajectories is invisible to a population aggregate. The E-step (w's
    softmax ascent over Lmaj/Lmin) is exactly MajorAgentEstimator's, unchanged.

    (2) PARAMETER estimation, in particular separating a and q, DOES benefit:
    averaging the optimal minor control law u_t^{i,*}=(q-phi_t)*[(F*xbar_t+
    G*x0_t)-x_t^i] over i gives u_bar_t=(q-phi_t)*[(F-1)*xbar_t+G*x0_t] --
    still LINEAR in q (unlike phi_t's own Riccati ODE, where q only enters as
    q^2), so comparing observed u_bar_t against this prediction is a second,
    w-independent estimating equation that should break the a+q ridge almost
    as well as full per-agent u observation, just with population-averaging
    noise instead of per-agent noise. See _ubar_residual/_loglik_ubar.

    (3) MEAN-FIELD reconstruction: overrides _xbar_signals to reconstruct
    xbar_t via the paper's eq (3.9), FORWARD-DRIVEN BY THE OBSERVED u_bar_t
    itself, rather than eq (4.5)'s fully model-implied version (which
    substitutes the model's own (q-phi_t)*(...) guess for the control).
    Eq (3.9) needs only 'a' among the unknown minor params -- not 'q' and
    not a Riccati solve at all -- so it's structurally less error-prone
    during early EM than eq (4.5), which needs BOTH 'a' AND 'q' (and a
    correctly-solved phi_t) to be right. The two are mathematically
    identical at the true optimal params (eq 4.5 IS eq 3.9 after
    substituting the optimal control law in); the choice only matters
    transiently, while parameters are still converging -- but that's
    precisely when the M-step gradient quality matters most. See
    _ubar_driven_xbar.

    Like XBarObservedEstimator, the u_bar residual is W-INDEPENDENT (one
    scalar-per-timestep constraint, not per-agent), so it can't be expressed
    as a _loglik_major/_loglik_minor override the way UObservedEstimator's
    PER-AGENT u-term can -- broadcasting it identically into every agent row
    before the (w_i, 1-w_i) sum would inflate its gradient weight by ~N.
    Plugs into MajorAgentEstimator.fit()'s _extra_M_term hook instead.
    """

    def __init__(self, mfg: MFG, unknown, u_bar, ubar_sigma=1e-3, ubar_sigma_loose=0.3, **kwargs):
        super().__init__(mfg, unknown, **kwargs)
        # u_bar: (Ndt+1,) observed AGGREGATE control of the minor-bank
        # population only (eq 3.4) -- NOT per-agent, excludes the major
        # bank's own control entirely. Full-length, not pre-sliced -- index
        # 0 is always 0/unset, same convention as UObservedEstimator's u
        # (mfg.py's simulate() loop only ever writes u[:, i+1]) -- sliced as
        # self.u_bar[1:] wherever it's used, aligned to xtp1/transitions.
        self.u_bar = torch.as_tensor(np.asarray(u_bar), dtype=torch.float64)
        # same "loose early, tight once beta~1" ramp rationale as
        # UObservedEstimator.u_sigma/u_sigma_loose -- see _ubar_sigma_eff.
        self.ubar_sigma = ubar_sigma
        self.ubar_sigma_loose = ubar_sigma_loose

    def _ubar_sigma_eff(self):
        beta = self._meanfield_beta
        return self.ubar_sigma_loose * (1 - beta) + self.ubar_sigma * beta

    def _ubar_driven_xbar(self, p, x0_hat, xbar0, dt):
        """
        Differentiable forward-Euler solve of eq (3.9), driven by the
        OBSERVED aggregate control u_bar_t (contrast _meanfield_xbar, which
        solves eq (4.5) using the model's OWN (q-phi_t)*(...) control-law
        guess instead of real data). x0_hat: (Ndt,) aligned to xt. Returns
        (Ndt,) xbar aligned to xt (xbar[0] == xbar0).
        """
        Ndt = x0_hat.shape[0]
        F = 1 - p['G']
        u_bar = self.u_bar[1:]
        xbar = [None] * Ndt
        xbar[0] = xbar0
        for k in range(Ndt - 1):
            xbar[k + 1] = xbar[k] + (p['a'] * (F - 1) * xbar[k] + p['a'] * p['G'] * x0_hat[k] + u_bar[k]) * dt
        return torch.stack(xbar)

    def _xbar_signals(self, p, phi_minor, x0_hat, xbar_emp, dt):
        """
        Same confidence-ramped blend as the base class, but toward the
        u_bar-driven eq (3.9) reconstruction instead of eq (4.5)'s -- see
        class docstring. Still gated by self._meanfield_beta since early-EM
        'a' guesses (the only param this needs) are unreliable too.
        """
        beta = self._meanfield_beta
        if beta <= 0.0:
            return xbar_emp, xbar_emp
        xbar_u = self._ubar_driven_xbar(p, x0_hat, xbar_emp[0], dt)
        xbar_control = beta * xbar_u + (1 - beta) * xbar_emp if beta < 1.0 else xbar_u
        return xbar_emp, xbar_control

    def _ubar_residual(self, p, phi_minor, xbar_control, x0_hat, dt):
        """
        u_bar_t^obs vs. the model-predicted aggregate minor control law
        (mean of eq 4.3 over i): u_bar_t = (q-phi_t)*[(F-1)*xbar_t+G*x0_t].
        Unlike _meanfield_residual (an xbar-INCREMENT residual), this
        compares observed u_bar directly against its own noise-free
        prediction -- one independent residual per timestep, no
        time-integration to compound error. phi_minor, xbar_control, x0_hat:
        (Ndt,) aligned to xt. Returns (Ndt,) residuals.
        """
        F = 1 - p['G']
        predicted = (p['q'] - phi_minor) * ((F - 1) * xbar_control + p['G'] * x0_hat)
        actual = self.u_bar[1:]
        return actual - predicted

    def _loglik_ubar(self, p, phi_minor, xbar_control, x0_hat, dt):
        resid = self._ubar_residual(p, phi_minor, xbar_control, x0_hat, dt)
        return dist.Normal(0.0, self._ubar_sigma_eff()).log_prob(resid).sum()

    def _extra_M_term(self, p, phi_minor, phi_major, xbar_control, x0_hat, dt):
        # w is frozen throughout the M-step, so this can't "cheat" by moving
        # w to compensate for wrong params -- mirrors XBarObservedEstimator's
        # _extra_M_term / the observed_xbar M-step-only rationale.
        return self._loglik_ubar(p, phi_minor, xbar_control, x0_hat, dt)


# ----------------------------------------------------------------------------
# TEMPORARY / experimental: hardcoded to unknown={'G','a','q'}, testing
# whether reparametrizing the M-step to (a+q, q) instead of (a, q) directly
# fixes the ridge-drift problem found in that case. Does NOT touch
# MajorAgentEstimator or PARAM_SPECS -- self-contained, reuses the class's
# private _riccati/_loglik_* methods via a throwaway unknown=['G','a','q']
# instance (needed so phi/phi_0 actually re-solve as functions of the
# current a,q; they're never read from est.raw here, a and q are tracked as
# a separate (a_plus_q, q) pair of raw tensors instead).
# ----------------------------------------------------------------------------
def fit_G_a_plus_q(mfg: MFG, X, true_major_idx=None, n_em_iters=100,
                    n_inner_E_steps=5, n_inner_M_steps=5, lr_E=0.05, lr_M=0.05,
                    lam_entropy=0.0, leave_one_out=True, verbose=False):
    """
    Returns (major_prob, fitted_dict, n_steps). fitted_dict has keys
    'G', 'a_plus_q', 'q', 'a' (a is derived: a = a_plus_q - q).
    """
    est = MajorAgentEstimator(mfg, unknown=['G', 'a', 'q'], leave_one_out=leave_one_out)
    dt, Ndt = mfg.dt, mfg.Ndt
    X = torch.as_tensor(np.asarray(X), dtype=torch.float64)
    Np1, _ = X.shape
    N = Np1 - 1

    sig_M = mfg.sigma_0 * dt ** 0.5
    sig_m = mfg.sigma * dt ** 0.5
    xt, xtp1 = X[:, :-1], X[:, 1:]

    theta = torch.zeros(Np1, dtype=torch.float64, requires_grad=True)
    opt_theta = torch.optim.Adam([theta], lr=lr_E)

    g_raw = torch.tensor(np.random.uniform(-1, 1), dtype=torch.float64, requires_grad=True)
    s_raw = torch.tensor(np.random.uniform(-1, 1), dtype=torch.float64, requires_grad=True)  # a+q
    q_raw = torch.tensor(np.random.uniform(-1, 1), dtype=torch.float64, requires_grad=True)  # q
    opt_M = torch.optim.Adam([g_raw, s_raw, q_raw], lr=lr_M)

    G_spec = PARAM_SPECS['G']
    S_spec = _Softplus()  # a+q > 0, same reasoning as a alone previously

    def current_params(detach):
        g_, s_, q_ = (t.detach() if detach else t for t in (g_raw, s_raw, q_raw))
        G = G_spec.to_value(g_)
        s = S_spec.to_value(s_)
        a = s - q_
        return {'G': G, 'a': a, 'a_0': mfg.a_0, 'q': q_, 'q_0': mfg.q_0,
                'epslon': mfg.epslon, 'epslon_0': mfg.epslon_0, 'c': mfg.c, 'c_0': mfg.c_0}, s, q_

    history_E, history_M, recent = [], [], []
    step = 0

    for em_iter in range(n_em_iters + 1):
        p_frozen, _, _ = current_params(detach=True)
        phi_minor_f, phi_major_f = est._riccati(p_frozen, track_grad=False)

        for _ in range(n_inner_E_steps):
            w = torch.softmax(theta, dim=0)
            wt = w.unsqueeze(1)
            x0_hat = (wt * xt).sum(dim=0)
            xbar = ((1 - w).unsqueeze(1) * xt).sum(dim=0) / N
            # unsplit (xbar used for both drift and control) -- this function
            # is temporary/experimental and out of scope for the drift-vs-
            # meanfield split, see MajorAgentEstimator.fit() for that.
            Lmaj = est._loglik_major(p_frozen, phi_major_f, xbar, xbar, xt, xtp1, dt, sig_M)
            Lmin = est._loglik_minor(p_frozen, phi_minor_f, xbar, xbar, x0_hat, xt, xtp1, wt, dt, sig_m)
            J = (w * Lmaj + (1 - w) * Lmin).sum()
            H_w = -(w * torch.log(w + 1e-12)).sum() / np.log(Np1)
            loss = -(J - lam_entropy * H_w)
            opt_theta.zero_grad(); loss.backward(); opt_theta.step()
            history_E.append(J.item())

        with torch.no_grad():
            w_frozen = torch.softmax(theta, dim=0)
            wt_frozen = w_frozen.unsqueeze(1)
            x0_hat_frozen = (wt_frozen * xt).sum(dim=0)
            xbar_frozen = ((1 - w_frozen).unsqueeze(1) * xt).sum(dim=0) / N

        for _ in range(n_inner_M_steps):
            p, s_val, q_val = current_params(detach=False)
            phi_minor, phi_major = est._riccati(p, track_grad=True)
            Lmaj = est._loglik_major(p, phi_major, xbar_frozen, xbar_frozen, xt, xtp1, dt, sig_M)
            Lmin = est._loglik_minor(p, phi_minor, xbar_frozen, xbar_frozen, x0_hat_frozen, xt, xtp1, wt_frozen, dt, sig_m)
            J = (w_frozen * Lmaj + (1 - w_frozen) * Lmin).sum()
            loss = -J
            opt_M.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([g_raw, s_raw, q_raw], max_norm=5.0)
            opt_M.step()
            history_M.append(J.item())

        with torch.no_grad():
            p_now, s_now, q_now = current_params(detach=True)
            cur = {'G': p_now['G'].item(), 'a_plus_q': s_now.item(), 'a': s_now.item()-q_now.item(), 'q': q_now.item()}
        recent.append(cur)
        if len(recent) > 5:
            recent.pop(0)
        stable = len(recent) == 5 and all(
            (max(r[k] for r in recent) - min(r[k] for r in recent)) < 1e-3 for k in cur
        )
        if w_frozen.max().item() >= 0.99 and stable:
            if verbose:
                print(f"  early stop at EM iter {em_iter}: max(w)={w_frozen.max().item():.4f}  {cur}")
            break
        if verbose:
            am = torch.softmax(theta, dim=0).argmax().item()
            print(f"  EM iter {em_iter:3d}  J={history_E[-1]:.4e}  argmax={am}  {cur}")
        step += 1

    with torch.no_grad():
        major_prob = torch.softmax(theta, dim=0)
        p_final, s_final, q_final = current_params(detach=True)
        fitted = {'G': p_final['G'].item(), 'a_plus_q': s_final.item(),
                  'q': q_final.item(), 'a': p_final['a'].item()}

    return major_prob, fitted, step


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

    print("=== MajorAgentEstimator: G unknown, a/a_0/q/q_0/epslon known ===")
    est = MajorAgentEstimator(mfg, unknown=['G'], lam_entropy=20.0)
    prob, fitted, n_steps = est.fit(X, true_major_idx=true_idx, n_em_iters=100,
                                     n_inner_E_steps=5, n_inner_M_steps=5, verbose=True)
    pred = int(prob.argmax().item())
    print(f"predicted={pred}  true={true_idx}  correct={pred==true_idx}  "
          f"fitted={fitted}  G_true={cfg.G}")
    print("top-3 prob:", np.round(np.sort(prob.numpy())[::-1][:3], 3))
    est.plot_loss(); est.plot_grad_norm(); est.plot_trajectory()
