"""
Oracle baseline: the log-likelihood J achieved with FULL knowledge -- the
true major agent (w = one-hot at true_major_idx) and the true physics
parameters (read straight from mfg, nothing estimated).

This is the ceiling J(fitted) is measured against: the gap between oracle_J
and the J an estimator actually converges to tells you how much likelihood
is lost purely from not knowing who's major / what G, a, ... are in
advance -- a diagnostic on the same absolute scale as history_E/history_M,
complementary to the parameter-RMSE metrics in summarize_results.py.
"""

import numpy as np
import torch

from mfg import MFG
from solver import MajorAgentEstimator


def oracle_J(mfg: MFG, X, lam_entropy, true_major_idx: int) -> float:
    """
    J at w = one-hot(true_major_idx), params = mfg's own true values.

    unknown=[] means every parameter is read straight from mfg (nothing
    estimated), and _riccati falls back to mfg's own pre-solved phi/phi_0 --
    exactly the "if we knew everything" setting.
    """
    est = MajorAgentEstimator(mfg, unknown=[], lam_entropy=lam_entropy)
    X = torch.as_tensor(np.asarray(X), dtype=torch.float64)
    Np1, _ = X.shape
    N = Np1 - 1
    dt = mfg.dt
    sig_M = mfg.sigma_0 * dt ** 0.5
    sig_m = mfg.sigma * dt ** 0.5

    xt, xtp1 = X[:, :-1], X[:, 1:]

    w = torch.zeros(Np1, dtype=torch.float64)
    w[true_major_idx] = 1.0
    wt = w.unsqueeze(1)
    x0_hat = (wt * xt).sum(dim=0)
    xbar = ((1 - w).unsqueeze(1) * xt).sum(dim=0) / N

    with torch.no_grad():
        p = est._param_values(detach=True)
        phi_minor, phi_major = est._riccati(p, track_grad=False)
        Lmaj = est._loglik_major(p, phi_major, xbar, xt, xtp1, dt, sig_M)
        Lmin = est._loglik_minor(p, phi_minor, xbar, x0_hat, xt, xtp1, wt, dt, sig_m)
        J = (w * Lmaj + (1 - w) * Lmin).sum()
        H_w = -(w * torch.log(w + 1e-12)).sum() / np.log(Np1)
        lossJ = J - lam_entropy * H_w
        
    return lossJ.item()


if __name__ == "__main__":
    from mfg import MFG_config
    from utility import make_example

    cfg = MFG_config(T=1, Ndt=512, a_0=2.5, sigma_0=1.0, c_0=0.0, epslon_0=10.0,
                      q_0=1.0, a=5, sigma=1.0, c=0.0, epslon=1.5, q=1.0, G=0.5)
    mfg = MFG(cfg)
    mfg.solve_ODE()
    X, x_bar_obs, true_idx = make_example(mfg, N=256, seed=0)
    print(f"oracle_J = {oracle_J(mfg, X, lam_entropy=20.0, true_major_idx=true_idx):.4e}  (true_idx={true_idx})")
