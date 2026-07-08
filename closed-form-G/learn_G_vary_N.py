#                                [Learn G and w] 
# Learns the major agents contribution to the market state which enters through the 
# parameter G alongside a soft assignment w to detection of the major agent in the 
# system. We learn G and w jointly on the same objective J(w)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import torch
import torch.distributions as dist
from mfg import MFG, MFG_config

# ----------------------------------------------------------------------------
# helper: simulates one system and permutes to remove positional label leakage
# ----------------------------------------------------------------------------
def make_example(mfg: MFG, N: int, seed=None):
    if seed is not None:
        np.random.seed(seed)
    # acquire a single simulated system
    x_bar, x_major, x_minor = mfg.simulate(N=N, N_sim=1, do_plot=False)
    # stack simulated universe 0 vertically with major on top: dim (N+1, Ndt+1)
    X_ordered = np.vstack([x_major[0:1, :], x_minor[0, :, :]])
    # vector of a shuffled order
    perm = np.random.permutation(N + 1)
    # actually reorders according to perm
    X = X_ordered[perm]
    # np.where returns a tuple: tuple[0] = np.array of indexes where cond. is true
    true_idx = int(np.where(perm == 0)[0][0])
    return X, x_bar[0], true_idx 


# ----------------------------------------------------------------------------
# L_Objective: objective of G and w which we seek to learn. 
# condition: X must be either numpy array or tensor
# alpha used to regularize to favour lower entropy
# ----------------------------------------------------------------------------
def L_objective(theta, G: float, X, mfg: MFG, alpha=100):
    # stores state at arbitrary time t and tp1
    X = torch.as_tensor(X, dtype=torch.float64)
    xt   = X[:, :-1]
    xtp1 = X[:, 1:]
    # number of minor agents in the system 
    N = X.shape[0] - 1
    # time step in system
    dt = mfg.dt
    # backward major ang minor ricatti equations
    phi  = torch.as_tensor(mfg.phi[:-1],   dtype=torch.float64)
    phi_0 = torch.as_tensor(mfg.phi_0[:-1], dtype=torch.float64)
    # system paramters from MFG configured in make example
    a, q, a0, q0 = mfg.a, mfg.q, mfg.a_0, mfg.q_0
    sig_m = mfg.sigma   * dt**0.5
    sig_M = mfg.sigma_0 * dt**0.5
    # softmax the theta logits to get probability that agent is major
    w  = torch.softmax(theta, dim=0)                       # (Np1,)
    wt = w.unsqueeze(1)                                    # (Np1,1)
    # xhar_hat(w): mean of minor agents in system dim (Ndt, )
    xbar = ((1 - w).unsqueeze(1) * xt).sum(dim=0) / N    
    # soft major with dependence on w 
    x0_hat = (wt * xt).sum(dim=0)                          # (Ndt,)
    # mu major: reverts to x_bar
    mu_maj = xt + (a0 + (q0 - phi_0)) * (xbar.unsqueeze(0) - xt) * dt
    Lmaj   = dist.Normal(mu_maj, sig_M).log_prob(xtp1).sum(dim=1)   # (Np1,)
    # minor reverts to leave-one-out market: G-coupled
    x0_loo   = x0_hat.unsqueeze(0) - wt * xt                        # (Np1,Ndt)
    market_i = (1 - G) * xbar.unsqueeze(0) + G * x0_loo             # (Np1,Ndt)
    mu_min   = xt + (a + (q - phi)) * (market_i - xt) * dt
    Lmin     = dist.Normal(mu_min, sig_m).log_prob(xtp1).sum(dim=1) # (Np1,)
    # return the L_objective
    L = (w * Lmaj + (1 - w) * Lmin).sum()
    # entropy of the softmax assignment: H(w) = -sum w_i log w_i
    H = -(w * torch.log(w + 1e-12)).sum() # added +1e-12 guards log(0)
    return L - alpha * H


# ----------------------------------------------------------------------------
# solve_G_star: solve for closed form G at fixed assignment w
# ----------------------------------------------------------------------------
def solve_G_star(w, xt, xtp1, gamma, N: int):
    # compute x_bar(w) and leave-one-out major agent  
    wt = w.unsqueeze(1)                                    # (Np1,1)
    xbar   = ((1 - w).unsqueeze(1) * xt).sum(dim=0) / N    # (Ndt,)
    x0_hat = (wt * xt).sum(dim=0)                          # (Ndt,)
    x0_loo = x0_hat.unsqueeze(0) - wt * xt                 # (Np1,Ndt)
    # terms in G* difference between loo-major and xbar
    M = x0_loo - xbar.unsqueeze(0)                         # (Np1,Ndt)
    g = gamma.unsqueeze(0)                                 # (1,Ndt)
    # residual: difference between xtp1 and xt + terms
    resid0 = xtp1 - xt - g * (xbar.unsqueeze(0) - xt)      # (Np1,Ndt)
    # weights for being a minor agent 
    weight = (1 - w).unsqueeze(1)                          # (Np1,1)
    # optimal G numerator and denominator
    num = (weight * g * M * resid0).sum()
    den = (weight * g * g * M * M).sum()
    # return optimal G and its denominator 
    return num / den, den


# ----------------------------------------------------------------------------
# learn_G_and_w: EM-style coordinate ascent  (w-step  <->  G-step)
# ----------------------------------------------------------------------------
def learn_G_and_w(mfg, X, init_G, outer_it=100, w_steps=15, lr=0.05, alpha=100):
    # mfg : MFG with solve_ODE() called.
    # X : (Np1, Ndt+1) permuted trajectories.
    X = torch.as_tensor(np.asarray(X), dtype=torch.float64)
    N = X.shape[0] - 1
    dt = mfg.dt
    # acquire two backward ricatti equations, not G free 
    phi   = torch.as_tensor(mfg.phi[:-1],   dtype=torch.float64)   # (Ndt,)
    phi_0 = torch.as_tensor(mfg.phi_0[:-1], dtype=torch.float64)   # (Ndt,)
    # critical parameter for solve_G_star
    a, q = mfg.a, mfg.q 
    # term in optimal G, minor reversion 
    gamma = (a + (q - phi)) * dt                                   # (Ndt,)
    # state at time t and tp1
    xt   = X[:, :-1]      # (Np1, Ndt)
    xtp1 = X[:, 1:]       # (Np1, Ndt)
    # logits to learn, all initialized at 0
    theta = torch.zeros(N+1, dtype=torch.float64)
    # history of opt. liklihood, G, and w accuracy.
    history = []
    # initialized G and previous likelihood value
    G = float(init_G)
    L_prev = -float("inf")

    # total number of learn-G/learn-w steps: 
    for it in range(outer_it):
        # warm start, learn w starting from previous theta not all zeros 
        # detach prev iteration computational graph, clone, start tracking grad.
        theta = theta.detach().clone().requires_grad_(True)
        # adam optimzer for theta with lr=0.05. gradient decent with extra info.
        opt = torch.optim.Adam([theta], lr=lr)
        # number of steps to reach optimal w (curr = 400)
        for _ in range(w_steps):
            # computes the current loss
            loss = -L_objective(theta, G, X, mfg, alpha)
            # clears, computes gradient (Audo. Diff,) wrt theta, updates gradient
            opt.zero_grad(); loss.backward(); opt.step()
        # theta is only the most optimal value after 400 steps,   
        theta = theta.detach()
        w = torch.softmax(theta, dim=0)
        # computes the closed form G-step at learned w
        G_star, den = solve_G_star(w, xt, xtp1, gamma, N)
        # optimal G*, not clamped to the region between 0.01 and 0.99: free
        G = G_star.item()
        # likelihood at outloops current learned theta (w) and G: non-decreasing
        L_curr = L_objective(theta, G, X, mfg, alpha).item()
        history.append((L_curr, G, den.item()))
        # convergence check: has L stopped moving, break if it has stopped
        if abs(L_curr - L_prev) < 1e-7:
            break
        L_prev = L_curr
    # return the optimal softmax assignment w, the optimal G* and history
    return torch.softmax(theta, dim=0), G, history

# ----------------------------------------------------------------------------
# across_seed_opt_G_and_w: reports stats including optimal G* mean & RMSE
# of G* and the detection accuracy of w across different seeds. 
# ----------------------------------------------------------------------------
def across_seed_opt_G_and_w(mfg, N, n_seed=50, init_G=0.9, seed0=0, alpha=100):
    # number of correctly identified major agent across seeds
    correct = 0
    # history of optimal G* across different seeds
    optimal_Gs = []
    # loop across the various seeds
    for s in range(n_seed):
        # make_example and track true_inx of major agent 
        X, _, true_idx = make_example(mfg, N, seed=seed0 + s)   
        # compute the learned G and w from our optimization 
        w, G_hat, _ = learn_G_and_w(mfg, X, init_G=init_G, alpha=alpha)
        # the predicted index of the major agent: argmax (index) of w
        pred = int(w.argmax())
        # compute total correctly identified
        correct += (pred == true_idx)
        # add G* to optimal G* history
        optimal_Gs.append(G_hat)

    # true G from mfg used to compute RMSE
    true_G = mfg.G
    # turn list into np.array to compute stats below
    optimal_Gs = np.array(optimal_Gs)
    # mean G*: unbiasedness check, should be around true_G
    mean_Gs = np.mean(optimal_Gs)
    # computes the mse and rmse of the Gs across seeds
    mse  = np.mean((optimal_Gs - true_G)**2)
    rmse = np.sqrt(mse)

    return N, mean_Gs, rmse, (correct / n_seed), optimal_Gs

# ----------------------------------------------------------------------------
# collect_Gs: for convience - return just the raw G* array at fixed N
#             (used by the KDE / distribution plots).
# ----------------------------------------------------------------------------
def collect_Gs(mfg, N, n_seed=100, init_G=0.9, seed0=0, alpha: float = 0.0):
    _, _, _, _, optimal_Gs = across_seed_opt_G_and_w(
        mfg, N, n_seed=n_seed, init_G=init_G, seed0=seed0, alpha=alpha)
    return optimal_Gs
 

def table_over_N(mfg, N_values, n_seed=50, init_G=0.9, alpha=100):
    print(f"{'N':>6} {'mean G*':>10} {'RMSE':>10} {'w accuracy':>12}")
    print("-" * 40)
    rows = []
    true_G = mfg.G

    # one figure: 2 rows x 3 cols = 5 plots (top row 3, bottom row 2) + 1 spare
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()            # so axes[i] indexes the i-th cell

    # iterate across all values of N to test
    for i, N in enumerate(N_values):
        N_out, mean_Gs, rmse, acc, optimal_Gs = across_seed_opt_G_and_w(
            mfg, N, n_seed=n_seed, init_G=init_G, alpha=alpha)
        # append all row information per N
        rows.append((N_out, mean_Gs, rmse, acc))
        print(f"{N_out:>6} {mean_Gs:>10.4f} {rmse:>10.4f} {acc:>12.1%}")
        # collect all optimal G*: used for the visual density 
        Gs = optimal_Gs[np.isfinite(optimal_Gs)]
        ax = axes[i] # plot into THIS cell, not a new figure

        # guard: KDE needs >=2 points with nonzero spread
        if len(Gs) < 2 or np.ptp(Gs) < 1e-9:
            ax.text(0.5, 0.5, f"N={N}\n(insufficient / degenerate G*)",
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'N={N}')
            continue
        # construct KDE density 
        kde  = gaussian_kde(Gs)
        grid = np.linspace(Gs.min() - 0.1, Gs.max() + 0.1, 400)
        ax.plot(grid, kde(grid), linewidth=2, label='KDE (normal)')
        ax.fill_between(grid, kde(grid), alpha=0.25)
        ax.axvline(true_G,    color='red',  linestyle='--', linewidth=2, label=f'true G={true_G}')
        ax.axvline(Gs.mean(), color='blue', linestyle='-',  linewidth=1.5, label=f'mean={Gs.mean():.3f}')
        ax.set_xlabel('G*'); ax.set_ylabel('density')
        ax.set_title(f'N={N}  (RMSE={rmse:.3f}, acc={acc:.0%})')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # hide any unused cells (with 5 N's in a 2x3 grid, the 6th is spare)
    for j in range(len(N_values), len(axes)):
        axes[j].axis('off')

    fig.suptitle(f'G* distributions across N  (true G={true_G}, α={alpha})', fontsize=14)
    plt.tight_layout()
    plt.savefig('G_distributions_over_N.png', dpi=150)
    plt.show()
    return rows

if __name__ == "__main__":
    TRUE_G = 0.5
    # specifies a configuration
    cfg = MFG_config(T=1.0, Ndt=512,
        a_0=2.5, sigma_0=1, c_0=0, epslon_0=10, q_0=1,
        a  =5, sigma  =1, c  =0, epslon  =1.5, q  =1, G=TRUE_G)
    mfg = MFG(cfg); mfg.solve_ODE()

    # prints the resulting table values
    table_over_N(mfg, N_values=[64, 128, 256, 512, 1028], n_seed=100, init_G=0.1, alpha=1000)
