#                                [Vary true G]
# Fixes N and sweeps the TRUE coupling strength G, rebuilding the MFG (and
# re-solving the Riccati) at each value. Produces the table AND the overlaid
# KDE plot from the same run.

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from mfg import MFG, MFG_config
from learn_G_and_w import across_seed_opt_G_and_w


# ----------------------------------------------------------------------------
# vary_true_G: one pass per true G. Prints the table row and collects the raw
# G* array, then draws the overlaid KDE from those same arrays.
# base_cfg: dict of all MFG_config args except G.
# ----------------------------------------------------------------------------
def vary_true_G(base_cfg, G_values, N, n_seed=100, init_G=0.9, alpha=0.0, seed0=0):
    print(f"{'true G':>8} {'mean G*':>10} {'RMSE':>10} {'w accuracy':>12}")
    print("-" * 44)

    rows = []
    results = []                       # (trueG, optimal_Gs) stored for the plot

    for trueG in G_values:
        cfg = MFG_config(G=trueG, **base_cfg)          # rebuild at this true G
        mfg = MFG(cfg); mfg.solve_ODE()                # re-solve Riccati for G

        # single estimation pass: returns stats AND the raw G* array
        N_out, mean_Gs, rmse, acc, optimal_Gs = across_seed_opt_G_and_w(
            mfg, N, n_seed=n_seed, init_G=init_G, seed0=seed0, alpha=alpha)

        rows.append((trueG, mean_Gs, rmse, acc))
        results.append((trueG, optimal_Gs))           # reuse the SAME array below
        print(f"{trueG:>8.2f} {mean_Gs:>10.4f} {rmse:>10.4f} {acc:>12.1%}")

    # --- plot from the arrays we already computed (no re-running) ---
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(G_values)))

    for (trueG, Gs), col in zip(results, colors):
        Gs = Gs[np.isfinite(Gs)]
        if len(Gs) < 2 or np.ptp(Gs) < 1e-9:          # KDE needs spread
            ax.axvline(trueG, color=col, linestyle='--', alpha=0.5)
            continue
        kde  = gaussian_kde(Gs)
        grid = np.linspace(Gs.min() - 0.1, Gs.max() + 0.1, 400)
        ax.plot(grid, kde(grid), color=col, linewidth=2,
                label=f'true G={trueG}  (mean {Gs.mean():.3f})')
        ax.axvline(trueG, color=col, linestyle='--', alpha=0.5)

    ax.set_xlabel('G*'); ax.set_ylabel('density')
    ax.set_title(f'G* distribution vs true G  (N={N}, {n_seed} seeds, α={alpha})')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('G_kde_over_true_G.png', dpi=150)
    plt.show()

    return rows


if __name__ == "__main__":
    base_cfg = dict(T=1.0, Ndt=512,
        a_0=2.5, sigma_0=1, c_0=0, epslon_0=10, q_0=1,
        a  =5,   sigma  =1, c  =0, epslon  =1.5, q  =1)
    G_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    # table printed and plot drawn, both from the same estimation runs
    vary_true_G(base_cfg, G_values, N=512, n_seed=50, alpha=1000.0)
