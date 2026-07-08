"""
Three sweeps for detect_major_G_closedform, each over one axis (G_true, N,
lam_entropy) with the other two held fixed at the paper's Section-6 baseline.
Each sweep runs n_seeds trials per value and reports:
    mean G*  -- average fitted G across seeds
    RMSE     -- root-mean-squared error of G* against G_true
    acc(w)   -- fraction of seeds where argmax(w) correctly identified the
                true major agent

Writes three plain-text tables: comparison_G.txt, comparison_N.txt, comparison_lambda.txt
"""

import numpy as np
from mfg import MFG, MFG_config
from solver import make_example, detect_major_G

# --- baseline config (paper Section 6 / Fig B.5-B.6 regime) -----------------
A        = 5
SIGMA_0, Q_0, EPSLON_0, C_0 = 1.0, 1.0, 10.0, 0.0
SIGMA, Q, EPSLON, C         = 1.0, 1.0, 1.5, 0.0
T, NDT   = 1, 512

N_BASE        = 256
G_TRUE_BASE   = 0.5
LAM_BASE      = 50.0

N_SEEDS = 10   # trials per swept value; bump up for tighter mean/RMSE/acc estimates

N_EM_ITERS      = 100
N_INNER_E_STEPS = 5
N_INNER_M_STEPS = 5


def make_mfg(G_true):
    cfg = MFG_config(T=T, Ndt=NDT,
                      a_0=A * G_true, sigma_0=SIGMA_0, c_0=C_0, epslon_0=EPSLON_0, q_0=Q_0,
                      a=A, sigma=SIGMA, c=C, epslon=EPSLON, q=Q,
                      G=G_true)
    mfg = MFG(cfg)
    mfg.solve_ODE()
    return mfg


def summarize(G_hats, correct_flags, steps, G_true):
    G_hats = np.array(G_hats)
    return {
        "mean_G_hat": G_hats.mean(),
        "rmse": np.sqrt(np.mean((G_hats - G_true) ** 2)),
        "acc_w": np.mean(correct_flags),
        "mean_step": np.mean(steps),
    }


def format_table(key_name, rows):
    lines = [f"{key_name:>10} | {'mean G*':>8} | {'RMSE':>8} | {'acc(w)':>8} | {'mean step':>9} | {'n_seeds':>7}"]
    lines.append("-" * 66)
    for r in rows:
        lines.append(f"{r[key_name]:>10.3g} | {r['mean_G_hat']:>8.4f} | {r['rmse']:>8.4f} | "
                      f"{r['acc_w']:>8.2%} | {r['mean_step']:>9.1f} | {r['n_seeds']:>7}")
    return "\n".join(lines)


def print_table(key_name, rows):
    print(format_table(key_name, rows))
    print()


def write_table(path, key_name, rows):
    with open(path, "w") as f:
        f.write(format_table(key_name, rows) + "\n")
    print(f"wrote {path}")


# -----------------------------------------------------------------------------
# Sweep 1: G_true, N and lam_entropy fixed at baseline
# -----------------------------------------------------------------------------
def sweep_G(G_true_values, n_seeds=N_SEEDS):
    rows = []
    for G_true in G_true_values:
        mfg = make_mfg(G_true)
        G_hats, correct_flags, steps = [], [], []
        for seed in range(n_seeds):
            print(f"  [G sweep] G_true={G_true}  seed={seed}", end="\r")
            X, x_bar_obs, true_idx = make_example(mfg, N_BASE, seed=seed)
            prob, G_hat, step = detect_major_G(
                mfg, X, true_idx,
                n_em_iters=N_EM_ITERS, n_inner_E_steps=N_INNER_E_STEPS,
                n_inner_M_steps=N_INNER_M_STEPS, lam_entropy=LAM_BASE, verbose=False,
            )
            G_hats.append(G_hat.item())
            correct_flags.append(int(prob.argmax()) == true_idx)
            steps.append(step)
        row = {"G_true": G_true, "n_seeds": n_seeds, **summarize(G_hats, correct_flags, steps, G_true)}
        rows.append(row)
    print()
    return rows


# -----------------------------------------------------------------------------
# Sweep 2: N, G_true and lam_entropy fixed at baseline
# -----------------------------------------------------------------------------
def sweep_N(N_values, n_seeds=N_SEEDS):
    rows = []
    for N in N_values:
        mfg = make_mfg(G_TRUE_BASE)
        G_hats, correct_flags, steps = [], [], []
        for seed in range(n_seeds):
            print(f"  [N sweep] N={N}  seed={seed}", end="\r")
            X, x_bar_obs, true_idx = make_example(mfg, N, seed=seed)
            prob, G_hat, step = detect_major_G(
                mfg, X, true_idx,fix_phi0=False,
                n_em_iters=N_EM_ITERS, n_inner_E_steps=N_INNER_E_STEPS,
                n_inner_M_steps=N_INNER_M_STEPS, lam_entropy=LAM_BASE, verbose=False,
            )
            G_hats.append(G_hat.item())
            correct_flags.append(int(prob.argmax()) == true_idx)
            steps.append(step)
        row = {"N": N, "n_seeds": n_seeds, **summarize(G_hats, correct_flags, steps, G_TRUE_BASE)}
        rows.append(row)
    print()
    return rows


# -----------------------------------------------------------------------------
# Sweep 3: lam_entropy, G_true and N fixed at baseline.
# Same X reused across all lam_entropy values within a seed, so the comparison
# isolates the effect of lam_entropy from sampling noise.
# -----------------------------------------------------------------------------
def sweep_lambda(lam_values, n_seeds=N_SEEDS):
    mfg = make_mfg(G_TRUE_BASE)
    per_lam = {lam: {"G_hats": [], "correct_flags": [], "steps": []} for lam in lam_values}

    for seed in range(n_seeds):
        X, x_bar_obs, true_idx = make_example(mfg, N_BASE, seed=seed)
        for lam in lam_values:
            print(f"  [lambda sweep] lam={lam}  seed={seed}", end="\r")
            prob, G_hat, step = detect_major_G(
                mfg, X, true_idx,
                n_em_iters=N_EM_ITERS, n_inner_E_steps=N_INNER_E_STEPS,
                n_inner_M_steps=N_INNER_M_STEPS, lam_entropy=lam, verbose=False,
            )
            per_lam[lam]["G_hats"].append(G_hat.item())
            per_lam[lam]["correct_flags"].append(int(prob.argmax()) == true_idx)
            per_lam[lam]["steps"].append(step)

    rows = []
    for lam in lam_values:
        d = per_lam[lam]
        row = {"lam_entropy": lam, "n_seeds": n_seeds,
               **summarize(d["G_hats"], d["correct_flags"], d["steps"], G_TRUE_BASE)}
        rows.append(row)
    print()
    return rows


if __name__ == "__main__":
    # print("=== Sweep 1/3: G_true ===")
    # rows_G = sweep_G([0.1, 0.3, 0.5, 0.7, 0.9])
    # print_table("G_true", rows_G)
    # write_table("comparison_G.txt", "G_true", rows_G)

    print("=== Sweep 2/3: N ===")
    rows_N = sweep_N([64, 128, 256, 512, 1024])
    print_table("N", rows_N)
    write_table("comparison_N.txt", "N", rows_N)

    # print("=== Sweep 3/3: lam_entropy ===")
    # rows_lambda = sweep_lambda([0.0, 5.0, 20.0, 50.0, 100.0, 200])
    # print_table("lam_entropy", rows_lambda)
    # write_table("comparison_lambda.txt", "lam_entropy", rows_lambda)
