"""
Progressive-relaxation experiment, crossed with a G_true sweep: gradually
declare more MFG parameters "unknown" and see how major-agent-detection
accuracy / parameter-recovery RMSE degrades as the estimator has to identify
more from data instead of being told -- and check whether that degradation
compounds with weak identification at low G (already known to be the
hardest regime for G alone).

Stages (each a strict superset of the previous):
    1. G
    2. G, a, a_0
    3. G, a, a_0, q, q_0
    4. G, a, a_0, q, q_0, c, c_0

G_true grid: 0.1, 0.3, 0.5, 0.7, 0.9

CONFIGS is the cross product (4 stages x 5 G_true values = 20 configs),
numbered 1-20 with G_true as the inner loop (configs 1-5 = stage 1 across
the G grid, 6-10 = stage 2, etc.). --config-id selects one entry directly,
so the SLURM array only ever needs a flat --array=1-20 -- the (stage,
G_true) decoding lives in ONE place (CONFIGS below), not duplicated as bash
arithmetic in the sbatch script.

One process runs ONE config's full n_seeds sweep, writing every seed's
result into ONE file (results_stage{K}_{name}_Gtrue{G}.csv) -- one row per
seed, flushed and fsync'd to disk immediately after that seed finishes, so
nothing is held in memory until the end. On startup, seeds already present
in that config's output file are skipped, so re-running the same command
(e.g. after hitting the walltime limit, or a node failure) RESUMES instead
of redoing finished work.

Run locally:
    python experiment.py --config-id 7      # stage 2, G_true=0.5

Run on a cluster with a SLURM job array, one array task per config (see
run_experiment.sbatch) -- each task writes to a DIFFERENT file, so there's
no shared-file contention between array tasks and all 20 run fully in
parallel.
"""

import argparse
import csv
import os
import time

from mfg import MFG, MFG_config
from utility import make_example
from solver import MajorAgentEstimator, UObservedEstimator

# --- baseline config (paper Section 6 / Fig B.5-B.6 regime) -----------------
A = 5
SIGMA_0, Q_0, EPSLON_0, C_0 = 1.0, 1.0, 10.0, 0.0
SIGMA, Q, EPSLON, C         = 1.0, 1.0, 1.5, 0.0
T, NDT   = 1, 512
N_BASE   = 256

N_SEEDS         = 50
N_EM_ITERS      = 200
N_INNER_E_STEPS = 10
N_INNER_M_STEPS = 5
LAM_ENTROPY     = 110.0   # normalized entropy penalty; validated good at N=256

G_TRUE_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9]

STAGES = [
    (1, "G",               ['G']),
    (2, "G_a",              ['G', 'a']),
    (3, "G_a_q",            ['G', 'a','q']),
    (4, "G_a_q_c",           ['G', 'a','q','c']),
    (5, "G_a_a0",           ['G', 'a', 'a_0']),
    (6, "G_a_a0_q_q0",      ['G', 'a', 'a_0', 'q', 'q_0']),
    (7, "G_a_a0_q_q0_c_c0", ['G', 'a', 'a_0', 'q', 'q_0', 'c', 'c_0']),
    
]

# cross product: (config_id, stage_idx, stage_name, unknown, G_true),
# G_true as the inner loop -- config_id 1-5 = stage 1 across the G grid,
# 6-10 = stage 2, 11-15 = stage 3, 16-20 = stage 4.
CONFIGS = [
    (5 * (stage_idx - 1) + g_i + 1, stage_idx, stage_name, unknown, G_true)
    for stage_idx, stage_name, unknown in STAGES
    for g_i, G_true in enumerate(G_TRUE_VALUES)
]


def config_lookup(config_id):
    return next(c for c in CONFIGS if c[0] == config_id)


def true_values_for(G_true):
    a_0 = A * G_true   # market-clearing baseline (eq 2.8) -- used only to
                        # GENERATE the synthetic data; the estimator is never
                        # told this relationship holds.
    return {
        'G': G_true, 'a': A, 'a_0': a_0, 'q': Q, 'q_0': Q_0,
        'epslon': EPSLON, 'epslon_0': EPSLON_0, 'c': C, 'c_0': C_0,
    }


def make_mfg(G_true):
    tv = true_values_for(G_true)
    cfg = MFG_config(T=T, Ndt=NDT, a_0=tv['a_0'], sigma_0=SIGMA_0, c_0=C_0,
                      epslon_0=EPSLON_0, q_0=Q_0, a=A, sigma=SIGMA, c=C,
                      epslon=EPSLON, q=Q, G=G_true)
    mfg = MFG(cfg)
    mfg.solve_ODE()
    return mfg


def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path


def result_path(stage_idx, stage_name, G_true):
    return _ensure_parent_dir(f"result_7_31/results_stage{stage_idx}_{stage_name}_Gtrue{G_true}.csv")


def result_path_u(stage_idx, stage_name, G_true):
    return _ensure_parent_dir(f"result_7_31/results_u_stage{stage_idx}_{stage_name}_Gtrue{G_true}.csv")


def load_completed_seeds(path):
    if not os.path.exists(path):
        return set()
    with open(path, newline='') as f:
        return {int(row['seed']) for row in csv.DictReader(f)}


def fieldnames_for(unknown):
    fixed = ['seed', 'config_id', 'stage', 'unknown_params', 'G_true',
              'predicted_idx', 'true_idx', 'correct', 'n_steps', 'wall_time_sec']
    # 'G' already has its own fixed 'G_true' column above (the data-generating
    # regime); skip it here to avoid a duplicate column name -- G_fit still
    # gets written normally, it's only the redundant *_true that's dropped.
    true_cols = [f"{p}_true" for p in unknown if p != 'G']
    fit_cols = [f"{p}_fit" for p in unknown]
    return fixed + true_cols + fit_cols


def run_config(config_id, n_seeds=N_SEEDS):
    config_id, stage_idx, stage_name, unknown, G_true = config_lookup(config_id)
    path = result_path(stage_idx, stage_name, G_true)
    done = load_completed_seeds(path)
    fieldnames = fieldnames_for(unknown)
    write_header = not os.path.exists(path)

    mfg = make_mfg(G_true)
    true_values = true_values_for(G_true)
    print(f"=== Config {config_id}: stage {stage_idx} ({stage_name}), "
          f"G_true={G_true}, unknown={unknown} ===", flush=True)
    print(f"  results -> {path}   ({len(done)}/{n_seeds} already done)", flush=True)

    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
            f.flush(); os.fsync(f.fileno())

        for seed in range(n_seeds):
            if seed in done:
                continue
            t0 = time.time()
            X, x_bar_obs, true_idx,_ = make_example(mfg, N_BASE, seed=seed)
            est = MajorAgentEstimator(mfg, unknown=unknown, lam_entropy=LAM_ENTROPY,
                                       lr_E=0.05, lr_M=0.05,lr_decay=0.97)
            prob, fitted, n_steps = est.fit(
                X, true_major_idx=true_idx, n_em_iters=N_EM_ITERS,
                n_inner_E_steps=N_INNER_E_STEPS, n_inner_M_steps=N_INNER_M_STEPS,
                verbose=False,
            )
            pred_idx = int(prob.argmax().item())
            row = {
                'seed': seed, 'config_id': config_id, 'stage': stage_idx,
                'unknown_params': "|".join(unknown), 'G_true': G_true,
                'predicted_idx': pred_idx, 'true_idx': true_idx,
                'correct': int(pred_idx == true_idx), 'n_steps': n_steps,
                'wall_time_sec': round(time.time() - t0, 2),
            }
            for p in unknown:
                row[f"{p}_true"] = true_values[p]
                row[f"{p}_fit"] = fitted[p]

            writer.writerow(row)
            f.flush(); os.fsync(f.fileno())   # survive a hard kill, not just clean exit

            print(f"  seed={seed:3d}  correct={bool(row['correct'])}  "
                  f"steps={n_steps:4d}  time={row['wall_time_sec']:.1f}s  "
                  + "  ".join(f"{p}={fitted[p]:.3f}" for p in unknown), flush=True)

    print(f"  config {config_id} done: {path}\n", flush=True)


def run_config_u(config_id, n_seeds=N_SEEDS):
    """Same as run_config, but with the control u observed (UObservedEstimator)
    -- identical hyperparameters (lam_entropy, lr_E, lr_M, n_em_iters,
    n_inner_E_steps, n_inner_M_steps) to run_config, so the two are an
    apples-to-apples comparison of the X-only vs. u-observed regimes."""
    config_id, stage_idx, stage_name, unknown, G_true = config_lookup(config_id)
    path = result_path_u(stage_idx, stage_name, G_true)
    done = load_completed_seeds(path)
    fieldnames = fieldnames_for(unknown)
    write_header = not os.path.exists(path)

    mfg = make_mfg(G_true)
    true_values = true_values_for(G_true)
    print(f"=== Config {config_id} (u observed): stage {stage_idx} ({stage_name}), "
          f"G_true={G_true}, unknown={unknown} ===", flush=True)
    print(f"  results -> {path}   ({len(done)}/{n_seeds} already done)", flush=True)

    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
            f.flush(); os.fsync(f.fileno())

        for seed in range(n_seeds):
            if seed in done:
                continue
            t0 = time.time()
            X, x_bar_obs, true_idx, u = make_example(mfg, N_BASE, seed=seed)
            est = UObservedEstimator(mfg, unknown=unknown, lam_entropy=LAM_ENTROPY,
                                      lr_E=0.05, lr_M=0.05, u=u,lr_decay=0.97)
            prob, fitted, n_steps = est.fit(
                X, true_major_idx=true_idx, n_em_iters=N_EM_ITERS,
                n_inner_E_steps=N_INNER_E_STEPS, n_inner_M_steps=N_INNER_M_STEPS,
                verbose=False,
            )
            pred_idx = int(prob.argmax().item())
            row = {
                'seed': seed, 'config_id': config_id, 'stage': stage_idx,
                'unknown_params': "|".join(unknown), 'G_true': G_true,
                'predicted_idx': pred_idx, 'true_idx': true_idx,
                'correct': int(pred_idx == true_idx), 'n_steps': n_steps,
                'wall_time_sec': round(time.time() - t0, 2),
            }
            for p in unknown:
                row[f"{p}_true"] = true_values[p]
                row[f"{p}_fit"] = fitted[p]

            writer.writerow(row)
            f.flush(); os.fsync(f.fileno())   # survive a hard kill, not just clean exit

            print(f"  seed={seed:3d}  correct={bool(row['correct'])}  "
                  f"steps={n_steps:4d}  time={row['wall_time_sec']:.1f}s  "
                  + "  ".join(f"{p}={fitted[p]:.3f}" for p in unknown), flush=True)

    print(f"  config {config_id} done: {path}\n", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config-id', type=int, required=True,
                         choices=[c[0] for c in CONFIGS],
                         help="1-20: (stage-1)*5 + g_index + 1, G_true as inner loop "
                              "over [0.1,0.3,0.5,0.7,0.9]. See CONFIGS for the full table.")
    parser.add_argument('--n-seeds', type=int, default=N_SEEDS)
    parser.add_argument('--method', choices=['x', 'u'], default='x',
                         help="'x': states only (MajorAgentEstimator, default). "
                              "'u': control also observed (UObservedEstimator), "
                              "same hyperparameters, writes to a results_u_*.csv.")
    args = parser.parse_args()

    if args.method == 'u':
        run_config_u(args.config_id, n_seeds=args.n_seeds)
    else:
        run_config(args.config_id, n_seeds=args.n_seeds)


if __name__ == "__main__":
    main()
