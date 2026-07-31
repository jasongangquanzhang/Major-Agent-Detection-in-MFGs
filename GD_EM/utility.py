import numpy as np
from mfg import MFG


def make_example(mfg: MFG, N: int, seed=None):
    if seed is not None:
        np.random.seed(seed)
    x_bar, x_major, x_minor, u_major, u_minor = mfg.simulate(N=N, N_sim=1, do_plot=False)
    # stack major as row 0, then minors
    X_ordered = np.vstack([x_major[0:1, :], x_minor[0, :, :]])   # (N+1, Ndt+1)
    u_ordered = np.vstack([u_major[0:1, :], u_minor[0, :, :]])   # (N+1, Ndt+1)
    perm = np.random.permutation(N + 1)
    X = X_ordered[perm]
    u = u_ordered[perm]
    true_idx = int(np.where(perm == 0)[0][0])                    # where major landed
    return X, x_bar[0], true_idx, u                                # x_bar[0]: (Ndt+1,)

