import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.distributions as dist


class MFG_config:
    def __init__(
        self,
        T,
        Ndt,
        a_0,
        sigma_0,
        c_0,
        epslon_0,
        q_0,
        a,
        sigma,
        c,
        epslon,
        q,
        G,
    ):
        self.T = T
        self.Ndt = Ndt
        self.dt = T / Ndt
        self.a_0 = a_0
        self.sigma_0 = sigma_0
        self.c_0 = c_0
        self.epslon_0 = epslon_0
        self.q_0 = q_0
        self.a = a
        self.sigma = sigma
        self.c = c
        self.epslon = epslon
        self.q = q
        self.G = G
        self.F = 1 - G

    def __str__(self):
        return f"MFG_config: {self.__dict__}"


class MFG:
    def __init__(self, mfg_config: MFG_config):
        self.config = mfg_config
        self.a_0 = mfg_config.a_0
        self.sigma_0 = mfg_config.sigma_0
        self.c_0 = mfg_config.c_0
        self.epslon_0 = mfg_config.epslon_0
        self.q_0 = mfg_config.q_0
        self.a = mfg_config.a
        self.sigma = mfg_config.sigma
        self.c = mfg_config.c
        self.epslon = mfg_config.epslon
        self.q = mfg_config.q
        self.G = mfg_config.G
        self.F = mfg_config.F
        self.T = mfg_config.T
        self.Ndt = mfg_config.Ndt
        self.dt = mfg_config.dt
        self.phi = np.zeros((self.Ndt + 1,))
        self.phi_0 = np.zeros((self.Ndt + 1,))

    def __str__(self):
        return f"MFG: {self.config}"

    def solve_ODE(self):
        # --- Minor bank phi_t: backward Euler from t=T to t=0 ---
        # ODE (eq 4.4): dphi/dt = 2(a+q)*phi - phi^2 + epsilon - q^2
        # Terminal condition: phi(T) = -c
        self.phi[self.Ndt] = -self.c
        for k in range(self.Ndt - 1, -1, -1):
            phi = self.phi[k + 1]
            dphi = 2 * (self.a + self.q) * phi - phi**2 + self.epslon - self.q**2
            self.phi[k] = phi - self.dt * dphi

        # --- Major bank phi_major_t: backward Euler from t=T to t=0 ---
        # ODE (eq 4.2): dphi0/dt = 2*((a0+q0) + G*(a+q-phi_t))*phi0 - phi0^2 + epsilon0 - q0^2
        # Terminal condition: phi0(T) = -c0
        # Note: depends on phi_t already solved above
        self.phi_0[self.Ndt] = -self.c_0
        for k in range(self.Ndt - 1, -1, -1):
            phi0 = self.phi_0[k + 1]
            phi = self.phi[k + 1]
            dphi0 = (
                2 * ((self.a_0 + self.q_0) + self.G * (self.a + self.q - phi)) * phi0
                - phi0**2
                + self.epslon_0
                - self.q_0**2
            )
            self.phi_0[k] = phi0 - self.dt * dphi0

    def state_transtition(self, t_step, x_bar, x_major, x_minor):
        # x_bar:   (N_sim,)
        # x_major: (N_sim,)
        # x_minor: (N_sim, N)
        phi_t   = self.phi[t_step]    # minor bank Riccati coefficient
        phi_0_t = self.phi_0[t_step]  # major bank Riccati coefficient

        # mean-field transition (eq 4.5): deterministic, driven by x_major
        x_bar_next = (
            x_bar
            + (self.a + self.q - phi_t)
            * ((self.F - 1) * x_bar + self.G * x_major)
            * self.dt
        )                                                                    # (N_sim,)

        # major bank: SDE (eq 3.5 / eq 2.1) + optimal control (eq 4.1)
        # eq 2.1 mean-revert term uses x^(N)_t, the actual empirical mean of
        # minor bank states, rather than the mean-field limit x_bar
        x_minor_mean = x_minor.mean(axis=1)                                 # (N_sim,)
        u_major = (self.q_0 - phi_0_t) * (x_bar - x_major)                 # (N_sim,)
        x_major_next = (
            x_major
            + self.a_0 * (x_minor_mean - x_major) * self.dt
            + u_major * self.dt
            + self.sigma_0 * np.sqrt(self.dt) * np.random.randn(*x_major.shape)
        )                                                                    # (N_sim,)

        # market state: (N_sim, 1) broadcasts against (N_sim, N)
        market_state = self.F * x_bar[:, None] + self.G * x_major[:, None] # (N_sim, 1)
        # eq 2.3 mean-revert term uses x^(N)_t, the actual empirical mean of
        # minor bank states, rather than the mean-field limit x_bar
        market_state_mean = self.F * x_minor_mean[:, None] + self.G * x_major[:, None]  # (N_sim, 1)

        # minor banks: SDE (eq 3.7) + optimal control (eq 4.3)
        u_minor = (self.q - phi_t) * (market_state - x_minor)               # (N_sim, N)
        x_minor_next = (
            x_minor
            + self.a * (market_state_mean - x_minor) * self.dt
            + u_minor * self.dt
            + self.sigma * np.sqrt(self.dt) * np.random.randn(*x_minor.shape)
        )                                                                    # (N_sim, N)

        return x_bar_next, x_major_next, x_minor_next,u_major,u_minor

    def simulate(self, N, N_sim, do_plot=True):
        self.solve_ODE()
        x_bar = np.zeros((N_sim, self.Ndt + 1))        # (N_sim, Ndt+1)
        x_major = np.zeros((N_sim, self.Ndt + 1))      # (N_sim, Ndt+1)
        x_minor = np.zeros((N_sim, N, self.Ndt + 1))   # (N_sim, N, Ndt+1)
        u_major = np.zeros((N_sim, self.Ndt + 1))      # (N_sim, Ndt+1)
        u_minor = np.zeros((N_sim, N, self.Ndt + 1))   # (N_sim, N, Ndt+1)
        for i in range(self.Ndt):
            x_bar[:, i + 1], x_major[:, i + 1], x_minor[:, :, i + 1], u_major[:, i + 1], u_minor[:, :, i + 1] = (
                self.state_transtition(i, x_bar[:, i], x_major[:, i], x_minor[:, :, i])
            )
        if do_plot:
            self.plot(x_bar, x_major, x_minor)
        return x_bar, x_major, x_minor, u_major, u_minor

    def plot(self, x_bar, x_major, x_minor):
        D      = -0.65
        N      = x_minor.shape[1]
        t_grid = np.linspace(0, self.T, self.Ndt + 1)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(str(self.config), fontsize=20)

        # --- top-left: Riccati ODE coefficients ---
        ax = axes[0, 0]
        ax.plot(t_grid, self.phi,   label=r'$\phi_t$ (minor)')
        ax.plot(t_grid, self.phi_0, label=r'$\phi^0_t$ (major)')
        ax.set_xlabel('Time')
        ax.set_title('Riccati ODE Coefficients')
        ax.legend()
        ax.grid(True)

        # --- top-right: effective mean-reversion rates after optimal control (Fig B.5) ---
        ax = axes[0, 1]
        ax.plot(t_grid, self.a   + self.q   - self.phi,   label=r'$a+q-\phi_t$ (minor)')
        ax.plot(t_grid, self.a_0 + self.q_0 - self.phi_0, label=r'$a_0+q_0-\phi^0_t$ (major)')
        ax.set_xlabel('Time')
        ax.set_title('Effective Mean-Reversion Rates')
        ax.legend()
        ax.grid(True)

        # --- bottom-left: sample trajectories for 1 simulation path (Fig B.3) ---
        ax = axes[1, 0]
        for n in range(N):
            ax.plot(t_grid, x_minor[0, n, :], color='orange', alpha=0.5, linewidth=0.8)
        ax.plot(t_grid, x_major[0, :], color='red',   linewidth=1.5, label='Major bank')
        market_state_0 = self.F * x_bar[0, :] + self.G * x_major[0, :]
        ax.plot(t_grid, market_state_0,   color='green', linewidth=1.5, label='Market state')
        ax.axhline(D, color='black', linestyle='--', label=f'Default threshold {D}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Log-monetary reserve')
        ax.set_title('Sample Trajectories (simulation 0)')
        ax.legend()
        ax.grid(True)

        # --- bottom-right: loss distribution across all simulations (Fig 1) ---
        ax = axes[1, 1]
        n_defaults = (x_minor.min(axis=2) <= D).sum(axis=1)   # (N_sim,): defaults per sim
        counts = np.bincount(n_defaults, minlength=N + 1)
        ax.bar(np.arange(N + 1), counts / counts.sum())
        ax.set_xlabel('# of minor bank defaults')
        ax.set_ylabel('Probability')
        ax.set_title('Loss Distribution (minor banks)')
        ax.grid(True, axis='y')

        plt.tight_layout()
        plt.show()


    def score_path_torch(self, x_bar, x_major, x_minor, per_step=False):
        """
        Joint log-likelihood of one path, via torch.distributions.
        x_bar:   (Ndt+1,)
        x_major: (Ndt+1,)
        x_minor: (N, Ndt+1)
        """
        dt = self.dt
        # to tensors (float64 to match numpy precision; enable grad if you'll optimize params)
        x_bar   = torch.as_tensor(x_bar,   dtype=torch.float64)
        x_major = torch.as_tensor(x_major, dtype=torch.float64)
        x_minor = torch.as_tensor(x_minor, dtype=torch.float64)
        phi     = torch.as_tensor(self.phi[:-1],   dtype=torch.float64)   # (Ndt,)
        phi_0   = torch.as_tensor(self.phi_0[:-1], dtype=torch.float64)

        sig_M = self.sigma_0 * np.sqrt(dt)
        sig_m = self.sigma   * np.sqrt(dt)

        # --- major bank ---
        gap_M = x_bar[:-1] - x_major[:-1]                                   # (Ndt,)
        mu_M  = x_major[:-1] + (self.a_0 + (self.q_0 - phi_0)) * gap_M * dt
        ll_M  = dist.Normal(mu_M, sig_M).log_prob(x_major[1:])             # (Ndt,)

        # --- minor banks ---
        market = self.F * x_bar + self.G * x_major                         # (Ndt+1,)
        gap_m  = market[:-1].unsqueeze(0) - x_minor[:, :-1]                # (N, Ndt)
        mu_m   = x_minor[:, :-1] + (self.a + (self.q - phi)) * gap_m * dt
        ll_m   = dist.Normal(mu_m, sig_m).log_prob(x_minor[:, 1:])         # (N, Ndt)

        step_ll = ll_M + ll_m.sum(dim=0)        # (Ndt,)
        score   = step_ll.sum()
        return (score, step_ll) if per_step else score