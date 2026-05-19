"""
Vectorized RK4 SEIR solver for ensemble simulation.

All n_runs trajectories are integrated simultaneously via NumPy broadcasting,
making ensemble generation fast enough for phase-diagram grids.
"""

import numpy as np


def seir_rk4_ensemble(
    beta_arr,
    sigma_arr,
    gamma_arr,
    N,
    S0, E0, I0, R0,
    T,
    dt=1.0,
    intervention_day=9999.0,
    beta_reduction=0.40,
    sigma_reduction=0.00,
):
    """
    Vectorized RK4 SEIR solver across an ensemble of parameter sets.

    Parameters
    ----------
    beta_arr, sigma_arr, gamma_arr : ndarray, shape (n_runs,)
        Per-run transmission, exposed-to-infectious, and recovery rates.
    N : int
        Total (fixed) population.
    S0, E0, I0, R0 : float
        Shared initial conditions.
    T : float
        Simulation horizon (days).
    dt : float
        Fixed time step (days). Default 1.0.
    intervention_day : float
        Day on which intervention begins. Use 9999 for no intervention.
    beta_reduction : float
        Fractional reduction in β at intervention (0.40 = 40 % reduction).
    sigma_reduction : float
        Fractional reduction in σ at intervention (0 = no σ intervention).

    Returns
    -------
    t : ndarray, shape (n_steps,)
        Time grid.
    I_traj : ndarray, shape (n_runs, n_steps)
        Infectious count for every run at every time step.
    """
    n_runs = len(beta_arr)
    n_steps = int(T / dt) + 1
    t = np.arange(n_steps, dtype=float) * dt

    # State: (n_runs, 4) columns → [S, E, I, R]
    state = np.column_stack([
        np.full(n_runs, float(S0)),
        np.full(n_runs, float(E0)),
        np.full(n_runs, float(I0)),
        np.full(n_runs, float(R0)),
    ])

    I_traj = np.zeros((n_runs, n_steps))
    R_traj = np.zeros((n_runs, n_steps))
    I_traj[:, 0] = I0
    R_traj[:, 0] = R0

    def _deriv(st, b, s, g):
        S, E, I, R = st[:, 0], st[:, 1], st[:, 2], st[:, 3]
        foi = b * S * I / N          # force of infection
        dS  = -foi
        dE  =  foi - s * E
        dI  =  s * E - g * I
        dR  =  g * I
        return np.stack([dS, dE, dI, dR], axis=1)

    for step in range(1, n_steps):
        t_cur = t[step - 1]

        if t_cur >= intervention_day:
            b = beta_arr  * (1.0 - beta_reduction)
            s = sigma_arr * (1.0 - sigma_reduction)
        else:
            b = beta_arr
            s = sigma_arr
        g = gamma_arr

        k1 = _deriv(state, b, s, g)
        k2 = _deriv(state + 0.5 * dt * k1, b, s, g)
        k3 = _deriv(state + 0.5 * dt * k2, b, s, g)
        k4 = _deriv(state + dt * k3, b, s, g)

        state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        np.clip(state, 0.0, float(N), out=state)      # numerical guard

        I_traj[:, step] = state[:, 2]
        R_traj[:, step] = state[:, 3]

    return t, I_traj, R_traj
