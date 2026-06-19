"""
simulate.py — Phase 4: Full iTEBD simulation loop.

WHAT THIS DOES
--------------
Runs the complete iTEBD time evolution for the TFIM quench from the Neel state:

    1. Initialize |ψ₀⟩ = |↑↓↑↓...⟩ (Neel product state, chi=1)
    2. Build Trotter gates for H(h1): gate_half = U(dt/2), gate_full = U(dt)
    3. At each time step:
       a. Apply 2nd-order Trotter step: AB(dt/2) → BA(dt) → AB(dt/2)
       b. Measure: λ(t), S_A(t), S_B(t), <Sz>_A(t), <Sz>_B(t), trunc_err
    4. Return all measurements as arrays

PARAMETERS
----------
J        : Ising coupling (set to 1.0 throughout the project)
h1       : post-quench transverse field (scan this to find DPT)
chi_max  : maximum Schmidt rank — controls accuracy vs speed tradeoff
dt       : time step (0.05 is good for dt^3 Trotter error with 2nd-order)
n_steps  : number of Trotter steps (total time = n_steps * dt)

CONVERGENCE GUIDE
-----------------
chi_max = 20 : quick, good for exploring parameter space
chi_max = 40 : production quality for short times (t < 5)
chi_max = 80 : high accuracy, needed for t > 10 or near DPT cusps
chi_max = 200: near-exact, overkill for this project

For chi_max = 40, n_steps = 100 (t_max = 5): runtime ~ 3 seconds on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tns.gates import make_tfim_gate
from tns.tebd import tebd_step_2nd_order
from tns.transfer import loschmidt_rate
from tns.vidal_state import (
    VidalMPS, entropy_A, entropy_B, init_neel, measure_sz, norm
)


@dataclass
class SimResult:
    """Output of run_itebd().

    All arrays have length n_steps (one entry per Trotter step).
    Entry i corresponds to time times[i] = (i+1) * dt.
    """
    times: np.ndarray       # shape (n_steps,) — time at each step
    lambda_t: np.ndarray    # shape (n_steps,) — Loschmidt rate λ(t) = -log|τ(t)|
    entropy_A: np.ndarray   # shape (n_steps,) — von Neumann entropy at A-bonds
    entropy_B: np.ndarray   # shape (n_steps,) — von Neumann entropy at B-bonds
    sz_A: np.ndarray        # shape (n_steps,) — ⟨Sz⟩ at A sublattice
    sz_B: np.ndarray        # shape (n_steps,) — ⟨Sz⟩ at B sublattice
    trunc_err: np.ndarray   # shape (n_steps,) — truncation error per step
    chi_A: np.ndarray       # shape (n_steps,) — bond dimension of Lambda_A at each step
    params: dict = field(default_factory=dict)   # J, h1, chi_max, dt, n_steps


def run_itebd(
    J: float,
    h1: float,
    chi_max: int,
    dt: float,
    n_steps: int,
    verbose: bool = False,
) -> SimResult:
    """Run iTEBD for the TFIM quench from Neel initial state.

    Parameters
    ----------
    J       : Ising coupling constant (J > 0 ferromagnetic)
    h1      : post-quench transverse field strength
    chi_max : maximum Schmidt rank (bond dimension cap)
    dt      : time step for 2nd-order Trotter
    n_steps : number of time steps; total time T = n_steps * dt
    verbose : if True, print progress every 10 steps

    Returns
    -------
    SimResult with all observables collected at each step.

    Notes
    -----
    Uses 2nd-order Trotter (AB(dt/2)→BA(dt)→AB(dt/2)) which is time-reversible
    and has O(dt^3) error per step.

    The Loschmidt rate λ(t) is computed from the chi_B × chi_B transfer matrix
    at every step — this is fast (O(chi_B^3) eigenvalue solve).
    """
    state = init_neel(d=2)

    # Build Trotter gates (computed once, reused every step)
    gate_half = make_tfim_gate(J, h1, dt / 2)
    gate_full = make_tfim_gate(J, h1, dt)

    # Pre-allocate output arrays
    times = np.arange(1, n_steps + 1, dtype=float) * dt
    lambda_arr = np.zeros(n_steps)
    entropy_A_arr = np.zeros(n_steps)
    entropy_B_arr = np.zeros(n_steps)
    sz_A_arr = np.zeros(n_steps)
    sz_B_arr = np.zeros(n_steps)
    trunc_arr = np.zeros(n_steps)
    chi_A_arr = np.zeros(n_steps, dtype=int)

    for step in range(n_steps):
        # Time evolution
        state, trunc = tebd_step_2nd_order(state, gate_half, gate_full, chi_max)

        # Observables
        lambda_arr[step] = loschmidt_rate(state)
        entropy_A_arr[step] = entropy_A(state)
        entropy_B_arr[step] = entropy_B(state)
        sz_A_arr[step], sz_B_arr[step] = measure_sz(state)
        trunc_arr[step] = trunc
        chi_A_arr[step] = state.chi_A

        if verbose and (step + 1) % 10 == 0:
            t = (step + 1) * dt
            print(
                f"  t={t:.2f}  λ={lambda_arr[step]:.4f}  "
                f"S_A={entropy_A_arr[step]:.4f}  "
                f"χ_A={chi_A_arr[step]}  "
                f"ε={trunc_arr[step]:.2e}"
            )

    return SimResult(
        times=times,
        lambda_t=lambda_arr,
        entropy_A=entropy_A_arr,
        entropy_B=entropy_B_arr,
        sz_A=sz_A_arr,
        sz_B=sz_B_arr,
        trunc_err=trunc_arr,
        chi_A=chi_A_arr,
        params=dict(J=J, h1=h1, chi_max=chi_max, dt=dt, n_steps=n_steps),
    )
