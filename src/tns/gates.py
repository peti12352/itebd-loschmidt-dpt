"""
gates.py — Phase 2: Trotter gate for the Transverse Field Ising Model.

THE TFIM HAMILTONIAN
--------------------
We study H = -J Σᵢ Zᵢ Zᵢ₊₁ - h Σᵢ Xᵢ  (spin-1/2, Pauli matrices scaled by 1/2).

Using Sz = [[1/2,0],[0,-1/2]] and Sx = [[0,1/2],[1/2,0]].

Quantum phase transition at h/J = 1 (gapless, Jordan-Wigner solvable).

THE 2-SITE BOND HAMILTONIAN
----------------------------
Trotter decomposition splits H into sums over even/odd bonds:
    H = H_even + H_odd = Σ_{even i} h_bond(i,i+1) + Σ_{odd i} h_bond(i,i+1)

Each bond Hamiltonian h_bond carries the full ZZ coupling and HALF the
transverse field for each site (to avoid double-counting when summing bonds):

    h_bond = -J * Sz⊗Sz  -  (h/2) * (Sx⊗I + I⊗Sx)

Basis ordering: {|↑↑⟩, |↑↓⟩, |↓↑⟩, |↓↓⟩} = {|00⟩, |01⟩, |10⟩, |11⟩}
where 0=up, 1=down throughout.

Explicit 4×4 matrix for J=1, h=0 (pure Ising):
    h_bond = -Sz⊗Sz = diag(-1/4, +1/4, +1/4, -1/4)
    (aligned spins have lower energy, ferromagnetic)

THE TROTTER GATE
----------------
First-order Trotter: exp(-i H dt) ≈ exp(-i H_even dt) * exp(-i H_odd dt)

For iTEBD with 2-site unit cell the two gate applications per step are:
    step 1: apply AB gate on all A-B bonds (even bonds)
    step 2: apply BA gate on all B-A bonds (odd bonds)

The AB and BA gates use the SAME h_bond (homogeneous chain) but act on
different pairs of Gamma tensors.

TENSOR CONVENTION
-----------------
Gate tensor U has shape (d, d, d, d) with:
    U[s1', s2', s1, s2] = <s1', s2'| exp(-i h_bond dt) |s1, s2>

This is obtained by C-order (row-major) reshape of the 4×4 unitary:
    U_mat = expm(-1j * dt * h_bond)          # 4×4 unitary
    U     = U_mat.reshape(d, d, d, d)         # U[s1',s2',s1,s2] = U_mat[2*s1'+s2', 2*s1+s2]

Row index 2*s1'+s2' = output state index (what the gate maps TO).
Col index 2*s1+s2   = input state index (what the gate maps FROM).

This is verified in tests: applying gate to |↑↑⟩ and checking the output
matches the analytically computed rotated state.

Reference: Werner's MATLAB iTEBD_XXZ_v2.m uses MATLAB Fortran-order reshape
    which gives different index arrangement. The physics is identical because
    the subsequent einsum contraction is also written to match.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm


def build_tfim_bond(J: float, h: float) -> np.ndarray:
    """2-site TFIM bond Hamiltonian as a 4×4 Hermitian matrix.

    h_bond = -J * Sz⊗Sz  -  (h/2) * (Sx⊗I + I⊗Sx)

    Parameters
    ----------
    J : Ising coupling. J > 0 is ferromagnetic.
    h : transverse field strength.

    Returns
    -------
    h_bond : complex ndarray, shape (4, 4), Hermitian.

    Notes
    -----
    The (h/2) factor distributes the single-body transverse field equally
    between left and right bonds of each site.  Summing h_bond over all bonds
    of the infinite chain recovers -J Σ Sz_i Sz_{i+1} - h Σ Sx_i exactly.
    """
    Sx = np.array([[0.0, 0.5],
                   [0.5, 0.0]])
    Sz = np.array([[0.5,  0.0],
                   [0.0, -0.5]])
    I2 = np.eye(2)

    h_bond = (-J * np.kron(Sz, Sz)
              - (h / 2.0) * (np.kron(Sx, I2) + np.kron(I2, Sx)))
    return h_bond.astype(complex)


def make_tfim_gate(J: float, h: float, dt: float) -> np.ndarray:
    """Trotter gate for the TFIM as a (d,d,d,d) tensor.

    Computes U = exp(-i h_bond dt) via exact matrix exponentiation (no
    approximation — no first-order Taylor truncation).

    Parameters
    ----------
    J   : Ising coupling.
    h   : transverse field.
    dt  : real time step (positive = forward in time).

    Returns
    -------
    U : complex ndarray, shape (2, 2, 2, 2)
        U[s1', s2', s1, s2] = <s1',s2'| exp(-i h_bond dt) |s1,s2>

    Usage in gate application (Phase 3):
        new_theta[s1',s2',a,c] = Σ_{s1,s2} U[s1',s2',s1,s2] * theta[s1,a,s2,c]
        which in einsum notation is:
        new_theta = np.einsum('pqrs,rasc->pqac', U, theta)
    """
    h_bond = build_tfim_bond(J, h)
    U_mat = expm(-1j * dt * h_bond)
    return U_mat.reshape(2, 2, 2, 2)
