"""
vidal_state.py — Phase 1: Vidal canonical form for a 2-site unit cell.

THE VIDAL FORM
--------------
Vidal (2003) showed that any infinite translation-invariant MPS with a 2-site
unit cell can be written in a canonical form that stores Schmidt values
(singular values of the bipartition) explicitly at every bond:

    ... -Λ^B- Γ^A -Λ^A- Γ^B -Λ^B- Γ^A -Λ^A- Γ^B -Λ^B- ...

The Λ vectors are the Schmidt values.  The Γ tensors are what remains after
dividing out the Schmidt values — they encode the left/right Schmidt eigenstates
but are NOT themselves normalized states.

WHY A 2-SITE UNIT CELL?
-----------------------
The Néel state |↑↓↑↓...⟩ breaks the 1-site translational symmetry: A sites
carry spin up, B sites spin down.  The TFIM Hamiltonian connects nearest
neighbors, so gate application alternates AB and BA bonds.  Two Gamma tensors
(one per sublattice) and two Lambda vectors (one per bond type) are sufficient
to exactly represent the entire infinite chain.

INDEX CONVENTIONS (1:1 with Werner's MATLAB iTEBD_XXZ_v2.m):
-------------------------------------------------------------
    Gamma_A[s, a, b]:  s = physical (0=↑, 1=↓), a = left bond (chi_B), b = right bond (chi_A)
    Gamma_B[s, a, b]:  s = physical (0=↑, 1=↓), a = left bond (chi_A), b = right bond (chi_B)
    Lambda_A[a]:       Schmidt values at A-bonds (between consecutive A and B sites)
    Lambda_B[a]:       Schmidt values at B-bonds (between consecutive B and A sites)

Physical index: 0 = spin up |↑⟩,  1 = spin down |↓⟩.
All tensors are complex (gates are unitary, act on complex space).

CANONICAL CONDITIONS
--------------------
The canonical form satisfies two orthonormality conditions (Vidal 2003, Eq. 3):

  (C1) Left-ortho at A bond:
       Σ_{s,a} (Λ^B_a)² |Γ^A[s,a,b]|² = δ_{bb'}
       i.e., M_L = (Λ^B · Γ^A).reshape(d·chi_B, chi_A) is a column-isometry: M_L† M_L = I

  (C2) Right-ortho at B bond:
       Σ_{s,b} |Γ^B[s,a,b]|² (Λ^B_b)² = δ_{aa'}
       i.e., M_R = (Γ^B · Λ^B).reshape(chi_A, d·chi_B) is a row-isometry: M_R M_R† = I

Consequence: the 1-site reduced density matrix at A simplifies to
    ρ_A[s,s'] = Σ_{a,b} (Λ^B_a)² Γ^A[s,a,b] Γ^A*[s',a,b] (Λ^A_b)²
(the environment collapses to the two adjacent Λ vectors by orthonormality).

STATE NORM
----------
When (C1) and (C2) hold and all Λ vectors satisfy Σ_a Λ_a² = 1, then ⟨ψ|ψ⟩ = 1.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from tns.svd_utils import von_neumann_entropy


@dataclass
class VidalMPS:
    """Vidal canonical form: 2-site unit cell, infinite chain.

    Fields
    ------
    Gamma_A  : complex ndarray, shape (d, chi_B, chi_A)
    Lambda_A : real ndarray,    shape (chi_A,)   — Schmidt values, sum(Λ²) = 1
    Gamma_B  : complex ndarray, shape (d, chi_A, chi_B)
    Lambda_B : real ndarray,    shape (chi_B,)   — Schmidt values, sum(Λ²) = 1
    d        : local Hilbert space dimension (2 for spin-1/2)
    """
    Gamma_A: np.ndarray
    Lambda_A: np.ndarray
    Gamma_B: np.ndarray
    Lambda_B: np.ndarray
    d: int = 2

    @property
    def chi_A(self) -> int:
        """Bond dimension of Lambda_A (A-bonds)."""
        return int(self.Lambda_A.shape[0])

    @property
    def chi_B(self) -> int:
        """Bond dimension of Lambda_B (B-bonds)."""
        return int(self.Lambda_B.shape[0])


def init_neel(d: int = 2) -> VidalMPS:
    """Initialize Néel state |↑↓↑↓...⟩ in Vidal canonical form.

    Product state: chi = 1 at all bonds.
    Physical indices: 0 = ↑,  1 = ↓.

    A sites carry spin up  → Gamma_A[0, 0, 0] = 1  (all others 0)
    B sites carry spin down → Gamma_B[1, 0, 0] = 1  (all others 0)
    Both Lambda vectors: [1.0]  (single Schmidt value of weight 1 = no entanglement)

    This is a product state: Schmidt rank 1 at every bond, entropy S = 0.
    """
    Gamma_A = np.zeros((d, 1, 1), dtype=complex)
    Gamma_A[0, 0, 0] = 1.0  # spin up at A sites

    Gamma_B = np.zeros((d, 1, 1), dtype=complex)
    Gamma_B[1, 0, 0] = 1.0  # spin down at B sites

    Lambda_A = np.array([1.0])
    Lambda_B = np.array([1.0])

    return VidalMPS(Gamma_A=Gamma_A, Lambda_A=Lambda_A,
                    Gamma_B=Gamma_B, Lambda_B=Lambda_B, d=d)


def norm(state: VidalMPS) -> float:
    """State norm = sqrt(sum(Lambda_A²)).

    In Vidal canonical form with all orthonormality conditions satisfied,
    ⟨ψ|ψ⟩ = sum(Lambda_A²) = sum(Lambda_B²).
    Since svd_truncate always normalizes Λ, this equals 1.0.
    """
    return float(np.sqrt(np.sum(state.Lambda_A ** 2)))


def entropy_A(state: VidalMPS) -> float:
    """Von Neumann entanglement entropy across an A-bond.

    S_A = -Σ_α (Λ^A_α)² log((Λ^A_α)²)

    This is the entanglement of the semi-infinite left chain vs the right.
    """
    return von_neumann_entropy(state.Lambda_A)


def entropy_B(state: VidalMPS) -> float:
    """Von Neumann entanglement entropy across a B-bond."""
    return von_neumann_entropy(state.Lambda_B)


def measure_one_site(Gamma: np.ndarray,
                     Lambda_left: np.ndarray,
                     Lambda_right: np.ndarray,
                     Op: np.ndarray) -> float:
    """Expectation value ⟨Op⟩ at a single site via reduced density matrix.

    Mirrors Werner's MATLAB measureOneSite(G, lL, lR, Op) exactly.

    The 1-site reduced density matrix in Vidal canonical form:
        GW[s, a, b] = Lambda_left[a] * Gamma[s, a, b] * Lambda_right[b]
        rho[s, s']  = Σ_{a,b} GW[s,a,b] * GW*[s',a,b]
        ⟨Op⟩        = Re Tr(Op @ rho)

    Parameters
    ----------
    Gamma       : shape (d, chi_left, chi_right)
    Lambda_left : shape (chi_left,)  — Schmidt values on the left bond of this site
    Lambda_right: shape (chi_right,) — Schmidt values on the right bond of this site
    Op          : shape (d, d)       — single-site operator

    Returns
    -------
    Real part of Tr(Op @ rho).

    Note: for A sites call as measure_one_site(Gamma_A, Lambda_B, Lambda_A, Op)
          for B sites call as measure_one_site(Gamma_B, Lambda_A, Lambda_B, Op)
    """
    # Weight Gamma by left and right Schmidt values — builds the dressed tensor
    GW = Gamma * Lambda_left[None, :, None] * Lambda_right[None, None, :]

    # rho[s, s'] = Σ_{a,b} GW[s,a,b] GW*[s',a,b]
    rho = np.einsum('sab,tab->st', GW, GW.conj())

    return float(np.real(np.trace(Op @ rho)))


def measure_sz(state: VidalMPS) -> tuple[float, float]:
    """⟨Sz⟩ on A and B sublattices.

    Returns (sz_A, sz_B).
    Expected for Néel: sz_A = +0.5, sz_B = -0.5.
    """
    Sz = np.array([[0.5, 0.0], [0.0, -0.5]])
    sz_A = measure_one_site(state.Gamma_A, state.Lambda_B, state.Lambda_A, Sz)
    sz_B = measure_one_site(state.Gamma_B, state.Lambda_A, state.Lambda_B, Sz)
    return sz_A, sz_B


def check_canonical(state: VidalMPS, atol: float = 1e-10) -> dict[str, bool]:
    """Verify Vidal canonical conditions (C1) and (C2).

    (C1) Left-ortho at A bond:
         M_L = (Λ^B · Γ^A).reshape(d·chi_B, chi_A)
         M_L† M_L  should be identity(chi_A).

    (C2) Right-ortho at B bond:
         M_R = (Γ^B · Λ^B).reshape(chi_A, d·chi_B)
         M_R M_R†  should be identity(chi_A).

    Returns a dict with keys 'left_ortho' and 'right_ortho', each True/False.
    """
    d, chi_B_A, chi_A = state.Gamma_A.shape  # chi_B_A = chi_B (left bond of A)

    # (C1): left-dressed Gamma_A
    LG = state.Gamma_A * state.Lambda_B[None, :, None]  # (d, chi_B, chi_A)
    M_L = LG.transpose(0, 1, 2).reshape(d * chi_B_A, chi_A)
    gram_left = M_L.conj().T @ M_L  # should be I_{chi_A}
    left_ok = bool(np.allclose(gram_left, np.eye(chi_A), atol=atol))

    # (C2): right-dressed Gamma_B
    d, chi_A_B, chi_B = state.Gamma_B.shape  # chi_A_B = chi_A (left bond of B)
    RG = state.Gamma_B * state.Lambda_B[None, None, :]  # (d, chi_A, chi_B)
    M_R = RG.transpose(1, 0, 2).reshape(chi_A_B, d * chi_B)
    gram_right = M_R @ M_R.conj().T  # should be I_{chi_A}
    right_ok = bool(np.allclose(gram_right, np.eye(chi_A_B), atol=atol))

    return {'left_ortho': left_ok, 'right_ortho': right_ok}
