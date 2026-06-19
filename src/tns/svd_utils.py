"""
svd_utils.py — Phase 0: Schmidt decomposition with controlled truncation.

WHAT THIS FILE IS FOR
---------------------
Every iTEBD step applies a 2-site gate and then must compress the resulting
tensor back to bond dimension chi_max.  That compression is exactly an SVD
followed by keeping only the top singular values — the Schmidt decomposition.

This module is the single place where that SVD + truncation lives.  All
higher-level modules (tebd, transfer matrix, ...) import from here.

THE MATH
--------
Any bipartite pure state |psi> on H_L ⊗ H_R can be written as

    |psi> = Σ_α  s_α  |u_α>_L  |v_α>_R        (Schmidt decomposition)

where:
  - s_α ≥ 0 are the Schmidt values (singular values of the coefficient matrix)
  - |u_α> and |v_α> are orthonormal bases on L and R respectively
  - the number of nonzero s_α is the Schmidt rank (= entanglement measure)

If we collect coefficients into a matrix  Psi[i, j] = <i|<j|psi>  then

    Psi = U  diag(s)  Vh        (standard SVD)

Truncating to chi_max terms: keep only the chi_max largest s_α.
Normalization after truncation: rescale so  Σ_α s_α² = 1  (unit state norm).

Truncation error:  ε = Σ_{α > chi_max}  s_α²   (weight lost from spectrum)
This equals  ||psi - psi_truncated||²  and is the key accuracy metric.

Von Neumann entropy:  S = -Σ_α  s_α²  log(s_α²)
This quantifies bipartite entanglement.  S=0 for product states, S=log(χ)
for maximally entangled states (all s_α = 1/√χ).
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np


@dataclass
class SVDResult:
    """Container for truncated Schmidt decomposition output.

    Represents  Psi ≈ U @ diag(S) @ Vh  with S normalized so  sum(S**2) = 1.

    Fields
    ------
    U         : shape (m, chi)  — left Schmidt vectors (columns orthonormal)
    S         : shape (chi,)    — Schmidt values, descending, normalized
    Vh        : shape (chi, n)  — right Schmidt vectors (rows orthonormal)
    chi       : int             — actual bond dimension kept
    trunc_err : float           — Σ_{discarded} s² — approximation error
    entropy   : float           — von Neumann entropy -Σ s² log(s²)
    """
    U: np.ndarray
    S: np.ndarray
    Vh: np.ndarray
    chi: int
    trunc_err: float
    entropy: float


def svd_truncate(psi: np.ndarray, chi_max: int, tol: float = 1e-12) -> SVDResult:
    """Truncated SVD (Schmidt decomposition) of a bipartite coefficient matrix.

    Parameters
    ----------
    psi     : np.ndarray, shape (m, n)
        Coefficient matrix of bipartite state.  Row index = left Hilbert space
        basis, column index = right Hilbert space basis.
    chi_max : int
        Maximum number of Schmidt values to keep.
    tol     : float
        Drop Schmidt values below this threshold (applied before normalization).
        Prevents keeping numerically zero singular values that just add noise.

    Returns
    -------
    SVDResult
        Truncated and normalized Schmidt decomposition.  The returned S satisfies
        sum(S**2) = 1 (state norm preserved).

    Notes
    -----
    We use numpy.linalg.svd with full_matrices=False (economy SVD), which gives
    at most min(m, n) singular values.  Then we apply chi_max and tol cutoffs.

    The 1/norm rescaling of S is the normalization step: after truncation the
    discarded weight must be compensated so the remaining state has unit norm.
    """
    # Economy SVD: U shape (m, k), S shape (k,), Vh shape (k, n), k = min(m, n)
    U, S, Vh = np.linalg.svd(psi, full_matrices=False)

    # Determine how many values to keep: top chi_max above threshold
    # np.linalg.svd returns singular values in descending order — guaranteed.
    keep = min(chi_max, np.sum(S > tol).item())
    keep = max(keep, 1)  # always keep at least one value to avoid empty state

    # Truncation error: squared norm of the discarded part of the spectrum
    trunc_err = float(np.sum(S[keep:] ** 2))

    # Slice to kept singular values
    S = S[:keep]
    U = U[:, :keep]
    Vh = Vh[:keep, :]

    # Normalize: rescale so sum(S**2) = 1 (state has unit norm after truncation)
    norm = np.linalg.norm(S)
    S = S / norm

    # Von Neumann entropy: S = -Σ s² log(s²), clip to avoid log(0)
    s_sq = S ** 2
    entropy = float(-np.sum(s_sq * np.log(np.clip(s_sq, 1e-300, None))))

    return SVDResult(U=U, S=S, Vh=Vh, chi=keep, trunc_err=trunc_err, entropy=entropy)


def von_neumann_entropy(singular_values: np.ndarray) -> float:
    """Von Neumann entropy from a vector of normalized Schmidt values.

    Parameters
    ----------
    singular_values : 1D array of Schmidt values s_α, must satisfy sum(s²) = 1.

    Returns
    -------
    S = -Σ_α  s_α²  log(s_α²)
    """
    s_sq = singular_values ** 2
    return float(-np.sum(s_sq * np.log(np.clip(s_sq, 1e-300, None))))
