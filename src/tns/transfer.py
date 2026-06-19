"""
transfer.py — Phase 5: Transfer matrix and Loschmidt echo rate function.

THE TRANSFER MATRIX
-------------------
For an infinite MPS |ψ(t)⟩ with 2-site unit cell and an initial product state
|ψ₀⟩ (Neel: chi=1), the Loschmidt amplitude factorizes across unit cells:

    G(t) = ⟨ψ₀|ψ(t)⟩ = ... × T(t) × T(t) × T(t) × ...  (infinite product)

In the thermodynamic limit (L→∞), this converges to:

    G(t) = τ(t)^{L/2}     (L/2 unit cells)

where τ(t) = dominant (largest-magnitude) eigenvalue of the transfer matrix T(t).

DERIVATION FOR NEEL INITIAL STATE
----------------------------------
Neel initial state: |ψ₀⟩ = |↑↓↑↓...⟩, chi=1 everywhere.
Initial tensors: Γ^A₀[↑,0,0]=1, Γ^B₀[↓,0,0]=1, Λ^A₀=[1], Λ^B₀=[1].

Transfer matrix element for one unit cell:

    T[α, β] = Σ_{σ₁,σ₂,γ}
              [Λ^B(t)_α Γ^A(t)^{σ₁}_{α,γ} Λ^A(t)_γ Γ^B(t)^{σ₂}_{γ,β} Λ^B(t)_β]
              × Γ^A₀*[σ₁] × Γ^B₀*[σ₂]

Since Γ^A₀[↑,0,0]=1 selects σ₁=↑ and Γ^B₀[↓,0,0]=1 selects σ₂=↓:

    T[α, β] = Σ_γ Λ^B(t)_α Γ^A(t)[↑, α, γ] Λ^A(t)_γ Γ^B(t)[↓, γ, β] Λ^B(t)_β

In matrix form (with shapes):

    T = diag(Λ^B) @ Γ^A(t)[↑, :, :] @ diag(Λ^A) @ Γ^B(t)[↓, :, :] @ diag(Λ^B)
          (chi_B,chi_B)   (chi_B,chi_A)  (chi_A,chi_A)  (chi_A,chi_B)  (chi_B,chi_B)

    Result: T has shape (chi_B, chi_B).

CORRESPONDENCE TO WERNER'S FORMULA (project sheet notation)
-------------------------------------------------------------
Werner writes T using absorbed tensors A(t) and B(t) (not Gamma-Lambda split):

    A(t)^{σ1}_{αγ} ≡ Λ^B_α Γ^A(t)^{σ1}_{αγ} Λ^A_γ       (= GA_dressed in code)
    B(t)^{σ2}_{γβ} ≡ Γ^B(t)^{σ2}_{γβ} Λ^B_β               (= GB_dressed in code)

So T[α,β] = Σ_γ A(t)^↑_{αγ} B(t)^↓_{γβ} = GA_dressed @ GB_dressed.
This is exactly Werner's T = Σ_{σ1,σ2,γ,γ̃} A(t)^{σ1}_{αγ}B(t)^{σ2}_{γβ}[A(0)^{σ1}_{ã=0,γ̃=0}]*[B(0)^{σ2}_{γ̃=0,b̃=0}]*
reduced by chi=1 initial state selecting σ1=↑, σ2=↓. ✓

THE RATE FUNCTION
-----------------
The Loschmidt rate function (free energy density analogue):

    λ(t) = -(1/L) log|G(t)|² = -log|τ(t)|

Physical behavior:
    - λ(0) = 0  (initial state exactly overlaps with itself: |τ(0)| = 1)
    - λ(t) ≥ 0  always (|τ(t)| ≤ 1 by Cauchy-Schwarz)
    - h₁ < 1 (quench within ordered phase): λ(t) smooth, no special features
    - h₁ > 1 (quench across QPT): λ(t) has cusps at critical times t*
                                   (DPT: dynamical phase transition)

DYNAMICAL PHASE TRANSITIONS
-----------------------------
At critical times t*, an eigenvalue of T passes through zero → |τ(t*)| = 0 → λ → ∞.
In practice (finite chi_max), cusps are rounded but appear as sharp local maxima.
The critical times t* = π(n+1/2)/ε* depend on post-quench quasi-particle dispersion.

For the Neel initial state (not the ferromagnet ground state used in Heyl 2013),
the critical times differ from the Heyl formula, but DPTs still occur when h₁ > 1.
"""

from __future__ import annotations

import numpy as np

from tns.vidal_state import VidalMPS


def build_transfer_matrix(
    state: VidalMPS,
    s1_neel: int = 0,
    s2_neel: int = 1,
) -> np.ndarray:
    """Transfer matrix for Loschmidt echo with Neel initial state.

    T[α, β] = Λ^B_α [Γ^A(t)[s1, α, γ]] Λ^A_γ [Γ^B(t)[s2, γ, β]] Λ^B_β

    Parameters
    ----------
    state    : current time-evolved VidalMPS
    s1_neel  : physical index at A sites in Neel state (0 = ↑)
    s2_neel  : physical index at B sites in Neel state (1 = ↓)

    Returns
    -------
    T : complex ndarray, shape (chi_B, chi_B)

    Notes
    -----
    T is computed as two matrix multiplies:
        GA_dressed[α, γ] = Λ^B_α Γ^A[s1, α, γ] Λ^A_γ    shape (chi_B, chi_A)
        GB_dressed[γ, β] = Γ^B[s2, γ, β] Λ^B_β            shape (chi_A, chi_B)
        T = GA_dressed @ GB_dressed                          shape (chi_B, chi_B)
    """
    # Dress Γ^A with environment Lambdas: Λ^B on left, Λ^A on right
    GA_dressed = (state.Lambda_B[:, None]
                  * state.Gamma_A[s1_neel, :, :]
                  * state.Lambda_A[None, :])                 # (chi_B, chi_A)

    # Dress Γ^B with right Lambda: Λ^B on right bond
    GB_dressed = (state.Gamma_B[s2_neel, :, :]
                  * state.Lambda_B[None, :])                 # (chi_A, chi_B)

    return GA_dressed @ GB_dressed                           # (chi_B, chi_B)


def loschmidt_rate(
    state: VidalMPS,
    s1_neel: int = 0,
    s2_neel: int = 1,
) -> float:
    """Loschmidt rate function λ(t) = -log|τ(t)|.

    τ(t) is the dominant (largest-magnitude) eigenvalue of the transfer matrix T.

    Parameters
    ----------
    state    : current time-evolved VidalMPS
    s1_neel  : physical index at A sites in Neel state (0 = ↑)
    s2_neel  : physical index at B sites in Neel state (1 = ↓)

    Returns
    -------
    lambda_t : float ≥ 0
        Rate function value. 0 at t=0, spikes at DPT critical times.

    Notes
    -----
    Uses numpy.linalg.eigvals (general complex eigenvalue solver).
    T is chi_B × chi_B, so this is fast even for chi_B = 200.
    The dominant eigenvalue is found by max(|eigenvalues|).
    """
    T = build_transfer_matrix(state, s1_neel, s2_neel)
    eigenvalues = np.linalg.eigvals(T)
    tau = float(np.max(np.abs(eigenvalues)))
    # Clip to avoid log(0) from numerical noise
    return float(-np.log(max(tau, 1e-300)))
