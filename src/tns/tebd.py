"""
tebd.py — Phase 3: Gate application in Vidal canonical form.

THE 5-STEP ALGORITHM (Vidal 2004, mirrored from Werner's iTEBD_XXZ_v2.m)
-------------------------------------------------------------------------
Given environment:  ... l2 -- G1 -- l1 -- G2 -- l2 ...

G1: shape (d, chi_L, chi_M)   left site tensor,  chi_L = len(l2)
l1: shape (chi_M,)             inner Schmidt values (bond being updated)
G2: shape (d, chi_M, chi_R)   right site tensor, chi_R = len(l2)
l2: shape (chi_L,) = (chi_R,) outer environment Schmidt values (SAME on both sides)

Step 1 — Absorb Schmidt values into dressed 2-site tensor Theta:
    T1[s1, a, b] = l2[a] * G1[s1, a, b] * l1[b]
    T2[s2, b, c] = G2[s2, b, c] * l2[c]

Step 2 — Contract on middle bond b:
    theta[s1, a, s2, c] = sum_b T1[s1,a,b] * T2[s2,b,c]   shape: (d, chi_L, d, chi_R)

Step 3 — Apply 2-site gate U[s1',s2',s1,s2]:
    new_theta[s1',s2',a,c] = sum_{s1,s2} U[s1',s2',s1,s2] * theta[s1,a,s2,c]

Step 4 — SVD with truncation:
    M = new_theta.transpose(0,2,1,3).reshape(d*chi_L, d*chi_R)
    M = U_svd @ diag(S_new) @ Vh_svd    (truncated to chi_max)
    l1_new = S_new / ||S_new||           (normalize: new inner Schmidt values)

Step 5 — Restore Gamma tensors (divide out outer environment l2):
    A           = U_svd.reshape(d, chi_L, chi_new)
    B           = Vh_svd.reshape(chi_new, d, chi_R).transpose(1,0,2)
    G1_new[s,a,b] = A[s,a,b] / l2[a]   (safe: set 0 where l2[a] < tol)
    G2_new[s,b,c] = B[s,b,c] / l2[c]   (safe: set 0 where l2[c] < tol)

WHY DIVIDE OUT l2 IN STEP 5?
------------------------------
In Vidal canonical form, the Gamma tensors are defined WITHOUT the adjacent
Schmidt values absorbed in.  Step 1 temporarily absorbs l2 to form the
properly normalized 2-site state Theta.  After SVD, U_svd has l2 baked in
on the left column index and Vh_svd has l2 baked in on the right row index.
Dividing by l2 restores the canonical Gamma definition.

The safe inversion (set 1/l2 = 0 for l2 < tol) prevents numerical explosion
on Schmidt values that are effectively zero — these will be truncated anyway.

AB vs BA BONDS
--------------
For a 2-site unit cell  ... LB - GA - LA - GB - LB - GA - LA - GB - LB ...:

  AB gate:  G1=Gamma_A, l1=Lambda_A, G2=Gamma_B, l2=Lambda_B
            Updates: Gamma_A, Lambda_A, Gamma_B.  Lambda_B unchanged.

  BA gate:  G1=Gamma_B, l1=Lambda_B, G2=Gamma_A, l2=Lambda_A
            Updates: Gamma_B, Lambda_B, Gamma_A.  Lambda_A unchanged.

2ND-ORDER TROTTER
-----------------
Suzuki-Trotter 2nd order (error O(dt^3) per step vs O(dt^2) for 1st order):
    exp(-i H dt) = exp(-i H_AB dt/2) exp(-i H_BA dt) exp(-i H_AB dt/2) + O(dt^3)

Implemented as: AB(gate_half) -> BA(gate_full) -> AB(gate_half)
where gate_half = make_tfim_gate(J, h, dt/2) and gate_full = make_tfim_gate(J, h, dt).

For long runs, consecutive steps can merge boundary AB(dt/2) gates:
    ... AB(dt/2) | AB(dt/2) ... -> ... AB(dt) ...
This optimization halves gate count but is NOT implemented here for clarity.
"""

from __future__ import annotations

import numpy as np

from tns.svd_utils import svd_truncate
from tns.vidal_state import VidalMPS


def apply_gate_vidal(
    G1: np.ndarray,
    l1: np.ndarray,
    G2: np.ndarray,
    l2: np.ndarray,
    U: np.ndarray,
    chi_max: int,
    tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Apply a 2-site gate U in Vidal canonical form (5-step algorithm).

    Environment: ... l2 -- G1 -- l1 -- G2 -- l2 ...

    Parameters
    ----------
    G1      : shape (d, chi_L, chi_M)  left site tensor, chi_L = len(l2)
    l1      : shape (chi_M,)           inner Schmidt values (bond updated)
    G2      : shape (d, chi_M, chi_R)  right site tensor, chi_R = len(l2)
    l2      : shape (chi_L,) = (chi_R,) outer environment (SAME on both ends)
    U       : shape (d, d, d, d)       gate, U[s1',s2',s1,s2] = <s1's2'|U|s1s2>
    chi_max : maximum Schmidt rank to keep after SVD
    tol     : threshold for Schmidt value truncation and safe l2 inversion

    Returns
    -------
    G1_new    : shape (d, chi_L, chi_new)  updated left tensor
    l1_new    : shape (chi_new,)           updated inner Schmidt values, sum(l**2)=1
    G2_new    : shape (d, chi_new, chi_R)  updated right tensor
    trunc_err : float                      truncation error = sum of discarded s^2
    """
    d = G1.shape[0]
    chi_L = G1.shape[1]
    chi_R = G2.shape[2]

    # ------------------------------------------------------------------
    # Step 1: Absorb Schmidt values (dress Gamma tensors with environment)
    # T1[s1, a, b] = l2[a] * G1[s1, a, b] * l1[b]
    # T2[s2, b, c] = G2[s2, b, c] * l2[c]
    # ------------------------------------------------------------------
    T1 = G1 * l2[None, :, None] * l1[None, None, :]   # (d, chi_L, chi_M)
    T2 = G2 * l2[None, None, :]                        # (d, chi_M, chi_R)

    # ------------------------------------------------------------------
    # Step 2: Contract on the middle bond
    # theta[s1, a, s2, c] = sum_b T1[s1,a,b] * T2[s2,b,c]
    # ------------------------------------------------------------------
    theta = np.einsum('sab,tbc->satc', T1, T2)         # (d, chi_L, d, chi_R)

    # ------------------------------------------------------------------
    # Step 3: Apply gate
    # new_theta[s1',s2',a,c] = sum_{s1,s2} U[s1',s2',s1,s2] * theta[s1,a,s2,c]
    # ------------------------------------------------------------------
    new_theta = np.einsum('pqrs,rasc->pqac', U, theta)  # (d, d, chi_L, chi_R)

    # ------------------------------------------------------------------
    # Step 4: SVD with truncation
    # Group (s1', a) as row, (s2', c) as column
    # transpose: (s1',s2',a,c) -> (s1',a,s2',c) then reshape to matrix
    # ------------------------------------------------------------------
    M = new_theta.transpose(0, 2, 1, 3).reshape(d * chi_L, d * chi_R)
    result = svd_truncate(M, chi_max, tol)
    chi_new = result.chi
    l1_new = result.S         # normalized Schmidt values: sum(l1_new**2) = 1
    trunc_err = result.trunc_err

    # ------------------------------------------------------------------
    # Step 5: Restore Gamma tensors (divide out outer environment l2)
    # A[s1', a, b_new]   = U_svd.reshape(d, chi_L, chi_new)
    # B[s2', b_new, c]   = Vh_svd.reshape(chi_new, d, chi_R).T
    # G1_new = A / l2[a] along dimension 1
    # G2_new = B / l2[c] along dimension 2
    # ------------------------------------------------------------------
    A = result.U.reshape(d, chi_L, chi_new)                          # (d, chi_L, chi_new)
    B = result.Vh.reshape(chi_new, d, chi_R).transpose(1, 0, 2)      # (d, chi_new, chi_R)

    # Safe inversion: set 1/l2 = 0 where l2 is below tol
    inv_l2 = np.where(l2 > tol, 1.0 / l2, 0.0)

    G1_new = A * inv_l2[None, :, None]   # divide left bond of G1 by l2
    G2_new = B * inv_l2[None, None, :]   # divide right bond of G2 by l2

    return G1_new, l1_new, G2_new, trunc_err


def tebd_step_1st_order(
    state: VidalMPS,
    gate: np.ndarray,
    chi_max: int,
    tol: float = 1e-12,
) -> tuple[VidalMPS, float]:
    """First-order Trotter step: AB gate then BA gate.

    Error is O(dt^2) per step.  Suitable for testing; use 2nd-order for production.

    Parameters
    ----------
    state   : current VidalMPS
    gate    : shape (d,d,d,d), Trotter gate for full time step dt
    chi_max : bond dimension cap
    tol     : SVD truncation threshold

    Returns
    -------
    new_state : VidalMPS after one Trotter step
    trunc_err : total truncation error (AB + BA combined)
    """
    # AB gate: inner = Lambda_A, outer environment = Lambda_B
    GA1, LA1, GB1, err1 = apply_gate_vidal(
        state.Gamma_A, state.Lambda_A, state.Gamma_B, state.Lambda_B,
        gate, chi_max, tol
    )

    # BA gate: inner = Lambda_B (OLD, unchanged by AB), outer = Lambda_A (NEW from AB)
    GB2, LB2, GA2, err2 = apply_gate_vidal(
        GB1, state.Lambda_B, GA1, LA1,
        gate, chi_max, tol
    )

    new_state = VidalMPS(
        Gamma_A=GA2,
        Lambda_A=LA1,   # updated by AB gate, unchanged by BA gate
        Gamma_B=GB2,
        Lambda_B=LB2,   # updated by BA gate
    )
    return new_state, err1 + err2


def tebd_step_2nd_order(
    state: VidalMPS,
    gate_half: np.ndarray,
    gate_full: np.ndarray,
    chi_max: int,
    tol: float = 1e-12,
) -> tuple[VidalMPS, float]:
    """Second-order Trotter step: AB(dt/2) -> BA(dt) -> AB(dt/2).

    Error is O(dt^3) per step.  Use this for production runs.

    Parameters
    ----------
    state     : current VidalMPS
    gate_half : make_tfim_gate(J, h, dt/2) — half-step gate for AB bonds
    gate_full : make_tfim_gate(J, h, dt)   — full-step gate for BA bond
    chi_max   : bond dimension cap
    tol       : SVD truncation threshold

    Returns
    -------
    new_state : VidalMPS after one 2nd-order Trotter step
    trunc_err : total truncation error (3 gate applications combined)

    Notes
    -----
    Chain structure at each stage:

      Initial:  ... LB -- GA  -- LA  -- GB  -- LB ...
      After AB(dt/2): ... LB -- GA1 -- LA1 -- GB1 -- LB ...
      After BA(dt):   ... LA1 -- GB2 -- LB2 -- GA2 -- LA1 ...
      After AB(dt/2): ... LB2 -- GA3 -- LA3 -- GB3 -- LB2 ...

    Lambda_B = LB2 after the BA gate; it is the outer environment for the
    second AB gate, not updated by it.
    """
    # --- AB gate with dt/2 ---
    # inner = Lambda_A (old), outer = Lambda_B (old)
    GA1, LA1, GB1, err1 = apply_gate_vidal(
        state.Gamma_A, state.Lambda_A, state.Gamma_B, state.Lambda_B,
        gate_half, chi_max, tol
    )

    # --- BA gate with dt ---
    # inner = Lambda_B (old, unchanged by AB), outer = Lambda_A (new = LA1)
    GB2, LB2, GA2, err2 = apply_gate_vidal(
        GB1, state.Lambda_B, GA1, LA1,
        gate_full, chi_max, tol
    )

    # --- AB gate with dt/2 ---
    # inner = Lambda_A (= LA1, unchanged by BA), outer = Lambda_B (new = LB2)
    GA3, LA3, GB3, err3 = apply_gate_vidal(
        GA2, LA1, GB2, LB2,
        gate_half, chi_max, tol
    )

    new_state = VidalMPS(
        Gamma_A=GA3,
        Lambda_A=LA3,   # updated by second AB gate
        Gamma_B=GB3,
        Lambda_B=LB2,   # updated by BA gate, unchanged by second AB gate
    )
    return new_state, err1 + err2 + err3
