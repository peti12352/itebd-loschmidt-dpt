# Phase 1: Vidal Canonical Form and Néel Initialization

**File:** `src/tns/vidal_state.py`  
**Tests:** `tests/test_phase1.py` — 26 tests, all pass  
**Status:** complete

---

## What this is

The `VidalMPS` dataclass is the central data structure of the entire project.
Every function from here on reads from or writes to it.  Getting the index
conventions right here prevents silent errors in all later phases.

---

## The Data Structure

```
... -Λ^B- Γ^A -Λ^A- Γ^B -Λ^B- Γ^A -Λ^A- Γ^B -Λ^B- ...
```

4 objects stored:

| Name | Shape | Meaning |
|---|---|---|
| `Gamma_A` | `(d, chi_B, chi_A)` | Tensors at A sites (spin up for Néel) |
| `Lambda_A` | `(chi_A,)` | Schmidt values at A-bonds |
| `Gamma_B` | `(d, chi_A, chi_B)` | Tensors at B sites (spin down for Néel) |
| `Lambda_B` | `(chi_B,)` | Schmidt values at B-bonds |

Index order for Gamma: `[physical, left_bond, right_bond]`.  This matches Werner's MATLAB
exactly (`Ga_A(s, a, b)` in `iTEBD_XXZ_v2.m`).

Physical index: `0 = ↑`,  `1 = ↓`.

---

## Néel State

Initial state for Project A4: `|↑↓↑↓...⟩`.

Product state = no entanglement = Schmidt rank 1 everywhere.  Only 4 scalars needed:

```
Gamma_A[0, 0, 0] = 1.0   (up at A sites)
Gamma_B[1, 0, 0] = 1.0   (down at B sites)
Lambda_A = [1.0]
Lambda_B = [1.0]
```

All other entries zero.

Why we chose Néel (not the ferromagnet `|↑↑↑...⟩`):
1. Product state: `chi = 1` → trivial to initialize, no truncation needed.
2. 2-site periodicity: A-up, B-down maps exactly onto the 2-site unit cell of iTEBD.
3. Project A4 specification: the course sheet prescribes this as the initial state.

---

## Canonical Conditions

After gate application, iTEBD restores the canonical form by dividing out Lambda
vectors (step 5 in the gate algorithm).  Two conditions must hold for this to be valid:

**(C1) Left orthonormality at A-bonds:**
```
Σ_{s,a} (Λ^B_a)² Γ^A[s,a,b] Γ^A*[s,a,b'] = δ_{bb'}
```
Equivalently: `M_L = (Λ^B · Γ^A).reshape(d·chi_B, chi_A)` satisfies `M_L† M_L = I`.

**(C2) Right orthonormality at B-bonds:**
```
Σ_{s,b} Γ^B[s,a,b] Γ^B*[s,a',b] (Λ^B_b)² = δ_{aa'}
```
Equivalently: `M_R = (Γ^B · Λ^B).reshape(chi_A, d·chi_B)` satisfies `M_R M_R† = I`.

Both hold trivially for the chi=1 Néel state (verified in `test_neel_is_canonical`).
They will be checked again after Phase 3 gate application to ensure iTEBD preserves them.

---

## Measurement Formula

The 1-site reduced density matrix at any site with tensor `Γ`, left Schmidt values `Λ_L`,
and right Schmidt values `Λ_R` is:

```
GW[s, a, b] = Λ_L[a] · Γ[s, a, b] · Λ_R[b]
ρ[s, s']    = Σ_{a,b} GW[s,a,b] · GW*[s',a,b]
⟨O⟩         = Re Tr(O · ρ)
```

For A sites: `measure_one_site(Gamma_A, Lambda_B, Lambda_A, Op)` — left is B-bond.  
For B sites: `measure_one_site(Gamma_B, Lambda_A, Lambda_B, Op)` — left is A-bond.

Verified results for Néel:
- `⟨Sz⟩_A = +0.5` (test: `test_sz_A_plus_half`)
- `⟨Sz⟩_B = -0.5` (test: `test_sz_B_minus_half`)

---

## Functions Exported

| Function | Purpose |
|---|---|
| `init_neel(d=2)` | Néel state in Vidal form |
| `norm(state)` | `sqrt(sum(Λ_A²))` — should stay 1.0 |
| `entropy_A(state)` | Von Neumann entropy from `Lambda_A` |
| `entropy_B(state)` | Von Neumann entropy from `Lambda_B` |
| `measure_one_site(Gamma, L_left, L_right, Op)` | `⟨Op⟩` at one site |
| `measure_sz(state)` | `(⟨Sz⟩_A, ⟨Sz⟩_B)` |
| `check_canonical(state, atol)` | Verify (C1) and (C2) |

---

**Next:** Phase 2 — TFIM gate `exp(-i h_bond dt)` and unitarity verification.
