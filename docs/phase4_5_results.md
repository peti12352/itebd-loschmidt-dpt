# Phase 4+5: Simulation Loop, Transfer Matrix, and DPT Results

**Files:** `src/tns/transfer.py`, `src/tns/simulate.py`, `run_a4.py`  
**Tests:** `tests/test_phase4_5.py` — 23 tests, all pass  
**Status:** complete

---

## Phase 5: Transfer Matrix

**File:** `src/tns/transfer.py`

### Derivation

For infinite MPS with 2-site unit cell and Neel initial state (chi=1), the
Loschmidt amplitude factorizes into unit-cell contributions:

```
G(t) = ⟨Neel|ψ(t)⟩ = τ(t)^{L/2}    (thermodynamic limit L→∞)
```

where τ(t) = dominant eigenvalue of the (chi_B × chi_B) transfer matrix:

```
T[α, β] = Σ_γ Λ^B(t)_α Γ^A(t)[↑, α, γ] Λ^A(t)_γ Γ^B(t)[↓, γ, β] Λ^B(t)_β
```

In code: two matrix multiplications, no loops:
```python
GA_dressed = Lambda_B[:, None] * Gamma_A[0, :, :] * Lambda_A[None, :]  # (chi_B, chi_A)
GB_dressed = Gamma_B[1, :, :] * Lambda_B[None, :]                       # (chi_A, chi_B)
T = GA_dressed @ GB_dressed                                              # (chi_B, chi_B)
```

Loschmidt rate: `λ(t) = -log|τ(t)|` where `|τ| = max(|eigenvalues(T)|)`.

### Key properties (all tested)

- At t=0: T = [[1+0j]], λ = 0
- λ(t) ≥ 0 always (|τ| ≤ 1 by Cauchy-Schwarz)
- T shape = (chi_B, chi_B) grows with entanglement

---

## Phase 4: Simulation Loop

**File:** `src/tns/simulate.py`

`run_itebd(J, h1, chi_max, dt, n_steps)` → `SimResult`

- Initializes Neel state
- Builds `gate_half = make_tfim_gate(J, h1, dt/2)` and `gate_full = make_tfim_gate(J, h1, dt)` once
- Loops n_steps times: `tebd_step_2nd_order` + measure all observables
- Returns `SimResult` with all time series

`SimResult` fields (all shape (n_steps,)):
`times`, `lambda_t`, `entropy_A`, `entropy_B`, `sz_A`, `sz_B`, `trunc_err`, `chi_A`

---

## Results: Dynamical Phase Transitions

**Run:** `uv run python run_a4.py`

Produces `results/loschmidt_scan.pdf` and `results/convergence.pdf`.

### λ(t) vs h1 (Figure 1)

| h1 | Behavior | Cusp times | max(λ) |
|---|---|---|---|
| 0.5 | Smooth, monotone | none | 0.370 |
| 0.8 | Mostly smooth | t ≈ 4.6 | 1.262 |
| 1.2 | Cusp visible | t ≈ 2.7 | 1.758 |
| 1.5 | Clear cusp | t ≈ 2.15 | 1.680 |
| 2.0 | Two cusps | t ≈ 1.65, 4.75 | 1.789 |
| 3.0 | Two cusps, faster | t ≈ 1.10, 3.15 | 2.095 |

DPT transition: cusp structure appears for h1 > 1 (quench crosses QPT at h=J=1). ✓

### Convergence (Figure 2, h1=2.0)

chi_max=20, 40, 80 all agree up to t ≈ 2.5. For t > 2.5:
- chi_max=20 saturates (trunc_err grows to 1e-15) → unreliable
- chi_max=40 saturates at t ≈ 4.5
- chi_max=80 still accurate at t = 5.0

For the exam, chi_max=40 is sufficient to show the first DPT cusp clearly.

### What the cusps mean

At critical time t*, the dominant eigenvalue of the transfer matrix approaches 0:
`|τ(t*)| → 0` → `λ(t*) → ∞`. In iTEBD with finite chi_max, the cusp is rounded
(not exactly infinity) but sharp enough to identify.

Physically: the time-evolved state |ψ(t*)⟩ is nearly orthogonal to the initial
Neel state. This is the quantum analogue of a first-order phase transition in time.

---

## Runtime

```
chi_max=40,  n_steps=100 (t_max=5.0):  ~6 h1 values in 20 seconds (3 sec/h1)
chi_max=80,  n_steps=100:              ~120 seconds total
chi_max=200, n_steps=100:              ~10 minutes (overkill for exam)
```

---

## Project A4: Complete

All 6 phases implemented and tested. Run `uv run python run_a4.py` to reproduce results.
```
Phase 0: svd_utils.py     — truncated SVD (24 tests)
Phase 1: vidal_state.py   — Vidal form, Neel init (26 tests)
Phase 2: gates.py         — TFIM Trotter gate (26 tests)
Phase 3: tebd.py          — 5-step gate application (22 tests)
Phase 4: simulate.py      — iTEBD run loop (23 tests, shared with Phase 5)
Phase 5: transfer.py      — transfer matrix, λ(t), DPT
Total:   121 tests, 0 failures
```
