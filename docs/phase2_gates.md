# Phase 2: TFIM Trotter Gate

**File:** `src/tns/gates.py`  
**Tests:** `tests/test_phase2.py` — 26 tests, all pass  
**Status:** complete

---

## What this is

The Trotter gate is the fundamental operation of iTEBD: it implements one
step of time evolution on a single bond.  All entanglement growth in the
simulation comes from gate applications.

---

## The Bond Hamiltonian

The full TFIM Hamiltonian: `H = -J Σᵢ Zᵢ Zᵢ₊₁ - h Σᵢ Xᵢ`

Trotter decomposition into even/odd bonds:
```
H = Σ_{even i} h_bond(i,i+1) + Σ_{odd i} h_bond(i,i+1)
```

Bond Hamiltonian (4×4 matrix, basis `{|↑↑⟩, |↑↓⟩, |↓↑⟩, |↓↓⟩}`):
```
h_bond = -J · Sz⊗Sz  -  (h/2) · (Sx⊗I + I⊗Sx)
```

The `(h/2)` factor: each site sits between two bonds (left and right neighbor).
Distributing half to each bond ensures summing over all bonds gives `-h Σᵢ Xᵢ` exactly.

Pure Ising limit (`h=0`):
```
h_bond = -J · Sz⊗Sz = diag(-J/4, +J/4, +J/4, -J/4)
```
Aligned spins `|↑↑⟩, |↓↓⟩` have lower energy `-J/4` (ferromagnetic).

### Eigenvalues for J=h=1

Computed numerically: `{-√5/4, -1/4, +1/4, +√5/4} ≈ {-0.559, -0.25, +0.25, +0.559}`

The `-1/4` eigenvalue corresponds to the antisymmetric singlet `(|↑↓⟩-|↓↑⟩)/√2`.

---

## The Gate

```
U[s1', s2', s1, s2] = ⟨s1' s2'| exp(-i h_bond dt) |s1 s2⟩
```

Computed via `scipy.linalg.expm` (exact, no Taylor truncation).

### Python Reshape Convention

```python
U_mat = expm(-1j * dt * h_bond)    # 4×4 unitary
U     = U_mat.reshape(2, 2, 2, 2)  # C-order (row-major)
```

C-order reshape maps: `U[s1', s2', s1, s2] = U_mat[2*s1' + s2', 2*s1 + s2]`

Row index `2*s1' + s2'` = output state. Column = input state. Correct physical convention. ✓

Note: MATLAB uses Fortran-order reshape which gives different 4D arrangement, but both are
correct when combined with their respective contraction formulas.

### How Gate is Applied (Phase 3 preview)

```python
# theta[s1, a, s2, c] = 2-site tensor on AB bond
new_theta = np.einsum('pqrs,rasc->pqac', U, theta)
# result new_theta[s1', s2', a, c] = Σ_{s1,s2} U[s1',s2',s1,s2] * theta[s1,a,s2,c]
```

---

## Key Properties (all verified in tests)

| Property | Formula | Test |
|---|---|---|
| h_bond Hermitian | `h_bond = h_bond†` | `test_hermitian_*` |
| h_bond traceless | `Tr(h_bond) = 0` | `test_traceless` |
| SWAP symmetry | `[h_bond, SWAP] = 0` | `test_symmetry_swap_spins` |
| Gate unitary | `U†U = UU† = I` | `test_unitary_*` |
| `|det(U)| = 1` | group property | `test_determinant_magnitude_one` |
| dt=0 → identity | `exp(0) = I` | `test_identity_at_dt0` |
| Composition | `U(dt/2)²= U(dt)` | `test_composition_two_half_steps` |
| Time reversal | `U(-dt) = U(dt)†` | `test_inverse_is_conjugate_transpose` |
| h=0 diagonal | exact formula | `test_pure_ising_*` |
| First-order Taylor | error O(dt²) | `test_first_order_small_dt` |
| Index convention | loop check all 16 | `test_index_convention_all_elements` |

---

## What to Watch For in Phase 3

**Trotter ordering:** iTEBD applies gates in order: `U_AB(dt/2) → U_BA(dt) → U_AB(dt/2)` for
second-order Trotter. Using `U_AB(dt)` twice gives first-order Trotter with twice the error.

**Same gate for AB and BA bonds:** the TFIM is homogeneous, so `make_tfim_gate` is called once
and reused for both bond types.  The gate only depends on J and h, not on which sublattice
the bond connects.

**No separate gate for 2nd-order:** `U_AB(dt/2)` is just `make_tfim_gate(J, h, dt/2)`. Build two
gates: `gate_half = make_tfim_gate(J, h1, dt/2)` and `gate_full = make_tfim_gate(J, h1, dt)`.

---

**Next:** Phase 3 — gate application in Vidal form (the 5-step algorithm) and full Trotter step.
