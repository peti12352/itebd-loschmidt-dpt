# Phase 0: Schmidt Decomposition and Truncated SVD

**File:** `src/tns/svd_utils.py`  
**Tests:** `tests/test_phase0.py` — 24 tests, all pass  
**Status:** complete

---

## What this is

The SVD truncation is the single most important primitive in the entire project.
Every iTEBD step does the following:

1. Apply a 2-site unitary gate → produces a rank-(d²·chi) tensor
2. **Reshape to matrix and run `svd_truncate`** → compress back to chi_max
3. Store the result as new Gamma tensors and Lambda vector

If this function is wrong, everything built on top of it is wrong.

---

## The Math

Any bipartite pure state `|ψ⟩` on `H_L ⊗ H_R` has a Schmidt decomposition:

```
|ψ⟩ = Σ_α  s_α  |u_α⟩_L  |v_α⟩_R
```

The Schmidt values `s_α ≥ 0` are the singular values of the coefficient
matrix `Ψ[i,j] = ⟨i|⟨j|ψ⟩`:

```
Ψ = U  diag(s)  Vh        (SVD)
```

The Schmidt rank (number of nonzero `s_α`) measures bipartite entanglement.

### Truncation

Keep only top `chi_max` Schmidt values. Discard the rest:

```
Ψ_approx = U[:, :chi_max]  diag(s[:chi_max])  Vh[:chi_max, :]
```

Truncation error (= squared approximation error):

```
ε = Σ_{α ≥ chi_max}  s_α²
```

This is the weight lost from the spectrum. For well-behaved states (gapped),
the spectrum decays fast and ε is tiny even at moderate chi_max.

### Normalization

After truncation, renormalize so `Σ s_α² = 1`. This keeps the state norm = 1
throughout the simulation.

### Von Neumann Entropy

```
S_vN = -Σ_α  s_α²  log(s_α²)
```

Measures entanglement:
- `S = 0`: product state (chi=1), no entanglement
- `S = log(2)`: Bell/singlet state (chi=2), maximally entangled 2-qubit
- `S = log(chi_max)`: maximally entangled at given chi — saturated, need larger chi

---

## Verified Cases

| State | Psi matrix | Schmidt values | Entropy |
|-------|-----------|----------------|---------|
| `\|↑⟩\|↑⟩` (product) | `[[1,0],[0,0]]` | `[1.0]` | 0 |
| Singlet `(\|↑↓⟩-\|↓↑⟩)/√2` | `[[0, 1/√2],[-1/√2, 0]]` | `[1/√2, 1/√2]` | log(2) |
| 4-value spectrum `[0.9, 0.3, 0.1, 0.05]` (raw) | random with those SVs | `[s]/norm` | computed |

Truncation to chi_max=2 on the 4-value state:
- Keeps `s[0], s[1]`, discards `s[2]=0.1, s[3]=0.05`
- `trunc_err = 0.1² + 0.05² = 0.0125` (verified in test)

---

## What to Watch For in Later Phases

**Numerical sensitivity:** In `tebd.py`, after restoring Gamma tensors we divide
by Lambda vectors: `Gamma[s,a,b] = A[s,a,b] / Lambda_env[a]`. If any `Lambda_env[a]`
is near zero (below `tol`), this division explodes. We handle this by
setting `1/lambda = 0` when `lambda < tol`.

**Complex matrices:** iTEBD uses complex gates `exp(-i H dt)`, so Psi becomes
complex after the first gate application. `svd_truncate` handles this correctly
(`numpy.linalg.svd` is defined for complex inputs). The test `test_complex_matrix`
verifies this.

**Rectangular Psi:** When `chi_L ≠ chi_R` (which happens after asymmetric
truncation), the matrix is not square. `svd_truncate` handles rectangular inputs
correctly. The test `test_rectangular_matrix` verifies this.

---

## Phase 0 Summary

`svd_truncate(psi, chi_max, tol=1e-12)` returns:
- `U, S, Vh`: truncated and normalized SVD factors
- `chi`: actual bond dimension kept
- `trunc_err`: approximation error introduced
- `entropy`: von Neumann entropy of the cut

All tests pass at machine precision (`atol=1e-12`).

**Next:** Phase 1 — Vidal state dataclass and Néel initialization.
