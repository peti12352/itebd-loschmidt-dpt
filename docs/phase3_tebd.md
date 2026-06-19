# Phase 3: Gate Application in Vidal Form

**File:** `src/tns/tebd.py`  
**Tests:** `tests/test_phase3.py` — 22 tests, all pass  
**Status:** complete

---

## What this is

`apply_gate_vidal` implements the 5-step algorithm from Vidal (2004), mirrored
exactly from Werner's MATLAB `iTEBD_XXZ_v2.m`.  It is the innermost loop of
the entire simulation: every iTEBD step calls it twice (AB bond, then BA bond).

---

## The 5-Step Algorithm

**Environment:**  `... l2 -- G1 -- l1 -- G2 -- l2 ...`

```
G1: (d, chi_L, chi_M),  l1: (chi_M,),  G2: (d, chi_M, chi_R),  l2: (chi_L,) = (chi_R,)
```

| Step | Operation | Why |
|---|---|---|
| 1 | `T1 = G1 * l2[:,] * l1[,:]`  `T2 = G2 * l2[,:]` | Absorb environment → Theta is normalized 2-site state |
| 2 | `theta[s1,a,s2,c] = einsum('sab,tbc->satc', T1, T2)` | Contract middle bond |
| 3 | `new_theta = einsum('pqrs,rasc->pqac', U, theta)` | Apply gate (rotate physical indices) |
| 4 | `M = new_theta.T(0,2,1,3).reshape(d*chi_L, d*chi_R)` → `svd_truncate` | Schmidt decomposition + truncation |
| 5 | `G1_new = U_svd.reshape(d,chi_L,chi_new) / l2` `G2_new = Vh_svd.reshape(chi_new,d,chi_R).T / l2` | Restore Gamma form (divide out environment) |

**Safe l2 inversion:** set `1/l2[a] = 0` when `l2[a] < tol`. Prevents explosion on effectively-zero Schmidt values that are about to be discarded.

### Why divide by l2 in step 5?

Gamma tensors are defined WITHOUT adjacent Schmidt values absorbed in (that's the whole point of Vidal form: separate Gammas from Lambdas). Step 1 temporarily absorbs l2 to form the normalized state Theta. After SVD, the left singular vectors have l2 baked in on their left index. Dividing out l2 restores the canonical Gamma definition.

---

## AB and BA Bonds

For `... LB - GA - LA - GB - LB - GA - LA - ...`:

```python
# AB gate (updates Lambda_A, Gamma_A, Gamma_B; Lambda_B unchanged)
GA1, LA1, GB1, err1 = apply_gate_vidal(GA, LA, GB, LB, U, chi_max)

# BA gate (updates Lambda_B, Gamma_B, Gamma_A; Lambda_A unchanged)
# CRITICAL: l1 = OLD Lambda_B (unchanged by AB), l2 = NEW Lambda_A (from AB)
GB2, LB2, GA2, err2 = apply_gate_vidal(GB1, LB, GA1, LA1, U, chi_max)
```

---

## 2nd-Order Trotter

`tebd_step_2nd_order(state, gate_half, gate_full, chi_max)` implements:

```
AB(dt/2) → BA(dt) → AB(dt/2)
```

**What changes at each stage:**

```
Initial:         ... LB  - GA  - LA  - GB  - LB  ...
After AB(dt/2):  ... LB  - GA1 - LA1 - GB1 - LB  ...
After BA(dt):    ... LA1 - GB2 - LB2 - GA2 - LA1 ...
After AB(dt/2):  ... LB2 - GA3 - LA3 - GB3 - LB2 ...
```

Final state: `VidalMPS(GA3, LA3, GB3, LB2)`.

**Lambda assignments:**
- `Lambda_A` (inner for AB) = LA1 from first AB, unchanged by BA, updated to LA3 by second AB
- `Lambda_B` (inner for BA) = original LB, updated to LB2 by BA, unchanged by second AB

### Time-reversal test

The symmetric decomposition is exactly time-reversible: applying forward then backward gives round-trip error = machine epsilon (1e-16). 1st-order gives error O(dt^2) ≈ 1e-4 for dt=0.3. This is the kill test for the AB-BA-AB ordering.

---

## Kill Tests

| Test | What it catches |
|---|---|
| `test_all_4_amplitudes_match_exact` | All 4 (s1,s2) amplitudes of post-gate state match `U_mat @ |↑↓⟩` exactly (1e-10) |
| `test_amplitude_match_several_parameters` | Same test for 4 different (J,h,dt) combinations |
| `test_multi_step_exact` | 3 consecutive AB gates match iterated `U_mat^3 @ |↑↓⟩` |
| `test_2nd_order_exact_time_reversal` | Round-trip error = machine epsilon (1e-16) |
| `test_chi_stays_one` | h=0 gate cannot entangle product state |
| `test_canonical_after_10_steps` | Canonical conditions (C1),(C2) preserved through 10 full steps |

---

## Trotter Error Note

For the TFIM, `[H_AB, H_BA]` is nonzero but small: ZZ terms commute across shared sites (Z commutes with Z), leaving only ZZ-X cross terms. In practice at chi_max=20-80, the truncation error dominates over Trotter discretization error. Use 2nd-order Trotter regardless — it costs nothing extra (3 gate applications vs 2 for 1st order) and guarantees better time-reversal properties.

---

**Next:** Phase 4 — full iTEBD loop, collecting `S(t)` and `⟨Sz(t)⟩`. Phase 5 — transfer matrix and `λ(t)`.
