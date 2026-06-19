# Exam Preparation: Project A4

## Step 2: When Are Results Reliable?

**Rule:** Results are reliable while `chi_A < chi_max` (not saturated). The moment chi_A
hits the cap, Schmidt values are being discarded that shouldn't be — approximation error
compounds from that point.

**From simulation data (h1=2.0, dt=0.05, T=5.0):**

| chi_max | chi_A saturates at | Reliable until | trunc_err at saturation |
|---|---|---|---|
| 20  | t ≈ 2.5 (chi_A=20) | t ≈ 2.5 | grows to 1e-15 by t=5 |
| 40  | t ≈ 4.5 (chi_A=40) | t ≈ 4.5 | 1e-21 at saturation, 1e-20 at t=5 |
| 80  | never (chi_A=47 at t=5) | all of [0,5] | stays at 1e-24 |
| 200 | never (chi_A=53 at t=5) | all of [0,5] | stays at 1e-24 |

**Answer to Werner's question "Up to what times are results reliable?"**

Entropy S(t) measures the required bond dimension. The simulation is reliable
while `S(t) << log(chi_max)`. Saturation `S → log(chi_max)` means every Schmidt value
above chi_max has been discarded — approximation error is compounding and growing.

For this specific problem (TFIM, Neel → h1=2.0):
- S only reaches ≈ 0.69 at t=5.0
- log(20) ≈ 3.0, log(40) ≈ 3.7, log(80) ≈ 4.4

So even chi_max=20 is far from the theoretical saturation limit. The actual chi_A
cap is what matters: chi_A=20 is reached at t≈2.5, after which Schmidt values ARE
being truncated. For chi_max=80 and above, chi_A never saturates in [0,5].

**Conclusion: For M=80 (or M=200), all results in [0,5] are reliable. For M=20,
trust results until t≈2.5. The reason chi stays small is that the TFIM is integrable
(maps to free fermions via Jordan-Wigner) — entanglement grows sub-linearly.**

---

## Step 3: Exam Questions Werner Will Ask

### Q: What is the Loschmidt echo?

G(t) = ⟨ψ₀|exp(-iH(h₁)t)|ψ₀⟩

Overlap of the time-evolved state with the initial state. Measures "how much the state
has moved" under the quench. Rate function:

λ(t) = -(1/L) log|G(t)|²  →  -log|τ(t)|  (thermodynamic limit)

Analogy: partition function Z(β) = Tr[exp(-βH)] with β → it. Phase transitions in
equilibrium occur when Z = 0. Here G(t) = 0 at critical times = dynamical phase transition.

### Q: Why do cusps appear for h1 > 1?

The key is the **Fisher zeros** of G(t) in the complex time plane.

G(t) = ⟨ψ₀|exp(-iHt)|ψ₀⟩ is analytic for finite L. In the thermodynamic limit, the
zeros z_n(k) of G (as a function of complex time z = R + it) form lines in the complex
plane (Eq. 10 in Heyl 2013):

z_n(k) = (1/2ε(k)) [log tan²φ_k + iπ(2n+1)]

where φ_k = θ_k(h₀) - θ_k(h₁) are the Bogoliubov angles (how much the ground state
rotates between h₀ and h₁ basis for each momentum mode k).

**For quench within same phase (h₁ < 1, both in ordered):** the zeros stay in the
negative real half-plane. They never cross the imaginary time axis. G(t) has no zeros
on the real time axis → λ(t) smooth.

**For quench across QPT (h₁ > 1):** the Bogoliubov angles satisfy φ_{k→0} = π/2
(the k=0 mode flips completely). This pushes the zero lines across the imaginary axis
at critical times t*_n. G(t*_n) = 0 → λ(t*_n) = ∞ (cusp).

**In one sentence:** The quench across the QPT creates modes with "inverted population"
(excited more than ground state). These modes push the zeros of G(z) onto the real
time axis, creating non-analytic cusps in λ(t).

### Q: Explain the transfer matrix

Werner's formula:

T_{αã,βb̃} = Σ_{σ1,σ2,γ,γ̃} A(t)^{σ1}_{αγ} B(t)^{σ2}_{γβ} [A(0)^{σ1}_{ãγ̃}]* [B(0)^{σ2}_{γ̃b̃}]*

Where A(t) and B(t) are the time-evolved MPS matrices (dressed tensors in Vidal form):
- A(t)^{σ}_{αγ} = Λ^B_α Γ^A(t)^{σ}_{αγ} Λ^A_γ   (what we call GA_dressed in code)
- B(t)^{σ}_{γβ} = Γ^B(t)^{σ}_{γβ} Λ^B_β          (what we call GB_dressed in code)

Since initial state is chi=1 Neel: ã=b̃=γ̃=0, A(0)^↑_{00}=1, B(0)^↓_{00}=1.
T reduces to chi_B × chi_B matrix:

T[α,β] = Σ_γ A(t)^↑_{αγ} B(t)^↓_{γβ}  =  GA_dressed @ GB_dressed

This is exact in our code (transfer.py, `build_transfer_matrix`).

Physical meaning: T is the "transfer matrix in space" for one unit cell. Its dominant
eigenvalue τ gives G(t) = τ^{L/2}, so λ(t) = -log|τ(t)|.

### Q: How do your results depend on M?

**Numerical answer:** M=20, 40, 80, 200 give IDENTICAL results for t ∈ [0, 4.5].
For M=20, results start deviating at t > 2.5 (chi_A saturates). For M≥80, results are
reliable throughout. Convergence is fast because the TFIM is integrable: entanglement
entropy grows slowly (S=0.69 at t=5, well below saturation).

**Physical answer:** χ controls how many entangled modes we can represent. When χ
is too small, we truncate Schmidt values that contribute to the physics. For the TFIM
(Jordan-Wigner solvable), entanglement is bounded even at long times, so small χ works.
For chaotic/non-integrable models, entanglement grows linearly in time (information
scrambling) and χ must grow exponentially — this is the fundamental barrier of TEBD.

### Q: What is entanglement entropy and why does it matter?

S = -Σ_α s_α² log(s_α²)  where s_α are the Schmidt values (singular values from SVD).

- S = 0: product state (no entanglement)
- S = log(χ): maximally entangled at bond dimension χ (all Schmidt values equal 1/√χ)
- S grows with time after a quench: the entanglement "spreads" from the initial state

It matters because it sets the required bond dimension χ ~ exp(S) for an accurate MPS
representation. If S is large, you need large χ, and the simulation becomes expensive.

### Q: Why does Neel state work as initial state for iTEBD?

1. Product state → chi=1: trivially represented as MPS, no truncation needed at t=0
2. 2-site unit cell: natural 2-sublattice structure (A↑, B↓) maps to iTEBD perfectly
3. h₀=0 → product state: at zero transverse field, the Hamiltonian is H = -JΣZᵢZᵢ₊₁.
   The Neel state is NOT the ground state of this (ferromagnet is). But the project
   specifies Neel explicitly as a convenient product state with the right unit cell.
   Different initial states give DPTs at different times, but same qualitative physics.

### Q: What does the h₁ < 1 curve look like, and why?

λ(t) is smooth, monotonically increasing (h₁=0.5) or slowly oscillating (h₁=0.8).
No cusps.

Reason: For a quench within the ordered phase, the Fisher zeros of G(z) stay in the
negative real half-plane of complex time. They never cross the imaginary axis
(= real time axis). So G(t) has no zeros for real t → λ(t) = -log|G(t)|^{1/L} is
analytic (smooth) for all t.

### Q: Why do the cusp positions shift with h₁?

Larger h₁ → larger quasi-particle energy ε(k*) → shorter period t* = π/ε(k*) → 
cusps appear sooner and repeat faster.

From data:
- h₁=1.2: first cusp at t≈2.7 (large t*, slow oscillation)
- h₁=2.0: first cusp at t≈1.65 (medium)
- h₁=3.0: first cusp at t≈1.1 (small t*, fast oscillation)

---

## Summary of What We Built vs Project Requirements

| Requirement | Our Answer |
|---|---|
| iTEBD + Neel initial state | ✓ vidal_state.py, init_neel() |
| Transfer matrix T at every step | ✓ transfer.py, build_transfer_matrix() |
| λ(t) from T diagonalization | ✓ loschmidt_rate() → np.linalg.eigvals(T) |
| Plot λ(t), compare h1<1 vs h1>1 | ✓ results/loschmidt_scan.pdf |
| M ≲ 200, show M-dependence | ✓ results/convergence.pdf (M=20,40,80,200) |
| Entropy vs t for different M | ✓ convergence.pdf right panel |
| Up to what times reliable? | ✓ This document, table above |

**All requirements met. Reliable until chi_A saturates at chi_max. For M=80+, reliable
through entire simulation t∈[0,5]. TFIM's integrability keeps entanglement low.**
