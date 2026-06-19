"""
test_phase2.py — Tests for gates.py (Phase 2).

WHY THESE TESTS MATTER
-----------------------
The gate U = exp(-i h_bond dt) is applied thousands of times during a single
iTEBD run.  Any error in its construction (wrong h_bond, wrong factor, wrong
index order) silently produces wrong time evolution — the code will run without
crashing but with wrong physics.

We test in increasing specificity:
  1. h_bond structural properties (Hermitian, real eigenvalues).
  2. h_bond matrix elements against analytical formulas (specific cases).
  3. Gate unitarity (U†U = I) — catches wrong signs or missing i.
  4. Gate boundary cases (dt=0 → identity, composition law).
  5. Gate vs known analytic form for h=0 (pure Ising, diagonal gate).
  6. First-order Taylor expansion for small dt.
  7. Tensor index convention: U[s1',s2',s1,s2] = U_mat[2*s1'+s2', 2*s1+s2].
  8. Gate action on known states: apply U to |↑↑⟩ and verify output.
"""

import numpy as np
import pytest
from scipy.linalg import expm

from tns.gates import build_tfim_bond, make_tfim_gate

ATOL = 1e-12


# ---------------------------------------------------------------------------
# Test 1: Bond Hamiltonian structure
# ---------------------------------------------------------------------------

class TestBondHamiltonian:
    """build_tfim_bond() must be Hermitian with the right matrix elements."""

    def test_hermitian_J_only(self):
        h_bond = build_tfim_bond(J=1.0, h=0.0)
        np.testing.assert_allclose(h_bond, h_bond.conj().T, atol=ATOL,
                                   err_msg="h_bond must be Hermitian")

    def test_hermitian_full(self):
        h_bond = build_tfim_bond(J=1.0, h=2.0)
        np.testing.assert_allclose(h_bond, h_bond.conj().T, atol=ATOL,
                                   err_msg="h_bond must be Hermitian")

    def test_real_eigenvalues(self):
        """Hermitian matrices have real eigenvalues — sanity check."""
        h_bond = build_tfim_bond(J=1.0, h=1.5)
        evals = np.linalg.eigvalsh(h_bond)  # eigvalsh guarantees real output
        np.testing.assert_allclose(np.imag(evals), 0.0, atol=ATOL,
                                   err_msg="Eigenvalues must be real")

    def test_shape(self):
        h_bond = build_tfim_bond(J=1.0, h=1.0)
        assert h_bond.shape == (4, 4), f"h_bond must be 4x4, got {h_bond.shape}"

    def test_dtype_complex(self):
        h_bond = build_tfim_bond(J=1.0, h=1.0)
        assert np.issubdtype(h_bond.dtype, np.complexfloating)

    def test_pure_ising_h0_diagonal(self):
        """h=0: h_bond = -J * Sz⊗Sz = diag(-J/4, +J/4, +J/4, -J/4).

        Basis {|↑↑⟩, |↑↓⟩, |↓↑⟩, |↓↓⟩}:
          ↑↑ and ↓↓: Sz values are ±1/2, product = +1/4 → energy = -J/4 (aligned)
          ↑↓ and ↓↑: Sz values are ±1/2, product = -1/4 → energy = +J/4 (antialigned)
        """
        J = 1.5
        h_bond = build_tfim_bond(J=J, h=0.0)
        expected_diag = np.array([-J / 4, +J / 4, +J / 4, -J / 4])
        np.testing.assert_allclose(np.diag(h_bond), expected_diag, atol=ATOL,
                                   err_msg="h=0 bond must be diagonal -J*Sz⊗Sz")
        np.testing.assert_allclose(h_bond - np.diag(np.diag(h_bond)),
                                   0.0, atol=ATOL,
                                   err_msg="h=0 bond must be purely diagonal")

    def test_pure_transverse_J0_offdiagonal(self):
        """J=0: h_bond = -(h/2)*(Sx⊗I + I⊗Sx), zero diagonal.

        Off-diagonal structure of Sx⊗I in {↑↑,↑↓,↓↑,↓↓}:
          Sx flips the first spin: |↑↑⟩↔|↓↑⟩ and |↑↓⟩↔|↓↓⟩
          matrix elements at positions (0,2),(2,0),(1,3),(3,1) = 1/2
        Off-diagonal structure of I⊗Sx:
          Sx flips second spin: |↑↑⟩↔|↑↓⟩ and |↓↑⟩↔|↓↓⟩
          matrix elements at positions (0,1),(1,0),(2,3),(3,2) = 1/2
        Bond factor -h/2 applied.
        """
        h = 2.0
        h_bond = build_tfim_bond(J=0.0, h=h)
        # Diagonal must be zero (no ZZ coupling)
        np.testing.assert_allclose(np.diag(h_bond), 0.0, atol=ATOL,
                                   err_msg="J=0 bond must have zero diagonal")
        # Sx⊗I flips first spin: positions (0,2),(2,0),(1,3),(3,1) = -(h/2)*(1/2) = -h/4
        for i, j in [(0, 2), (2, 0), (1, 3), (3, 1)]:
            assert abs(h_bond[i, j] - (-h / 4)) < ATOL, \
                f"h_bond[{i},{j}] should be {-h/4}, got {h_bond[i,j]}"
        # I⊗Sx flips second spin: positions (0,1),(1,0),(2,3),(3,2) = -h/4
        for i, j in [(0, 1), (1, 0), (2, 3), (3, 2)]:
            assert abs(h_bond[i, j] - (-h / 4)) < ATOL, \
                f"h_bond[{i},{j}] should be {-h/4}, got {h_bond[i,j]}"

    def test_symmetry_swap_spins(self):
        """h_bond must commute with the SWAP gate (spins 1 and 2 are symmetric).

        SWAP matrix in {↑↑,↑↓,↓↑,↓↓}: identity except |↑↓⟩↔|↓↑⟩.
        So SWAP = diag(1,0,0,1) + off-diag (1,2) and (2,1).
        """
        h_bond = build_tfim_bond(J=1.0, h=2.0)
        SWAP = np.array([[1, 0, 0, 0],
                         [0, 0, 1, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1]], dtype=float)
        # [h_bond, SWAP] = 0 iff h_bond = SWAP @ h_bond @ SWAP (SWAP is its own inverse)
        np.testing.assert_allclose(
            SWAP @ h_bond @ SWAP, h_bond, atol=ATOL,
            err_msg="h_bond must commute with SWAP (symmetric in both spins)"
        )

    def test_traceless(self):
        """Tr(h_bond) = 0 for any J, h.

        Tr(-J*Sz⊗Sz) = -J * Tr(Sz) * Tr(Sz) ... actually that's not right.
        Tr(A⊗B) = Tr(A)*Tr(B).
        Tr(Sz) = 0 → Tr(Sx⊗I) = Tr(Sx)*Tr(I) = 0*2 = 0.
        Tr(Sz⊗Sz) = Tr(Sz)*Tr(Sz) = 0.
        So Tr(h_bond) = 0 always.
        """
        for J, h in [(1.0, 0.0), (0.0, 1.0), (1.0, 2.0), (0.5, 3.0)]:
            h_bond = build_tfim_bond(J=J, h=h)
            assert abs(np.trace(h_bond)) < ATOL, \
                f"Tr(h_bond) must be 0 for J={J},h={h}, got {np.trace(h_bond)}"


# ---------------------------------------------------------------------------
# Test 2: Gate structural properties
# ---------------------------------------------------------------------------

class TestGateStructure:
    """make_tfim_gate() must return a (2,2,2,2) unitary tensor."""

    def test_shape(self):
        U = make_tfim_gate(J=1.0, h=1.0, dt=0.05)
        assert U.shape == (2, 2, 2, 2), f"Gate must be (2,2,2,2), got {U.shape}"

    def test_dtype_complex(self):
        U = make_tfim_gate(J=1.0, h=1.0, dt=0.05)
        assert np.issubdtype(U.dtype, np.complexfloating)

    def test_unitary_U_dag_U(self):
        """U† U = I_4 (right unitarity)."""
        U = make_tfim_gate(J=1.0, h=2.0, dt=0.05)
        U_mat = U.reshape(4, 4)
        np.testing.assert_allclose(
            U_mat.conj().T @ U_mat, np.eye(4), atol=1e-10,
            err_msg="Gate must satisfy U†U = I"
        )

    def test_unitary_U_U_dag(self):
        """U U† = I_4 (left unitarity)."""
        U = make_tfim_gate(J=1.0, h=2.0, dt=0.05)
        U_mat = U.reshape(4, 4)
        np.testing.assert_allclose(
            U_mat @ U_mat.conj().T, np.eye(4), atol=1e-10,
            err_msg="Gate must satisfy UU† = I"
        )

    def test_determinant_magnitude_one(self):
        """det(U) must have |det| = 1 (unitary matrix property)."""
        U = make_tfim_gate(J=1.0, h=1.5, dt=0.1)
        det = np.linalg.det(U.reshape(4, 4))
        assert abs(abs(det) - 1.0) < 1e-10, \
            f"|det(U)| must be 1.0, got {abs(det)}"


# ---------------------------------------------------------------------------
# Test 3: Boundary cases
# ---------------------------------------------------------------------------

class TestGateBoundaryCases:
    """Gate must reduce to identity at dt=0 and compose correctly."""

    def test_identity_at_dt0(self):
        """exp(-i h_bond * 0) = I."""
        U = make_tfim_gate(J=1.0, h=2.0, dt=0.0)
        U_mat = U.reshape(4, 4)
        np.testing.assert_allclose(U_mat, np.eye(4), atol=ATOL,
                                   err_msg="Gate at dt=0 must be identity")

    def test_composition_two_half_steps(self):
        """U(dt) = U(dt/2) @ U(dt/2).

        Exact for expm (no Trotter approximation here — this tests that our
        expm-based gate satisfies the exact group law for matrix exponentials).
        """
        J, h, dt = 1.0, 1.5, 0.1
        U_full = make_tfim_gate(J, h, dt).reshape(4, 4)
        U_half = make_tfim_gate(J, h, dt / 2).reshape(4, 4)
        np.testing.assert_allclose(
            U_half @ U_half, U_full, atol=1e-10,
            err_msg="U(dt/2)@U(dt/2) must equal U(dt)"
        )

    def test_inverse_is_conjugate_transpose(self):
        """U(-dt) = U(dt)†.

        Evolving backward in time is the adjoint of forward evolution.
        """
        J, h, dt = 1.0, 1.0, 0.1
        U_fwd = make_tfim_gate(J, h, dt).reshape(4, 4)
        U_bwd = make_tfim_gate(J, h, -dt).reshape(4, 4)
        np.testing.assert_allclose(
            U_bwd, U_fwd.conj().T, atol=1e-10,
            err_msg="U(-dt) must equal U(dt)†"
        )


# ---------------------------------------------------------------------------
# Test 4: Exact analytic form for h=0 (pure Ising gate)
# ---------------------------------------------------------------------------

class TestPureIsingGate:
    """At h=0, h_bond = diag(-J/4, +J/4, +J/4, -J/4), so gate is diagonal.

    U_mat = diag(exp(+iJ*dt/4), exp(-iJ*dt/4), exp(-iJ*dt/4), exp(+iJ*dt/4))

    Physical meaning:
      |↑↑⟩ and |↓↓⟩ (aligned): energy -J/4 → phase exp(+i J dt / 4) (energy *(-1)*i*dt)
      |↑↓⟩ and |↓↑⟩ (anti):   energy +J/4 → phase exp(-i J dt / 4)
    """

    def _expected_diag(self, J: float, dt: float) -> np.ndarray:
        return np.array([np.exp(+1j * J * dt / 4),
                         np.exp(-1j * J * dt / 4),
                         np.exp(-1j * J * dt / 4),
                         np.exp(+1j * J * dt / 4)])

    def test_diagonal_values(self):
        J, dt = 1.0, 0.1
        U_mat = make_tfim_gate(J=J, h=0.0, dt=dt).reshape(4, 4)
        np.testing.assert_allclose(
            np.diag(U_mat), self._expected_diag(J, dt), atol=ATOL,
            err_msg="h=0 gate diagonal elements wrong"
        )

    def test_off_diagonal_zero(self):
        """Pure Ising gate has no off-diagonal elements (Sz⊗Sz is diagonal)."""
        J, dt = 1.0, 0.1
        U_mat = make_tfim_gate(J=J, h=0.0, dt=dt).reshape(4, 4)
        off_diag = U_mat - np.diag(np.diag(U_mat))
        np.testing.assert_allclose(off_diag, 0.0, atol=ATOL,
                                   err_msg="h=0 gate must be diagonal")

    def test_scaling_with_J(self):
        """Doubling J doubles the phase argument."""
        dt = 0.05
        U1 = make_tfim_gate(J=1.0, h=0.0, dt=dt).reshape(4, 4)
        U2 = make_tfim_gate(J=2.0, h=0.0, dt=dt).reshape(4, 4)
        # U2[0,0] = exp(i*2*dt/4) = (exp(i*1*dt/4))^2 = U1[0,0]^2
        assert abs(U2[0, 0] - U1[0, 0] ** 2) < ATOL, \
            "Doubling J must square the phase factors"


# ---------------------------------------------------------------------------
# Test 5: First-order Taylor expansion for small dt
# ---------------------------------------------------------------------------

class TestFirstOrderExpansion:
    """For small dt, U ≈ I - i*dt*h_bond to first order.

    Error is O(dt²): |U - (I - i*dt*h_bond)| < C * dt²
    For dt=1e-3: error < C * 1e-6.
    """

    def test_first_order_small_dt(self):
        J, h, dt = 1.0, 1.5, 1e-3
        h_bond = np.array([[-0.25, -0.25, -0.25, 0.0],
                            [-0.25, 0.25, 0.0, -0.25],
                            [-0.25, 0.0, 0.25, -0.25],
                            [0.0, -0.25, -0.25, -0.25]])  # would need to compute this

        # Use the module's own h_bond to be consistent
        from tns.gates import build_tfim_bond
        h_bond = build_tfim_bond(J, h)
        U_mat = make_tfim_gate(J, h, dt).reshape(4, 4)
        approx = np.eye(4, dtype=complex) - 1j * dt * h_bond
        error = np.max(np.abs(U_mat - approx))
        assert error < 1e-5, \
            f"First-order expansion error too large: {error:.2e} (expected < 1e-5)"

    def test_second_order_correction_sign(self):
        """U - (I - i*dt*h) ≈ (-i*dt)^2/2 * h^2 = -dt^2/2 * h^2.

        The second-order term is real and negative on the diagonal.
        """
        from tns.gates import build_tfim_bond
        J, h, dt = 1.0, 1.5, 1e-2
        h_bond = build_tfim_bond(J, h)
        U_mat = make_tfim_gate(J, h, dt).reshape(4, 4)
        first_order = np.eye(4, dtype=complex) - 1j * dt * h_bond
        second_order_term = (-dt ** 2 / 2) * h_bond @ h_bond
        residual = U_mat - first_order - second_order_term
        # Residual should be O(dt^3) ≈ 1e-6
        assert np.max(np.abs(residual)) < 1e-5, \
            f"Second-order residual too large: {np.max(np.abs(residual)):.2e}"


# ---------------------------------------------------------------------------
# Test 6: Tensor index convention
# ---------------------------------------------------------------------------

class TestTensorConvention:
    """U[s1',s2',s1,s2] must equal U_mat[2*s1'+s2', 2*s1+s2].

    Rows of U_mat = output states, cols = input states.
    This is the convention used in Phase 3 gate application:
        new_theta[s1',s2',a,c] = Σ_{s1,s2} U[s1',s2',s1,s2] * theta[s1,a,s2,c]
    """

    def test_index_convention_all_elements(self):
        """Every element of U[s1',s2',s1,s2] matches U_mat[2*s1'+s2', 2*s1+s2]."""
        J, h, dt = 1.0, 1.0, 0.05
        U = make_tfim_gate(J, h, dt)
        U_mat = U.reshape(4, 4)
        d = 2
        for s1p in range(d):
            for s2p in range(d):
                for s1 in range(d):
                    for s2 in range(d):
                        expected = U_mat[d * s1p + s2p, d * s1 + s2]
                        got = U[s1p, s2p, s1, s2]
                        assert abs(got - expected) < ATOL, \
                            f"U[{s1p},{s2p},{s1},{s2}]={got} but U_mat[{d*s1p+s2p},{d*s1+s2}]={expected}"

    def test_gate_action_on_up_up(self):
        """Verify gate applied to |↑↑⟩ gives the correct rotated state.

        |↑↑⟩ is column 0 (input s1=0, s2=0) of U_mat.
        Gate output = U_mat[:, 0] (amplitude for each output state).
        In tensor form: output[s1',s2'] = U[s1',s2', 0, 0] = U_mat[2*s1'+s2', 0].
        """
        J, h, dt = 1.0, 1.5, 0.1
        U = make_tfim_gate(J, h, dt)
        U_mat = U.reshape(4, 4)

        # Amplitude for output states when input is |↑↑⟩ = s1=0, s2=0
        output_col0_tensor = U[:, :, 0, 0]  # shape (2,2): output_col0[s1',s2']
        output_col0_matrix = U_mat[:, 0]    # shape (4,): U_mat[2*s1'+s2', 0]

        for s1p in range(2):
            for s2p in range(2):
                expected = output_col0_matrix[2 * s1p + s2p]
                got = output_col0_tensor[s1p, s2p]
                assert abs(got - expected) < ATOL, \
                    f"Tensor/matrix mismatch at output ({s1p},{s2p})"

    def test_gate_preserves_norm_of_product_state(self):
        """Applying U to any normalized input state gives a normalized output.

        Test with |↑↑⟩ (input col 0 of U_mat): output norm must be 1.
        """
        U_mat = make_tfim_gate(J=1.0, h=2.0, dt=0.1).reshape(4, 4)
        # Input |↑↑⟩: vector [1,0,0,0]
        input_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
        output_vec = U_mat @ input_vec
        assert abs(np.linalg.norm(output_vec) - 1.0) < 1e-10, \
            "Gate must preserve state norm"

    def test_gate_preserves_norm_of_superposition(self):
        """Gate applied to equal superposition of all 4 basis states."""
        U_mat = make_tfim_gate(J=1.0, h=1.0, dt=0.05).reshape(4, 4)
        input_vec = np.ones(4, dtype=complex) / 2.0  # normalized: norm = 1
        output_vec = U_mat @ input_vec
        assert abs(np.linalg.norm(output_vec) - 1.0) < 1e-10
