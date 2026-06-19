"""
test_phase1.py — Tests for vidal_state.py (Phase 1).

WHY THESE TESTS MATTER
-----------------------
The VidalMPS dataclass and its initialization are the scaffold that all later
code rests on.  If the index conventions or Néel initialization are wrong,
every gate application will produce wrong results silently (gate will run
without error, but on the wrong physical state).

We test five independent claims about the Néel state:
  1. Tensor shapes are correct.
  2. State norm equals 1.
  3. Entanglement entropy equals 0 (product state, no entanglement).
  4. <Sz>_A = +0.5 (spin up at A sites).
  5. <Sz>_B = -0.5 (spin down at B sites).
  6. Canonical conditions (C1) and (C2) are satisfied.
  7. check_canonical detects a violation when tensors are modified.

We also test that measure_one_site gives correct results for states with
known expectation values (maximally mixed, eigenstates of Sz).
"""

import numpy as np
import pytest
from tns.vidal_state import (
    VidalMPS, init_neel, norm, entropy_A, entropy_B,
    measure_sz, measure_one_site, check_canonical
)

ATOL = 1e-12


class TestNeelInit:
    """Structural correctness of init_neel()."""

    def setup_method(self):
        self.state = init_neel(d=2)

    def test_gamma_A_shape(self):
        # (d=2, chi_B=1, chi_A=1)
        assert self.state.Gamma_A.shape == (2, 1, 1), \
            f"Gamma_A shape wrong: {self.state.Gamma_A.shape}"

    def test_gamma_B_shape(self):
        # (d=2, chi_A=1, chi_B=1)
        assert self.state.Gamma_B.shape == (2, 1, 1), \
            f"Gamma_B shape wrong: {self.state.Gamma_B.shape}"

    def test_lambda_A_shape(self):
        assert self.state.Lambda_A.shape == (1,)

    def test_lambda_B_shape(self):
        assert self.state.Lambda_B.shape == (1,)

    def test_chi_A_property(self):
        assert self.state.chi_A == 1

    def test_chi_B_property(self):
        assert self.state.chi_B == 1

    def test_gamma_A_spin_up(self):
        """A sites must be spin up: Gamma_A[0,0,0] = 1, all others = 0."""
        assert abs(self.state.Gamma_A[0, 0, 0] - 1.0) < ATOL, \
            "Gamma_A[0,0,0] must be 1 (spin up)"
        assert abs(self.state.Gamma_A[1, 0, 0]) < ATOL, \
            "Gamma_A[1,0,0] must be 0 (no spin down at A site)"

    def test_gamma_B_spin_down(self):
        """B sites must be spin down: Gamma_B[1,0,0] = 1, all others = 0."""
        assert abs(self.state.Gamma_B[1, 0, 0] - 1.0) < ATOL, \
            "Gamma_B[1,0,0] must be 1 (spin down)"
        assert abs(self.state.Gamma_B[0, 0, 0]) < ATOL, \
            "Gamma_B[0,0,0] must be 0 (no spin up at B site)"

    def test_lambda_A_value(self):
        assert abs(self.state.Lambda_A[0] - 1.0) < ATOL

    def test_lambda_B_value(self):
        assert abs(self.state.Lambda_B[0] - 1.0) < ATOL

    def test_dtype_complex(self):
        """Gamma tensors must be complex — gates act in complex space."""
        assert np.issubdtype(self.state.Gamma_A.dtype, np.complexfloating), \
            "Gamma_A must be complex dtype"
        assert np.issubdtype(self.state.Gamma_B.dtype, np.complexfloating), \
            "Gamma_B must be complex dtype"


class TestNormAndEntropy:
    """Norm = 1 and entropy = 0 for Néel state (product state)."""

    def setup_method(self):
        self.state = init_neel()

    def test_norm_one(self):
        assert abs(norm(self.state) - 1.0) < ATOL, \
            f"Norm must be 1.0 for Néel state, got {norm(self.state)}"

    def test_entropy_A_zero(self):
        S = entropy_A(self.state)
        assert abs(S) < ATOL, f"Entropy_A must be 0 for product state, got {S}"

    def test_entropy_B_zero(self):
        S = entropy_B(self.state)
        assert abs(S) < ATOL, f"Entropy_B must be 0 for product state, got {S}"

    def test_lambda_A_normalized(self):
        """sum(Lambda_A**2) must equal 1."""
        assert abs(np.sum(self.state.Lambda_A ** 2) - 1.0) < ATOL

    def test_lambda_B_normalized(self):
        assert abs(np.sum(self.state.Lambda_B ** 2) - 1.0) < ATOL


class TestMeasureSz:
    """Local <Sz> on Néel state.

    Expected: <Sz>_A = +0.5 (spin up), <Sz>_B = -0.5 (spin down).
    """

    def setup_method(self):
        self.state = init_neel()

    def test_sz_A_plus_half(self):
        sz_A, _ = measure_sz(self.state)
        assert abs(sz_A - 0.5) < ATOL, \
            f"<Sz>_A must be +0.5 for Néel state, got {sz_A}"

    def test_sz_B_minus_half(self):
        _, sz_B = measure_sz(self.state)
        assert abs(sz_B - (-0.5)) < ATOL, \
            f"<Sz>_B must be -0.5 for Néel state, got {sz_B}"

    def test_total_magnetization_zero(self):
        """Néel state has zero total magnetization by symmetry."""
        sz_A, sz_B = measure_sz(self.state)
        assert abs(sz_A + sz_B) < ATOL, \
            f"<Sz>_A + <Sz>_B must be 0, got {sz_A + sz_B}"


class TestMeasureOneSite:
    """Test measure_one_site against analytically known states.

    Uses a chi=1 Vidal state where the physical state is a known eigenstate
    or superposition, so we can compute expectation values by hand.
    """

    def _make_product_state(self, d: int, phys_state: np.ndarray) -> tuple:
        """Make chi=1 Vidal tensors for a given physical state vector.

        phys_state: shape (d,), normalized.
        Returns (Gamma, Lambda_left=[1], Lambda_right=[1]).
        """
        Gamma = np.zeros((d, 1, 1), dtype=complex)
        Gamma[:, 0, 0] = phys_state
        return Gamma, np.array([1.0]), np.array([1.0])

    def test_spin_up_sz(self):
        """Pure |↑⟩ state: <Sz> = +0.5."""
        G, L, R = self._make_product_state(2, np.array([1.0, 0.0]))
        Sz = np.array([[0.5, 0.0], [0.0, -0.5]])
        val = measure_one_site(G, L, R, Sz)
        assert abs(val - 0.5) < ATOL

    def test_spin_down_sz(self):
        """Pure |↓⟩ state: <Sz> = -0.5."""
        G, L, R = self._make_product_state(2, np.array([0.0, 1.0]))
        Sz = np.array([[0.5, 0.0], [0.0, -0.5]])
        val = measure_one_site(G, L, R, Sz)
        assert abs(val - (-0.5)) < ATOL

    def test_sx_eigenstate(self):
        """State |+x⟩ = (|↑⟩ + |↓⟩)/√2: <Sz> = 0, <Sx> = +0.5."""
        G, L, R = self._make_product_state(2, np.array([1.0, 1.0]) / np.sqrt(2))
        Sz = np.array([[0.5, 0.0], [0.0, -0.5]])
        Sx = np.array([[0.0, 0.5], [0.5, 0.0]])
        assert abs(measure_one_site(G, L, R, Sz)) < ATOL, "<Sz> should be 0 for |+x>"
        assert abs(measure_one_site(G, L, R, Sx) - 0.5) < ATOL, "<Sx> should be +0.5 for |+x>"

    def test_identity_gives_one(self):
        """<I> = 1 for any normalized state."""
        G, L, R = self._make_product_state(2, np.array([0.6, 0.8]))  # norm = 1
        I2 = np.eye(2)
        val = measure_one_site(G, L, R, I2)
        assert abs(val - 1.0) < ATOL, f"<I> must equal 1.0, got {val}"

    def test_schmidt_environment_weighted(self):
        """Test that Lambda weights enter correctly for chi=2 state.

        Build a chi=2 Vidal tensor manually where we know the exact rho.
        State at site A: Γ^A[↑,0,:] = [1,0], Γ^A[↓,1,:] = [0,1]
        Lambda_B = [1/√2, 1/√2], Lambda_A = [1/√2, 1/√2]

        rho_A[s,s'] = Σ_{a,b} (1/2) Γ[s,a,b] (1/2) Γ*[s',a,b]
            = (1/2)^2 * 2  *  diagonal
        Actually: rho_A = (1/4)*Γ[↑,0,:]*Γ*[↑,0,:] + (1/4)*Γ[↓,1,:]*Γ*[↓,1,:]
                       = (1/4)*[1,0]*[1,0] + (1/4)*[0,1]*[0,1] summed over b
        Wait, let me redo:
          GW[s,a,b] = Λ_B[a] Γ[s,a,b] Λ_A[b]
          GW[↑,0,0] = (1/√2) * 1 * (1/√2) = 1/2
          GW[↓,1,1] = (1/√2) * 1 * (1/√2) = 1/2
          others = 0
          rho[↑,↑] = |GW[↑,0,0]|² + |GW[↑,1,1]|² = 1/4
          rho[↓,↓] = |GW[↓,0,0]|² + |GW[↓,1,1]|² = 1/4
          -> sum of rho diagonal = 1/4 + 1/4 = 1/2 ≠ 1 ??

        Hmm, that's wrong. Let me reconsider.

        The GW tensor sums over BOTH (a,b) — so rho[↑,↑] includes contributions
        from all (a,b) pairs where Γ[↑,a,b] ≠ 0.

        With Γ[↑,0,0]=1, Γ[↓,1,0]=1 (not [1,1]):
        Gamma_A shape (d=2, chi_B=2, chi_A=1) — single right bond
        Γ[↑,0,0] = 1 (a=0, b=0)
        Γ[↓,1,0] = 1 (a=1, b=0)
        Lambda_B = [1/√2, 1/√2], Lambda_A = [1.0] (chi_A=1, normalized)
        GW[↑,0,0] = (1/√2) * 1 * 1.0 = 1/√2
        GW[↓,1,0] = (1/√2) * 1 * 1.0 = 1/√2
        rho[↑,↑] = |GW[↑,0,0]|² + |GW[↑,1,0]|² = 1/2 + 0 = 1/2
        rho[↓,↓] = |GW[↓,0,0]|² + |GW[↓,1,0]|² = 0 + 1/2 = 1/2
        rho = diag(1/2, 1/2) — maximally mixed! <Sz> = 0.
        """
        d, chi_B, chi_A = 2, 2, 1
        Gamma = np.zeros((d, chi_B, chi_A), dtype=complex)
        Gamma[0, 0, 0] = 1.0  # spin up at left bond 0
        Gamma[1, 1, 0] = 1.0  # spin down at left bond 1
        Lambda_left = np.array([1.0 / np.sqrt(2), 1.0 / np.sqrt(2)])  # Lambda_B
        Lambda_right = np.array([1.0])  # Lambda_A (chi_A=1, normalized)

        Sz = np.array([[0.5, 0.0], [0.0, -0.5]])
        val = measure_one_site(Gamma, Lambda_left, Lambda_right, Sz)
        assert abs(val) < ATOL, \
            f"Maximally mixed reduced state should give <Sz>=0, got {val}"


class TestCanonicalConditions:
    """Verify check_canonical() on the Néel state and detect violations."""

    def setup_method(self):
        self.state = init_neel()

    def test_neel_is_canonical(self):
        result = check_canonical(self.state)
        assert result['left_ortho'], \
            "Néel state must satisfy left orthonormality (C1)"
        assert result['right_ortho'], \
            "Néel state must satisfy right orthonormality (C2)"

    def test_violation_detected(self):
        """Breaking Gamma_A should violate the canonical condition."""
        import copy
        bad = copy.deepcopy(self.state)
        bad.Gamma_A[0, 0, 0] = 2.0  # unnormalized — violates (C1)
        result = check_canonical(bad, atol=1e-6)
        assert not result['left_ortho'], \
            "Modified Gamma_A should fail left orthonormality check"
