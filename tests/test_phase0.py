"""
test_phase0.py — Tests for svd_utils.py (Phase 0).

WHY THESE TESTS MATTER
-----------------------
The SVD truncation in svd_truncate() is the lowest-level operation in the
entire iTEBD pipeline.  Every bug here propagates into every gate application
and every measurement.  We test against states with known exact Schmidt spectra
so we can verify correctness analytically before touching MPS code.

TEST STATES
-----------
1. Product state |up>|up>:
   Psi = [[1, 0], [0, 0]]  — rank 1, one Schmidt value = 1, entropy = 0.

2. Bell state (singlet) (|up,down> - |down,up>) / sqrt(2):
   Psi = [[0, 1/sqrt(2)], [-1/sqrt(2), 0]]
   Schmidt values: [1/sqrt(2), 1/sqrt(2)] — both equal, entropy = log(2).
   This is the maximally entangled 2-qubit state.

3. General state with known singular values [0.9, 0.3, 0.1, 0.05] (normalized):
   Used to verify truncation error and chi_max cutoff are applied correctly.

4. Truncation test: chi_max = 2 on a chi = 4 matrix.
   Verify trunc_err = sum of discarded s^2 values.
"""

import numpy as np
import pytest
from tns.svd_utils import svd_truncate, von_neumann_entropy

# Tolerance for floating-point comparisons throughout
ATOL = 1e-12


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def make_matrix_with_singular_values(s_values: np.ndarray) -> np.ndarray:
    """Build a square matrix with prescribed singular values via random SVD.

    The matrix  M = U diag(s) Vh  where U, Vh are random orthogonal matrices.
    This lets us test svd_truncate against a known spectrum without needing to
    hand-craft the matrix entries.
    """
    n = len(s_values)
    rng = np.random.default_rng(seed=42)
    # Random orthogonal matrices via QR decomposition
    U, _ = np.linalg.qr(rng.standard_normal((n, n)))
    Vh, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return U @ np.diag(s_values) @ Vh


# ---------------------------------------------------------------------------
# Test 1: Product state — zero entanglement
# ---------------------------------------------------------------------------

class TestProductState:
    """Product state |up>|up>: Psi[up, up] = 1, all others = 0.

    Expected Schmidt decomposition: single value s = 1, entropy = 0.
    """

    def setup_method(self):
        # Psi[sigma_L, sigma_R], basis: 0=up, 1=down
        self.psi = np.array([[1.0, 0.0],
                              [0.0, 0.0]])

    def test_chi_is_one(self):
        result = svd_truncate(self.psi, chi_max=4)
        assert result.chi == 1, f"Product state must have chi=1, got {result.chi}"

    def test_singular_value_is_one(self):
        result = svd_truncate(self.psi, chi_max=4)
        assert abs(result.S[0] - 1.0) < ATOL, f"Schmidt value must be 1.0, got {result.S[0]}"

    def test_zero_entropy(self):
        result = svd_truncate(self.psi, chi_max=4)
        assert abs(result.entropy) < ATOL, f"Entropy must be 0 for product state, got {result.entropy}"

    def test_zero_truncation_error(self):
        result = svd_truncate(self.psi, chi_max=4)
        assert abs(result.trunc_err) < ATOL, f"No truncation for chi=1 state, got {result.trunc_err}"

    def test_reconstruction(self):
        """U @ diag(S) @ Vh must reconstruct original psi (up to norm)."""
        result = svd_truncate(self.psi, chi_max=4)
        psi_reconstructed = result.U @ np.diag(result.S) @ result.Vh
        # Original psi has norm 1, reconstructed has norm 1 (normalized), so equal
        np.testing.assert_allclose(
            np.abs(psi_reconstructed), np.abs(self.psi), atol=ATOL,
            err_msg="Reconstruction failed for product state"
        )

    def test_norm_is_one(self):
        result = svd_truncate(self.psi, chi_max=4)
        assert abs(np.sum(result.S ** 2) - 1.0) < ATOL, "sum(S^2) must equal 1"


# ---------------------------------------------------------------------------
# Test 2: Singlet (Bell state) — maximal entanglement
# ---------------------------------------------------------------------------

class TestSingletState:
    """Singlet state (|up,down> - |down,up>) / sqrt(2).

    Coefficient matrix Psi[sigma_L, sigma_R]:
        Psi[up=0, down=1] = +1/sqrt(2)
        Psi[down=1, up=0] = -1/sqrt(2)

    Expected:
        Schmidt values: [1/sqrt(2), 1/sqrt(2)]   (both equal)
        Entropy:  log(2) ≈ 0.6931
        chi:      2
    """
    SQRT2_INV = 1.0 / np.sqrt(2)
    ENTROPY_SINGLET = np.log(2)  # = log(2) ≈ 0.6931...

    def setup_method(self):
        self.psi = np.array([[0.0, self.SQRT2_INV],
                              [-self.SQRT2_INV, 0.0]])

    def test_chi_is_two(self):
        result = svd_truncate(self.psi, chi_max=4)
        assert result.chi == 2, f"Singlet must have chi=2, got {result.chi}"

    def test_schmidt_values_equal(self):
        """Both Schmidt values must equal 1/sqrt(2) — maximally entangled."""
        result = svd_truncate(self.psi, chi_max=4)
        np.testing.assert_allclose(
            np.sort(result.S)[::-1],
            [self.SQRT2_INV, self.SQRT2_INV],
            atol=ATOL,
            err_msg="Singlet Schmidt values must both be 1/sqrt(2)"
        )

    def test_entropy_log2(self):
        result = svd_truncate(self.psi, chi_max=4)
        assert abs(result.entropy - self.ENTROPY_SINGLET) < 1e-10, \
            f"Singlet entropy must be log(2)={self.ENTROPY_SINGLET:.6f}, got {result.entropy:.6f}"

    def test_zero_truncation_error(self):
        result = svd_truncate(self.psi, chi_max=4)
        assert abs(result.trunc_err) < ATOL, \
            f"No truncation needed for chi=2 with chi_max=4, got {result.trunc_err}"

    def test_truncation_to_chi1_loses_half(self):
        """Forcing chi_max=1 on singlet must discard half the weight.

        Keeps s_0 = 1/sqrt(2) — but normalized to 1.0.
        Truncation error = s_1^2 = 1/2.
        """
        result = svd_truncate(self.psi, chi_max=1)
        assert result.chi == 1
        assert abs(result.trunc_err - 0.5) < ATOL, \
            f"Truncation error must be 0.5, got {result.trunc_err}"
        assert abs(result.S[0] - 1.0) < ATOL, \
            "After truncation + normalization, single Schmidt value must be 1.0"

    def test_norm_is_one(self):
        result = svd_truncate(self.psi, chi_max=4)
        assert abs(np.sum(result.S ** 2) - 1.0) < ATOL, "sum(S^2) must equal 1"

    def test_reconstruction(self):
        """Reconstructed matrix magnitude must match original (up to global phase)."""
        result = svd_truncate(self.psi, chi_max=4)
        psi_reconstructed = result.U @ np.diag(result.S) @ result.Vh
        np.testing.assert_allclose(
            np.abs(psi_reconstructed), np.abs(self.psi), atol=ATOL,
            err_msg="Reconstruction failed for singlet state"
        )


# ---------------------------------------------------------------------------
# Test 3: Truncation accuracy on a 4-Schmidt-value state
# ---------------------------------------------------------------------------

class TestTruncation:
    """Verify chi_max cutoff and truncation error on known spectrum.

    Unnormalized Schmidt values: [0.9, 0.3, 0.1, 0.05]
    Norm^2 = 0.81 + 0.09 + 0.01 + 0.0025 = 0.9125
    Normalized: s_k = [0.9, 0.3, 0.1, 0.05] / sqrt(0.9125)
    """

    def setup_method(self):
        # Unnormalized Schmidt values (descending)
        s_raw = np.array([0.9, 0.3, 0.1, 0.05])
        norm = np.linalg.norm(s_raw)
        self.s_normalized = s_raw / norm
        self.norm_sq = norm ** 2  # 0.9125
        self.psi = make_matrix_with_singular_values(s_raw)

    def test_full_spectrum_no_truncation(self):
        result = svd_truncate(self.psi, chi_max=4)
        assert result.chi == 4
        np.testing.assert_allclose(result.S, self.s_normalized, atol=1e-10,
                                   err_msg="Full SVD must recover normalized singular values")

    def test_chi_max_two_truncation_error(self):
        """Keep top 2 values, discard s[2] and s[3].

        After truncation (before renorm), discarded weight = s[2]^2 + s[3]^2
        measured in the original (unnormalized) scale.
        trunc_err = (0.1^2 + 0.05^2) = 0.01 + 0.0025 = 0.0125
        """
        result = svd_truncate(self.psi, chi_max=2)
        expected_trunc = 0.1 ** 2 + 0.05 ** 2  # = 0.0125
        assert result.chi == 2
        assert abs(result.trunc_err - expected_trunc) < 1e-10, \
            f"Truncation error must be {expected_trunc}, got {result.trunc_err}"

    def test_chi_max_two_normalization(self):
        """After keeping top 2 and renormalizing, sum(S^2) must equal 1."""
        result = svd_truncate(self.psi, chi_max=2)
        assert abs(np.sum(result.S ** 2) - 1.0) < ATOL

    def test_chi_max_one_truncation_error(self):
        """Keep only largest singular value.

        trunc_err = s[1]^2 + s[2]^2 + s[3]^2 = 0.09 + 0.01 + 0.0025 = 0.1125
        """
        result = svd_truncate(self.psi, chi_max=1)
        expected_trunc = 0.3 ** 2 + 0.1 ** 2 + 0.05 ** 2  # = 0.1125
        assert result.chi == 1
        assert abs(result.trunc_err - expected_trunc) < 1e-10, \
            f"Truncation error must be {expected_trunc}, got {result.trunc_err}"


# ---------------------------------------------------------------------------
# Test 4: von Neumann entropy helper
# ---------------------------------------------------------------------------

class TestEntropy:
    """Unit tests for von_neumann_entropy()."""

    def test_product_state_zero_entropy(self):
        s = np.array([1.0])
        assert abs(von_neumann_entropy(s)) < ATOL

    def test_singlet_log2(self):
        s = np.array([1.0 / np.sqrt(2), 1.0 / np.sqrt(2)])
        assert abs(von_neumann_entropy(s) - np.log(2)) < 1e-10

    def test_equal_superposition_4(self):
        """4 equal Schmidt values: s_k = 1/2, entropy = log(4)."""
        s = np.array([0.5, 0.5, 0.5, 0.5])
        assert abs(von_neumann_entropy(s) - np.log(4)) < 1e-10

    def test_entropy_positive(self):
        s = np.array([0.8, 0.6])  # sum(s^2) = 0.64 + 0.36 = 1.0
        assert von_neumann_entropy(s) > 0


# ---------------------------------------------------------------------------
# Test 5: Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Numerically tricky inputs."""

    def test_already_rank_1_with_chi_max_1(self):
        """Product state: chi_max=1 should give trunc_err=0."""
        psi = np.array([[1.0, 0.0], [0.0, 0.0]])
        result = svd_truncate(psi, chi_max=1)
        assert result.chi == 1
        assert abs(result.trunc_err) < ATOL

    def test_complex_matrix(self):
        """svd_truncate must handle complex-valued Psi (iTEBD uses complex gates)."""
        psi = np.array([[1.0 + 0j, 0.0 + 1j],
                        [0.0 - 1j, 1.0 + 0j]]) / 2.0
        result = svd_truncate(psi, chi_max=4)
        # Verify norm preservation
        assert abs(np.sum(result.S ** 2) - 1.0) < 1e-10, "Complex matrix norm must be 1"

    def test_rectangular_matrix(self):
        """Non-square Psi: shape (d*chi_L, d*chi_R) is common in TEBD."""
        psi = np.zeros((4, 8))
        psi[0, 0] = 1.0 / np.sqrt(2)
        psi[1, 1] = 1.0 / np.sqrt(2)
        result = svd_truncate(psi, chi_max=10)
        assert result.chi == 2
        assert abs(result.entropy - np.log(2)) < 1e-10
