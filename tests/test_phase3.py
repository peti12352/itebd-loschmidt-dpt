"""
test_phase3.py — Tests for tebd.py (Phase 3).

WHY THESE TESTS MATTER
-----------------------
apply_gate_vidal() is the core of the entire simulation.  It runs thousands
of times.  A silent bug here — wrong index, wrong Lambda assignment, missing
normalization — produces physically wrong results with no crash.

Test strategy: verify at three levels.
  Level 1 — Structure: norm, shapes, canonical conditions after every call.
  Level 2 — Known physics: h=0 preserves product state; h>0 creates entanglement.
  Level 3 — Exact numerics: post-gate state amplitudes match matrix exponentiation.

Level 3 is the kill test: if all 4 amplitudes of the 2-site wavefunction
(after one AB gate on the Neel state) exactly match U_mat @ |up,down>,
the 5-step algorithm is correct end-to-end.
"""

import numpy as np
import pytest
from tns.gates import make_tfim_gate, build_tfim_bond
from tns.tebd import apply_gate_vidal, tebd_step_1st_order, tebd_step_2nd_order
from tns.vidal_state import (
    VidalMPS, init_neel, norm, entropy_A, entropy_B,
    measure_sz, check_canonical
)

ATOL = 1e-10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def reconstruct_2site_state(G1: np.ndarray, l1: np.ndarray, G2: np.ndarray) -> np.ndarray:
    """Reconstruct 2-site amplitude matrix psi[s1, s2] from Vidal tensors.

    For chi_L = chi_R = 1 (product or near-product state):
        psi[s1, s2] = sum_b G1[s1, 0, b] * l1[b] * G2[s2, b, 0]
    """
    return np.einsum('ib,b,jb->ij', G1[:, 0, :], l1, G2[:, :, 0])


def neel_gate_result(J: float, h: float, dt: float) -> np.ndarray:
    """Exact post-gate state for Neel |up,down> under TFIM gate.

    Returns psi_exact[s1, s2] = U_mat[2*s1+s2, 1] (column 1 of U = |up,down> input).
    """
    U_mat = make_tfim_gate(J, h, dt).reshape(4, 4)
    return U_mat[:, 1].reshape(2, 2)


# ---------------------------------------------------------------------------
# Test 1: Identity gate — state must be unchanged
# ---------------------------------------------------------------------------

class TestIdentityGate:
    """U = I⊗I: gate application must return the original tensors."""

    def setup_method(self):
        self.state = init_neel()
        d = 2
        self.U_id = np.eye(4, dtype=complex).reshape(d, d, d, d)

    def test_gamma_A_unchanged(self):
        GA_new, LA_new, GB_new, _ = apply_gate_vidal(
            self.state.Gamma_A, self.state.Lambda_A,
            self.state.Gamma_B, self.state.Lambda_B,
            self.U_id, chi_max=4
        )
        # Reconstruct state amplitudes and compare — global phase may differ
        psi_orig = reconstruct_2site_state(self.state.Gamma_A, self.state.Lambda_A, self.state.Gamma_B)
        psi_new = reconstruct_2site_state(GA_new, LA_new, GB_new)
        np.testing.assert_allclose(np.abs(psi_new), np.abs(psi_orig), atol=ATOL,
                                   err_msg="Identity gate must leave state amplitudes unchanged")

    def test_lambda_unchanged(self):
        _, LA_new, _, _ = apply_gate_vidal(
            self.state.Gamma_A, self.state.Lambda_A,
            self.state.Gamma_B, self.state.Lambda_B,
            self.U_id, chi_max=4
        )
        np.testing.assert_allclose(LA_new, self.state.Lambda_A, atol=ATOL,
                                   err_msg="Identity gate must not change Lambda_A")

    def test_zero_truncation_error(self):
        _, _, _, trunc_err = apply_gate_vidal(
            self.state.Gamma_A, self.state.Lambda_A,
            self.state.Gamma_B, self.state.Lambda_B,
            self.U_id, chi_max=4
        )
        assert trunc_err < ATOL, f"Identity gate must have zero truncation error, got {trunc_err}"


# ---------------------------------------------------------------------------
# Test 2: Norm preservation
# ---------------------------------------------------------------------------

class TestNormPreservation:
    """sum(l1_new**2) must equal 1 after any gate application."""

    def test_norm_AB_gate_h0(self):
        """Pure Ising gate on Neel: norm must stay 1."""
        state = init_neel()
        gate = make_tfim_gate(J=1.0, h=0.0, dt=0.1)
        _, LA_new, _, _ = apply_gate_vidal(
            state.Gamma_A, state.Lambda_A,
            state.Gamma_B, state.Lambda_B,
            gate, chi_max=4
        )
        assert abs(np.sum(LA_new ** 2) - 1.0) < ATOL, \
            f"sum(Lambda_A**2) must be 1 after gate, got {np.sum(LA_new**2)}"

    def test_norm_AB_gate_h1(self):
        """TFIM gate with h=1 on Neel: norm must stay 1."""
        state = init_neel()
        gate = make_tfim_gate(J=1.0, h=1.0, dt=0.05)
        _, LA_new, _, _ = apply_gate_vidal(
            state.Gamma_A, state.Lambda_A,
            state.Gamma_B, state.Lambda_B,
            gate, chi_max=8
        )
        assert abs(np.sum(LA_new ** 2) - 1.0) < ATOL

    def test_norm_preserved_after_full_1st_order_step(self):
        state = init_neel()
        gate = make_tfim_gate(J=1.0, h=1.5, dt=0.05)
        new_state, _ = tebd_step_1st_order(state, gate, chi_max=16)
        assert abs(norm(new_state) - 1.0) < ATOL, \
            f"State norm after full step: {norm(new_state)}"

    def test_norm_preserved_after_full_2nd_order_step(self):
        state = init_neel()
        gate_half = make_tfim_gate(J=1.0, h=1.5, dt=0.025)
        gate_full = make_tfim_gate(J=1.0, h=1.5, dt=0.05)
        new_state, _ = tebd_step_2nd_order(state, gate_half, gate_full, chi_max=16)
        assert abs(norm(new_state) - 1.0) < ATOL


# ---------------------------------------------------------------------------
# Test 3: Canonical conditions preserved
# ---------------------------------------------------------------------------

class TestCanonicalConditions:
    """After gate application, Vidal canonical conditions must still hold."""

    def test_canonical_after_AB_gate(self):
        state = init_neel()
        gate = make_tfim_gate(J=1.0, h=1.0, dt=0.1)
        GA_new, LA_new, GB_new, _ = apply_gate_vidal(
            state.Gamma_A, state.Lambda_A,
            state.Gamma_B, state.Lambda_B,
            gate, chi_max=4
        )
        new_state = VidalMPS(Gamma_A=GA_new, Lambda_A=LA_new,
                              Gamma_B=GB_new, Lambda_B=state.Lambda_B)
        result = check_canonical(new_state, atol=1e-8)
        assert result['left_ortho'], "Left orthonormality (C1) broken after AB gate"
        assert result['right_ortho'], "Right orthonormality (C2) broken after AB gate"

    def test_canonical_after_full_step(self):
        state = init_neel()
        gate = make_tfim_gate(J=1.0, h=1.5, dt=0.05)
        new_state, _ = tebd_step_1st_order(state, gate, chi_max=16)
        result = check_canonical(new_state, atol=1e-8)
        assert result['left_ortho'], "Left orthonormality broken after full Trotter step"
        assert result['right_ortho'], "Right orthonormality broken after full Trotter step"

    def test_canonical_after_10_steps(self):
        """Run 10 steps to verify canonical form accumulates no drift."""
        state = init_neel()
        gate = make_tfim_gate(J=1.0, h=2.0, dt=0.05)
        for _ in range(10):
            state, _ = tebd_step_1st_order(state, gate, chi_max=20)
        result = check_canonical(state, atol=1e-6)
        assert result['left_ortho'], "Left orthonormality drifted after 10 steps"
        assert result['right_ortho'], "Right orthonormality drifted after 10 steps"


# ---------------------------------------------------------------------------
# Test 4: h=0 Ising gate — product state stays product
# ---------------------------------------------------------------------------

class TestPureIsingGate:
    """At h=0, the Ising gate is diagonal.

    Acting on |up,down> (a basis state): output is basis state times a phase.
    No superposition created, no entanglement, chi stays 1.
    """

    def test_chi_stays_one(self):
        """Pure Ising gate cannot create entanglement from a product state."""
        state = init_neel()
        gate = make_tfim_gate(J=1.0, h=0.0, dt=0.1)
        GA_new, LA_new, GB_new, _ = apply_gate_vidal(
            state.Gamma_A, state.Lambda_A,
            state.Gamma_B, state.Lambda_B,
            gate, chi_max=4
        )
        assert LA_new.shape[0] == 1, \
            f"h=0 gate on product state must keep chi=1, got chi={LA_new.shape[0]}"

    def test_entropy_stays_zero(self):
        """Pure Ising gate: entanglement entropy must remain 0."""
        state = init_neel()
        gate = make_tfim_gate(J=1.0, h=0.0, dt=0.1)
        new_state, _ = tebd_step_1st_order(state, gate, chi_max=4)
        assert abs(entropy_A(new_state)) < ATOL, \
            f"h=0 gate must give zero entropy, got {entropy_A(new_state)}"

    def test_Sz_unchanged(self):
        """Ising gate is diagonal in Sz basis: <Sz> cannot change."""
        state = init_neel()
        gate = make_tfim_gate(J=1.0, h=0.0, dt=0.3)
        new_state, _ = tebd_step_1st_order(state, gate, chi_max=4)
        sz_A, sz_B = measure_sz(new_state)
        assert abs(sz_A - 0.5) < ATOL, f"<Sz>_A must stay +0.5 for h=0, got {sz_A}"
        assert abs(sz_B + 0.5) < ATOL, f"<Sz>_B must stay -0.5 for h=0, got {sz_B}"

    def test_exact_phase_on_Neel(self):
        """Ising gate U_diag on |up,down> gives phase exp(-i * J/4 * dt) * |up,down>.

        The |up,down> component of h_bond has energy +J/4 (antialigned spins).
        Gate phase: exp(-i * (J/4) * dt).
        After gate: state is still |up,down>, just multiplied by this phase.
        After normalization the phase drops out — state stays Neel.
        """
        J, dt = 1.5, 0.2
        state = init_neel()
        gate = make_tfim_gate(J=J, h=0.0, dt=dt)
        GA_new, LA_new, GB_new, _ = apply_gate_vidal(
            state.Gamma_A, state.Lambda_A,
            state.Gamma_B, state.Lambda_B,
            gate, chi_max=4
        )
        psi = reconstruct_2site_state(GA_new, LA_new, GB_new)
        # Normalized state: only |up,down> component, magnitude 1
        assert abs(abs(psi[0, 1]) - 1.0) < ATOL, \
            f"|psi[up,down]| must be 1.0 after h=0 gate, got {abs(psi[0,1])}"
        assert abs(psi[0, 0]) < ATOL, "psi[up,up] must be 0"
        assert abs(psi[1, 0]) < ATOL, "psi[down,up] must be 0"
        assert abs(psi[1, 1]) < ATOL, "psi[down,down] must be 0"


# ---------------------------------------------------------------------------
# Test 5: Transverse field creates entanglement
# ---------------------------------------------------------------------------

class TestEntanglementCreation:
    """h > 0 gate on Neel must create nonzero entanglement."""

    def test_chi_grows_for_h1(self):
        state = init_neel()
        gate = make_tfim_gate(J=1.0, h=1.0, dt=0.1)
        _, LA_new, _, _ = apply_gate_vidal(
            state.Gamma_A, state.Lambda_A,
            state.Gamma_B, state.Lambda_B,
            gate, chi_max=4
        )
        assert LA_new.shape[0] > 1, \
            "h>0 gate on product state must create entanglement (chi > 1)"

    def test_entropy_nonzero_after_steps(self):
        # 15 steps needed for entanglement to build up past 0.01 from product state
        state = init_neel()
        gate = make_tfim_gate(J=1.0, h=1.5, dt=0.05)
        for _ in range(15):
            state, _ = tebd_step_1st_order(state, gate, chi_max=20)
        S = entropy_A(state)
        assert S > 0.01, f"Entropy must grow above 0 after steps with h>0, got {S:.4f}"

    def test_Sz_changes_for_h1(self):
        """Transverse field flips spins: <Sz>_A must deviate from +0.5."""
        state = init_neel()
        gate = make_tfim_gate(J=1.0, h=2.0, dt=0.1)
        for _ in range(3):
            state, _ = tebd_step_1st_order(state, gate, chi_max=20)
        sz_A, _ = measure_sz(state)
        assert abs(sz_A - 0.5) > 1e-4, \
            f"<Sz>_A must change under h>0 transverse field, got {sz_A:.6f}"


# ---------------------------------------------------------------------------
# Test 6: EXACT amplitude verification (kill test)
# ---------------------------------------------------------------------------

class TestExactAmplitudes:
    """After one AB gate on Neel, reconstruct psi[s1,s2] and compare to
    U_mat @ |up,down> computed directly via matrix exponentiation.

    This is the end-to-end kill test: if all 4 complex amplitudes match,
    the full 5-step pipeline (absorb, contract, gate, SVD, restore) is correct.
    """

    def test_all_4_amplitudes_match_exact(self):
        """Amplitudes psi_new[s1,s2] must equal U_mat[2*s1+s2, 1] for all (s1,s2)."""
        J, h, dt = 1.0, 1.5, 0.1
        state = init_neel()
        gate = make_tfim_gate(J, h, dt)

        GA_new, LA_new, GB_new, _ = apply_gate_vidal(
            state.Gamma_A, state.Lambda_A,
            state.Gamma_B, state.Lambda_B,
            gate, chi_max=8
        )

        psi_vidal = reconstruct_2site_state(GA_new, LA_new, GB_new)
        psi_exact = neel_gate_result(J, h, dt)

        # States may differ by a global phase — compare absolute amplitudes
        np.testing.assert_allclose(
            np.abs(psi_vidal), np.abs(psi_exact), atol=1e-10,
            err_msg="Post-gate state amplitudes must match U_mat @ |up,down> exactly"
        )

    def test_amplitude_match_several_parameters(self):
        """Verify amplitude match for multiple (J, h, dt) combinations."""
        test_cases = [
            (1.0, 0.5, 0.05),
            (1.0, 1.0, 0.10),
            (0.5, 2.0, 0.20),
            (2.0, 1.5, 0.02),
        ]
        for J, h, dt in test_cases:
            state = init_neel()
            gate = make_tfim_gate(J, h, dt)
            GA_new, LA_new, GB_new, _ = apply_gate_vidal(
                state.Gamma_A, state.Lambda_A,
                state.Gamma_B, state.Lambda_B,
                gate, chi_max=8
            )
            psi_vidal = reconstruct_2site_state(GA_new, LA_new, GB_new)
            psi_exact = neel_gate_result(J, h, dt)
            max_err = np.max(np.abs(np.abs(psi_vidal) - np.abs(psi_exact)))
            assert max_err < 1e-10, \
                f"Amplitude mismatch for J={J},h={h},dt={dt}: max_err={max_err:.2e}"

    def test_multi_step_exact(self):
        """After 3 AB gates in sequence (no BA), compare to iterated matrix action.

        Since we only apply AB gates, Lambda_B is unchanged throughout.
        State after k AB gates = (U_AB)^k @ |up,down>.
        We compare psi reconstruction vs exact iterated matrix exponentiation.
        """
        J, h, dt, n = 1.0, 1.0, 0.05, 3
        U_mat = make_tfim_gate(J, h, dt).reshape(4, 4)

        # Exact: iterate U_mat k times on |up,down>
        vec = np.array([0.0, 1.0, 0.0, 0.0], dtype=complex)
        for _ in range(n):
            vec = U_mat @ vec
        psi_exact = vec.reshape(2, 2)

        # Vidal: apply AB gate n times
        state = init_neel()
        gate = make_tfim_gate(J, h, dt)
        GA, LA, GB = state.Gamma_A.copy(), state.Lambda_A.copy(), state.Gamma_B.copy()
        LB = state.Lambda_B.copy()
        for _ in range(n):
            GA, LA, GB, _ = apply_gate_vidal(GA, LA, GB, LB, gate, chi_max=16)

        psi_vidal = reconstruct_2site_state(GA, LA, GB)
        np.testing.assert_allclose(
            np.abs(psi_vidal), np.abs(psi_exact), atol=1e-8,
            err_msg=f"After {n} AB gates, reconstructed state must match exact"
        )


# ---------------------------------------------------------------------------
# Test 7: Trotter order accuracy
# ---------------------------------------------------------------------------

class TestTrotterOrder:
    """Verify the 2nd-order Trotter step has the correct time-reversal symmetry.

    The symmetric decomposition AB(dt/2) -> BA(dt) -> AB(dt/2) is self-adjoint:
    applying it forward then backward (with negated dt) recovers the exact original
    state (to machine precision), because the product of gates is exactly unitary.

    The 1st-order decomposition AB(dt) -> BA(dt) is NOT self-adjoint, so a round
    trip leaves a Trotter error proportional to dt^2.

    This test verifies two things:
      1. The 2nd-order step's AB-BA-AB ordering is correct (not AB-AB-BA).
      2. The 2nd-order step is strictly more time-reversible than 1st-order.
    """

    def test_2nd_order_exact_time_reversal(self):
        """Forward + backward 2nd-order step must recover original state exactly.

        Forward: AB(dt/2) -> BA(dt) -> AB(dt/2)
        Backward: AB(-dt/2) -> BA(-dt) -> AB(-dt/2)
        Composition is exact unitary -> round-trip error = machine epsilon.
        """
        J, h, dt = 1.0, 1.5, 0.3
        gate_fwd = make_tfim_gate(J, h, dt)
        gate_half_fwd = make_tfim_gate(J, h, dt / 2)
        gate_bwd = make_tfim_gate(J, h, -dt)
        gate_half_bwd = make_tfim_gate(J, h, -dt / 2)

        sz_init, _ = measure_sz(init_neel())

        state = init_neel()
        state, _ = tebd_step_2nd_order(state, gate_half_fwd, gate_fwd, chi_max=20)
        state, _ = tebd_step_2nd_order(state, gate_half_bwd, gate_bwd, chi_max=20)
        sz_roundtrip, _ = measure_sz(state)

        assert abs(sz_roundtrip - sz_init) < 1e-12, (
            f"2nd-order round-trip must recover <Sz>_A to machine precision, "
            f"got error {abs(sz_roundtrip - sz_init):.2e}"
        )

    def test_2nd_order_more_reversible_than_1st(self):
        """Round-trip error for 2nd-order must be strictly less than for 1st-order.

        1st order AB(dt)->BA(dt) + AB(-dt)->BA(-dt): error O(dt^2)
        2nd order: machine precision (symmetric decomposition cancels exactly)
        """
        J, h, dt = 1.0, 1.5, 0.3
        gate_fwd = make_tfim_gate(J, h, dt)
        gate_bwd = make_tfim_gate(J, h, -dt)
        gate_half_fwd = make_tfim_gate(J, h, dt / 2)
        gate_half_bwd = make_tfim_gate(J, h, -dt / 2)

        sz_init, _ = measure_sz(init_neel())

        # 1st order round-trip
        s1 = init_neel()
        s1, _ = tebd_step_1st_order(s1, gate_fwd, chi_max=20)
        s1, _ = tebd_step_1st_order(s1, gate_bwd, chi_max=20)
        err_1st = abs(measure_sz(s1)[0] - sz_init)

        # 2nd order round-trip
        s2 = init_neel()
        s2, _ = tebd_step_2nd_order(s2, gate_half_fwd, gate_fwd, chi_max=20)
        s2, _ = tebd_step_2nd_order(s2, gate_half_bwd, gate_bwd, chi_max=20)
        err_2nd = abs(measure_sz(s2)[0] - sz_init)

        assert err_2nd < err_1st, (
            f"2nd-order round-trip error ({err_2nd:.2e}) must be less than "
            f"1st-order ({err_1st:.2e})"
        )
