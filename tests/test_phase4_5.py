"""
test_phase4_5.py — Tests for simulate.py (Phase 4) and transfer.py (Phase 5).

PHASE 4 TESTS: iTEBD simulation loop
    Verify structure, norm, physically correct behavior of the full run loop.

PHASE 5 TESTS: Transfer matrix and Loschmidt rate
    Verify λ(0)=0, λ≥0, correct matrix shape, DPT signatures detectable.

DPT SIGNATURE TEST STRATEGY
-----------------------------
For the Neel initial state under TFIM quench:
    h1 < 1: quench within ordered phase → λ(t) stays small, smooth, low amplitude
    h1 > 1: quench crosses QPT → λ(t) reaches large values (DPT: dominant
             eigenvalue approaches 0, λ → large)

We test:
    1. max(λ(t)) for h1=2.0 >> max(λ(t)) for h1=0.5
    2. max jump in consecutive λ values much larger for h1=2.0 (cusp signature)
    3. Local maxima exist in λ(t) for h1=2.0 (cusp peaks)

These are numerically verified from a prior exploratory run:
    h1=0.5: λ max ≈ 0.37, max jump ≈ 0.006
    h1=2.0: λ max ≈ 1.79, max jump ≈ 0.099, local maxima at t ≈ 1.6, 4.7
"""

import numpy as np
import pytest

from tns.gates import make_tfim_gate
from tns.simulate import run_itebd
from tns.tebd import tebd_step_2nd_order
from tns.transfer import build_transfer_matrix, loschmidt_rate
from tns.vidal_state import init_neel, norm


# ---------------------------------------------------------------------------
# Phase 5: Transfer matrix structural tests (run first — no loop needed)
# ---------------------------------------------------------------------------

class TestTransferMatrixStructure:
    """build_transfer_matrix() structural properties."""

    def test_t0_shape_is_1x1(self):
        """Neel state has chi_B=1 → T is 1×1 at t=0."""
        state = init_neel()
        T = build_transfer_matrix(state)
        assert T.shape == (1, 1), f"T at t=0 must be 1x1, got {T.shape}"

    def test_t0_value_is_one(self):
        """At t=0: T = [[1+0j]], eigenvalue = 1, λ = 0."""
        state = init_neel()
        T = build_transfer_matrix(state)
        assert abs(T[0, 0] - 1.0) < 1e-12, f"T[0,0] at t=0 must be 1.0, got {T[0,0]}"

    def test_t0_lambda_is_zero(self):
        """Loschmidt rate at t=0 must be exactly 0."""
        state = init_neel()
        lam = loschmidt_rate(state)
        assert abs(lam) < 1e-12, f"λ(0) must be 0, got {lam}"

    def test_shape_grows_with_chi(self):
        """After gate application chi_B grows: T shape must grow accordingly."""
        state = init_neel()
        gate_half = make_tfim_gate(J=1.0, h=2.0, dt=0.025)
        gate_full = make_tfim_gate(J=1.0, h=2.0, dt=0.05)
        for _ in range(5):
            state, _ = tebd_step_2nd_order(state, gate_half, gate_full, chi_max=20)
        T = build_transfer_matrix(state)
        chi_B = state.chi_B
        assert T.shape == (chi_B, chi_B), \
            f"T shape must be ({chi_B},{chi_B}), got {T.shape}"

    def test_lambda_nonnegative(self):
        """λ(t) = -log|τ| must be ≥ 0 since |τ| ≤ 1."""
        state = init_neel()
        gate_half = make_tfim_gate(J=1.0, h=1.5, dt=0.025)
        gate_full = make_tfim_gate(J=1.0, h=1.5, dt=0.05)
        for _ in range(20):
            state, _ = tebd_step_2nd_order(state, gate_half, gate_full, chi_max=20)
            lam = loschmidt_rate(state)
            assert lam >= -1e-10, f"λ must be ≥ 0, got {lam:.6f}"

    def test_lambda_monotone_early(self):
        """λ(t) must increase from 0 for first few steps (state moving away from Neel)."""
        state = init_neel()
        gate_half = make_tfim_gate(J=1.0, h=2.0, dt=0.025)
        gate_full = make_tfim_gate(J=1.0, h=2.0, dt=0.05)
        prev_lam = 0.0
        for i in range(5):
            state, _ = tebd_step_2nd_order(state, gate_half, gate_full, chi_max=20)
            lam = loschmidt_rate(state)
            assert lam > prev_lam - 1e-8, \
                f"λ must increase at step {i+1}: prev={prev_lam:.4f}, now={lam:.4f}"
            prev_lam = lam


# ---------------------------------------------------------------------------
# Phase 5: DPT signature tests
# ---------------------------------------------------------------------------

# Module-level DPT fixture (avoids class-scoped fixture deprecation in pytest>=10)
@pytest.fixture(scope='module')
def dpt_results():
    """Run simulation for h1=0.5 and h1=2.0. Module scope: computed once."""
    J, chi_max, dt, n = 1.0, 40, 0.05, 100
    result_low  = run_itebd(J, h1=0.5, chi_max=chi_max, dt=dt, n_steps=n)
    result_high = run_itebd(J, h1=2.0, chi_max=chi_max, dt=dt, n_steps=n)
    return result_low, result_high


class TestDPTSignatures:
    """Verify DPT signatures: h1>1 gives much larger λ than h1<1."""

    def test_h1_high_reaches_larger_lambda(self, dpt_results):
        """h1=2.0 must have max(λ) >> max(λ) for h1=0.5."""
        result_low, result_high = dpt_results
        max_low  = np.max(result_low.lambda_t)
        max_high = np.max(result_high.lambda_t)
        assert max_high > 3 * max_low, (
            f"h1=2.0 max λ ({max_high:.3f}) must be > 3x h1=0.5 max λ ({max_low:.3f})"
        )

    def test_h1_high_has_larger_jumps(self, dpt_results):
        """Cusps manifest as large consecutive differences in λ."""
        result_low, result_high = dpt_results
        max_jump_low  = np.max(np.abs(np.diff(result_low.lambda_t)))
        max_jump_high = np.max(np.abs(np.diff(result_high.lambda_t)))
        assert max_jump_high > 5 * max_jump_low, (
            f"h1=2.0 max jump ({max_jump_high:.4f}) must be > 5x h1=0.5 ({max_jump_low:.4f})"
        )

    def test_h1_high_has_local_maxima(self, dpt_results):
        """h1=2.0 must have local maxima in λ(t) — the DPT cusps."""
        _, result_high = dpt_results
        lam = result_high.lambda_t
        local_max = np.where((lam[1:-1] > lam[:-2]) & (lam[1:-1] > lam[2:]))[0]
        assert len(local_max) >= 1, \
            "h1=2.0 must have at least 1 local maximum in λ(t) (DPT cusp)"

    def test_h1_low_smooth(self, dpt_results):
        """h1=0.5 must have small max jump (smooth curve, no cusps)."""
        result_low, _ = dpt_results
        max_jump = np.max(np.abs(np.diff(result_low.lambda_t)))
        assert max_jump < 0.02, (
            f"h1=0.5 λ(t) must be smooth (max jump < 0.02), got {max_jump:.4f}"
        )

    def test_lambda_starts_near_zero(self, dpt_results):
        """Both h1 values: λ at first step must be near 0 (no instantaneous jump)."""
        for result in dpt_results:
            assert result.lambda_t[0] < 0.01, \
                f"λ at first step must be near 0, got {result.lambda_t[0]:.4f}"


# ---------------------------------------------------------------------------
# Phase 4: Simulation loop structure
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def sim_result():
    return run_itebd(J=1.0, h1=1.5, chi_max=20, dt=0.05, n_steps=30)


class TestSimResultStructure:
    """run_itebd() output must have correct shapes and param storage."""

    def test_array_lengths(self, sim_result):
        n = 30
        assert len(sim_result.times) == n
        assert len(sim_result.lambda_t) == n
        assert len(sim_result.entropy_A) == n
        assert len(sim_result.entropy_B) == n
        assert len(sim_result.sz_A) == n
        assert len(sim_result.sz_B) == n
        assert len(sim_result.trunc_err) == n
        assert len(sim_result.chi_A) == n

    def test_times_correct(self, sim_result):
        np.testing.assert_allclose(
            sim_result.times, np.arange(1, 31) * 0.05, atol=1e-12,
            err_msg="times must be [dt, 2dt, ..., n_steps*dt]"
        )

    def test_params_stored(self, sim_result):
        assert sim_result.params['J'] == 1.0
        assert sim_result.params['h1'] == 1.5
        assert sim_result.params['chi_max'] == 20
        assert sim_result.params['n_steps'] == 30

    def test_trunc_err_nonnegative(self, sim_result):
        assert np.all(sim_result.trunc_err >= 0), "Truncation error must be ≥ 0"

    def test_chi_A_bounded(self, sim_result):
        assert np.all(sim_result.chi_A <= 20), "chi_A must never exceed chi_max"
        assert np.all(sim_result.chi_A >= 1), "chi_A must be ≥ 1"


# ---------------------------------------------------------------------------
# Phase 4: Physical correctness of the simulation
# ---------------------------------------------------------------------------

class TestPhysicsCorrectness:
    """Physically motivated tests on full simulation runs."""

    def test_h0_ising_only_no_entanglement(self):
        """Pure Ising (h1=0): Neel stays product state, S=0, Sz unchanged."""
        result = run_itebd(J=1.0, h1=0.0, chi_max=4, dt=0.1, n_steps=20)
        np.testing.assert_allclose(result.entropy_A, 0.0, atol=1e-10,
                                   err_msg="h1=0: entropy_A must stay 0")
        np.testing.assert_allclose(result.sz_A, 0.5, atol=1e-10,
                                   err_msg="h1=0: <Sz>_A must stay 0.5")
        np.testing.assert_allclose(result.sz_B, -0.5, atol=1e-10,
                                   err_msg="h1=0: <Sz>_B must stay -0.5")

    def test_h0_lambda_zero_throughout(self):
        """Pure Ising: state stays Neel, Loschmidt echo = 1, λ = 0."""
        result = run_itebd(J=1.0, h1=0.0, chi_max=4, dt=0.1, n_steps=20)
        np.testing.assert_allclose(result.lambda_t, 0.0, atol=1e-10,
                                   err_msg="h1=0: λ(t) must stay 0 (state unchanged)")

    def test_entropy_grows_from_zero(self):
        """h1>0: entropy_A must grow from 0 during time evolution.

        After one step with dt=0.05, entanglement is tiny (O(dt^2)) but nonzero.
        After 30 steps it must be substantially larger than at step 1.
        """
        result = run_itebd(J=1.0, h1=2.0, chi_max=20, dt=0.05, n_steps=30)
        # Nonzero after first step (h1>0 creates entanglement immediately)
        assert result.entropy_A[0] > 1e-10, "Entropy must be > 0 after first step"
        # Grows over time
        assert result.entropy_A[-1] > result.entropy_A[0], \
            "Entropy must grow over time for h1>0"

    def test_sz_changes_for_h1_nonzero(self):
        """h1>0: <Sz>_A must deviate from +0.5 (transverse field flips spins)."""
        result = run_itebd(J=1.0, h1=2.0, chi_max=20, dt=0.05, n_steps=20)
        assert abs(result.sz_A[-1] - 0.5) > 1e-3, \
            f"<Sz>_A must change under h1=2.0, got {result.sz_A[-1]:.4f}"

    def test_sz_antisymmetric(self):
        """By Neel symmetry: <Sz>_A + <Sz>_B ≈ 0 always (total Sz conserved)."""
        result = run_itebd(J=1.0, h1=1.5, chi_max=20, dt=0.05, n_steps=30)
        total_sz = result.sz_A + result.sz_B
        np.testing.assert_allclose(total_sz, 0.0, atol=1e-10,
                                   err_msg="Total <Sz>_A + <Sz>_B must be 0")

    def test_lambda_first_step_near_zero(self):
        """λ at t=dt must be near 0: state barely moved from Neel."""
        result = run_itebd(J=1.0, h1=2.0, chi_max=20, dt=0.01, n_steps=1)
        assert result.lambda_t[0] < 1e-3, \
            f"λ at first tiny step must be near 0, got {result.lambda_t[0]:.6f}"

    def test_chi_grows_with_entanglement(self):
        """Bond dimension chi_A must grow from 1 as entanglement builds."""
        result = run_itebd(J=1.0, h1=2.0, chi_max=40, dt=0.05, n_steps=30)
        assert result.chi_A[0] > 1, "chi_A must exceed 1 after first step with h1>0"
        assert result.chi_A[-1] >= result.chi_A[0], \
            "chi_A must be non-decreasing (entanglement grows)"
