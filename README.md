# Loschmidt Echo and Dynamical Phase Transitions via iTEBD

Simulation of dynamical quantum phase transitions (DPTs) in the transverse-field Ising model (TFIM) using the infinite time-evolving block decimation (iTEBD) algorithm. The Loschmidt echo rate function lambda(t) is computed for quantum quenches across the equilibrium quantum critical point at h=1, reproducing non-analytic cusps characteristic of DPTs in the thermodynamic limit.

Submitted as Project A4 for the Tensor Network States and DMRG course, BME Spring 2026 (Werner).

## Model

H = -J sum_i Z_i Z_{i+1} - h sum_i X_i

Quench: system initialized in ground state at h_0=0.5 (ferromagnetic phase), then evolved under post-quench Hamiltonian H(h_1) for t in [0,5]. Loschmidt echo G(t) = <psi_0|e^{-iHt}|psi_0>; rate function lambda(t) = -(1/N) log|G(t)|.

## Requirements

Python 3.14+, [uv](https://github.com/astral-sh/uv).

```
uv sync
```

## Usage

```bash
# Run all simulations (generates results/data/*.h5)
uv run python run_a4.py

# Finer dt and extended chi scan for convergence analysis
uv run python run_a4_long.py

# Produce all figures (results/figures/*.png)
uv run python plot_a4.py
uv run python plot_extra.py

# Analyse convergence and print error budget
uv run python analyse_a4.py

# Run test suite
uv run pytest
```

Pre-computed data and figures are included in `results/`.

## Structure

```
src/tns/
    vidal_state.py   -- Vidal canonical form (Gamma, Lambda tensors)
    tebd.py          -- 2nd-order Suzuki-Trotter TEBD update
    gates.py         -- two-site Ising gates
    transfer.py      -- transfer matrix and Loschmidt echo
    simulate.py      -- top-level iTEBD driver
    svd_utils.py     -- truncated SVD with Schmidt error tracking
    io.py            -- HDF5 save/load

run_a4.py            -- h_1 scan at chi_max=40, convergence check at h_1=2
run_a4_long.py       -- fine dt=0.025 run and extended chi range
plot_a4.py           -- Fig 1-3: lambda(t), error budget, entanglement entropy
plot_extra.py        -- Fig 4-5: order parameter, cusp sharpness vs chi
analyse_a4.py        -- quantitative convergence and cusp analysis
tests/               -- unit tests per implementation phase
docs/report_a4.tex   -- full written report (LaTeX)
docs/report_a4.pdf   -- compiled report
results/data/        -- simulation output (HDF5)
results/figures/     -- publication figures (PNG)
```

## Key results

- DPTs confirmed for h_1 > 1; no cusps for h_1 < 1
- First cusp time scales as t*_0 * epsilon_{k*} approx 3.3 (consistent with exact fermion dispersion)
- Bond dimension chi_max = 40 converges lambda(t) to < 10^{-14} vs chi_max = 200; Trotter error (dt=0.05) dominates at ~10^{-4}
- Von Neumann entropy S_A(t=5) approx 0.76 nats, well below log(40) = 3.69; simulation is exact throughout

## Report

`docs/report_a4.pdf` gives a self-contained derivation of the iTEBD algorithm, Vidal canonical form, Suzuki-Trotter decomposition, and quantitative analysis of all results with error budgets.

## Reference implementation

M. Werner, [TNSClass](https://github.com/wernermiklos/TNSClass) — course materials for the BME TNS/DMRG course. The Vidal canonical form conventions and index layout in `src/tns/vidal_state.py` follow `Class10/iTEBD_XXZ.m` from that repository (MATLAB). This project extends it to the TFIM, adds 2nd-order Trotter, the Loschmidt echo observable, and convergence analysis.

## References

- M. Heyl, A. Polkovnikov, S. Kehrein, PRL 110, 135704 (2013)
- G. Vidal, PRL 98, 070201 (2007)
- M. Suzuki, Commun. Math. Phys. 51, 183 (1976)
- N. Hatano, M. Suzuki, in *Quantum Annealing and Other Optimization Methods*, Springer (2005), pp. 37-68
- S. Sachdev, *Quantum Phase Transitions*, 2nd ed., Cambridge (2011)
