"""
run_a4.py: Run all iTEBD simulations and save raw data to results/data/.

Does NOT produce any plots. Run this once to generate data, then use
plot_a4.py to produce figures from the saved HDF5 files.

Output files:
    results/data/scan_h{h1}_chi{chi_max}.h5  -- h1 scan at chi_max=40
    results/data/conv_h2.0_chi{chi_max}.h5   -- convergence check at h1=2.0

Parameters:
    J       = 1.0   (Ising coupling)
    dt      = 0.05  (Trotter time step, 2nd-order error O(dt^3))
    n_steps = 100   (total time T = 5.0)

Approximate runtimes on Ryzen 9 CPU:
    chi_max=40,  100 steps:   3 s
    chi_max=80,  100 steps:  20 s
    chi_max=200, 100 steps: 120 s
"""

import os
from tns.io import save_result
from tns.simulate import run_itebd

os.makedirs('results/data', exist_ok=True)

J = 1.0
dt = 0.05
n_steps = 100

# ---------------------------------------------------------------------------
# Scan over h1 at fixed chi_max
# ---------------------------------------------------------------------------
H1_VALUES = [0.5, 0.8, 1.2, 1.5, 2.0, 3.0]
CHI_SCAN = 40

print("h1 scan (chi_max={})".format(CHI_SCAN))
for h1 in H1_VALUES:
    path = f'results/data/scan_h{h1}_chi{CHI_SCAN}.h5'
    print(f"  h1={h1}  ->  {path}")
    result = run_itebd(J, h1, CHI_SCAN, dt, n_steps, verbose=True)
    save_result(result, path)

# ---------------------------------------------------------------------------
# Convergence check at h1=2.0 across chi_max values
# ---------------------------------------------------------------------------
CHI_CONV = [20, 40, 80, 200]
H1_CONV = 2.0

print(f"\nConvergence check (h1={H1_CONV})")
for chi in CHI_CONV:
    path = f'results/data/conv_h{H1_CONV}_chi{chi}.h5'
    print(f"  chi_max={chi}  ->  {path}")
    result = run_itebd(J, H1_CONV, chi, dt, n_steps, verbose=True)
    save_result(result, path)

print("\nAll data saved to results/data/")
print("Run:  uv run python plot_a4.py  to generate figures.")
