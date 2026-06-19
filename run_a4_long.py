"""
run_a4_long.py — Extended runs needed for analysis not covered by run_a4.py.

    1. h1=0.8, chi_max=80, t_max=15  -- resolves DPT ambiguity
    2. h1=2.0, chi_max=80, dt=0.025  -- fine Trotter reference
    3. h1=2.0, chi_max=80, dt=0.1    -- coarse Trotter (ratio check)

Run:
    uv run python run_a4_long.py
"""

import os
from tns.simulate import run_itebd
from tns.io import save_result

os.makedirs('results/data', exist_ok=True)

runs = [
    dict(J=1.0, h1=0.8,  chi_max=80,  dt=0.05,  n_steps=300,
         path='results/data/scan_h0.8_chi80_long.h5',
         desc='h1=0.8 long (t_max=15, resolves DPT ambiguity)'),
    dict(J=1.0, h1=2.0,  chi_max=80,  dt=0.025, n_steps=200,
         path='results/data/scan_h2.0_chi80_dt0.025.h5',
         desc='h1=2.0 fine Trotter reference (dt=0.025)'),
    dict(J=1.0, h1=2.0,  chi_max=80,  dt=0.1,   n_steps=50,
         path='results/data/scan_h2.0_chi80_dt0.1.h5',
         desc='h1=2.0 coarse Trotter (dt=0.1, ratio check)'),
]

for run in runs:
    path = run.pop('path')
    desc = run.pop('desc')
    if os.path.exists(path):
        print(f'SKIP (exists): {path}')
        continue
    print(f'Running: {desc}')
    result = run_itebd(**run, verbose=True)
    save_result(result, path)
    print(f'Saved:   {path}')

print('\nDone. Run analyse_a4.py to compute metrics.')
