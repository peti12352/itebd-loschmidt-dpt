"""
analyse_a4.py — Load saved HDF5 simulation data, compute all metrics,
write results/analysis_report.txt as the authoritative source of numbers.

Nothing is computed analytically or assumed. Every number in the output
comes from the HDF5 files produced by run_a4.py and the h1=0.8 long run.

Run:
    uv run python analyse_a4.py

Output:
    results/analysis_report.txt
"""

import os
import numpy as np
from scipy.interpolate import interp1d
from tns.io import load_result

os.makedirs('results', exist_ok=True)
lines = []

def log(s=''):
    print(s)
    lines.append(s)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def find_peaks(times, lam, min_height=0.3):
    """Return (time, lambda) at local maxima above min_height."""
    idx = np.where(
        (lam[1:-1] > lam[:-2]) &
        (lam[1:-1] > lam[2:]) &
        (lam[1:-1] > min_height)
    )[0] + 1
    return [(times[i], lam[i]) for i in idx]

def saturation_time(times, chi_A, chi_max):
    """First time chi_A >= chi_max. Returns None if never saturated."""
    idx = np.where(chi_A >= chi_max)[0]
    return times[idx[0]] if len(idx) > 0 else None

# ---------------------------------------------------------------------------
# 1. Cusp positions
# ---------------------------------------------------------------------------
log('=== 1. CUSP POSITIONS (h1 scan, chi_max=40, dt=0.05) ===')
H1_VALUES = [0.5, 0.8, 1.2, 1.5, 2.0, 3.0]
for h1 in H1_VALUES:
    path = f'results/data/scan_h{h1}_chi40.h5'
    if not os.path.exists(path):
        log(f'  h1={h1}: MISSING ({path})')
        continue
    r = load_result(path)
    peaks = find_peaks(r.times, r.lambda_t)
    if peaks:
        peak_str = '  '.join([f't*={t:.2f} lambda={l:.4f}' for t, l in peaks])
        log(f'  h1={h1}: {peak_str}')
    else:
        log(f'  h1={h1}: no peaks (smooth)')

# ---------------------------------------------------------------------------
# 2. h1=0.8 long run (t_max=15, chi_max=80)
# ---------------------------------------------------------------------------
log()
log('=== 2. h1=0.8 EXTENDED RUN (chi_max=80, dt=0.05, t_max=15) ===')
path_long = 'results/data/scan_h0.8_chi80_long.h5'
if os.path.exists(path_long):
    r08 = load_result(path_long)
    peaks = find_peaks(r08.times, r08.lambda_t, min_height=0.3)
    log(f'  Peaks found: {len(peaks)}')
    for i, (t, l) in enumerate(peaks):
        idx = np.argmin(np.abs(r08.times - t))
        log(f'    n={i}: t*={t:.2f}  lambda={l:.4f}  chi_A={r08.chi_A[idx]}  trunc_err={r08.trunc_err[idx]:.2e}')
    if len(peaks) >= 2:
        spacing = peaks[1][0] - peaks[0][0]
        log(f'  Observed spacing (t*_1 - t*_0): {spacing:.2f}')
        predicted_t1 = 3 * peaks[0][0]   # t*_n = t*(n+0.5), so t1/t0 = 1.5/0.5 = 3
        log(f'  Predicted t*_1 (= 3 * t*_0): {predicted_t1:.2f}')
        log(f'  Shift from prediction: {peaks[1][0] - predicted_t1:+.2f}')
    log(f'  S_A at t=15: {r08.entropy_A[-1]:.4f}')
    log(f'  chi_A at t=15: {r08.chi_A[-1]}')
    log(f'  trunc_err at t=15: {r08.trunc_err[-1]:.2e}')
else:
    log(f'  MISSING: {path_long}  -- run run_a4_long.py first')

# ---------------------------------------------------------------------------
# 3. Chi convergence errors (chi=20,40,80 vs chi=200)
# ---------------------------------------------------------------------------
log()
log('=== 3. CHI CONVERGENCE ERRORS (h1=2.0, vs chi_max=200) ===')
ref_path = 'results/data/conv_h2.0_chi200.h5'
if os.path.exists(ref_path):
    ref = load_result(ref_path)
    for chi in [20, 40, 80]:
        path = f'results/data/conv_h2.0_chi{chi}.h5'
        if not os.path.exists(path):
            log(f'  chi={chi}: MISSING')
            continue
        r = load_result(path)
        err_all = np.abs(r.lambda_t - ref.lambda_t)
        max_err = float(np.max(err_all))
        max_t   = float(ref.times[np.argmax(err_all)])
        sat_t   = saturation_time(ref.times, r.chi_A, chi)
        chiA_t5 = int(r.chi_A[-1])
        log(f'  chi={chi}: max_err={max_err:.3e} at t={max_t:.2f}'
            f'  chi_A(t=5)={chiA_t5}  saturates_at={sat_t}')
    log(f'  chi=200: chi_A(t=5)={int(ref.chi_A[-1])}  (reference)')
else:
    log('  MISSING: conv_h2.0_chi200.h5')

# ---------------------------------------------------------------------------
# 4. Trotter error (dt=0.05 vs dt=0.025 reference)
# ---------------------------------------------------------------------------
log()
log('=== 4. TROTTER ERROR (h1=2.0, chi_max=80) ===')
path_fine = 'results/data/scan_h2.0_chi80_dt0.025.h5'
path_mid  = 'results/data/conv_h2.0_chi80.h5'
path_coarse = 'results/data/scan_h2.0_chi80_dt0.1.h5'
if os.path.exists(path_fine) and os.path.exists(path_mid):
    r_fine = load_result(path_fine)
    r_mid  = load_result(path_mid)
    f_interp = interp1d(r_fine.times, r_fine.lambda_t, kind='cubic',
                        fill_value='extrapolate')
    err_05 = np.abs(r_mid.lambda_t - f_interp(r_mid.times))
    log(f'  dt=0.05 vs dt=0.025: max={float(np.max(err_05)):.3e}  rms={float(np.sqrt(np.mean(err_05**2))):.3e}')
    if os.path.exists(path_coarse):
        r_coarse = load_result(path_coarse)
        err_10 = np.abs(r_coarse.lambda_t - f_interp(r_coarse.times))
        log(f'  dt=0.1  vs dt=0.025: max={float(np.max(err_10)):.3e}  rms={float(np.sqrt(np.mean(err_10**2))):.3e}')
        ratio = float(np.max(err_10)) / float(np.max(err_05))
        log(f'  Ratio dt=0.1/dt=0.05: {ratio:.2f}  (expected ~4 for 2nd-order Trotter)')
else:
    log('  MISSING fine/mid Trotter files')

# ---------------------------------------------------------------------------
# 5. Truncation error (chi=40, h1=2.0)
# ---------------------------------------------------------------------------
log()
log('=== 5. TRUNCATION ERROR PER STEP (h1=2.0, chi_max=40) ===')
path40 = 'results/data/conv_h2.0_chi40.h5'
if os.path.exists(path40):
    r40 = load_result(path40)
    for t_check in [1.0, 2.0, 3.0, 4.0, 4.5, 5.0]:
        idx = np.argmin(np.abs(r40.times - t_check))
        log(f'  t={t_check:.1f}: trunc_err={r40.trunc_err[idx]:.2e}  chi_A={r40.chi_A[idx]}')

# ---------------------------------------------------------------------------
# 6. Entanglement entropy at key times
# ---------------------------------------------------------------------------
log()
log('=== 6. ENTANGLEMENT ENTROPY S_A (chi_max=200, h1=2.0) ===')
if os.path.exists(ref_path):
    ref = load_result(ref_path)
    for t_check in [1.0, 2.0, 3.0, 4.0, 5.0]:
        idx = np.argmin(np.abs(ref.times - t_check))
        log(f'  t={t_check:.1f}: S_A={ref.entropy_A[idx]:.4f}  log(chi_max)=log(200)={np.log(200):.4f}')

log()
log('=== 7. SATURATION LIMITS ===')
for chi in [20, 40, 80, 200]:
    log(f'  log({chi}) = {np.log(chi):.4f}')

# ---------------------------------------------------------------------------
# Write report
# ---------------------------------------------------------------------------
report_path = 'results/analysis_report.txt'
with open(report_path, 'w') as f:
    f.write('\n'.join(lines) + '\n')

print()
print(f'Report written to: {report_path}')
