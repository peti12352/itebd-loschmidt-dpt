"""
plot_a4.py: Publication-quality figures from HDF5 data produced by run_a4.py.

Figures:
    fig1_dpt_physics.png    -- lambda(t) + DPT cusp phase diagram
    fig2_error_analysis.png -- chi convergence error + error budget
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.interpolate import interp1d

from tns.io import load_result

os.makedirs('results/figures', exist_ok=True)

# ---------------------------------------------------------------------------
# Style: Nature/PRL-like — serif font, inward ticks, minimal ink
# ---------------------------------------------------------------------------
sns.set_theme(style='ticks', context='paper', font='serif', font_scale=1.35)
plt.rcParams.update({
    'figure.dpi': 200,
    'axes.linewidth': 0.8,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'xtick.minor.visible': True, 'ytick.minor.visible': True,
    'legend.frameon': True, 'legend.framealpha': 0.95,
    'legend.edgecolor': '0.75', 'legend.fontsize': 9,
    'axes.formatter.use_mathtext': True,
})

# Colorblind-safe palette (Wong 2011)
C = {
    'blue':   '#0072B2',
    'orange': '#E69F00',
    'green':  '#009E73',
    'red':    '#D55E00',
    'purple': '#CC79A7',
    'sky':    '#56B4E9',
    'yellow': '#F0E442',
    'black':  '#000000',
}


def local_maxima(times, values, min_height=0.3):
    """Indices of local maxima above min_height threshold."""
    idx = np.where(
        (values[1:-1] > values[:-2]) &
        (values[1:-1] > values[2:]) &
        (values[1:-1] > min_height)
    )[0] + 1  # +1 to account for slicing offset
    return idx


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
H1_ALL = [0.5, 0.8, 1.2, 1.5, 2.0, 3.0]
CHI_SCAN = 40

scan = {h1: load_result(f'results/data/scan_h{h1}_chi{CHI_SCAN}.h5')
        for h1 in H1_ALL
        if os.path.exists(f'results/data/scan_h{h1}_chi{CHI_SCAN}.h5')}

conv = {chi: load_result(f'results/data/conv_h2.0_chi{chi}.h5')
        for chi in [20, 40, 80, 200]
        if os.path.exists(f'results/data/conv_h2.0_chi{chi}.h5')}

r_fine = load_result('results/data/scan_h2.0_chi80_dt0.025.h5')
r_mid  = load_result('results/data/conv_h2.0_chi80.h5')   # dt=0.05

f_interp = interp1d(r_fine.times, r_fine.lambda_t, kind='cubic',
                    fill_value='extrapolate')
trotter_err = np.abs(r_mid.lambda_t - f_interp(r_mid.times))

ref = conv[200]

# ---------------------------------------------------------------------------
# Figure 1
# ---------------------------------------------------------------------------
fig1, (ax_A, ax_B) = plt.subplots(1, 2, figsize=(10.5, 4.2))

# ---- Panel A: lambda(t) for three h1 values --------------------------------
show_h1  = [0.5,       2.0,         3.0]
colors_a = [C['blue'], C['orange'], C['green']]
dashes_a = [(4, 2),    [],          []]           # dashed for h1<1

for h1, col, dash in zip(show_h1, colors_a, dashes_a):
    r = scan[h1]
    lkw = dict(color=col, linewidth=1.6)
    if dash:
        ax_A.plot(r.times, r.lambda_t, '--', dashes=dash, **lkw,
                  label=fr'$h_1 = {h1}$  (no DPT)')
    else:
        ax_A.plot(r.times, r.lambda_t, '-', **lkw,
                  label=fr'$h_1 = {h1}$')

    # Mark cusp peaks with a triangle marker only — no floating text
    idx = local_maxima(r.times, r.lambda_t)
    for i in idx:
        ax_A.plot(r.times[i], r.lambda_t[i], '^',
                  color=col, markersize=6, zorder=5)

# Custom legend entry for the triangle markers
ax_A.plot([], [], '^k', markersize=6, label=r'cusp $t^*_n$')

ax_A.set_xlabel(r'time $t$')
ax_A.set_ylabel(r'$\lambda(t) = -\log\,|\tau(t)|$')
ax_A.set_xlim(0, 5)
ax_A.set_ylim(0, 2.4)
ax_A.legend(loc='upper right', handlelength=1.8)
ax_A.set_title('(A)', loc='left', fontsize=11, pad=4)
sns.despine(ax=ax_A)

# ---- Panel B: t*_0 vs h1 (DPT phase diagram) --------------------------------
h1_arr, tstar_arr = [], []
for h1 in H1_ALL:
    if h1 not in scan:
        continue
    r = scan[h1]
    idx = local_maxima(r.times, r.lambda_t)
    if len(idx):
        h1_arr.append(h1)
        tstar_arr.append(r.times[idx[0]])

h1_arr   = np.array(h1_arr)
tstar_arr = np.array(tstar_arr)
dt_err   = 0.025   # half-step resolution uncertainty

# Split: confirmed DPT (h1 > 1) vs ambiguous (h1 = 0.8, near edge of window)
confirmed = h1_arr > 1.0
ambiguous = ~confirmed

# Guide line through confirmed points only
if confirmed.sum() > 1:
    ax_B.plot(h1_arr[confirmed], tstar_arr[confirmed],
              '-', color='0.75', linewidth=1.0, zorder=1)

ax_B.errorbar(h1_arr[confirmed], tstar_arr[confirmed],
              yerr=dt_err, fmt='o', color=C['blue'],
              markersize=6, capsize=3, capthick=1.0, linewidth=1.2,
              zorder=3, label=r'$t^*_0$, confirmed DPT ($h_1>1$)')

if ambiguous.sum():
    ax_B.errorbar(h1_arr[ambiguous], tstar_arr[ambiguous],
                  yerr=dt_err, fmt='o', color=C['blue'],
                  markersize=6, capsize=3, capthick=1.0, linewidth=1.2,
                  mfc='white', zorder=3,          # open marker = ambiguous
                  label=r'$t^*_0$, ambiguous ($h_1 < h_{\rm QPT}$)')

# Mark equilibrium QPT boundary
ax_B.axvline(1.0, color='0.5', linewidth=0.8, linestyle='--', zorder=0)
ax_B.text(1.05, 5.1, r'QPT ($h=1$)', fontsize=8, color='0.5', va='top')

ax_B.set_xlabel(r'post-quench field $h_1$')
ax_B.set_ylabel(r'first cusp time $t^*_0$')
ax_B.set_xlim(0.4, 3.6)
ax_B.set_ylim(0, 5.5)
ax_B.legend(loc='upper right', fontsize=8.5, bbox_to_anchor=(0.98, 0.98))
ax_B.set_title('(B)', loc='left', fontsize=11, pad=4)
# Error bar explanation in axis label, not floating text
ax_B.set_xlabel(r'post-quench field $h_1$  (error bars: $\delta t/2 = 0.025$)')
sns.despine(ax=ax_B)

fig1.tight_layout(w_pad=3.5)
fig1.savefig('results/figures/fig1_dpt_physics.png', bbox_inches='tight')
print("Saved fig1_dpt_physics")

# ---------------------------------------------------------------------------
# Figure 2
# ---------------------------------------------------------------------------
fig2, (ax_C, ax_D) = plt.subplots(1, 2, figsize=(10.5, 4.2))

# ---- Panel C: convergence error vs time (log scale) -------------------------
# Only plot chi=20 (chi=40 and chi=80 are at machine precision < 1e-14,
# which is 7 orders of magnitude below the Trotter error and carries no info).
err20 = np.abs(conv[20].lambda_t - ref.lambda_t)
err20_plot = np.where(err20 > 1e-16, err20, np.nan)

ax_C.semilogy(ref.times, err20_plot, '-', color=C['blue'], linewidth=1.5,
              label=r'$|\lambda_{\chi=20} - \lambda_{\chi=200}|$')
ax_C.semilogy(r_mid.times, np.maximum(trotter_err, 1e-16),
              '--', color=C['red'], linewidth=1.5,
              label=r'Trotter: $|\lambda_{\delta t=0.05} - \lambda_{\delta t=0.025}|$')

# Annotate: chi=40 and chi=80 are at machine precision — top-left, clear of all lines
ax_C.text(0.03, 0.97, r'$\chi=40,\,80\!:\;<10^{-14}$',
          transform=ax_C.transAxes, fontsize=7.5, color='0.5',
          ha='left', va='top')

ax_C.set_xlabel(r'time $t$')
ax_C.set_ylabel(r'$|\Delta\lambda(t)|$')
ax_C.set_xlim(0, 5)
ax_C.set_ylim(1e-16, 1e-3)
ax_C.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=7))
ax_C.legend(loc='lower left', fontsize=8.5)
ax_C.set_title('(C)', loc='left', fontsize=11, pad=4)
sns.despine(ax=ax_C)

# ---- Panel D: Error budget bar chart ----------------------------------------
labels = [
    r'$\chi_{\max}=20$',
    r'$\chi_{\max}=40$',
    'Trotter\n' + r'$\delta t=0.05$',
]
vals = [
    float(np.max(np.abs(conv[20].lambda_t - ref.lambda_t))),   # 3.85e-11
    float(np.max(np.abs(conv[40].lambda_t - ref.lambda_t))),   # 4.89e-15
    float(np.max(trotter_err)),                                  # 8.16e-05
]
bar_cols = [C['blue'], C['sky'], C['red']]

x = np.arange(len(labels))
bars = ax_D.bar(x, vals, color=bar_cols, width=0.5,
                edgecolor='white', linewidth=0.5)
ax_D.set_yscale('log')
ax_D.set_xticks(x)
ax_D.set_xticklabels(labels, fontsize=9)
ax_D.set_ylabel(r'max $|\Delta\lambda|$ over $t\in[0,5]$')
ax_D.set_title('(D)', loc='left', fontsize=11, pad=4)

def sci_tex(val):
    """Format float as LaTeX scientific notation: $M.M \times 10^{e}$."""
    exp = int(np.floor(np.log10(abs(val))))
    mantissa = val / 10**exp
    return fr'${mantissa:.1f}{{\times}}10^{{{exp}}}$'

for bar, val in zip(bars, vals):
    ax_D.text(bar.get_x() + bar.get_width() / 2,
              val * 5,
              sci_tex(val), ha='center', va='bottom', fontsize=8)

ax_D.set_ylim(1e-16, 1e-2)
ax_D.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=8))
sns.despine(ax=ax_D)

fig2.tight_layout(w_pad=3.5)
fig2.savefig('results/figures/fig2_error_analysis.png', bbox_inches='tight')
print("Saved fig2_error_analysis")

# ---------------------------------------------------------------------------
# Figure 3: Entanglement entropy growth + saturation limits
# Two panels:
#   (E) S_A(t) zoomed on actual data — all chi identical (converged)
#   (F) Same curve vs log(chi_max) saturation limits — shows margin to failure
# ---------------------------------------------------------------------------
fig3, (ax_E, ax_F) = plt.subplots(1, 2, figsize=(10.5, 4.2))

palette_ent = sns.color_palette('crest', n_colors=4)

# Panel E: entropy data (all chi collapse — plot only chi=200 to keep clean)
r_ent = conv[200]
ax_E.plot(r_ent.times, r_ent.entropy_A, '-', color=C['blue'], linewidth=1.6,
          label=r'$S_A(t)$  (all $\chi_{\max}$ identical)')
ax_E.set_xlabel(r'time $t$')
ax_E.set_ylabel(r'$S_A(t)$ (von Neumann entropy)')
ax_E.set_xlim(0, 5)
ax_E.set_ylim(0, 0.85)
ax_E.legend(loc='upper left', fontsize=9)
ax_E.set_title('(E)', loc='left', fontsize=11, pad=4)
sns.despine(ax=ax_E)

# Panel F: entropy vs saturation limits on shared scale
ax_F.plot(r_ent.times, r_ent.entropy_A, '-', color=C['black'], linewidth=1.8,
          label=r'$S_A(t)$', zorder=5)

sat_labels = [20, 40, 80, 200]
for chi, col in zip(sat_labels, palette_ent):
    ax_F.axhline(np.log(chi), color=col, linewidth=0.9, linestyle='--',
                 label=fr'$\log({chi}) = {np.log(chi):.2f}$')

ax_F.set_xlabel(r'time $t$')
ax_F.set_ylabel(r'entropy')
ax_F.set_xlim(0, 5)
ax_F.set_ylim(0, 5.8)
ax_F.legend(loc='upper right', fontsize=8.5)
ax_F.set_title('(F)', loc='left', fontsize=11, pad=4)
sns.despine(ax=ax_F)

fig3.tight_layout(w_pad=3.5)
fig3.savefig('results/figures/fig3_entropy.png', bbox_inches='tight')
print("Saved fig3_entropy")

plt.close('all')

# ---- summary ----------------------------------------------------------------
print("\n=== Error budget ===")
for chi in [20, 40, 80]:
    err = np.max(np.abs(conv[chi].lambda_t - ref.lambda_t))
    print(f"  chi={chi:3d} vs chi=200: {err:.2e}")
print(f"  Trotter dt=0.05:        {np.max(trotter_err):.2e}")
print("\n=== DPT cusp positions ===")
for h1, tc in zip(h1_arr, tstar_arr):
    flag = '' if h1 > 1 else '  [ambiguous: h1 < QPT]'
    print(f"  h1={h1}: t*_0 = {tc:.2f} ± 0.025{flag}")
