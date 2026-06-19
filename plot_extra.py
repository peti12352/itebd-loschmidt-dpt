"""
plot_extra.py — Additional figures for the project report.

Fig 4: Order parameter <Sz_A>(t) for h=0.5, 2.0, 3.0
Fig 5: Cusp sharpness vs chi near first cusp of h=2.0

Run: uv run python plot_extra.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from tns.io import load_result
import os

os.makedirs('results/figures', exist_ok=True)

# ---- style (matches plot_a4.py) ----------------------------------------
sns.set_theme(style='ticks', context='paper', font='serif', font_scale=1.35)
plt.rcParams.update({
    'figure.dpi': 200,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'xtick.minor.visible': True, 'ytick.minor.visible': True,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.5,
    'legend.frameon': False,
    'legend.fontsize': 8.5,
    'font.family': 'serif',
})

# Wong (2011) colorblind-safe palette
C = {
    'black':  '#000000',
    'orange': '#E69F00',
    'sky':    '#56B4E9',
    'green':  '#009E73',
    'yellow': '#F0E442',
    'blue':   '#0072B2',
    'red':    '#D55E00',
    'purple': '#CC79A7',
}

# -----------------------------------------------------------------------
# Figure 4: Order parameter <Sz_A>(t)
# -----------------------------------------------------------------------
fig4, ax = plt.subplots(figsize=(5.2, 3.5))

h1_plot = [0.5, 2.0, 3.0]
colors   = [C['blue'], C['orange'], C['green']]
labels   = [r'$h=0.5$ (no DPT)', r'$h=2.0$', r'$h=3.0$']
ls       = ['--', '-', '-']

for h1, col, lab, style in zip(h1_plot, colors, labels, ls):
    r = load_result(f'results/data/scan_h{h1}_chi40.h5')
    ax.plot(r.times, r.sz_A, linestyle=style, color=col, label=lab, linewidth=1.6)

ax.axhline(0, color='0.7', linewidth=0.7, linestyle=':')
ax.set_xlabel(r'time $t$')
ax.set_ylabel(r'$\langle S^z_A \rangle(t)$')
ax.set_xlim(0, 5)
ax.set_ylim(-0.55, 0.55)
ax.legend(loc='lower right', fontsize=8.5)
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.125))
sns.despine(ax=ax)

fig4.tight_layout()
fig4.savefig('results/figures/fig4_order_parameter.png', bbox_inches='tight')
print('Saved fig4_order_parameter')

# -----------------------------------------------------------------------
# Figure 5: Cusp sharpness vs chi (zoom near first cusp, h=2.0, t*~1.65)
# -----------------------------------------------------------------------
fig5, ax2 = plt.subplots(figsize=(5.2, 3.5))

chi_vals = [5, 10, 20, 40]
pal = sns.color_palette('flare', n_colors=len(chi_vals))

t_lo, t_hi = 0.5, 3.2
for chi, col in zip(chi_vals, pal):
    r = load_result(f'results/data/conv_h2.0_chi{chi}.h5')
    mask = (r.times >= t_lo) & (r.times <= t_hi)
    ax2.plot(r.times[mask], r.lambda_t[mask], color=col,
             label=fr'$\chi_{{\max}}={chi}$', linewidth=1.6)

# reference chi=200
r200 = load_result('results/data/conv_h2.0_chi200.h5')
mask = (r200.times >= t_lo) & (r200.times <= t_hi)
ax2.plot(r200.times[mask], r200.lambda_t[mask], color='k', linestyle='--',
         linewidth=1.1, label=r'$\chi_{\max}=200$ (ref)', zorder=5)

ax2.axvline(1.65, color='0.6', linewidth=0.8, linestyle=':', zorder=0)
ax2.set_xlabel(r'time $t$')
ax2.set_ylabel(r'$\lambda(t) = -\ln|\tau(t)|$')
ax2.set_xlim(t_lo, t_hi)
ax2.legend(loc='upper right', fontsize=8.5)
sns.despine(ax=ax2)

fig5.tight_layout()
fig5.savefig('results/figures/fig5_cusp_sharpness.png', bbox_inches='tight')
print('Saved fig5_cusp_sharpness')

plt.close('all')
