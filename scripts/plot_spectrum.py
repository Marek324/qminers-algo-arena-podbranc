"""Plot the npz written by scripts/spectrum.py (run in a throwaway env: uv run --no-project --with matplotlib --with numpy)."""
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

d = np.load(sys.argv[1])
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), facecolor='#fcfcfb')
for k, ax in enumerate(axes):
    per = 1 / d[f'fc{k}']
    ax.set_facecolor('#fcfcfb')
    ax.fill_between(per, d[f'lo{k}'], d[f'hi{k}'], color='#86b6ef', alpha=0.35, lw=0, label='shuffled 60 s blocks, 95% band')
    ax.plot(per, d[f'med{k}'], color='#2a78d6', lw=1.2, ls='--', label='shuffled, median')
    ax.plot(per, d[f'pc{k}'], color='#eb6834', lw=2, marker='o', ms=4, label=str(d['tape']))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.invert_xaxis()
    ax.set_xlabel('period (s), long on the left', color='#52514e')
    ax.set_ylabel('power (log-binned)', color='#52514e')
    ax.set_title(str(d[f'name{k}']), color='#0b0b0b', fontsize=11, loc='left')
    ax.grid(True, which='major', color='#e6e5e1', lw=0.6)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#bdbcb6')
    ax.tick_params(colors='#52514e')
    ax.legend(frameon=False, fontsize=8, loc='lower left' if k == 0 else 'upper right')
fig.suptitle(f"{d['tape']}: spectrum after removing the lowest-frequency wave", x=0.01, ha='left', fontsize=12, color='#0b0b0b')
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=130)
print('saved', sys.argv[2])
