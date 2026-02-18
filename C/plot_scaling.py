"""
MAP55672 Case 1 — Question 3: TSQR Scaling Plots
Run via: python3 plot_scaling.py  (after run_scaling.sh has finished)
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def read_results(filename):
    data = defaultdict(list)
    with open(filename) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 3:
                m, n, t = int(parts[0]), int(parts[1]), float(parts[2])
                data[(m, n)].append(t)
    return data

data_m = read_results("results_vary_m.csv")
data_n = read_results("results_vary_n.csv")

ms  = sorted(set(k[0] for k in data_m))
t_m = [np.mean(data_m[(m, 8)]) for m in ms]

ns  = sorted(set(k[1] for k in data_n))
t_n = [np.mean(data_n[(4000, n)]) for n in ns]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

ax = axes[0]
ax.loglog(ms, t_m, 'o-', color='steelblue', linewidth=2, markersize=7, label='TSQR')
c = t_m[0] / ms[0]
ax.loglog(ms, [c * m for m in ms], '--', color='gray', label='O(m) reference')
ax.set_xlabel('m  (number of rows)', fontsize=12)
ax.set_ylabel('Wall time (s)', fontsize=12)
ax.set_title('Scaling with m  (n = 8 fixed)', fontsize=12)
ax.legend()
ax.grid(True, which='both', alpha=0.3)

ax = axes[1]
ax.loglog(ns, t_n, 's-', color='darkorange', linewidth=2, markersize=7, label='TSQR')
c2 = t_n[0] / ns[0]**2
ax.loglog(ns, [c2 * n**2 for n in ns], '--', color='gray', label='O(n²) reference')
ax.set_xlabel('n  (number of columns)', fontsize=12)
ax.set_ylabel('Wall time (s)', fontsize=12)
ax.set_title('Scaling with n  (m = 4000 fixed)', fontsize=12)
ax.legend()
ax.grid(True, which='both', alpha=0.3)

plt.suptitle('TSQR Scaling (4 MPI processes, Seagull cluster)', fontsize=13)
plt.tight_layout()
plt.savefig('tsqr_scaling.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: tsqr_scaling.png")
