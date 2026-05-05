"""
quick_test.py — Mini dry-run of the full pipeline.

Runs all 4 scenarios with only 2 trials each.
If this completes without errors and produces CSV + PNG files,
you're safe to kick off the real 100-trial run overnight.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hamiltonian import verify_gradient, build_pauli_hamiltonian
from experiments import scenario_snr, scenario_N, scenario_speed, scenario_K
from simulator import DEFAULT_PARAMS

N_TRIALS = 2  # tiny — just enough to verify everything runs

print("=" * 60)
print("QUICK PIPELINE TEST (2 trials per scenario)")
print("=" * 60)

# 1. Gradient check
print("\n[1] Gradient check...")
ok = verify_gradient(gamma=0.4, beta=0.7, a=1.2, b=0.9, c=0.5)
assert ok, "Gradient check FAILED"
print("    ✓ Passed\n")

# 2. All four scenarios (tiny)
print("[2] Scenario 1 — SNR sweep")
df_snr = scenario_snr(snr_db_range=range(0, 11, 5), n_trials=N_TRIALS)
df_snr.to_csv('qt_results_snr.csv', index=False)
print("    ✓ CSV saved: qt_results_snr.csv\n")

print("[3] Scenario 2 — N sweep")
df_N = scenario_N(N_values=(16, 32), n_trials=N_TRIALS)
df_N.to_csv('qt_results_N.csv', index=False)
print("    ✓ CSV saved: qt_results_N.csv\n")

print("[4] Scenario 3 — Speed sweep")
df_spd = scenario_speed(speed_values=(5, 30), n_trials=N_TRIALS)
df_spd.to_csv('qt_results_speed.csv', index=False)
print("    ✓ CSV saved: qt_results_speed.csv\n")

print("[5] Scenario 4 — K sweep")
df_K = scenario_K(K_values=((2, 2), (4, 4)), n_trials=N_TRIALS)
df_K.to_csv('qt_results_K.csv', index=False)
print("    ✓ CSV saved: qt_results_K.csv\n")

# 3. Plot check
print("[6] Generating test plot...")
COLORS  = {'ddpg': '#D4380D', 'qaoa': '#0F6E56'}
MARKERS = {'ddpg': 's',       'qaoa': '^'}
fig, ax = plt.subplots(figsize=(6, 4))
for m in ['ddpg', 'qaoa']:
    s = df_snr[df_snr.method == m].sort_values('snr_db')
    ax.plot(s['snr_db'], s['sum_rate_mean'],
            color=COLORS[m], marker=MARKERS[m], label=m.upper())
ax.set_xlabel('SNR (dB)'); ax.set_ylabel('Avg sum-rate (bits/s/Hz)')
ax.set_title('Quick Test — Sum-rate vs SNR'); ax.legend(); ax.grid(True)
plt.tight_layout()
plt.savefig('qt_fig_snr.png', dpi=100)
plt.close()
print("    ✓ Plot saved: qt_fig_snr.png\n")

print("=" * 60)
print("ALL PIPELINE CHECKS PASSED — safe to run 100-trial overnight job")
print("=" * 60)
print("\nFiles produced:")
import os
for f in ['qt_results_snr.csv','qt_results_N.csv','qt_results_speed.csv',
          'qt_results_K.csv','qt_fig_snr.png']:
    size = os.path.getsize(f) if os.path.exists(f) else 0
    print(f"  {f}  ({size} bytes)")
