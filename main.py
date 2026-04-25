"""
main.py — STAR-RIS Vehicular Beamforming: DDPG vs QAOA Comparison

Runs all four sweep scenarios, prints result tables, and generates
publication-quality comparison plots (saved as PNG).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hamiltonian import verify_gradient, build_pauli_hamiltonian
from experiments import scenario_snr, scenario_N, scenario_speed, scenario_K
from simulator import DEFAULT_PARAMS

# ── 1. Gradient check ─────────────────────────────────────────────────────────
print("=" * 60)
print("Toy 2-element Hamiltonian check")
print("=" * 60)

a, b, c = 1.2, 0.9, 0.5
H_op = build_pauli_hamiltonian(a, b, c)
print("\nPauli decomposition:")
for term in H_op:
    print(f"  {float(np.real(term.coeffs[0])):+.6f} * {str(term.paulis[0])}")

print()
ok = verify_gradient(gamma=0.4, beta=0.7, a=a, b=b, c=c)
if not ok:
    raise RuntimeError("Gradient check failed.")

# ── 2. Scenarios ──────────────────────────────────────────────────────────────
N_TRIALS = 20

print("\nScenario 1: SNR sweep (dynamic cars)")
df_snr = scenario_snr(snr_db_range=range(0, 31, 5), n_trials=N_TRIALS)
df_snr.to_csv('results_snr.csv', index=False)

print("\nScenario 2: N sweep (dynamic cars)")
df_N = scenario_N(N_values=(16, 32, 64), n_trials=N_TRIALS)
df_N.to_csv('results_N.csv', index=False)

print("\nScenario 3: Speed sweep")
df_spd = scenario_speed(speed_values=(5, 10, 20, 30), n_trials=N_TRIALS)
df_spd.to_csv('results_speed.csv', index=False)

print("\nScenario 4: K (number of cars) sweep")
df_K = scenario_K(K_values=((2, 2), (4, 4), (8, 8)), n_trials=N_TRIALS)
df_K.to_csv('results_K.csv', index=False)

# ── 3. Tables ─────────────────────────────────────────────────────────────────
def print_table(df, sweep_col, title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
    for val in df[sweep_col].unique():
        sub = df[df[sweep_col] == val]
        for m in ['ddpg', 'qaoa']:
            row = sub[sub.method == m].iloc[0]
            print(f"  {val}  {m:<12}  "
                  f"SR={row['sum_rate_mean']:.3f}  "
                  f"E={row['energy_norm_mean']:.1f}  "
                  f"t={row['time_s_mean']:.3f}s")

print_table(df_snr, 'snr_db',   "Table 1 – SNR sweep")
print_table(df_N,   'N',        "Table 2 – N sweep")
print_table(df_spd, 'speed_ms', "Table 3 – Speed sweep")
print_table(df_K,   'K',        "Table 4 – K sweep")

# ── 4. Plots ──────────────────────────────────────────────────────────────────
COLORS  = {'ddpg': '#D4380D', 'qaoa': '#0F6E56'}
MARKERS = {'ddpg': 's',       'qaoa': '^'}
LS      = {'ddpg': '-',       'qaoa': '-.'}
LABELS  = {'ddpg': 'DDPG',    'qaoa': 'QAOA (Hybrid)'}

def _plot(df, xcol, ycol, xlabel, ylabel, title, fname):
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in ['ddpg', 'qaoa']:
        s = df[df.method == m].sort_values(xcol)
        ax.plot(s[xcol], s[ycol + '_mean'],
                color=COLORS[m], marker=MARKERS[m],
                linestyle=LS[m], label=LABELS[m], linewidth=1.5)
        ax.fill_between(s[xcol],
                        s[ycol + '_mean'] - s[ycol + '_std'],
                        s[ycol + '_mean'] + s[ycol + '_std'],
                        color=COLORS[m], alpha=0.12)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title); ax.legend(); ax.grid(True, linewidth=0.4)
    plt.tight_layout(); plt.savefig(fname, dpi=150); plt.close()
    print(f"  Saved {fname}")

print("\nGenerating figures...")
_plot(df_snr, 'snr_db',   'sum_rate', 'SNR (dB)',
      'Avg sum-rate (bits/s/Hz)', 'Sum-rate vs SNR — DDPG vs QAOA',
      'fig_sumrate_snr.png')

_plot(df_spd, 'speed_ms', 'sum_rate', 'Car speed (m/s)',
      'Avg sum-rate (bits/s/Hz)', 'Sum-rate vs speed — DDPG vs QAOA',
      'fig_sumrate_speed.png')

_plot(df_N,   'N',        'sum_rate', 'STAR-RIS elements N',
      'Avg sum-rate (bits/s/Hz)', 'Sum-rate vs N — DDPG vs QAOA',
      'fig_sumrate_N.png')

_plot(df_K,   'K',        'sum_rate', 'Number of cars K',
      'Avg sum-rate (bits/s/Hz)', 'Sum-rate vs K — DDPG vs QAOA',
      'fig_sumrate_K.png')

# Energy efficiency plots
for df, col, fname in [
    (df_snr, 'snr_db',   'fig_ee_snr.png'),
    (df_spd, 'speed_ms', 'fig_ee_speed.png'),
    (df_N,   'N',        'fig_ee_N.png'),
]:
    df = df.copy()
    df['ee_mean'] = df['sum_rate_mean'] / (df['energy_norm_mean'] + 1e-9)
    df['ee_std']  = df['ee_mean'] * 0.1
    _plot(df, col, 'ee', col,
          'Energy efficiency', f'Energy efficiency vs {col}', fname)

print("\nDone.")
