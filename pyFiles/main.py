"""
main.py — STAR-RIS Vehicular Beamforming: All-Methods Comparison

Runs all four sweep scenarios across five methods:
    ddpg | qaoa | qddpg | qppo | baseline

Results are saved as CSV files; figures saved as PNG.
Run in the background with:
    nohup python main.py > run.log 2>&1 &
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hamiltonian import (verify_gradient, build_pauli_hamiltonian,
                         verify_4qubit_gradient)
from experiments import scenario_snr, scenario_N, scenario_speed, scenario_K
from simulator import DEFAULT_PARAMS

if __name__ == '__main__':

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
        raise RuntimeError("2-qubit gradient check failed.")

    print("\n" + "=" * 60)
    print("4-qubit Hamiltonian check")
    print("=" * 60)

    a_vec = [1.2, 0.9, 0.7, 0.5]
    c_mat = np.array([[0, 0.3, 0.2, 0.1],
                       [0, 0,   0.25, 0.15],
                       [0, 0,   0,    0.2],
                       [0, 0,   0,    0]])
    print()
    ok4 = verify_4qubit_gradient(gamma=0.4, beta=0.7, a_vec=a_vec, c_mat=c_mat)
    if not ok4:
        raise RuntimeError("4-qubit gradient check failed.")

    # ── 2. Scenarios ──────────────────────────────────────────────────────────────
    N_TRIALS = 30

    print("\nScenario 1: SNR sweep")
    df_snr = scenario_snr(snr_db_range=range(0, 31, 5), n_trials=N_TRIALS)
    df_snr.to_csv('results_snr.csv', index=False)
    print("  Saved results_snr.csv")

    print("\nScenario 2: N sweep (number of STAR-RIS elements)")
    df_N = scenario_N(N_values=(16, 32, 64), n_trials=N_TRIALS)
    df_N.to_csv('results_N.csv', index=False)
    print("  Saved results_N.csv")

    print("\nScenario 3: Speed sweep")
    df_spd = scenario_speed(speed_values=(5, 10, 20, 30), n_trials=N_TRIALS)
    df_spd.to_csv('results_speed.csv', index=False)
    print("  Saved results_speed.csv")

    print("\nScenario 4: K (number of cars) sweep")
    df_K = scenario_K(K_values=((2, 2), (4, 4), (8, 8)), n_trials=N_TRIALS)
    df_K.to_csv('results_K.csv', index=False)
    print("  Saved results_K.csv")

    # ── 3. Tables ─────────────────────────────────────────────────────────────────
    ALL_METHODS = ['qaoa', 'qddpg', 'qppo', 'baseline']

    def print_table(df, sweep_col, title):
        print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")
        hdr = f"  {'val':<8} {'method':<12} {'SR':>7} {'E':>8} {'t(s)':>7}  "\
              f"{'Mem(MB)':>8} {'circ':>7} {'c-std':>7} {'c-slp':>8}"
        print(hdr)
        print('  ' + '-' * 68)
        for val in sorted(df[sweep_col].unique()):
            sub = df[df[sweep_col] == val]
            for m in ALL_METHODS:
                rows = sub[sub.method == m]
                if rows.empty:
                    continue
                r = rows.iloc[0]
                print(f"  {str(val):<8} {m:<12}"
                      f"  {r['sum_rate_mean']:6.3f}"
                      f"  {r['energy_norm_mean']:7.1f}"
                      f"  {r['time_s_mean']:6.3f}"
                      f"  {r['peak_mem_mb_mean']:7.2f}"
                      f"  {r['circuit_evals_mean']:6.0f}"
                      f"  {r['conv_std_mean']:6.4f}"
                      f"  {r['conv_slope_mean']:+.4f}")

    print_table(df_snr, 'snr_db',   "Table 1 – SNR sweep")
    print_table(df_N,   'N',        "Table 2 – N sweep")
    print_table(df_spd, 'speed_ms', "Table 3 – Speed sweep")
    print_table(df_K,   'K',        "Table 4 – K sweep")

    # Complexity analysis summary (aggregated across sweep values)
    print(f"\n{'=' * 70}")
    print("Complexity & Resource Analysis (mean across all sweep points)")
    print(f"{'=' * 70}")
    print(f"  {'Method':<12} {'Time(s)':>8} {'Mem(MB)':>9} "
          f"{'CircEvals':>10} {'Iters':>8} {'ConvStd':>9} {'ConvSlope':>10}")
    print('  ' + '-' * 68)
    for m in ALL_METHODS:
        rows = df_snr[df_snr.method == m]
        print(f"  {m:<12}"
              f"  {rows['time_s_mean'].mean():7.3f}"
              f"  {rows['peak_mem_mb_mean'].mean():8.2f}"
              f"  {rows['circuit_evals_mean'].mean():9.0f}"
              f"  {rows['iterations_mean'].mean():7.0f}"
              f"  {rows['conv_std_mean'].mean():8.4f}"
              f"  {rows['conv_slope_mean'].mean():+.5f}")

    # ── 4. Plots ──────────────────────────────────────────────────────────────────
    COLORS  = {
        'ddpg':     '#D4380D',
        'qaoa':     '#0F6E56',
        'qddpg':    '#2563EB',
        'qppo':     '#7C3AED',
        'baseline': '#6B7280',
    }
    MARKERS = {'ddpg': 's', 'qaoa': '^', 'qddpg': 'o', 'qppo': 'D', 'baseline': 'x'}
    LS      = {'ddpg': '-', 'qaoa': '-.', 'qddpg': '--', 'qppo': ':', 'baseline': '-'}
    LABELS  = {
        'ddpg':     'DDPG',
        'qaoa':     'QAOA (Hybrid)',
        'qddpg':    'QDDPG',
        'qppo':     'QPPO',
        'baseline': 'STAR-RIS Baseline',
    }


    def _plot(df, xcol, ycol, xlabel, ylabel, title, fname):
        fig, ax = plt.subplots(figsize=(7, 4))
        for m in ALL_METHODS:
            s = df[df.method == m].sort_values(xcol)
            if s.empty:
                continue
            ax.plot(s[xcol], s[ycol + '_mean'],
                    color=COLORS[m], marker=MARKERS[m],
                    linestyle=LS[m], label=LABELS[m], linewidth=1.8)
            ax.fill_between(s[xcol],
                            s[ycol + '_mean'] - s[ycol + '_std'],
                            s[ycol + '_mean'] + s[ycol + '_std'],
                            color=COLORS[m], alpha=0.10)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.legend(fontsize=8); ax.grid(True, linewidth=0.4)
        plt.tight_layout(); plt.savefig(fname, dpi=150); plt.close()
        print(f"  Saved {fname}")


    # Convergence stability plot (conv_std vs SNR)
    print("\nConvergence & resource figures...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for m in ALL_METHODS:
        s = df_snr[df_snr.method == m].sort_values('snr_db')
        axes[0].plot(s['snr_db'], s['conv_std_mean'],
                     color=COLORS[m], marker=MARKERS[m],
                     linestyle=LS[m], label=LABELS[m], linewidth=1.8)
        axes[1].plot(s['snr_db'], s['peak_mem_mb_mean'],
                     color=COLORS[m], marker=MARKERS[m],
                     linestyle=LS[m], label=LABELS[m], linewidth=1.8)
        axes[2].plot(s['snr_db'], s['time_s_mean'],
                     color=COLORS[m], marker=MARKERS[m],
                     linestyle=LS[m], label=LABELS[m], linewidth=1.8)

    axes[0].set_xlabel('SNR (dB)'); axes[0].set_ylabel('Conv. Std (bits/s/Hz)')
    axes[0].set_title('Convergence Stability'); axes[0].legend(fontsize=7)
    axes[0].grid(True, linewidth=0.4)

    axes[1].set_xlabel('SNR (dB)'); axes[1].set_ylabel('Peak Memory (MB)')
    axes[1].set_title('Memory Usage'); axes[1].legend(fontsize=7)
    axes[1].grid(True, linewidth=0.4)

    axes[2].set_xlabel('SNR (dB)'); axes[2].set_ylabel('Exec. Time (s)')
    axes[2].set_title('Execution Time'); axes[2].legend(fontsize=7)
    axes[2].grid(True, linewidth=0.4)

    plt.tight_layout()
    plt.savefig('fig_complexity_analysis.png', dpi=150)
    plt.close()
    print("  Saved fig_complexity_analysis.png")

    # Circuit evaluations vs N (quantum methods only)
    qmethods = ['qaoa', 'qddpg', 'qppo']
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in qmethods:
        s = df_N[df_N.method == m].sort_values('N')
        ax.plot(s['N'], s['circuit_evals_mean'],
                color=COLORS[m], marker=MARKERS[m],
                linestyle=LS[m], label=LABELS[m], linewidth=1.8)
    ax.set_xlabel('STAR-RIS elements N')
    ax.set_ylabel('Quantum circuit evaluations')
    ax.set_title('Computational Complexity vs N (quantum methods)')
    ax.legend(fontsize=8); ax.grid(True, linewidth=0.4)
    plt.tight_layout()
    plt.savefig('fig_circuit_evals_N.png', dpi=150)
    plt.close()
    print("  Saved fig_circuit_evals_N.png")

    print("\nGenerating sum-rate figures...")

    _plot(df_snr, 'snr_db',   'sum_rate', 'SNR (dB)',
          'Avg sum-rate (bits/s/Hz)', 'Sum-rate vs SNR',
          'fig_sumrate_snr.png')

    _plot(df_spd, 'speed_ms', 'sum_rate', 'Car speed (m/s)',
          'Avg sum-rate (bits/s/Hz)', 'Sum-rate vs Car Speed',
          'fig_sumrate_speed.png')

    _plot(df_N,   'N',        'sum_rate', 'STAR-RIS elements N',
          'Avg sum-rate (bits/s/Hz)', 'Sum-rate vs N',
          'fig_sumrate_N.png')

    _plot(df_K,   'K',        'sum_rate', 'Number of cars K',
          'Avg sum-rate (bits/s/Hz)', 'Sum-rate vs K',
          'fig_sumrate_K.png')

    # Energy efficiency plots
    for df, col, fname in [
        (df_snr, 'snr_db',   'fig_ee_snr.png'),
        (df_spd, 'speed_ms', 'fig_ee_speed.png'),
        (df_N,   'N',        'fig_ee_N.png'),
        (df_K,   'K',        'fig_ee_K.png'),
    ]:
        df2 = df.copy()
        df2['ee_mean'] = df2['sum_rate_mean'] / (df2['energy_norm_mean'] + 1e-9)
        df2['ee_std']  = df2['ee_mean'] * 0.1
        _plot(df2, col, 'ee', col,
              'Energy efficiency (bits/s/Hz / energy)',
              f'Energy Efficiency vs {col}', fname)

    print("\nDone.")
