"""
plot_analysis.py
─────────────────────────────────────────────────────────────────────
Output structure:
  results/analysis_output/
    K/
      graph_all_methods.png ← NEW: all 5 methods, 4 metric subplots overlaid
      graph_ddpg.png        ← 4 metric subplots for DDPG vs K
      graph_qaoa.png
      ...
      table_ddpg.png        ← table: DDPG × K sweep (rows=K values, cols=4 metrics)
      table_qaoa.png
      ...
    N/    (same layout)
    SNR/  (same layout)
    Speed/(same layout)

Total: 4 scenarios × (1 combined + 5 individual graphs + 5 tables) = 44 files
─────────────────────────────────────────────────────────────────────
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np

# ─── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
BASE_OUT    = os.path.join(RESULTS_DIR, 'analysis_output')

# ─── Config ────────────────────────────────────────────────────────────────────
METHODS = ['ddpg', 'qaoa', 'qddpg', 'qppo', 'baseline']

# (display label, category tag, use log-y)
METRICS = {
    'sum_rate_mean':    ('Sum Rate (bps/Hz)',   'Performance', False),
    'conv_std_mean':    ('Convergence Std',     'Stability',   False),
    'iterations_mean':  ('Iterations',          'Speed',       False),
    'energy_norm_mean': ('Energy (normalised)', 'Energy',      True),
}

SWEEPS = {
    'K':     {'file': 'results_K.csv',     'x_col': 'K',        'label': 'Number of Users (K)'},
    'N':     {'file': 'results_N.csv',     'x_col': 'N',        'label': 'RIS Elements (N)'},
    'SNR':   {'file': 'results_snr.csv',   'x_col': 'snr_db',   'label': 'SNR (dB)'},
    'Speed': {'file': 'results_speed.csv', 'x_col': 'speed_ms', 'label': 'Vehicle Speed (m/s)'},
}

METHOD_COLORS = {
    'ddpg':     '#3A86FF',
    'qaoa':     '#FF6B6B',
    'qddpg':    '#2EC4B6',
    'qppo':     '#F7931E',
    'baseline': '#8338EC',
}

METHOD_MARKERS = {
    'ddpg':     'o',
    'qaoa':     's',
    'qddpg':    '^',
    'qppo':     'D',
    'baseline': 'P',
}

METHOD_LABELS = {
    'ddpg':     'DDPG',
    'qaoa':     'QAOA',
    'qddpg':    'Q-DDPG',
    'qppo':     'Q-PPO',
    'baseline': 'Baseline',
}

HEADER_COLOR = '#1E2D40'

SWEEP_ACCENT = {
    'K':     '#D6EAF8',
    'N':     '#D5F5E3',
    'SNR':   '#FDEDEC',
    'Speed': '#FAF0E6',
}

# ─── Load all CSVs once ────────────────────────────────────────────────────────
print("Loading data...")
sweep_data = {}
for sweep_name, cfg in SWEEPS.items():
    path = os.path.join(RESULTS_DIR, cfg['file'])
    df   = pd.read_csv(path).dropna()
    df   = df.sort_values(cfg['x_col'])
    sweep_data[sweep_name] = df
print(f"  Loaded {len(SWEEPS)} sweep files.\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  Helper: draw combined all-methods graph for a scenario
# ═══════════════════════════════════════════════════════════════════════════════
def plot_all_methods(sweep_name, df_sweep, cfg, out_dir):
    """
    One figure with 4 metric subplots.
    Each subplot shows all 5 methods overlaid with distinct colour+marker.
    """
    x_col   = cfg['x_col']
    x_label = cfg['label']

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
    fig.patch.set_facecolor('#F5F7FA')

    fig.suptitle(
        f"Scenario: {sweep_name}  ·  All Methods Comparison",
        fontsize=15, fontweight='bold', color=HEADER_COLOR, y=1.03,
    )

    metric_items = list(METRICS.items())
    for col_idx, (metric, (ylabel, category, use_log)) in enumerate(metric_items):
        ax = axes[col_idx]
        ax.set_facecolor('#FFFFFF')

        for method in METHODS:
            df_m   = df_sweep[df_sweep['method'] == method].sort_values(x_col)
            color  = METHOD_COLORS[method]
            marker = METHOD_MARKERS[method]
            label  = METHOD_LABELS[method]

            x = df_m[x_col].values
            y = df_m[metric].values
            std_col = metric.replace('_mean', '_std')
            y_err   = df_m[std_col].values if std_col in df_m.columns else np.zeros_like(y)

            ax.plot(
                x, y,
                marker=marker, linewidth=2.0, markersize=7,
                color=color, label=label, zorder=3,
            )
            ax.fill_between(
                x, y - y_err, y + y_err,
                alpha=0.10, color=color, zorder=2,
            )

        ax.set_xlabel(x_label,  fontsize=10, color='#444444', labelpad=6)
        ax.set_ylabel(ylabel,   fontsize=10, color='#444444')
        ax.set_title(f'{category}\n{ylabel}', fontsize=11,
                     fontweight='semibold', color=HEADER_COLOR, pad=8)

        # x-ticks: all unique sweep values
        x_all = sorted(df_sweep[x_col].unique())
        ax.set_xticks(x_all)
        ax.tick_params(labelsize=9)
        ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.5, color='#CCCCCC')
        ax.spines[['top', 'right']].set_visible(False)

        if use_log:
            ax.set_yscale('log')
            ax.set_ylabel(ylabel + ' [log scale]', fontsize=9, color='#444444')

    # ── Shared legend below the figure ────────────────────────────────────────
    legend_handles = [
        mlines.Line2D(
            [], [],
            color=METHOD_COLORS[m], marker=METHOD_MARKERS[m],
            linewidth=2, markersize=8, label=METHOD_LABELS[m],
        )
        for m in METHODS
    ]
    fig.legend(
        handles=legend_handles,
        loc='lower center',
        ncol=5,
        fontsize=11,
        framealpha=0.92,
        edgecolor='#CCCCCC',
        bbox_to_anchor=(0.5, -0.08),
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out_path = os.path.join(out_dir, 'graph_all_methods.png')
    fig.savefig(out_path, dpi=160, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓  {sweep_name}/graph_all_methods.png  ← NEW combined graph")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main loop — one subdir per scenario
# ═══════════════════════════════════════════════════════════════════════════════
for sweep_name, cfg in SWEEPS.items():

    out_dir = os.path.join(BASE_OUT, sweep_name)
    os.makedirs(out_dir, exist_ok=True)

    df_sweep = sweep_data[sweep_name]
    x_col    = cfg['x_col']
    x_label  = cfg['label']
    x_vals   = sorted(df_sweep[x_col].unique())

    print(f"── Scenario: {sweep_name}  ({x_label}) ──────────────────────────")

    # ── COMBINED: all 5 methods on one figure ──────────────────────────────────
    plot_all_methods(sweep_name, df_sweep, cfg, out_dir)

    for method in METHODS:
        color  = METHOD_COLORS[method]
        marker = METHOD_MARKERS[method]
        df_m   = df_sweep[df_sweep['method'] == method].sort_values(x_col)

        # ── GRAPH: 1 figure × 4 metric subplots ────────────────────────────────
        fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
        fig.patch.set_facecolor('#FAFAFA')
        fig.suptitle(
            f"Scenario: {sweep_name}  ·  Method: {method.upper()}",
            fontsize=14, fontweight='bold', color=HEADER_COLOR, y=1.03,
        )

        for ax, (metric, (ylabel, category, use_log)) in zip(axes, METRICS.items()):
            x = df_m[x_col].values
            y = df_m[metric].values
            std_col = metric.replace('_mean', '_std')
            y_err   = df_m[std_col].values if std_col in df_m.columns else np.zeros_like(y)

            ax.set_facecolor('#F7F9FC')
            ax.plot(x, y, marker=marker, linewidth=2.2, markersize=7,
                    color=color, zorder=3)
            ax.fill_between(x, y - y_err, y + y_err,
                            alpha=0.18, color=color, zorder=2)

            ax.set_xlabel(x_label,  fontsize=10, color='#333333')
            ax.set_ylabel(ylabel,   fontsize=10, color='#333333')
            ax.set_title(f'{category}: {ylabel}', fontsize=11,
                         fontweight='semibold', color=HEADER_COLOR)
            ax.set_xticks(x)
            ax.tick_params(labelsize=9)
            ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.6, color='#CCCCCC')
            ax.spines[['top', 'right']].set_visible(False)

            if use_log:
                ax.set_yscale('log')
                ax.set_ylabel(ylabel + ' [log scale]', fontsize=9)

        plt.tight_layout(rect=[0, 0, 1, 1])
        graph_path = os.path.join(out_dir, f'graph_{method}.png')
        fig.savefig(graph_path, dpi=160, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  ✓  {sweep_name}/graph_{method}.png")

        # ── TABLE: rows = x-values, cols = 4 metrics ───────────────────────────
        metric_keys = list(METRICS.keys())
        col_labels  = [
            x_label,
            'Sum Rate\n(bps/Hz)\n[Performance]',
            'Conv Std\n[Stability]',
            'Iterations\n[Speed]',
            'Energy Norm\n[Energy]',
        ]

        cell_data  = []
        row_colors = []
        accent     = SWEEP_ACCENT[sweep_name]

        for _, row in df_m.iterrows():
            cell_data.append([
                f"{row[x_col]}",
                f"{row['sum_rate_mean']:.4f}",
                f"{row['conv_std_mean']:.4f}",
                f"{row['iterations_mean']:.0f}",
                f"{row['energy_norm_mean']:.1f}",
            ])
            row_colors.append([accent] * len(col_labels))

        n_rows = len(cell_data)
        n_cols = len(col_labels)

        fig_h = 1.4 + n_rows * 0.6
        fig, ax = plt.subplots(figsize=(12, fig_h))
        fig.patch.set_facecolor('#FAFAFA')
        ax.axis('off')

        tbl = ax.table(
            cellText=cell_data,
            colLabels=col_labels,
            cellLoc='center',
            loc='center',
            cellColours=row_colors,
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(11)
        tbl.scale(1.2, 2.2)

        # Header styling
        for j in range(n_cols):
            cell = tbl[0, j]
            cell.set_facecolor(HEADER_COLOR)
            cell.set_text_props(color='white', fontweight='bold', fontsize=11)
            cell.set_edgecolor('#FFFFFF')

        # Data row borders + x-value column bold
        for i in range(1, n_rows + 1):
            for j in range(n_cols):
                tbl[i, j].set_edgecolor('#CCCCCC')
                tbl[i, j].set_linewidth(0.5)
            tbl[i, 0].set_text_props(fontweight='bold', color=HEADER_COLOR)

        # Method colour patch legend
        legend_patch = mpatches.Patch(color=color, label=method.upper())
        ax.legend(handles=[legend_patch], loc='upper right',
                  fontsize=11, framealpha=0.9, edgecolor=color,
                  bbox_to_anchor=(1.0, 1.10))

        fig.suptitle(
            f'Method: {method.upper()}  |  Scenario: {sweep_name} ({x_label})',
            fontsize=14, fontweight='bold', color=HEADER_COLOR, y=1.02,
        )

        plt.tight_layout()
        table_path = os.path.join(out_dir, f'table_{method}.png')
        fig.savefig(table_path, dpi=160, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  ✓  {sweep_name}/table_{method}.png")

    print()

print(f"✅  Done!  44 files saved →  {BASE_OUT}/{{K,N,SNR,Speed}}/")
