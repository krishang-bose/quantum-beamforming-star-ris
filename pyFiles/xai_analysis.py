"""
xai_analysis.py  —  Explainable AI for STAR-RIS Beamforming Benchmark
═══════════════════════════════════════════════════════════════════════
Inputs : results/results_{K,N,snr,speed}.csv
Outputs: results/analysis_output/xai/
  ├── shap/
  │   ├── shap_beeswarm_{metric}.png      — global feature importance per metric
  │   ├── shap_heatmap.png                — SHAP value heatmap (features × metrics)
  │   ├── shap_waterfall_{metric}_{method}.png  — per-instance local explanation
  │   └── shap_dependence_{feature}_{metric}.png — how one feature drives a metric
  └── lime/
      ├── lime_explanation_{metric}_{method}.png  — LIME bar chart per method × metric
      └── lime_summary_heatmap.png                — LIME importance summary
═══════════════════════════════════════════════════════════════════════
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from lime import lime_tabular

warnings.filterwarnings('ignore')

# ── Paths ───────────────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
XAI_DIR     = os.path.join(RESULTS_DIR, 'analysis_output', 'xai')
SHAP_DIR    = os.path.join(XAI_DIR, 'shap')
LIME_DIR    = os.path.join(XAI_DIR, 'lime')
for d in [SHAP_DIR, LIME_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Style ───────────────────────────────────────────────────────────────────────
HEADER_COLOR = '#1E2D40'
BG_COLOR     = '#F5F7FA'
ACCENT       = '#3A86FF'

METHOD_COLORS = {
    'ddpg':     '#3A86FF',
    'qaoa':     '#FF6B6B',
    'qddpg':    '#2EC4B6',
    'qppo':     '#F7931E',
    'baseline': '#8338EC',
}
METHODS = ['ddpg', 'qaoa', 'qddpg', 'qppo', 'baseline']

METRICS = {
    'sum_rate_mean':    'Sum Rate (bps/Hz)',
    'energy_norm_mean': 'Energy (normalised)',
    'conv_std_mean':    'Convergence Std',
    'iterations_mean':  'Iterations',
}

# ── Load & merge all CSVs into one unified dataframe ───────────────────────────
print("Loading and merging datasets...")

def load_sweep(fname, x_col):
    df = pd.read_csv(os.path.join(RESULTS_DIR, fname)).dropna()
    df['sweep_param'] = x_col
    df.rename(columns={x_col: 'param_value'}, inplace=True)
    return df

frames = [
    load_sweep('results_K.csv',     'K'),
    load_sweep('results_N.csv',     'N'),
    load_sweep('results_snr.csv',   'snr_db'),
    load_sweep('results_speed.csv', 'speed_ms'),
]
raw = pd.concat(frames, ignore_index=True)

# One-hot encode sweep_param so the model knows WHICH axis is being swept
sweep_dummies = pd.get_dummies(raw['sweep_param'], prefix='sweep')
raw = pd.concat([raw, sweep_dummies], axis=1)

# Encode method as integer
le_method = LabelEncoder()
raw['method_enc'] = le_method.fit_transform(raw['method'])

# Feature columns for the XAI models
FEAT_COLS = ['param_value', 'method_enc',
             'sweep_K', 'sweep_N', 'sweep_snr_db', 'sweep_speed_ms']
FEAT_LABELS = ['Param Value', 'Method',
               'Scenario: K', 'Scenario: N', 'Scenario: SNR', 'Scenario: Speed']

X = raw[FEAT_COLS].values

print(f"  Merged dataset: {X.shape[0]} rows × {len(FEAT_COLS)} features\n")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  1.  SHAP  —  TreeExplainer on Random Forest per metric                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
print("═" * 60)
print("  SHAP ANALYSIS")
print("═" * 60)

shap_values_store  = {}   # metric → SHAP matrix (n × p)
models_store       = {}   # metric → fitted RF

for metric, metric_label in METRICS.items():
    print(f"\n  ▸ Fitting Random Forest for: {metric_label}")
    y = raw[metric].values

    rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    models_store[metric] = rf

    explainer  = shap.TreeExplainer(rf)
    sv         = explainer.shap_values(X)
    shap_values_store[metric] = sv

    # ── 1a. Beeswarm / Summary Plot ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    shap.summary_plot(
        sv, X,
        feature_names=FEAT_LABELS,
        plot_type='dot',
        show=False,
        max_display=len(FEAT_COLS),
        color_bar=True,
    )
    plt.title(f'SHAP Beeswarm — {metric_label}',
              fontsize=14, fontweight='bold', color=HEADER_COLOR, pad=12)
    plt.tight_layout()
    out = os.path.join(SHAP_DIR, f'shap_beeswarm_{metric}.png')
    plt.savefig(out, dpi=160, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close()
    print(f"    ✓  shap/shap_beeswarm_{metric}.png")

    # ── 1b. Dependence plots — param_value vs each metric, coloured by method ─
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    # Use integer index 0 = 'Param Value'; interaction_index 1 = 'Method'
    shap.dependence_plot(
        0, sv, X,
        feature_names=FEAT_LABELS,
        interaction_index=1,
        ax=ax, show=False,
        dot_size=40, alpha=0.7,
    )
    ax.set_title(f'SHAP Dependence — Param Value → {metric_label}',
                 fontsize=13, fontweight='bold', color=HEADER_COLOR)
    ax.set_facecolor(BG_COLOR)
    plt.tight_layout()
    out = os.path.join(SHAP_DIR, f'shap_dependence_paramvalue_{metric}.png')
    plt.savefig(out, dpi=160, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close()
    print(f"    ✓  shap/shap_dependence_paramvalue_{metric}.png")

    # ── 1c. Waterfall plots — one representative instance per method ──────────
    raw_reset = raw.reset_index(drop=True)
    for method in METHODS:
        mask  = raw_reset['method'] == method
        if mask.sum() == 0:
            continue
        idx   = raw_reset.index[mask][0]    # positional index into X
        sv_i  = sv[idx]

        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor(BG_COLOR)

        # Build manual waterfall bar chart (shap.plots.waterfall needs Explanation obj)
        base_val  = float(np.array(explainer.expected_value).ravel()[0])
        feat_vals = X[idx]

        sorted_idx  = np.argsort(np.abs(sv_i))[::-1]
        sorted_sv   = sv_i[sorted_idx]
        sorted_feat = [FEAT_LABELS[i] for i in sorted_idx]
        sorted_fval = feat_vals[sorted_idx]

        colors = ['#3A86FF' if v > 0 else '#FF6B6B' for v in sorted_sv]
        y_pos  = np.arange(len(sorted_sv))

        bars = ax.barh(y_pos, sorted_sv, color=colors, edgecolor='white',
                       linewidth=0.5, height=0.65)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(
            [f'{f}\n= {v:.2g}' for f, v in zip(sorted_feat, sorted_fval)],
            fontsize=10,
        )
        ax.axvline(0, color='#333333', linewidth=1.2)
        ax.set_xlabel('SHAP Value (impact on model output)', fontsize=10)
        ax.set_title(
            f'SHAP Waterfall — {method.upper()} | {metric_label}\n'
            f'Base value: {base_val:.4f}  →  Prediction: {base_val + sv_i.sum():.4f}',
            fontsize=12, fontweight='bold', color=HEADER_COLOR,
        )
        ax.set_facecolor(BG_COLOR)
        ax.spines[['top', 'right']].set_visible(False)
        fig.patch.set_facecolor(BG_COLOR)
        plt.tight_layout()

        out = os.path.join(SHAP_DIR, f'shap_waterfall_{metric}_{method}.png')
        plt.savefig(out, dpi=160, bbox_inches='tight', facecolor=BG_COLOR)
        plt.close()
    print(f"    ✓  shap/shap_waterfall_{metric}_{{method}}.png  (×5)")

# ── 1d. SHAP Heatmap — mean |SHAP| per feature × metric ─────────────────────
print("\n  Building SHAP importance heatmap...")
importance_matrix = np.zeros((len(FEAT_LABELS), len(METRICS)))
for col_i, metric in enumerate(METRICS):
    sv = shap_values_store[metric]
    importance_matrix[:, col_i] = np.abs(sv).mean(axis=0)

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(BG_COLOR)

im = ax.imshow(importance_matrix, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(len(METRICS)))
ax.set_xticklabels(list(METRICS.values()), fontsize=10, rotation=15, ha='right')
ax.set_yticks(range(len(FEAT_LABELS)))
ax.set_yticklabels(FEAT_LABELS, fontsize=11)

for i in range(len(FEAT_LABELS)):
    for j in range(len(METRICS)):
        val = importance_matrix[i, j]
        ax.text(j, i, f'{val:.4f}', ha='center', va='center',
                fontsize=9, color='black' if val < importance_matrix.max() * 0.6 else 'white',
                fontweight='semibold')

plt.colorbar(im, ax=ax, label='Mean |SHAP value|', shrink=0.8)
ax.set_title('SHAP Global Feature Importance Heatmap\n(mean |SHAP| across all instances)',
             fontsize=13, fontweight='bold', color=HEADER_COLOR, pad=14)
plt.tight_layout()
out = os.path.join(SHAP_DIR, 'shap_heatmap.png')
plt.savefig(out, dpi=160, bbox_inches='tight', facecolor=BG_COLOR)
plt.close()
print(f"  ✓  shap/shap_heatmap.png")

# ── 1e. SHAP Method Contribution — mean SHAP for method_enc per metric ────────
print("  Building SHAP method contribution bar chart...")
fig, axes = plt.subplots(1, len(METRICS), figsize=(20, 5), sharey=False)
fig.patch.set_facecolor(BG_COLOR)
fig.suptitle('SHAP — Method Feature Contribution per Metric',
             fontsize=14, fontweight='bold', color=HEADER_COLOR, y=1.03)

for ax, (metric, metric_label) in zip(axes, METRICS.items()):
    sv   = shap_values_store[metric]
    feat_idx = FEAT_COLS.index('method_enc')

    # Group SHAP values by method (use positional indices into X)
    raw_pos = raw.reset_index(drop=True)
    method_shap = {}
    for m in METHODS:
        indices = raw_pos.index[raw_pos['method'] == m].tolist()
        if indices:
            method_shap[m] = sv[indices, feat_idx].mean()

    methods_list = list(method_shap.keys())
    shap_vals    = [method_shap[m] for m in methods_list]
    colors       = [METHOD_COLORS[m] for m in methods_list]

    bars = ax.bar(
        [m.upper() for m in methods_list], shap_vals,
        color=colors, edgecolor='white', linewidth=0.8,
    )
    ax.axhline(0, color='#666666', linewidth=0.9, linestyle='--')
    ax.set_title(metric_label, fontsize=11, fontweight='semibold', color=HEADER_COLOR)
    ax.set_xlabel('Method', fontsize=9)
    ax.set_ylabel('Mean SHAP (method feature)', fontsize=9)
    ax.set_facecolor(BG_COLOR)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=9)

    for bar, val in zip(bars, shap_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + (0.002 if val >= 0 else -0.005),
                f'{val:+.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
out = os.path.join(SHAP_DIR, 'shap_method_contributions.png')
plt.savefig(out, dpi=160, bbox_inches='tight', facecolor=BG_COLOR)
plt.close()
print(f"  ✓  shap/shap_method_contributions.png\n")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  2.  LIME  —  Local Interpretable Model-agnostic Explanations              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
print("═" * 60)
print("  LIME ANALYSIS")
print("═" * 60)

lime_explainer = lime_tabular.LimeTabularExplainer(
    training_data   = X,
    feature_names   = FEAT_LABELS,
    mode            = 'regression',
    discretize_continuous = True,
    random_state    = 42,
)

# Store: metric → method → dict of {feature: weight}
lime_importance = {m: {method: {} for method in METHODS} for m in METRICS}

for metric, metric_label in METRICS.items():
    rf   = models_store[metric]
    print(f"\n  ▸ LIME explanations for: {metric_label}")

    fig, axes = plt.subplots(1, len(METHODS), figsize=(24, 6))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle(f'LIME Local Explanations — {metric_label}',
                 fontsize=14, fontweight='bold', color=HEADER_COLOR, y=1.03)

    for ax, method in zip(axes, METHODS):
        # Use median param_value for this method as the representative instance
        mask   = raw['method'] == method
        df_m   = raw[mask]
        if df_m.empty:
            ax.axis('off')
            continue

        # Pick the instance closest to the median param_value for this method
        raw_pos2   = raw.reset_index(drop=True)
        pos_mask   = raw_pos2['method'] == method
        pos_indices = raw_pos2.index[pos_mask].tolist()
        param_vals  = raw_pos2.loc[pos_indices, 'param_value']
        med_val     = param_vals.median()
        row_idx     = (param_vals - med_val).abs().idxmin()
        instance    = X[row_idx]

        exp = lime_explainer.explain_instance(
            instance,
            rf.predict,
            num_features=len(FEAT_COLS),
            num_samples=500,
        )

        feat_weights = dict(exp.as_list())
        lime_importance[metric][method] = feat_weights

        # Sort by absolute weight
        items = sorted(feat_weights.items(), key=lambda x: abs(x[1]), reverse=True)
        feats, weights = zip(*items)
        colors = ['#3A86FF' if w > 0 else '#FF6B6B' for w in weights]
        y_pos  = np.arange(len(feats))

        ax.barh(y_pos, weights, color=colors, edgecolor='white',
                linewidth=0.5, height=0.65)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feats, fontsize=8)
        ax.axvline(0, color='#333333', linewidth=1.0)
        ax.set_title(method.upper(), fontsize=11,
                     fontweight='bold', color=METHOD_COLORS[method])
        ax.set_xlabel('LIME Weight', fontsize=9)
        ax.set_facecolor(BG_COLOR)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=8)

        predicted_val = rf.predict([instance])[0]
        ax.annotate(f'Predicted: {predicted_val:.4f}',
                    xy=(0.97, 0.03), xycoords='axes fraction',
                    fontsize=8, ha='right', color=HEADER_COLOR,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

    plt.tight_layout()
    out = os.path.join(LIME_DIR, f'lime_explanation_{metric}.png')
    plt.savefig(out, dpi=160, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close()
    print(f"    ✓  lime/lime_explanation_{metric}.png")

# ── 2b. LIME Summary Heatmap — feature × (method × metric) ─────────────────
print("\n  Building LIME summary heatmap...")

# For each metric, show top-feature weights across methods
for metric, metric_label in METRICS.items():
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor(BG_COLOR)

    # Gather all feature labels seen across methods
    all_feat_keys = set()
    for method in METHODS:
        all_feat_keys.update(lime_importance[metric][method].keys())
    all_feat_keys = sorted(all_feat_keys)

    matrix = np.zeros((len(all_feat_keys), len(METHODS)))
    for j, method in enumerate(METHODS):
        weights = lime_importance[metric][method]
        for i, feat in enumerate(all_feat_keys):
            matrix[i, j] = weights.get(feat, 0.0)

    vmax = np.abs(matrix).max() or 1.0
    im   = ax.imshow(matrix, cmap='RdBu_r', aspect='auto',
                     vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels([m.upper() for m in METHODS], fontsize=11, fontweight='bold')
    ax.set_yticks(range(len(all_feat_keys)))
    ax.set_yticklabels(all_feat_keys, fontsize=8)

    for i in range(len(all_feat_keys)):
        for j in range(len(METHODS)):
            val = matrix[i, j]
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=7.5,
                    color='white' if abs(val) > vmax * 0.55 else 'black')

    plt.colorbar(im, ax=ax, label='LIME Weight', shrink=0.85)
    ax.set_title(f'LIME Feature Importance — {metric_label}\n(feature conditions × method)',
                 fontsize=13, fontweight='bold', color=HEADER_COLOR, pad=12)
    plt.tight_layout()
    out = os.path.join(LIME_DIR, f'lime_heatmap_{metric}.png')
    plt.savefig(out, dpi=160, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close()
    print(f"  ✓  lime/lime_heatmap_{metric}.png")

# ── 2c. LIME vs SHAP Agreement Plot — do both agree on top features? ─────────
print("\n  Building LIME vs SHAP agreement plots...")
fig, axes = plt.subplots(2, len(METRICS), figsize=(22, 10))
fig.patch.set_facecolor(BG_COLOR)
fig.suptitle('LIME vs SHAP — Feature Importance Agreement per Metric',
             fontsize=14, fontweight='bold', color=HEADER_COLOR, y=1.02)

for col_i, (metric, metric_label) in enumerate(METRICS.items()):
    sv          = shap_values_store[metric]
    shap_global = np.abs(sv).mean(axis=0)   # shape: (n_features,)

    # LIME global: average absolute weight across all methods
    lime_global = np.zeros(len(FEAT_COLS))
    for method in METHODS:
        weights = lime_importance[metric][method]
        for feat_label, val in weights.items():
            # Match LIME discretised feature back to original feature
            for fi, fl in enumerate(FEAT_LABELS):
                if fl in feat_label:
                    lime_global[fi] += abs(val)
                    break
    lime_global /= len(METHODS)

    # Normalise both to [0,1]
    def norm(v):
        mx = v.max()
        return v / mx if mx > 0 else v

    shap_n = norm(shap_global)
    lime_n = norm(lime_global)

    # Top bar chart
    ax_top = axes[0, col_i]
    x_pos  = np.arange(len(FEAT_LABELS))
    width  = 0.38
    ax_top.bar(x_pos - width/2, shap_n, width, label='SHAP', color='#3A86FF',
               edgecolor='white', linewidth=0.5)
    ax_top.bar(x_pos + width/2, lime_n, width, label='LIME', color='#FF6B6B',
               edgecolor='white', linewidth=0.5)
    ax_top.set_xticks(x_pos)
    ax_top.set_xticklabels(FEAT_LABELS, rotation=25, ha='right', fontsize=8)
    ax_top.set_title(metric_label, fontsize=11, fontweight='semibold',
                     color=HEADER_COLOR)
    ax_top.set_ylabel('Normalised Importance', fontsize=9)
    ax_top.legend(fontsize=9)
    ax_top.set_facecolor(BG_COLOR)
    ax_top.spines[['top', 'right']].set_visible(False)

    # Correlation scatter
    ax_bot = axes[1, col_i]
    ax_bot.scatter(shap_n, lime_n, color=ACCENT, s=70, edgecolors=HEADER_COLOR,
                   linewidths=0.8, zorder=3)
    for i, label in enumerate(FEAT_LABELS):
        ax_bot.annotate(label, (shap_n[i], lime_n[i]),
                        fontsize=7, xytext=(4, 4),
                        textcoords='offset points', color=HEADER_COLOR)
    # Perfect agreement line
    ax_bot.plot([0, 1], [0, 1], 'k--', linewidth=0.9, alpha=0.4, label='Perfect agreement')
    corr = np.corrcoef(shap_n, lime_n)[0, 1]
    ax_bot.set_title(f'Correlation r = {corr:.3f}', fontsize=10,
                     color=HEADER_COLOR, fontweight='semibold')
    ax_bot.set_xlabel('SHAP Importance', fontsize=9)
    ax_bot.set_ylabel('LIME Importance', fontsize=9)
    ax_bot.set_facecolor(BG_COLOR)
    ax_bot.spines[['top', 'right']].set_visible(False)
    ax_bot.legend(fontsize=8)

plt.tight_layout()
out = os.path.join(XAI_DIR, 'lime_vs_shap_agreement.png')
plt.savefig(out, dpi=160, bbox_inches='tight', facecolor=BG_COLOR)
plt.close()
print(f"  ✓  xai/lime_vs_shap_agreement.png\n")

# ── Summary ──────────────────────────────────────────────────────────────────
print("═" * 60)
print("✅  XAI Analysis Complete!")
print(f"    Output → {XAI_DIR}")
print()
print("  SHAP outputs:")
print("    shap_beeswarm_*.png          — global feature importance (dot)")
print("    shap_dependence_*.png        — how param_value drives each metric")
print("    shap_waterfall_*_{method}.png — per-instance local explanations")
print("    shap_heatmap.png             — features × metrics importance matrix")
print("    shap_method_contributions.png— method SHAP contribution per metric")
print()
print("  LIME outputs:")
print("    lime_explanation_{metric}.png— per-method local bar charts")
print("    lime_heatmap_{metric}.png    — feature × method weight matrix")
print("    lime_vs_shap_agreement.png   — LIME / SHAP importance correlation")
print("═" * 60)
