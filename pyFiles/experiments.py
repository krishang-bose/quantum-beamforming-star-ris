"""
experiments.py  — STAR-RIS beamforming benchmark

Compares five methods:
    1. ddpg     — Deep Deterministic Policy Gradient
    2. qaoa     — QAOA coarse search + classical gradient refinement
    3. qddpg    — Quantum-enhanced DDPG (PQC actor, classical critic)
    4. qppo     — Quantum PPO with STAR-RIS baseline advantage reference
    5. baseline — STAR-RIS random phases + MRT (no learning)

Scenario 1 — vary SNR
Scenario 2 — vary number of STAR-RIS elements N
Scenario 3 — vary car speed
Scenario 4 — vary number of cars K

Note: The "SNR" parameter controls the noise variance as
sigma2 = sigma2_ref / (10^(snr_db/10)), where sigma2_ref is the
default noise floor from DEFAULT_PARAMS. Higher snr_db = lower noise.
"""

import numpy as np
import pandas as pd
from simulator import DEFAULT_PARAMS, init_cars
from methods import (method_ddpg, method_qaoa,
                     method_qddpg, method_qppo,
                     method_star_ris_baseline)


# All methods: (label, callable)
_METHODS = [
    ('ddpg',     method_ddpg),
    ('qaoa',     method_qaoa),
    ('qddpg',    method_qddpg),
    ('qppo',     method_qppo),
    ('baseline', method_star_ris_baseline),
]


def _run_one_trial(p):
    """
    Create fresh cars for every method (same initial snapshot),
    run all 5 methods, return list of result dicts in _METHODS order.
    """
    # Reference car set for copying initial positions
    ref_cars = init_cars(p)
    results = []
    for label, fn in _METHODS:
        cars = init_cars(p)
        for cr, ce in zip(cars, ref_cars):
            cr.pos[:] = ce.pos
            cr.vel[:] = ce.vel
        results.append(fn(cars, p))
    return results   # list of 5 dicts


def _aggregate(records):
    """Aggregate a list of result dicts into mean ± std."""
    keys = [
        'sum_rate', 'iterations', 'energy_norm', 'time_s',
        'circuit_evals', 'peak_mem_mb', 'conv_std', 'conv_slope',
    ]
    out = {}
    for k in keys:
        vals = [r[k] for r in records]
        out[k + '_mean'] = float(np.mean(vals))
        out[k + '_std']  = float(np.std(vals))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Scenario helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run_scenario(sweep_key, sweep_values, set_params_fn,
                  n_trials, base_params, label_key):
    """
    Generic scenario runner.
    set_params_fn(p, val) mutates param dict p for each sweep value.
    label_key is the column name for the sweep variable in the output df.
    """
    p0   = (base_params or DEFAULT_PARAMS).copy()
    rows = []
    for val in sweep_values:
        p = p0.copy()
        set_params_fn(p, val)
        print(f"  {label_key}={val}", end='', flush=True)

        # Collect per-method lists across trials
        per_method = {name: [] for name, _ in _METHODS}
        for _ in range(n_trials):
            trial_results = _run_one_trial(p)
            for (name, _), res in zip(_METHODS, trial_results):
                per_method[name].append(res)
            print('.', end='', flush=True)
        print()

        for name, recs in per_method.items():
            agg = _aggregate(recs)
            agg.update(method=name, **{label_key: val})
            rows.append(agg)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Public scenario functions
# ─────────────────────────────────────────────────────────────────────────────

def scenario_snr(snr_db_range=range(0, 31, 5), n_trials=20, base_params=None):
    """
    Sweep SNR from low to high.
    sigma2 = sigma2_ref / SNR_linear, where sigma2_ref is the default.
    """
    p0 = (base_params or DEFAULT_PARAMS).copy()
    sigma2_ref = p0['sigma2']

    def _set(p, snr_db):
        p['sigma2'] = sigma2_ref / (10 ** (snr_db / 10))

    return _run_scenario('snr_db', list(snr_db_range), _set,
                         n_trials, base_params, 'snr_db')


def scenario_N(N_values=(16, 32, 64), n_trials=20, base_params=None):
    """Sweep number of STAR-RIS elements (fixed SNR = default)."""
    return _run_scenario('N', list(N_values),
                         lambda p, v: p.update({'N': v}),
                         n_trials, base_params, 'N')


def scenario_speed(speed_values=(5, 10, 20, 30), n_trials=20,
                   base_params=None):
    """
    Sweep car speed (v_min = v_max = v so all cars go same speed).
    Shows how fast channel variation affects each method.
    """
    def _set(p, v):
        p['v_min'] = v; p['v_max'] = v

    return _run_scenario('speed_ms', list(speed_values), _set,
                         n_trials, base_params, 'speed_ms')


def scenario_K(K_values=((2, 2), (4, 4), (8, 8)), n_trials=20,
               base_params=None):
    """Sweep number of cars (K = Kr + Kt)."""
    p0   = (base_params or DEFAULT_PARAMS).copy()
    rows = []
    for Kr, Kt in K_values:
        p = p0.copy(); p['Kr'] = Kr; p['Kt'] = Kt
        K = Kr + Kt
        print(f"  K={K}", end='', flush=True)

        per_method = {name: [] for name, _ in _METHODS}
        for _ in range(n_trials):
            trial_results = _run_one_trial(p)
            for (name, _), res in zip(_METHODS, trial_results):
                per_method[name].append(res)
            print('.', end='', flush=True)
        print()

        for name, recs in per_method.items():
            agg = _aggregate(recs)
            agg.update(K=K, method=name)
            rows.append(agg)
    return pd.DataFrame(rows)
