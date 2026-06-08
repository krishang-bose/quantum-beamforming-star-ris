"""
experiments.py  — STAR-RIS beamforming benchmark (parallelized)

Compares three methods:
    1. qaoa     — QAOA coarse search + classical gradient refinement
    2. qddpg    — Quantum-enhanced DDPG (PQC actor, classical critic)
    3. qppo     — Quantum PPO with STAR-RIS baseline advantage reference

Scenario 1 — vary SNR
Scenario 2 — vary number of STAR-RIS elements N
Scenario 3 — vary car speed
Scenario 4 — vary number of cars K

Uses multiprocessing with a pool initializer that safely redirects stdio,
preventing "Bad file descriptor" crashes when running under nohup on macOS.
"""

import os
import sys
import numpy as np
import pandas as pd
import multiprocessing as mp
from multiprocessing import Pool
from simulator import DEFAULT_PARAMS, init_cars

from methods import (method_ddpg, method_qaoa,
                     method_qddpg, method_qppo)


# All methods: (label, callable)
_METHODS = [
    # ('ddpg',  method_ddpg),  # removed — too slow for batch runs
    ('qaoa',  method_qaoa),
    ('qddpg', method_qddpg),
    ('qppo',  method_qppo),
]

# 4 workers — each handles independent Monte-Carlo trials in parallel
N_WORKERS = 4


# ─────────────────────────────────────────────────────────────────────────────
# Worker pool initializer — called once per worker process at startup.
# Redirects stdin/stdout/stderr to /dev/null so that workers spawned
# under nohup don't crash with "Bad file descriptor" on stdio access.
# ─────────────────────────────────────────────────────────────────────────────
def _worker_init():
    try:
        devnull_fd = os.open(os.devnull, os.O_RDWR)
        for fd in (0, 1, 2):       # stdin, stdout, stderr
            try:
                os.dup2(devnull_fd, fd)
            except Exception:
                pass
        os.close(devnull_fd)
        sys.stdin  = open(os.devnull, 'r')
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
    except Exception:
        pass


def _run_one_trial(p):
    """
    Create fresh cars for every method (same initial snapshot),
    run all methods, return list of result dicts in _METHODS order.
    """
    ref_cars = init_cars(p)
    results = []
    for label, fn in _METHODS:
        cars = init_cars(p)
        for cr, ce in zip(cars, ref_cars):
            cr.pos[:] = ce.pos
            cr.vel[:] = ce.vel
        results.append(fn(cars, p))
    return results


def _run_one_trial_wrapper(args):
    """Top-level picklable wrapper for multiprocessing."""
    p, trial_idx = args
    try:
        return _run_one_trial(p)
    except Exception:
        return None


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
# Generic scenario runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_scenario(sweep_key, sweep_values, set_params_fn,
                  n_trials, base_params, label_key):
    """
    Generic scenario runner with parallel trial execution.
    set_params_fn(p, val) mutates param dict p for each sweep value.
    label_key is the column name for the sweep variable in the output df.
    """
    p0   = (base_params or DEFAULT_PARAMS).copy()
    rows = []
    for val in sweep_values:
        p = p0.copy()
        set_params_fn(p, val)
        print(f"  {label_key}={val}  [{n_trials} trials × {N_WORKERS} workers]",
              flush=True)

        args = [(p.copy(), i) for i in range(n_trials)]

        with Pool(processes=N_WORKERS, initializer=_worker_init) as pool:
            all_results = pool.map(_run_one_trial_wrapper, args)

        all_results = [r for r in all_results if r is not None]
        n_ok = len(all_results)
        if n_ok < n_trials:
            print(f"    WARNING: {n_trials - n_ok} trials failed", flush=True)
        print(f"    done ({n_ok}/{n_trials} trials)", flush=True)

        per_method = {name: [] for name, _ in _METHODS}
        for trial_results in all_results:
            if trial_results is None:
                continue
            for (name, _), res in zip(_METHODS, trial_results):
                per_method[name].append(res)

        for name, recs in per_method.items():
            if recs:
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
        print(f"  K={K}  [{n_trials} trials × {N_WORKERS} workers]",
              flush=True)

        args = [(p.copy(), i) for i in range(n_trials)]

        with Pool(processes=N_WORKERS, initializer=_worker_init) as pool:
            all_results = pool.map(_run_one_trial_wrapper, args)

        all_results = [r for r in all_results if r is not None]
        n_ok = len(all_results)
        if n_ok < n_trials:
            print(f"    WARNING: {n_trials - n_ok} trials failed", flush=True)
        print(f"    done ({n_ok}/{n_trials} trials)", flush=True)

        per_method = {name: [] for name, _ in _METHODS}
        for trial_results in all_results:
            if trial_results is None:
                continue
            for (name, _), res in zip(_METHODS, trial_results):
                per_method[name].append(res)

        for name, recs in per_method.items():
            if recs:
                agg = _aggregate(recs)
                agg.update(K=K, method=name)
                rows.append(agg)

    return pd.DataFrame(rows)
