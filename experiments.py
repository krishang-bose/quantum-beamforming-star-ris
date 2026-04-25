"""
experiments.py  (DDPG vs QAOA comparison)

Compares DDPG (deep RL) vs QAOA Hybrid (QAOA + gradient refinement).

Scenario 1 — vary SNR
Scenario 2 — vary number of STAR-RIS elements N
Scenario 3 — vary car speed
Scenario 4 — vary number of cars K

Note: The "SNR" parameter here controls the noise variance as
sigma2 = sigma2_ref / (10^(snr_db/10)), where sigma2_ref is the
default noise floor from DEFAULT_PARAMS. This means snr_db=0
corresponds to the default noise level, and higher snr_db
reduces noise proportionally.
"""

import numpy as np
import pandas as pd
from simulator import DEFAULT_PARAMS, init_cars
from methods import method_ddpg, method_qaoa


def _run_one_trial(p):
    """
    Create fresh cars, run both methods, return results.
    Cars are re-created per trial so mobility is independent.
    Both methods start from the same initial car positions/velocities.
    """
    cars1 = init_cars(p)
    cars2 = init_cars(p)

    # copy initial positions so both methods start from same snapshot
    for ca, cb in zip(cars1, cars2):
        cb.pos = ca.pos.copy()
        cb.vel = ca.vel.copy()

    r1 = method_ddpg(cars1, p)
    r2 = method_qaoa(cars2, p)
    return r1, r2


def _aggregate(records):
    """Aggregate a list of result dicts into mean ± std."""
    keys = ['sum_rate', 'iterations', 'energy_norm', 'time_s']
    out = {}
    for k in keys:
        vals = [r[k] for r in records]
        out[k + '_mean'] = float(np.mean(vals))
        out[k + '_std']  = float(np.std(vals))
    return out


def scenario_snr(snr_db_range=range(0, 31, 5), n_trials=20,
                 base_params=None):
    """
    Sweep SNR from low to high.
    sigma2 = sigma2_ref / SNR_linear, where sigma2_ref is the default.
    """
    p0 = (base_params or DEFAULT_PARAMS).copy()
    sigma2_ref = p0['sigma2']  # baseline noise
    rows = []
    for snr_db in snr_db_range:
        p = p0.copy()
        # Higher snr_db → lower noise
        p['sigma2'] = sigma2_ref / (10 ** (snr_db / 10))
        print(f"  SNR={snr_db} dB (sigma2={p['sigma2']:.2e})",
              end='', flush=True)
        r1s, r2s = [], []
        for _ in range(n_trials):
            r1, r2 = _run_one_trial(p)
            r1s.append(r1); r2s.append(r2)
            print('.', end='', flush=True)
        print()
        for name, recs in [('ddpg', r1s), ('qaoa', r2s)]:
            agg = _aggregate(recs)
            agg.update(snr_db=snr_db, method=name)
            rows.append(agg)
    return pd.DataFrame(rows)


def scenario_N(N_values=(16, 32, 64), n_trials=20, base_params=None):
    """Sweep number of STAR-RIS elements (fixed SNR = default)."""
    p0 = (base_params or DEFAULT_PARAMS).copy()
    rows = []
    for N in N_values:
        p = p0.copy(); p['N'] = N
        print(f"  N={N}", end='', flush=True)
        r1s, r2s = [], []
        for _ in range(n_trials):
            r1, r2 = _run_one_trial(p)
            r1s.append(r1); r2s.append(r2)
            print('.', end='', flush=True)
        print()
        for name, recs in [('ddpg', r1s), ('qaoa', r2s)]:
            agg = _aggregate(recs)
            agg.update(N=N, method=name)
            rows.append(agg)
    return pd.DataFrame(rows)


def scenario_speed(speed_values=(5, 10, 20, 30), n_trials=20,
                   base_params=None):
    """
    Sweep car speed (v_min = v_max = v so all cars go same speed).
    Shows how fast channel variation affects each method.
    """
    p0 = (base_params or DEFAULT_PARAMS).copy()
    rows = []
    for v in speed_values:
        p = p0.copy(); p['v_min'] = v; p['v_max'] = v
        print(f"  speed={v} m/s", end='', flush=True)
        r1s, r2s = [], []
        for _ in range(n_trials):
            r1, r2 = _run_one_trial(p)
            r1s.append(r1); r2s.append(r2)
            print('.', end='', flush=True)
        print()
        for name, recs in [('ddpg', r1s), ('qaoa', r2s)]:
            agg = _aggregate(recs)
            agg.update(speed_ms=v, method=name)
            rows.append(agg)
    return pd.DataFrame(rows)


def scenario_K(K_values=((2, 2), (4, 4), (8, 8)), n_trials=20,
               base_params=None):
    """Sweep number of cars (K = Kr + Kt)."""
    p0 = (base_params or DEFAULT_PARAMS).copy()
    rows = []
    for Kr, Kt in K_values:
        p = p0.copy(); p['Kr'] = Kr; p['Kt'] = Kt
        print(f"  K={Kr + Kt}", end='', flush=True)
        r1s, r2s = [], []
        for _ in range(n_trials):
            r1, r2 = _run_one_trial(p)
            r1s.append(r1); r2s.append(r2)
            print('.', end='', flush=True)
        print()
        for name, recs in [('ddpg', r1s), ('qaoa', r2s)]:
            agg = _aggregate(recs)
            agg.update(K=Kr + Kt, method=name)
            rows.append(agg)
    return pd.DataFrame(rows)
