"""
methods.py  (DDPG vs QAOA comparison)

Two methods for STAR-RIS beamforming comparison:

    1. DDPG   — Deep Deterministic Policy Gradient (deep RL)
    2. QAOA   — QAOA coarse search + classical gradient refinement (hybrid)

Each method runs over T_horizon time slots.
Cars move between slots — channels are regenerated each slot.
"""

import time
import numpy as np
from scipy.optimize import minimize

from simulator import (
    generate_channels_at_slot, effective_channels,
    compute_sum_rate, init_beamformers, project_power,
    phase_choices_to_coeffs, PHASE_LEVELS_2BIT,
)
from hamiltonian import (
    build_pauli_hamiltonian, expectation_value, analytic_gradient,
)
from ddpg import DDPGAgent

from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector
import scipy.linalg as la


# ═════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═════════════════════════════════════════════════════════════════════════════

def _extract_state(H_BR, H_r, H_t, prev_rate):
    """
    Build a flat state vector from the current channel snapshot.
    Contains: |H_BR| flattened, |H_r| flattened, |H_t| flattened,
              previous sum-rate (scalar).
    """
    features = np.concatenate([
        np.abs(H_BR).ravel(),
        np.abs(H_r).ravel(),
        np.abs(H_t).ravel(),
        [prev_rate],
    ])
    return features


def _mrt_beamformer(h_eff, P_max):
    """
    Maximum-Ratio Transmission beamformer from effective channels.
    W_k = h_eff[k]^H / ||h_eff[k]||, then scale to P_max.
    """
    K, Nt = h_eff.shape
    W = h_eff.conj().T  # (Nt, K)
    norm = np.sqrt(np.sum(np.abs(W) ** 2))
    if norm > 0:
        W = W / norm * np.sqrt(P_max)
    return W


def _gradient_update_slot(H_BR, H_r, H_t, p, W, phases,
                           lr_w=0.05, lr_phi=0.03, n_iter=10):
    """
    Run n_iter gradient steps on (W, phases) for a single channel slot.
    Returns updated (W, phases, R_final, history).
    Warm-starts from supplied W and phases so it tracks across slots.
    """
    N, Nt = p['N'], p['Nt']
    K     = p['Kr'] + p['Kt']
    eps   = 1e-4
    hist  = []

    for _ in range(n_iter):
        beta  = np.sqrt(0.5) * np.exp(1j * phases)
        h     = effective_channels(H_BR, H_r, H_t, beta, beta)
        R     = compute_sum_rate(h, W, p['sigma2'])
        hist.append(R)

        # gradient on W (finite difference)
        gW = np.zeros_like(W)
        for i in range(Nt):
            for j in range(K):
                for part in (1, 1j):
                    W[i, j] += eps * part
                    Rp = compute_sum_rate(
                        effective_channels(H_BR, H_r, H_t, beta, beta),
                        W, p['sigma2'])
                    W[i, j] -= eps * part
                    if part == 1:
                        gW[i, j] += (Rp - R) / eps
                    else:
                        gW[i, j] += 1j * (Rp - R) / eps
        W = project_power(W + lr_w * gW, p['P_max'])

        # gradient on phases (finite difference)
        gp = np.zeros(N)
        for n in range(N):
            phases[n] += eps
            b2 = np.sqrt(0.5) * np.exp(1j * phases)
            Rp = compute_sum_rate(
                effective_channels(H_BR, H_r, H_t, b2, b2), W, p['sigma2'])
            phases[n] -= eps
            gp[n] = (Rp - R) / eps
        phases = (phases + lr_phi * gp) % (2 * np.pi)

    beta  = np.sqrt(0.5) * np.exp(1j * phases)
    h     = effective_channels(H_BR, H_r, H_t, beta, beta)
    R_fin = compute_sum_rate(h, W, p['sigma2'])
    return W, phases, R_fin, hist


# ═════════════════════════════════════════════════════════════════════════════
# METHOD 1 — DDPG (Deep Deterministic Policy Gradient)
# ═════════════════════════════════════════════════════════════════════════════

def method_ddpg(cars, p, n_pretrain_episodes=5, train_steps_per_slot=4):
    """
    DDPG-based STAR-RIS beamforming.

    The agent learns a mapping from channel state to RIS phase
    configuration. The beamformer is derived via MRT from the
    resulting effective channel (closed-form, no learning needed).

    Training schedule:
      1. Pre-training: run n_pretrain_episodes of T_horizon slots to
         populate the replay buffer and learn an initial policy.
      2. Evaluation: run one final episode of T_horizon slots using
         the learned policy (no exploration noise).

    Returns per-slot metrics from the evaluation episode.
    """
    t0 = time.time()
    N, Nt = p['N'], p['Nt']
    K = p['Kr'] + p['Kt']
    T = p['T_horizon']

    # state dim = |H_BR| + |H_r| + |H_t| + prev_rate
    state_dim = N * Nt + p['Kr'] * N + p['Kt'] * N + 1
    action_dim = N  # continuous phase angles

    agent = DDPGAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden=(128, 64),
        lr_actor=5e-4,
        lr_critic=1e-3,
        gamma=0.95,
        tau=0.01,
        buffer_size=5000,
        batch_size=min(64, max(16, T * 2)),
        noise_sigma=0.3,
    )

    total_train_iters = 0

    # ── Pre-training episodes ────────────────────────────────────────────
    for ep in range(n_pretrain_episodes):
        # create fresh cars for each training episode
        from simulator import init_cars as _init_cars
        train_cars = _init_cars(p)
        prev_rate = 0.0
        agent.reset_noise()

        for t in range(T):
            for car in train_cars:
                car.step(p['T_slot'])

            H_BR, H_r, H_t = generate_channels_at_slot(train_cars, p)
            state = _extract_state(H_BR, H_r, H_t, prev_rate)

            # agent selects phases
            phases = agent.select_action(state, explore=True)

            # build RIS coefficients and compute effective channel
            beta = np.sqrt(0.5) * np.exp(1j * phases)
            h_eff = effective_channels(H_BR, H_r, H_t, beta, beta)
            W = _mrt_beamformer(h_eff, p['P_max'])
            rate = compute_sum_rate(h_eff, W, p['sigma2'])

            # next state (for transition storage)
            done = (t == T - 1)
            if not done:
                # peek at next slot (we store transition now)
                next_H_BR, next_H_r, next_H_t = H_BR, H_r, H_t
                next_state = _extract_state(next_H_BR, next_H_r, next_H_t,
                                            rate)
            else:
                next_state = state  # terminal

            agent.store(state, phases, rate, next_state, float(done))

            # train on mini-batches
            for _ in range(train_steps_per_slot):
                agent.train()
                total_train_iters += 1

            prev_rate = rate

    # ── Evaluation episode (no exploration noise) ────────────────────────
    sr_ts = []
    prev_rate = 0.0

    for t in range(T):
        for car in cars:
            car.step(p['T_slot'])

        H_BR, H_r, H_t = generate_channels_at_slot(cars, p)
        state = _extract_state(H_BR, H_r, H_t, prev_rate)

        # deterministic action (no noise)
        phases = agent.select_action(state, explore=False)

        beta = np.sqrt(0.5) * np.exp(1j * phases)
        h_eff = effective_channels(H_BR, H_r, H_t, beta, beta)
        W = _mrt_beamformer(h_eff, p['P_max'])
        rate = compute_sum_rate(h_eff, W, p['sigma2'])

        sr_ts.append(rate)
        prev_rate = rate

    # energy: count training iterations + evaluation forward passes
    c_train = 1.0   # cost per training step
    c_eval  = 0.1   # cost per eval forward pass
    energy = c_train * total_train_iters + c_eval * T

    return dict(
        sum_rate    = float(np.mean(sr_ts)),
        sum_rate_ts = sr_ts,
        iterations  = total_train_iters + T,
        energy_norm = energy,
        time_s      = time.time() - t0,
    )


# ═════════════════════════════════════════════════════════════════════════════
# QAOA sub-routine (used inside method_qaoa)
# ═════════════════════════════════════════════════════════════════════════════

def _qaoa_one_slot(H_BR, H_r, H_t, p,
                   gamma_init, beta_init,
                   n_shots=100, max_opt_iter=15):
    """
    Run QAOA variational optimisation for one channel slot.
    Warm-starts from (gamma_init, beta_init) passed in from previous slot.
    Returns (best_choices, best_W, best_R, gamma_opt, beta_opt, iters)
    """
    N, Nt, K = p['N'], p['Nt'], p['Kr'] + p['Kt']

    # build Hamiltonian coefficients from current channel snapshot
    beta_r = phase_choices_to_coeffs(np.zeros(N, int))
    h_ref  = effective_channels(H_BR, H_r, H_t, beta_r, beta_r)
    W_ref  = init_beamformers(Nt, K, p['P_max'])
    hn     = np.mean(np.abs(h_ref) ** 2, axis=1)
    a = float(np.sum(hn[:K // 2]))
    b = float(np.sum(hn[K // 2:]))
    c = float(np.mean(np.abs(h_ref)) ** 2 * N)

    iters_used = [0]

    def obj(params):
        iters_used[0] += 1
        return expectation_value(params[0], params[1], a, b, c)

    def jac(params):
        dg, db = analytic_gradient(params[0], params[1], a, b, c)
        return np.array([dg, db])

    res = minimize(obj, [gamma_init, beta_init],
                   jac=jac, method='L-BFGS-B',
                   options={'maxiter': max_opt_iter})

    gamma_opt, beta_opt = res.x

    # build and sample QAOA circuit
    H_op = build_pauli_hamiltonian(a, b, c)
    pauli_list = [(str(t.paulis[0]), float(np.real(t.coeffs[0])))
                  for t in H_op]

    qc = QuantumCircuit(2)
    qc.h([0, 1])
    for label, coeff in pauli_list:
        if abs(coeff) < 1e-12:
            continue
        active = [(2 - 1 - i, pp)
                  for i, pp in enumerate(label) if pp != 'I']
        if len(active) == 1:
            q, typ = active[0]
            if typ == 'Z':
                qc.rz(2 * gamma_opt * coeff, q)
        elif len(active) == 2:
            (q0, _), (q1, _) = active
            qc.cx(q0, q1)
            qc.rz(2 * gamma_opt * coeff, q1)
            qc.cx(q0, q1)
    for q in range(2):
        qc.rx(2 * beta_opt, q)
    qc.measure_all()

    sampler = StatevectorSampler()
    counts  = sampler.run([qc], shots=n_shots).result()[0] \
                     .data.meas.get_counts()

    best_R, best_choices, best_W = -np.inf, np.zeros(N, int), W_ref
    for bstr, _ in sorted(counts.items(), key=lambda x: -x[1]):
        x0 = int(bstr[-1]); x1 = int(bstr[-2])
        ch = np.zeros(N, int)
        ch[:N // 2] = x0 * 2; ch[N // 2:] = x1 * 2
        bc = phase_choices_to_coeffs(ch)
        W_c = init_beamformers(Nt, K, p['P_max'])
        R   = compute_sum_rate(
            effective_channels(H_BR, H_r, H_t, bc, bc),
            W_c, p['sigma2'])
        if R > best_R:
            best_R, best_choices, best_W = R, ch, W_c

    return best_choices, best_W, best_R, gamma_opt, beta_opt, iters_used[0]


# ═════════════════════════════════════════════════════════════════════════════
# METHOD 2 — QAOA Hybrid (QAOA coarse + classical gradient refinement)
# ═════════════════════════════════════════════════════════════════════════════

def method_qaoa(cars, p, n_shots=80, max_opt_iter=10, refine_iter=10):
    """
    Hybrid quantum-classical beamforming:
      Step 1: QAOA coarse search for phase configuration
      Step 2: Classical gradient refinement starting from QAOA solution

    This combines the global exploration capability of QAOA with the
    local optimisation strength of gradient descent.
    """
    t0     = time.time()
    T      = p['T_horizon']
    N, Nt  = p['N'], p['Nt']
    K      = p['Kr'] + p['Kt']
    c1, c2 = 1.0, 1.5   # energy cost coefficients
    sr_ts, iter_ts_q, iter_ts_c = [], [], []

    gamma, beta_q = np.random.uniform(0, np.pi), np.random.uniform(0, np.pi)
    W = init_beamformers(Nt, K, p['P_max'])

    for t in range(T):
        for car in cars:
            car.step(p['T_slot'])
        H_BR, H_r, H_t = generate_channels_at_slot(cars, p)

        # step 1: QAOA coarse search
        choices, W_q, R_q, gamma, beta_q, it_q = _qaoa_one_slot(
            H_BR, H_r, H_t, p,
            gamma_init=gamma, beta_init=beta_q,
            n_shots=n_shots, max_opt_iter=max_opt_iter)

        # step 2: classical refinement (warm-start W from QAOA)
        phases = PHASE_LEVELS_2BIT[choices].copy().astype(float)
        W      = W_q.copy()
        W, phases, R_fin, hist = _gradient_update_slot(
            H_BR, H_r, H_t, p, W, phases,
            n_iter=refine_iter)

        sr_ts.append(R_fin)
        iter_ts_q.append(it_q)
        iter_ts_c.append(len(hist))

    iq = sum(iter_ts_q)
    ic = sum(iter_ts_c)
    return dict(
        sum_rate     = float(np.mean(sr_ts)),
        sum_rate_ts  = sr_ts,
        iterations   = iq + ic,
        energy_norm  = c2 * iq + c1 * ic,
        time_s       = time.time() - t0,
    )
