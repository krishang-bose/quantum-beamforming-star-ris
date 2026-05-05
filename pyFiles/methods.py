"""
methods.py  — STAR-RIS beamforming method library

Five methods compared:
    1. DDPG          — Deep Deterministic Policy Gradient
    2. QAOA          — QAOA coarse search + classical gradient refinement
    3. QDDPG         — Quantum-enhanced DDPG (PQC actor, classical critic)
    4. QPPO          — Quantum Proximal Policy Optimisation (PQC policy)
    5. STAR-RIS Baseline — Random phases + MRT (no learning)

Each method runs over T_horizon time slots.
Cars move between slots — channels are regenerated each slot.
"""

import time
import tracemalloc
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
    tracemalloc.start()
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

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    m = _method_metrics(sr_ts)
    return dict(
        sum_rate     = float(np.mean(sr_ts)),
        sum_rate_ts  = sr_ts,
        iterations   = total_train_iters + T,
        energy_norm  = energy,
        time_s       = time.time() - t0,
        circuit_evals= 0,
        peak_mem_mb  = peak / 1024 / 1024,
        conv_std     = m['conv_std'],
        conv_slope   = m['conv_slope'],
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
    tracemalloc.start()
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
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # QAOA circuit evals: each _qaoa_one_slot runs L-BFGS iters (obj calls)
    # plus one final circuit sample — approximate as iq optimizer calls
    m = _method_metrics(sr_ts)
    return dict(
        sum_rate      = float(np.mean(sr_ts)),
        sum_rate_ts   = sr_ts,
        iterations    = iq + ic,
        energy_norm   = c2 * iq + c1 * ic,
        time_s        = time.time() - t0,
        circuit_evals = iq,          # one Qiskit circuit per QAOA slot
        peak_mem_mb   = peak / 1024 / 1024,
        conv_std      = m['conv_std'],
        conv_slope    = m['conv_slope'],
    )


# ═════════════════════════════════════════════════════════════════════════════
# Parameterised Quantum Circuit (PQC) — numpy statevector simulator
# Used by both QDDPG and QPPO as the policy / actor network.
# ═════════════════════════════════════════════════════════════════════════════

def _ry_mat(theta):
    """2×2 Ry rotation matrix."""
    c, s = float(np.cos(theta / 2)), float(np.sin(theta / 2))
    return np.array([[c, -s], [s, c]], dtype=complex)


def _apply_sq(sv, gate, q, n):
    """Apply 2×2 gate to qubit q of n-qubit state vector."""
    sv = sv.reshape([2] * n)
    # tensordot(gate, sv, axes=([1],[q])) contracts axis 1 of gate with axis q
    # of sv and places the new axis at position 0 of the result.
    sv = np.tensordot(gate, sv, axes=([1], [q]))
    # Move the new axis (0) back to position q.
    sv = np.moveaxis(sv, 0, q)
    return sv.reshape(-1)


def _apply_cnot(sv, ctrl, tgt, n):
    """CNOT(ctrl→tgt) on n-qubit state vector."""
    sv = sv.reshape([2] * n)
    sv_new = sv.copy()
    idx = [slice(None)] * n
    idx[ctrl] = 1
    slc = sv_new[tuple(idx)]
    tgt_adj = tgt if tgt < ctrl else tgt - 1
    sv_new[tuple(idx)] = np.flip(slc, axis=tgt_adj)
    return sv_new.reshape(-1)


def _z_exp(sv, q, n):
    """⟨Z_q⟩ = P(0) - P(1) for qubit q."""
    sv = sv.reshape([2] * n)
    p0 = float(np.sum(np.abs(sv.take(0, axis=q)) ** 2))
    return 2.0 * p0 - 1.0


# Module-level counter: incremented every time _run_pqc is called.
# Reset by each quantum method before its run; read at the end.
_PQC_EVALS = 0


def _reset_pqc_counter():
    global _PQC_EVALS
    _PQC_EVALS = 0


def _run_pqc(enc_angles, theta, n_qubits, n_layers):
    """
    Parameterised quantum circuit.
    Architecture per layer: Ry(theta[l,q]) on each qubit, then ring CNOT.
    Angle encoding: Ry(enc_angles[q]) applied after H-layer.
    Returns Z-expectation values: shape (n_qubits,).
    """
    global _PQC_EVALS
    _PQC_EVALS += 1
    n = n_qubits
    sv = np.zeros(2 ** n, dtype=complex)
    sv[0] = 1.0
    H_gate = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    for q in range(n):
        sv = _apply_sq(sv, H_gate, q, n)
    for q in range(n):
        sv = _apply_sq(sv, _ry_mat(float(enc_angles[q])), q, n)
    for l in range(n_layers):
        for q in range(n):
            sv = _apply_sq(sv, _ry_mat(float(theta[l, q])), q, n)
        for q in range(n):
            sv = _apply_cnot(sv, q, (q + 1) % n, n)
    return np.array([_z_exp(sv, q, n) for q in range(n)])


def _method_metrics(sr_ts):
    """
    Compute convergence & stability metrics from a per-slot rate trajectory.

    Returns dict with:
      conv_std   — std of sum_rate_ts (lower = more stable / converged)
      conv_slope — linear-regression slope over slots (bits/s/Hz per slot);
                   near-zero means converged, positive means still improving
    """
    ts = np.array(sr_ts, dtype=float)
    T  = len(ts)
    conv_std = float(np.std(ts))
    if T > 1:
        x = np.arange(T, dtype=float)
        slope = float(np.polyfit(x, ts, 1)[0])
    else:
        slope = 0.0
    return dict(conv_std=conv_std, conv_slope=slope)


def _pqc_grad_theta(enc_angles, theta, n_qubits, n_layers, d_q_out):
    """Parameter-shift gradient of Z-expectations w.r.t. theta."""
    shift = np.pi / 2
    d_theta = np.zeros_like(theta)
    for l in range(n_layers):
        for q in range(n_qubits):
            t_p = theta.copy(); t_p[l, q] += shift
            t_m = theta.copy(); t_m[l, q] -= shift
            dq = (_run_pqc(enc_angles, t_p, n_qubits, n_layers)
                  - _run_pqc(enc_angles, t_m, n_qubits, n_layers)) / 2.0
            d_theta[l, q] = float(np.dot(d_q_out, dq))
    return d_theta


def _pqc_grad_enc(enc_angles, theta, n_qubits, n_layers, d_q_out):
    """Parameter-shift gradient of Z-expectations w.r.t. encoding angles."""
    shift = np.pi / 2
    d_enc = np.zeros(n_qubits)
    for q in range(n_qubits):
        e_p = enc_angles.copy(); e_p[q] += shift
        e_m = enc_angles.copy(); e_m[q] -= shift
        dq = (_run_pqc(e_p, theta, n_qubits, n_layers)
              - _run_pqc(e_m, theta, n_qubits, n_layers)) / 2.0
        d_enc[q] = float(np.dot(d_q_out, dq))
    return d_enc


class _QuantumActor:
    """
    Hybrid PQC actor: state → (linear encoder) → PQC → (linear decoder) → tanh.
    Trained with Adam; quantum params use parameter-shift rule.
    """

    def __init__(self, state_dim, action_dim, n_qubits=4, n_layers=2, lr=5e-4):
        self.nq, self.nl = n_qubits, n_layers
        self.action_dim = action_dim
        self.lr = lr
        self.t = 0

        sc1 = np.sqrt(2.0 / state_dim)
        self.W_enc = np.random.randn(state_dim, n_qubits) * sc1
        self.b_enc = np.zeros(n_qubits)
        self.theta  = np.random.uniform(0, 2 * np.pi, (n_layers, n_qubits))

        sc2 = np.sqrt(2.0 / n_qubits)
        self.W_dec = np.random.randn(n_qubits, action_dim) * sc2
        self.b_dec = np.zeros(action_dim)

        # Adam moments
        self._m = {k: np.zeros_like(v) for k, v in self._p().items()}
        self._v = {k: np.zeros_like(v) for k, v in self._p().items()}

    def _p(self):
        return dict(W_enc=self.W_enc, b_enc=self.b_enc,
                    theta=self.theta, W_dec=self.W_dec, b_dec=self.b_dec)

    def forward(self, state):
        enc       = state @ self.W_enc + self.b_enc
        enc_a     = np.pi * np.tanh(enc)
        q_out     = _run_pqc(enc_a, self.theta, self.nq, self.nl)
        out       = q_out @ self.W_dec + self.b_dec
        return np.tanh(out)  # (action_dim,) in [-1, 1]

    def _forward_cache(self, state):
        enc       = state @ self.W_enc + self.b_enc
        enc_a     = np.pi * np.tanh(enc)
        q_out     = _run_pqc(enc_a, self.theta, self.nq, self.nl)
        out       = q_out @ self.W_dec + self.b_dec
        action    = np.tanh(out)
        return enc, enc_a, q_out, out, action

    def gradient(self, state, d_action):
        """Return param gradients given upstream d_action (action_dim,)."""
        enc, enc_a, q_out, out, action = self._forward_cache(state)

        d_out   = d_action * (1 - action ** 2)          # through tanh
        dW_dec  = np.outer(q_out, d_out)
        db_dec  = d_out.copy()

        d_q_out = self.W_dec @ d_out                    # (n_qubits,)

        d_theta = _pqc_grad_theta(enc_a, self.theta, self.nq, self.nl, d_q_out)
        d_enc_a = _pqc_grad_enc(enc_a, self.theta, self.nq, self.nl, d_q_out)

        d_enc   = d_enc_a * np.pi * (1 - np.tanh(enc) ** 2)
        dW_enc  = np.outer(state, d_enc)
        db_enc  = d_enc.copy()

        return dict(W_enc=dW_enc, b_enc=db_enc,
                    theta=d_theta, W_dec=dW_dec, b_dec=db_dec)

    def _adam_step(self, grads, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        p = self._p()
        for k in p:
            g = grads[k]
            self._m[k] = beta1 * self._m[k] + (1 - beta1) * g
            self._v[k] = beta2 * self._v[k] + (1 - beta2) * g ** 2
            mh = self._m[k] / (1 - beta1 ** self.t)
            vh = self._v[k] / (1 - beta2 ** self.t)
            p[k][:] -= self.lr * mh / (np.sqrt(vh) + eps)

    def update(self, grads):
        self._adam_step(grads)


# ═════════════════════════════════════════════════════════════════════════════
# METHOD 3 — QDDPG (Quantum-enhanced DDPG)
# ═════════════════════════════════════════════════════════════════════════════

def method_qddpg(cars, p, n_pretrain_episodes=2, train_steps_per_slot=2):
    """
    Quantum DDPG: PQC actor + classical MLP critic (from ddpg.py).

    Actor:  state → PQC (4 qubits, 2 layers) → N continuous phases
    Critic: (state, action) → Q-value  (identical to classical DDPG critic)
    Training uses replay buffer + soft target updates (same as DDPG).
    """
    tracemalloc.start()
    _reset_pqc_counter()
    t0 = time.time()
    N, Nt = p['N'], p['Nt']
    K  = p['Kr'] + p['Kt']
    T  = p['T_horizon']

    state_dim  = N * Nt + p['Kr'] * N + p['Kt'] * N + 1
    action_dim = N

    # Quantum actor (PQC) — classical critic reused from DDPGAgent
    qa = _QuantumActor(state_dim, action_dim, n_qubits=4, n_layers=2, lr=5e-4)

    # Classical critic only (borrow DDPGAgent internals)
    agent_c = DDPGAgent(
        state_dim=state_dim, action_dim=action_dim,
        hidden=(128, 64), lr_actor=5e-4, lr_critic=1e-3,
        gamma=0.95, tau=0.01,
        buffer_size=5000, batch_size=min(64, max(16, T * 2)),
        noise_sigma=0.0,   # noise applied externally
    )

    noise_sigma = 0.3
    total_iters = 0

    def _select(state, explore=True):
        raw = qa.forward(state)                       # tanh in [-1,1]
        phases = (raw + 1.0) * np.pi                  # → [0, 2π]
        if explore:
            phases += noise_sigma * np.random.randn(action_dim)
            phases %= 2 * np.pi
        return phases

    # Pre-training
    from simulator import init_cars as _ic
    for _ in range(n_pretrain_episodes):
        tc = _ic(p); prev_rate = 0.0
        for t in range(T):
            for car in tc:
                car.step(p['T_slot'])
            H_BR, H_r, H_t = generate_channels_at_slot(tc, p)
            state  = _extract_state(H_BR, H_r, H_t, prev_rate)
            phases = _select(state, explore=True)

            beta   = np.sqrt(0.5) * np.exp(1j * phases)
            h_eff  = effective_channels(H_BR, H_r, H_t, beta, beta)
            W      = _mrt_beamformer(h_eff, p['P_max'])
            rate   = compute_sum_rate(h_eff, W, p['sigma2'])

            done = float(t == T - 1)
            ns   = _extract_state(H_BR, H_r, H_t, rate)
            agent_c.store(state, phases, rate, ns, done)

            # Critic update + actor (quantum) policy gradient
            for _ts in range(train_steps_per_slot):
                if len(agent_c.buffer) < agent_c.batch_size:
                    break
                agent_c.train_step += 1
                ts_ = agent_c.train_step
                ss, aa, rr, nss, dd = agent_c.buffer.sample(agent_c.batch_size)

                # Critic targets
                na_raw = np.array([qa.forward(s) for s in nss])
                na     = (na_raw + 1.0) * np.pi
                ci_t   = np.hstack([nss, na])
                Q_t    = agent_c.critic_target.forward(ci_t)
                y      = rr + agent_c.gamma * (1 - dd) * Q_t

                ci     = np.hstack([ss, aa])
                Q_c    = agent_c.critic.forward(ci)
                d_Q    = 2.0 * (Q_c - y) / agent_c.batch_size
                agent_c.critic.backward(d_Q)
                agent_c.critic.step(agent_c.lr_critic, ts_)
                agent_c.critic_target.soft_update(agent_c.critic, agent_c.tau)

                # Actor (quantum) policy gradient — average over batch
                # to avoid per-sample PQC gradient calls (too slow).
                d_a_accum = np.zeros(action_dim)
                for si in ss:
                    raw_a = qa.forward(si)
                    a_sc  = (raw_a + 1.0) * np.pi
                    Q_val = agent_c.critic.forward(
                        np.hstack([si, a_sc]).reshape(1, -1))
                    d_Q_da = agent_c.critic.backward(
                        -np.ones_like(Q_val) / agent_c.batch_size
                    )
                    d_a_accum += d_Q_da[0, state_dim:] * np.pi
                # One quantum gradient step with averaged signal
                grads = qa.gradient(ss[0], d_a_accum / len(ss))
                qa.update(grads)
                total_iters += 1

            prev_rate = rate

    # Evaluation
    sr_ts = []; prev_rate = 0.0
    for t in range(T):
        for car in cars:
            car.step(p['T_slot'])
        H_BR, H_r, H_t = generate_channels_at_slot(cars, p)
        state  = _extract_state(H_BR, H_r, H_t, prev_rate)
        phases = _select(state, explore=False)
        beta   = np.sqrt(0.5) * np.exp(1j * phases)
        h_eff  = effective_channels(H_BR, H_r, H_t, beta, beta)
        W      = _mrt_beamformer(h_eff, p['P_max'])
        rate   = compute_sum_rate(h_eff, W, p['sigma2'])
        sr_ts.append(rate); prev_rate = rate

    energy = 1.0 * total_iters + 0.1 * T
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    pqc_evals = _PQC_EVALS
    m = _method_metrics(sr_ts)
    return dict(sum_rate=float(np.mean(sr_ts)), sum_rate_ts=sr_ts,
                iterations=total_iters + T, energy_norm=energy,
                time_s=time.time() - t0,
                circuit_evals=pqc_evals,
                peak_mem_mb=peak / 1024 / 1024,
                conv_std=m['conv_std'], conv_slope=m['conv_slope'])


# ═════════════════════════════════════════════════════════════════════════════
# METHOD 4 — QPPO (Quantum Proximal Policy Optimisation)
#            with STAR-RIS Baseline advantage normalisation reference
# ═════════════════════════════════════════════════════════════════════════════

class _ClassicalValueNet:
    """Small 2-layer MLP value function (state → scalar), trained with Adam."""

    def __init__(self, state_dim, hidden=64, lr=1e-3):
        s = np.sqrt(2.0 / state_dim)
        self.W1 = np.random.randn(state_dim, hidden) * s
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(hidden, 1) * np.sqrt(2.0 / hidden)
        self.b2 = np.zeros(1)
        self.lr = lr; self.t = 0
        self._m = {k: np.zeros_like(v)
                   for k, v in (('W1', self.W1), ('b1', self.b1),
                                ('W2', self.W2), ('b2', self.b2))}
        self._v = {k: np.zeros_like(v)
                   for k, v in (('W1', self.W1), ('b1', self.b1),
                                ('W2', self.W2), ('b2', self.b2))}

    def forward(self, s):
        h  = np.maximum(0, s @ self.W1 + self.b1)
        return float((h @ self.W2 + self.b2).ravel()[0])

    def update(self, states, targets, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        for s, tgt in zip(states, targets):
            h     = np.maximum(0, s @ self.W1 + self.b1)
            v     = float((h @ self.W2 + self.b2).ravel()[0])
            d_out = 2.0 * (v - tgt)
            dW2   = h[:, None] * d_out
            db2   = np.array([d_out])
            d_h   = self.W2.ravel() * d_out * (h > 0)
            dW1   = np.outer(s, d_h)
            db1   = d_h
            for k, g in (('W1', dW1), ('b1', db1),
                         ('W2', dW2), ('b2', db2)):
                self._m[k] = beta1 * self._m[k] + (1 - beta1) * g
                self._v[k] = beta2 * self._v[k] + (1 - beta2) * g ** 2
                mh = self._m[k] / (1 - beta1 ** self.t)
                vh = self._v[k] / (1 - beta2 ** self.t)
                getattr(self, k)[:] -= self.lr * mh / (np.sqrt(vh) + eps)


def method_qppo(cars, p, n_epochs=2, clip_eps=0.2, gamma_gae=0.99,
                lam_gae=0.95, lr_actor=5e-4, lr_value=1e-3,
                noise_sigma=0.25):
    """
    Quantum PPO (on-policy).

    Policy : PQC actor (_QuantumActor) — 4 qubits, 2 variational layers.
    Value  : Classical 1-hidden-layer MLP.
    Advantage: GAE(λ) computed over T_horizon trajectory.

    A STAR-RIS baseline (random phases + MRT) is used as a lower-bound
    reference to normalise advantages, ensuring the quantum agent learns
    to beat trivial random beamforming from the very first update.
    """
    tracemalloc.start()
    _reset_pqc_counter()
    t0 = time.time()
    N, Nt = p['N'], p['Nt']
    K  = p['Kr'] + p['Kt']
    T  = p['T_horizon']

    state_dim  = N * Nt + p['Kr'] * N + p['Kt'] * N + 1
    action_dim = N

    actor = _QuantumActor(state_dim, action_dim, n_qubits=4, n_layers=2,
                          lr=lr_actor)
    critic = _ClassicalValueNet(state_dim, hidden=64, lr=lr_value)

    # ── STAR-RIS baseline reference (random phases + MRT) ────────────────
    # Run once over T slots to get a reference sum-rate lower bound.
    # This is used to shift advantages so the agent learns relative gains.
    from simulator import init_cars as _ic
    ref_cars = _ic(p)
    for c_r, c_e in zip(ref_cars, cars):
        c_r.pos[:] = c_e.pos; c_r.vel[:] = c_e.vel
    ref_rates = []
    for _ in range(T):
        for car in ref_cars:
            car.step(p['T_slot'])
        H_BR, H_r, H_t = generate_channels_at_slot(ref_cars, p)
        phases_rnd = np.random.uniform(0, 2 * np.pi, N)
        beta_rnd   = np.sqrt(0.5) * np.exp(1j * phases_rnd)
        h_ref      = effective_channels(H_BR, H_r, H_t, beta_rnd, beta_rnd)
        W_ref      = _mrt_beamformer(h_ref, p['P_max'])
        ref_rates.append(compute_sum_rate(h_ref, W_ref, p['sigma2']))
    baseline_mean = float(np.mean(ref_rates))

    total_iters = 0

    def _act(state, explore=True):
        raw    = actor.forward(state)
        phases = (raw + 1.0) * np.pi
        if explore:
            phases += noise_sigma * np.random.randn(action_dim)
            phases %= 2 * np.pi
        # log-prob under Gaussian for PPO ratio (diagonal)
        log_prob = -0.5 * np.sum(((phases - (raw + 1.0) * np.pi)
                                  / noise_sigma) ** 2) if explore else 0.0
        return phases, log_prob

    # ── Collect one on-policy trajectory ────────────────────────────────
    states, actions, log_probs_old = [], [], []
    rewards, values, dones = [], [], []
    sr_ts = []; prev_rate = 0.0

    for t in range(T):
        for car in cars:
            car.step(p['T_slot'])
        H_BR, H_r, H_t = generate_channels_at_slot(cars, p)
        state  = _extract_state(H_BR, H_r, H_t, prev_rate)
        phases, lp = _act(state, explore=True)

        beta  = np.sqrt(0.5) * np.exp(1j * phases)
        h_eff = effective_channels(H_BR, H_r, H_t, beta, beta)
        W     = _mrt_beamformer(h_eff, p['P_max'])
        rate  = compute_sum_rate(h_eff, W, p['sigma2'])

        # Reward = rate minus STAR-RIS baseline (encourage beating baseline)
        reward = rate - baseline_mean

        states.append(state); actions.append(phases)
        log_probs_old.append(lp); rewards.append(reward)
        values.append(critic.forward(state)); dones.append(float(t == T - 1))
        sr_ts.append(rate); prev_rate = rate

    # GAE advantage
    advantages = np.zeros(T)
    gae = 0.0
    last_val = 0.0
    for i in reversed(range(T)):
        nv  = last_val if dones[i] else values[min(i + 1, T - 1)]
        delta = rewards[i] + gamma_gae * nv - values[i]
        gae   = delta + gamma_gae * lam_gae * (1 - dones[i]) * gae
        advantages[i] = gae
    returns  = advantages + np.array(values)
    adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # ── PPO update epochs ────────────────────────────────────────────────
    for _ in range(n_epochs):
        for i in range(T):
            state  = states[i]
            raw    = actor.forward(state)
            phases = (raw + 1.0) * np.pi

            # new log prob
            diff    = actions[i] - phases
            lp_new  = -0.5 * np.sum((diff / noise_sigma) ** 2)
            ratio   = np.exp(lp_new - log_probs_old[i])

            # clipped PPO objective gradient
            A = adv_norm[i]
            clip_r = float(np.clip(ratio, 1 - clip_eps, 1 + clip_eps))
            # d/d_action of -min(r*A, clip*A)
            if ratio * A <= clip_r * A:
                d_lp   = -A * ratio
            else:
                d_lp   = 0.0
            # d_log_prob / d_action = -(action - phases) / sigma^2
            d_phases = d_lp * (-(actions[i] - phases) / noise_sigma ** 2)
            # chain through (phases = (tanh+1)*pi): d/d_raw = pi
            d_raw = d_phases * np.pi
            grads = actor.gradient(state, d_raw)
            actor.update(grads)
            total_iters += 1

        critic.update(states, returns.tolist())

    energy = 1.2 * total_iters + 0.1 * T
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    pqc_evals = _PQC_EVALS
    m = _method_metrics(sr_ts)
    return dict(sum_rate=float(np.mean(sr_ts)), sum_rate_ts=sr_ts,
                iterations=total_iters, energy_norm=energy,
                time_s=time.time() - t0,
                circuit_evals=pqc_evals,
                peak_mem_mb=peak / 1024 / 1024,
                conv_std=m['conv_std'], conv_slope=m['conv_slope'])


# ═════════════════════════════════════════════════════════════════════════════
# METHOD 5 — STAR-RIS Baseline (random phases + MRT, no learning)
# ═════════════════════════════════════════════════════════════════════════════

def method_star_ris_baseline(cars, p):
    """
    Lower-bound baseline: random RIS phase configuration each slot + MRT.
    No training, no optimisation — pure random beamforming.
    Models a naive STAR-RIS deployment without any intelligent control.
    """
    tracemalloc.start()
    t0 = time.time()
    N  = p['N']
    T  = p['T_horizon']
    sr_ts = []; prev_rate = 0.0

    for _ in range(T):
        for car in cars:
            car.step(p['T_slot'])
        H_BR, H_r, H_t = generate_channels_at_slot(cars, p)
        phases = np.random.uniform(0, 2 * np.pi, N)
        beta   = np.sqrt(0.5) * np.exp(1j * phases)
        h_eff  = effective_channels(H_BR, H_r, H_t, beta, beta)
        W      = _mrt_beamformer(h_eff, p['P_max'])
        rate   = compute_sum_rate(h_eff, W, p['sigma2'])
        sr_ts.append(rate); prev_rate = rate

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    m = _method_metrics(sr_ts)
    return dict(sum_rate=float(np.mean(sr_ts)), sum_rate_ts=sr_ts,
                iterations=T, energy_norm=0.1 * T,
                time_s=time.time() - t0,
                circuit_evals=0,
                peak_mem_mb=peak / 1024 / 1024,
                conv_std=m['conv_std'], conv_slope=m['conv_slope'])
