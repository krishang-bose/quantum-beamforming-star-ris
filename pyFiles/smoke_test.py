"""Quick smoke test — verify imports, gradient, and one mini trial."""
import numpy as np

print("=" * 60)
print("SMOKE TEST")
print("=" * 60)

# 1. Gradient check
from hamiltonian import verify_gradient, build_pauli_hamiltonian
print("\n[1] Gradient check...")
ok = verify_gradient(gamma=0.4, beta=0.7, a=1.2, b=0.9, c=0.5)
assert ok, "Gradient check FAILED"
print("    ✓ Gradient check passed\n")

# 2. Simulator check
from simulator import DEFAULT_PARAMS, init_cars, generate_channels_at_slot
from simulator import effective_channels, compute_sum_rate, init_beamformers
p = DEFAULT_PARAMS.copy()
p['T_horizon'] = 3  # tiny
cars = init_cars(p)
for car in cars:
    car.step(p['T_slot'])
H_BR, H_r, H_t = generate_channels_at_slot(cars, p)
beta = np.sqrt(0.5) * np.exp(1j * np.random.uniform(0, 2*np.pi, p['N']))
h_eff = effective_channels(H_BR, H_r, H_t, beta, beta)
W = init_beamformers(p['Nt'], p['Kr'] + p['Kt'], p['P_max'])
R = compute_sum_rate(h_eff, W, p['sigma2'])
print(f"[2] Simulator check: sum-rate = {R:.4f}")
assert R > 0, "Sum-rate should be positive"
print("    ✓ Simulator works\n")

# 3. DDPG agent check
from ddpg import DDPGAgent
print("[3] DDPG agent check...")
state_dim = p['N'] * p['Nt'] + p['Kr'] * p['N'] + p['Kt'] * p['N'] + 1
action_dim = p['N']
agent = DDPGAgent(state_dim, action_dim, hidden=(32, 16))
dummy_state = np.random.randn(state_dim)
action = agent.select_action(dummy_state, explore=True)
assert action.shape == (action_dim,), f"Bad action shape: {action.shape}"
assert np.all(action >= 0) and np.all(action < 2 * np.pi), \
    "Action out of [0, 2π) range"
print(f"    Action sample (first 4): {action[:4]}")
print("    ✓ DDPG agent works\n")

# 4. One mini trial (1 trial, 3 time slots)
from experiments import _run_one_trial
p_mini = DEFAULT_PARAMS.copy()
p_mini['T_horizon'] = 3
print("[4] Mini trial (T_horizon=3)...")
r1, r2 = _run_one_trial(p_mini)
print(f"    DDPG:   SR={r1['sum_rate']:.4f}  time={r1['time_s']:.2f}s  "
      f"iters={r1['iterations']}")
print(f"    QAOA:   SR={r2['sum_rate']:.4f}  time={r2['time_s']:.2f}s  "
      f"iters={r2['iterations']}")
print("    ✓ Both methods ran successfully\n")

print("=" * 60)
print("ALL SMOKE TESTS PASSED")
print("=" * 60)
