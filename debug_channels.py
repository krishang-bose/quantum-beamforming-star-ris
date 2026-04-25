"""Debug: check channel magnitudes and SINR values with fixed params."""
import numpy as np
from simulator import DEFAULT_PARAMS, init_cars, generate_channels_at_slot
from simulator import effective_channels, compute_sum_rate, compute_sinr
from simulator import init_beamformers, _path_loss

p = DEFAULT_PARAMS.copy()
cars = init_cars(p)

bs_pos  = np.array([p['d_BS'], 0.0])
ris_pos = np.array([p['d_RIS'], p['d_RIS']])
d_ref   = p.get('d_ref', 1.0)
print(f"BS position:  {bs_pos}")
print(f"RIS position: {ris_pos}")
d_BR = np.linalg.norm(bs_pos - ris_pos)
print(f"BS-RIS distance: {d_BR:.1f} m")
print(f"BS-RIS path loss: {_path_loss(d_BR, p['alpha'], d_ref):.6e}")

for i, car in enumerate(cars):
    d_car = np.linalg.norm(ris_pos - car.pos)
    print(f"Car {i} ({car.side}): pos={car.pos}, dist_to_RIS={d_car:.1f} m, "
          f"PL={_path_loss(d_car, p['alpha'], d_ref):.6e}")

H_BR, H_r, H_t = generate_channels_at_slot(cars, p)
print(f"\n|H_BR| mean: {np.mean(np.abs(H_BR)):.6e}")
print(f"|H_r|  mean: {np.mean(np.abs(H_r)):.6e}")
print(f"|H_t|  mean: {np.mean(np.abs(H_t)):.6e}")

beta = np.sqrt(0.5) * np.exp(1j * np.random.uniform(0, 2*np.pi, p['N']))
h_eff = effective_channels(H_BR, H_r, H_t, beta, beta)
print(f"|h_eff| mean: {np.mean(np.abs(h_eff)):.6e}")

W = init_beamformers(p['Nt'], p['Kr'] + p['Kt'], p['P_max'])
print(f"|W| mean: {np.mean(np.abs(W)):.6e}")
print(f"P_max: {p['P_max']}")
print(f"sigma2: {p['sigma2']:.6e}")

sinr = compute_sinr(h_eff, W, p['sigma2'])
print(f"\nSINR per user: {sinr}")
print(f"Sum-rate: {compute_sum_rate(h_eff, W, p['sigma2']):.6f}")
