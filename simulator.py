"""
simulator.py  (dynamic / vehicular STAR-RIS)

Mobility model:   constant-velocity with wrap-around (torus boundary)
Channel model:    Rayleigh fading + distance-dependent path loss
                  + Doppler shift based on car velocity and direction
Time axis:        T slots of duration T_slot seconds
"""

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# Default simulation parameters
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_PARAMS = dict(
    Nt        = 4,          # BS transmit antennas
    N         = 16,         # STAR-RIS elements
    Kr        = 2,          # reflection-side cars
    Kt        = 2,          # transmission-side cars
    P_max     = 10.0,       # total BS transmit power  [W]
    sigma2    = 1e-4,       # noise variance
    fc        = 5.9e9,      # carrier frequency  [Hz]  (V2X band)
    c_light   = 3e8,        # speed of light     [m/s]
    T_slot    = 1e-3,       # slot duration      [s]   (1 ms)
    T_horizon = 50,         # number of time slots per trial
    v_min     = 5.0,        # min car speed      [m/s]  (~18 km/h)
    v_max     = 30.0,       # max car speed      [m/s] (~108 km/h)
    area_size = 60.0,       # simulation area    [m] (square)
    d_BS      = 30.0,       # BS x-position      [m]
    d_RIS     = 15.0,       # RIS position at (15, 15) [m]
    d_ref     = 1.0,        # reference distance [m]  for path-loss model
    alpha     = 2.2,        # path-loss exponent (urban micro)
)

# 2-bit phase quantisation levels (0, pi/2, pi, 3pi/2)
PHASE_LEVELS_2BIT = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])


# ═══════════════════════════════════════════════════════════════════════════════
# Mobility model
# ═══════════════════════════════════════════════════════════════════════════════

class Car:
    """
    Single car with constant-velocity mobility.
    Wraps around the simulation area (torus boundary).
    """
    def __init__(self, area_size, v_min, v_max, side='reflection'):
        self.area = area_size
        self.side = side  # 'reflection' or 'transmission'

        # random initial position
        self.pos = np.random.uniform(0, area_size, size=2)

        # random initial velocity
        speed = np.random.uniform(v_min, v_max)
        angle = np.random.uniform(0, 2 * np.pi)
        self.vel = speed * np.array([np.cos(angle), np.sin(angle)])

    def step(self, dt):
        """Advance position by one time step dt [s]."""
        self.pos = (self.pos + self.vel * dt) % self.area

    @property
    def speed(self):
        return float(np.linalg.norm(self.vel))

    @property
    def direction(self):
        return self.vel / (self.speed + 1e-12)


def init_cars(p):
    """
    Create Kr reflection-side cars and Kt transmission-side cars.
    Returns list of Car objects.
    """
    cars = []
    for _ in range(p['Kr']):
        cars.append(Car(p['area_size'], p['v_min'], p['v_max'],
                        side='reflection'))
    for _ in range(p['Kt']):
        cars.append(Car(p['area_size'], p['v_min'], p['v_max'],
                        side='transmission'))
    return cars


# ═══════════════════════════════════════════════════════════════════════════════
# Channel generation (one time slot)
# ═══════════════════════════════════════════════════════════════════════════════

def _path_loss(d, alpha, d_ref=1.0):
    """
    Distance-dependent path loss (linear scale).
    PL(d) = (d / d_ref)^(-alpha), clipped so d >= d_ref.
    """
    return (max(d, d_ref) / d_ref) ** (-alpha)


def _doppler_factor(car, rx_pos, p):
    """
    Scalar Doppler phase rotation per slot.
    fd = (v/c) * fc * cos(angle between velocity and LOS)
    phase_rotation = exp(j * 2*pi * fd * T_slot)
    """
    los_vec  = rx_pos - car.pos
    los_dist = np.linalg.norm(los_vec) + 1e-6
    los_unit = los_vec / los_dist
    cos_theta = np.dot(car.direction, los_unit)
    fd = (car.speed / p['c_light']) * p['fc'] * cos_theta
    return np.exp(1j * 2 * np.pi * fd * p['T_slot'])


def generate_channels_at_slot(cars, p):
    """
    Generate instantaneous channels for all cars at their current positions.

    Returns:
        H_BR : (N, Nt)   BS -> RIS  (quasi-static, updates slowly)
        H_r  : (Kr, N)   RIS -> reflection cars
        H_t  : (Kt, N)   RIS -> transmission cars
    """
    N, Nt = p['N'], p['Nt']
    Kr = p['Kr']
    Kt = p['Kt']

    bs_pos  = np.array([p['d_BS'], 0.0])
    ris_pos = np.array([p['d_RIS'], p['d_RIS']])
    d_ref   = p.get('d_ref', 1.0)

    # ── BS -> RIS (quasi-static for the horizon) ──
    d_BR = np.linalg.norm(bs_pos - ris_pos)
    pl = _path_loss(d_BR, p['alpha'], d_ref)
    H_BR = (np.random.randn(N, Nt) + 1j * np.random.randn(N, Nt)) \
           * np.sqrt(pl / 2)

    # ── RIS -> reflection cars ──
    ref_cars = [c for c in cars if c.side == 'reflection']
    H_r = np.zeros((Kr, N), dtype=complex)
    for k, car in enumerate(ref_cars):
        d_RU = np.linalg.norm(ris_pos - car.pos) + 1e-6
        pl_u = _path_loss(d_RU, p['alpha'], d_ref)
        h_raw = (np.random.randn(N) + 1j * np.random.randn(N)) \
                * np.sqrt(pl_u / 2)
        dop = _doppler_factor(car, ris_pos, p)
        H_r[k] = h_raw * dop

    # ── RIS -> transmission cars ──
    trans_cars = [c for c in cars if c.side == 'transmission']
    H_t = np.zeros((Kt, N), dtype=complex)
    for k, car in enumerate(trans_cars):
        d_RU = np.linalg.norm(ris_pos - car.pos) + 1e-6
        pl_u = _path_loss(d_RU, p['alpha'], d_ref)
        h_raw = (np.random.randn(N) + 1j * np.random.randn(N)) \
                * np.sqrt(pl_u / 2)
        dop = _doppler_factor(car, ris_pos, p)
        H_t[k] = h_raw * dop

    return H_BR, H_r, H_t


# ═══════════════════════════════════════════════════════════════════════════════
# STAR-RIS coefficients, effective channel, SINR, sum-rate
# ═══════════════════════════════════════════════════════════════════════════════

def phase_choices_to_coeffs(choices, amp=None):
    """Convert integer phase choices to complex STAR-RIS coefficients."""
    if amp is None:
        amp = np.sqrt(0.5)
    thetas = PHASE_LEVELS_2BIT[choices]
    return amp * np.exp(1j * thetas)


def effective_channels(H_BR, H_r, H_t, beta_r, beta_t):
    """
    Compute effective end-to-end channels through the STAR-RIS.
    
    H_eff_r = H_r @ diag(beta_r) @ H_BR   for reflection users
    H_eff_t = H_t @ diag(beta_t) @ H_BR   for transmission users
    
    Returns: (K, Nt) stacked effective channel matrix
    """
    Phi_r = np.diag(beta_r)
    Phi_t = np.diag(beta_t)
    h_eff_r = H_r @ Phi_r @ H_BR
    h_eff_t = H_t @ Phi_t @ H_BR
    return np.vstack([h_eff_r, h_eff_t])


def compute_sinr(h_eff, W, sigma2):
    """Compute per-user SINR given effective channels and beamformers."""
    K = h_eff.shape[0]
    sinr = np.zeros(K)
    for k in range(K):
        sig = np.abs(h_eff[k] @ W[:, k]) ** 2
        intf = sum(np.abs(h_eff[k] @ W[:, j]) ** 2
                   for j in range(K) if j != k)
        sinr[k] = sig / (intf + sigma2)
    return sinr


def compute_sum_rate(h_eff, W, sigma2):
    """Sum-rate in bits/s/Hz."""
    return float(np.sum(np.log2(1 + compute_sinr(h_eff, W, sigma2))))


def init_beamformers(Nt, K, P_max):
    """Initialise random beamformers with total power = P_max."""
    W = np.random.randn(Nt, K) + 1j * np.random.randn(Nt, K)
    return W / np.sqrt(np.sum(np.abs(W) ** 2)) * np.sqrt(P_max)


def project_power(W, P_max):
    """Project beamformers so that total power <= P_max."""
    n = np.sum(np.abs(W) ** 2)
    return W * np.sqrt(P_max / n) if n > P_max else W
