"""
ddpg.py — Deep Deterministic Policy Gradient (NumPy implementation)

A lightweight DDPG agent for STAR-RIS beamforming optimisation.
No PyTorch / TensorFlow dependency — pure NumPy + SciPy.

Architecture
────────────
  Actor  : state → continuous action (RIS phases + beamformer)
  Critic : (state, action) → Q-value (predicted sum-rate)

Training uses:
  • Experience replay buffer
  • Ornstein–Uhlenbeck exploration noise
  • Soft target-network updates (Polyak averaging)

References
──────────
  Lillicrap et al., "Continuous control with deep reinforcement learning",
  ICLR 2016.
"""

import numpy as np
from collections import deque


# ═════════════════════════════════════════════════════════════════════════════
# Helper: simple dense layer
# ═════════════════════════════════════════════════════════════════════════════

def _he_init(fan_in, fan_out):
    """He (Kaiming) initialisation for ReLU networks."""
    std = np.sqrt(2.0 / fan_in)
    return np.random.randn(fan_in, fan_out).astype(np.float64) * std


class DenseLayer:
    """Single fully-connected layer with optional activation."""

    def __init__(self, in_dim, out_dim, activation='relu'):
        self.W = _he_init(in_dim, out_dim)
        self.b = np.zeros(out_dim)
        self.activation = activation
        # Adam state
        self.mW, self.vW = np.zeros_like(self.W), np.zeros_like(self.W)
        self.mb, self.vb = np.zeros_like(self.b), np.zeros_like(self.b)

    def forward(self, x):
        """x: (batch, in_dim) → (batch, out_dim)"""
        self.x = x
        self.z = x @ self.W + self.b
        if self.activation == 'relu':
            self.a = np.maximum(0, self.z)
        elif self.activation == 'tanh':
            self.a = np.tanh(self.z)
        elif self.activation == 'none':
            self.a = self.z.copy()
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
        return self.a

    def backward(self, d_out):
        """Backprop through this layer; returns d_input and stores grads."""
        if self.activation == 'relu':
            d_act = d_out * (self.z > 0).astype(float)
        elif self.activation == 'tanh':
            d_act = d_out * (1 - self.a ** 2)
        elif self.activation == 'none':
            d_act = d_out
        else:
            d_act = d_out

        self.dW = self.x.T @ d_act
        self.db = d_act.sum(axis=0)
        d_input = d_act @ self.W.T
        return d_input

    def step_adam(self, lr, t, beta1=0.9, beta2=0.999, eps=1e-8):
        """Adam optimiser update."""
        for param, grad, m, v in [
            ('W', 'dW', 'mW', 'vW'),
            ('b', 'db', 'mb', 'vb'),
        ]:
            g = getattr(self, grad)
            m_arr = getattr(self, m)
            v_arr = getattr(self, v)
            m_arr[:] = beta1 * m_arr + (1 - beta1) * g
            v_arr[:] = beta2 * v_arr + (1 - beta2) * g ** 2
            m_hat = m_arr / (1 - beta1 ** t)
            v_hat = v_arr / (1 - beta2 ** t)
            getattr(self, param)[:] -= lr * m_hat / (np.sqrt(v_hat) + eps)


# ═════════════════════════════════════════════════════════════════════════════
# Multi-layer perceptron
# ═════════════════════════════════════════════════════════════════════════════

class MLP:
    """Simple multi-layer perceptron (list of DenseLayers)."""

    def __init__(self, dims, out_activation='none'):
        """
        dims: list of layer widths, e.g. [state_dim, 128, 64, action_dim]
        out_activation: activation on the final layer
        """
        self.layers = []
        for i in range(len(dims) - 1):
            act = 'relu' if i < len(dims) - 2 else out_activation
            self.layers.append(DenseLayer(dims[i], dims[i + 1], activation=act))

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, d_out):
        for layer in reversed(self.layers):
            d_out = layer.backward(d_out)
        return d_out

    def step(self, lr, t):
        for layer in self.layers:
            layer.step_adam(lr, t)

    def get_params(self):
        """Return flat list of (W, b) tuples."""
        return [(l.W.copy(), l.b.copy()) for l in self.layers]

    def set_params(self, params):
        for l, (W, b) in zip(self.layers, params):
            l.W[:] = W
            l.b[:] = b

    def soft_update(self, source, tau):
        """Polyak averaging: self ← tau * source + (1-tau) * self."""
        for l_self, l_src in zip(self.layers, source.layers):
            l_self.W[:] = tau * l_src.W + (1 - tau) * l_self.W
            l_self.b[:] = tau * l_src.b + (1 - tau) * l_self.b


# ═════════════════════════════════════════════════════════════════════════════
# Ornstein–Uhlenbeck noise process
# ═════════════════════════════════════════════════════════════════════════════

class OUNoise:
    """Ornstein–Uhlenbeck process for temporally-correlated exploration."""

    def __init__(self, dim, mu=0.0, theta=0.15, sigma=0.2):
        self.dim = dim
        self.mu = mu * np.ones(dim)
        self.theta = theta
        self.sigma = sigma
        self.state = self.mu.copy()

    def reset(self):
        self.state = self.mu.copy()

    def sample(self):
        dx = self.theta * (self.mu - self.state) + \
             self.sigma * np.random.randn(self.dim)
        self.state += dx
        return self.state.copy()


# ═════════════════════════════════════════════════════════════════════════════
# Replay buffer
# ═════════════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        idxs = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in idxs]
        states = np.array([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch]).reshape(-1, 1)
        next_states = np.array([b[3] for b in batch])
        dones = np.array([b[4] for b in batch]).reshape(-1, 1)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# ═════════════════════════════════════════════════════════════════════════════
# DDPG Agent
# ═════════════════════════════════════════════════════════════════════════════

class DDPGAgent:
    """
    Deep Deterministic Policy Gradient agent for STAR-RIS beamforming.

    State  : flattened channel magnitudes + previous sum-rate
    Action : continuous RIS phase angles ∈ [0, 2π]
             (beamformer is derived analytically from phases via MRT)
    Reward : sum-rate at the current slot
    """

    def __init__(self, state_dim, action_dim,
                 hidden=(128, 64),
                 lr_actor=1e-3, lr_critic=1e-3,
                 gamma=0.95, tau=0.005,
                 buffer_size=10000, batch_size=64,
                 noise_sigma=0.2):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.train_step = 0

        # Actor:  state → action (tanh output, scaled to [0, 2π])
        actor_dims = [state_dim] + list(hidden) + [action_dim]
        self.actor = MLP(actor_dims, out_activation='tanh')
        self.actor_target = MLP(actor_dims, out_activation='tanh')
        self.actor_target.set_params(self.actor.get_params())

        # Critic: (state, action) → Q-value
        critic_dims = [state_dim + action_dim] + list(hidden) + [1]
        self.critic = MLP(critic_dims, out_activation='none')
        self.critic_target = MLP(critic_dims, out_activation='none')
        self.critic_target.set_params(self.critic.get_params())

        self.buffer = ReplayBuffer(buffer_size)
        self.noise = OUNoise(action_dim, sigma=noise_sigma)

    def select_action(self, state, explore=True):
        """
        Select action given state.
        Actor outputs tanh ∈ [-1, 1], scaled to [0, 2π].
        """
        s = state.reshape(1, -1)
        raw = self.actor.forward(s)[0]          # tanh ∈ [-1, 1]
        action = (raw + 1.0) * np.pi           # → [0, 2π]
        if explore:
            action += self.noise.sample()
            action = action % (2 * np.pi)      # keep in [0, 2π]
        return action

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def train(self):
        """One gradient step on a mini-batch from replay buffer."""
        if len(self.buffer) < self.batch_size:
            return 0.0

        self.train_step += 1
        t = self.train_step

        states, actions, rewards, next_states, dones = \
            self.buffer.sample(self.batch_size)

        # ── Critic update ────────────────────────────────────────────────
        # target actions from target actor
        raw_next = self.actor_target.forward(next_states)
        next_actions = (raw_next + 1.0) * np.pi

        # target Q
        critic_target_input = np.hstack([next_states, next_actions])
        Q_target = self.critic_target.forward(critic_target_input)
        y = rewards + self.gamma * (1 - dones) * Q_target

        # current Q
        critic_input = np.hstack([states, actions])
        Q_current = self.critic.forward(critic_input)

        # MSE loss gradient: d_loss/d_Q = 2*(Q - y) / batch
        critic_loss = np.mean((Q_current - y) ** 2)
        d_Q = 2.0 * (Q_current - y) / self.batch_size
        self.critic.backward(d_Q)
        self.critic.step(self.lr_critic, t)

        # ── Actor update ─────────────────────────────────────────────────
        # maximise Q(s, actor(s)) → minimise -Q
        raw_a = self.actor.forward(states)
        a_scaled = (raw_a + 1.0) * np.pi
        actor_critic_input = np.hstack([states, a_scaled])
        Q_val = self.critic.forward(actor_critic_input)

        # d(-Q)/d_action = -d_Q/d_action
        d_Q_da = self.critic.backward(
            -np.ones_like(Q_val) / self.batch_size
        )
        # d_action is the last action_dim columns of the critic input grad
        d_action = d_Q_da[:, self.state_dim:]

        # chain through scaling: a = (tanh + 1) * π → da/d_raw = π
        d_raw = d_action * np.pi
        self.actor.backward(d_raw)
        self.actor.step(self.lr_actor, t)

        # ── Soft target update ───────────────────────────────────────────
        self.actor_target.soft_update(self.actor, self.tau)
        self.critic_target.soft_update(self.critic, self.tau)

        return float(critic_loss)

    def reset_noise(self):
        self.noise.reset()
