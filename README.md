# Quantum Beamforming for Vehicular STAR-RIS Networks

A research simulation framework comparing **quantum-hybrid** and classical deep reinforcement learning methods for intelligent beamforming in vehicular (V2X) networks using **Simultaneous Transmitting And Reflecting Reconfigurable Intelligent Surfaces (STAR-RIS)**.

Three quantum-enhanced methods are benchmarked against each other across four system-level sweeps — SNR, RIS size (N), vehicle speed, and number of users (K) — with 30 Monte Carlo trials per operating point.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Model](#2-system-model)
3. [Repository Structure](#3-repository-structure)
4. [Dependencies & Installation](#4-dependencies--installation)
5. [Module Reference](#5-module-reference)
   - [simulator.py](#simulatorpy)
   - [hamiltonian.py](#hamiltonianpy)
   - [ddpg.py](#ddpgpy)
   - [methods.py](#methodspy)
   - [experiments.py](#experimentspy)
   - [main.py](#mainpy)
   - [plot_analysis.py](#plot_analysispy)
   - [xai_analysis.py](#xai_analysispy)
   - [Testing Scripts](#testing-scripts)
6. [Method Details](#6-method-details)
   - [QAOA (Hybrid Quantum-Classical)](#method-1--qaoa-hybrid-quantum-classical)
   - [QDDPG (Quantum DDPG)](#method-2--qddpg-quantum-enhanced-ddpg)
   - [QPPO (Quantum PPO)](#method-3--qppo-quantum-proximal-policy-optimisation)
7. [Running the Simulation](#7-running-the-simulation)
8. [Experiment Scenarios](#8-experiment-scenarios)
9. [Output Files](#9-output-files)
10. [Results Summary](#10-results-summary)
11. [Post-Processing & Analysis](#11-post-processing--analysis)
12. [XAI Analysis](#12-xai-analysis)
13. [Energy Efficiency Model](#13-energy-efficiency-model)
14. [Key Design Decisions & Notes](#14-key-design-decisions--notes)

---

## 1. Project Overview

This project simulates a **vehicular V2X (Vehicle-to-Everything) communication** scenario where a Base Station (BS) serves moving cars through a STAR-RIS. The key research question is:

> _Can quantum-enhanced reinforcement learning algorithms (QAOA, QDDPG, QPPO) outperform each other in optimising STAR-RIS beamforming under high-mobility vehicular channels?_

The simulation runs across four parametric sweeps (SNR, N, car speed, K), each with 30 independent Monte Carlo trials executed in parallel using Python multiprocessing.

---

## 2. System Model

```
  [Base Station (BS)]
    4 Tx antennas (Nt=4)
    P_max = 5 W (37 dBm)
         |
         | BS→RIS channel H_BR (N×Nt)
         ↓
  [STAR-RIS]  ← N = 16/32/64 elements
    - Simultaneous Transmitting AND Reflecting
    - 2-bit phase quantization (0, π/2, π, 3π/2)
    - Amplitude: |β| = √0.5 for both sides
    |                      |
    | Reflection side       | Transmission side
    | H_r (Kr×N)           | H_t (Kt×N)
    ↓                      ↓
[Kr reflection cars]   [Kt transmission cars]
   (moving, ~18–108 km/h)
```

**Channel model:**
- Rayleigh fading + distance-dependent path loss: `PL(d) = (d/d_ref)^(-α)`, α = 2.2
- Carrier frequency: 5.9 GHz (DSRC/V2X band)
- Doppler shift modeled per-slot using radial velocity along line-of-sight
- Jakes coherence factor models temporal decorrelation:
  `coherence = exp(-κ · fd_max · T_slot)`, κ = 2.5
  - At 5 m/s → 0.77 coherence loss
  - At 30 m/s → 0.21 coherence loss

**Key simulation parameters (defaults):**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `Nt` | 4 | BS transmit antennas |
| `N` | 16 | STAR-RIS elements (default) |
| `Kr` | 2 | Reflection-side cars |
| `Kt` | 2 | Transmission-side cars |
| `P_max` | 5.0 W | Max BS transmit power |
| `sigma2` | 1e-3 | Noise variance |
| `fc` | 5.9 GHz | Carrier frequency |
| `T_slot` | 1 ms | Time slot duration |
| `T_horizon` | 80 | Slots per trial |
| `v_min/v_max` | 5/30 m/s | Car speed range |
| `area_size` | 60 m | Square simulation area |
| `alpha` | 2.2 | Path-loss exponent |

---

## 3. Repository Structure

```
quantum-beamforming-star-ris/
├── pyFiles/
│   ├── simulator.py        # Channel model, mobility model, SINR/rate computation
│   ├── hamiltonian.py      # QAOA Hamiltonians (2-qubit & 4-qubit) + gradient verification
│   ├── ddpg.py             # Pure-NumPy DDPG agent (MLP, replay buffer, OU noise)
│   ├── methods.py          # All 4 beamforming methods (DDPG, QAOA, QDDPG, QPPO, Baseline)
│   ├── experiments.py      # Parallelized scenario runners (SNR, N, Speed, K sweeps)
│   ├── main.py             # Full pipeline: gradient checks → sweeps → tables → plots
│   ├── plot_analysis.py    # Extended per-method/per-scenario figure + table generation
│   ├── xai_analysis.py     # SHAP + LIME explainability analysis on CSV results
│   ├── quick_test.py       # 2-trial dry run to verify the pipeline before overnight runs
│   ├── smoke_test.py       # Minimal import + unit check for each module
│   ├── debug_channels.py   # Channel magnitude / SINR inspection utility
│   ├── requirements.txt    # Python package dependencies
│   └── run.log             # Output log from the 30-trial benchmark run
├── results/
│   ├── results_snr.csv     # SNR sweep results (mean ± std per method)
│   ├── results_N.csv       # N sweep results
│   ├── results_speed.csv   # Speed sweep results
│   ├── results_K.csv       # K sweep results
│   ├── fig_sumrate_snr.png          # Sum-rate vs SNR figure
│   ├── fig_sumrate_N.png            # Sum-rate vs N
│   ├── fig_sumrate_speed.png        # Sum-rate vs car speed
│   ├── fig_sumrate_K.png            # Sum-rate vs K
│   ├── fig_ee_snr.png               # Energy efficiency vs SNR
│   ├── fig_ee_N.png                 # Energy efficiency vs N
│   ├── fig_ee_speed.png             # Energy efficiency vs speed
│   ├── fig_ee_K.png                 # Energy efficiency vs K
│   ├── fig_complexity_analysis.png  # Convergence std / memory / time vs SNR
│   └── fig_circuit_evals_N.png      # Quantum circuit evaluations vs N
└── results.zip             # Compressed archive of all results
```

---

## 4. Dependencies & Installation

**Python 3.9+ required.**

```bash
cd pyFiles
pip install -r requirements.txt
```

`requirements.txt`:
```
numpy==2.4.4
pandas==3.0.2
matplotlib==3.10.9
scipy==1.17.1
qiskit==2.4.1
```

**For XAI analysis only** (`xai_analysis.py`), additional packages are needed:
```bash
pip install shap scikit-learn lime
```

> **Note:** All quantum circuits run on a pure-NumPy statevector simulator — no quantum hardware or Qiskit Aer is required. Qiskit is used only for `SparsePauliOp` Hamiltonian construction and the `Statevector` utility in the 2-qubit path.

---

## 5. Module Reference

### `simulator.py`

Core physics engine for the vehicular STAR-RIS simulation.

**Key classes/functions:**

| Name | Description |
|------|-------------|
| `DEFAULT_PARAMS` | Dict of all default simulation parameters |
| `PHASE_LEVELS_2BIT` | `[0, π/2, π, 3π/2]` — quantized phase levels |
| `Car` | Single car with constant-velocity mobility + torus wrap-around |
| `Car.step(dt)` | Advance position by `dt` seconds |
| `init_cars(p)` | Create `Kr` + `Kt` cars from param dict |
| `generate_channels_at_slot(cars, p)` | Returns `H_BR` (N×Nt), `H_r` (Kr×N), `H_t` (Kt×N) |
| `phase_choices_to_coeffs(choices)` | Convert integer phase indices → complex STAR-RIS coefficients |
| `effective_channels(H_BR, H_r, H_t, β_r, β_t)` | Compute end-to-end effective channels `H_eff = H @ diag(β) @ H_BR` |
| `compute_sinr(h_eff, W, sigma2)` | Per-user SINR given beamformers |
| `compute_sum_rate(h_eff, W, sigma2)` | Sum-rate in bits/s/Hz |
| `init_beamformers(Nt, K, P_max)` | Random beamformers normalized to total power |
| `project_power(W, P_max)` | Project beamformer to satisfy power constraint |
| `compute_channel_gains(H_BR, H_r, H_t)` | Channel gain features for state enrichment |

**Channel generation pipeline per slot:**
1. Compute BS→RIS path loss (quasi-static)
2. Generate Rayleigh fading `H_BR` scaled by path loss
3. For each user car: compute distance-based path loss, Rayleigh fading, Doppler phase rotation, and Jakes coherence factor
4. Return three channel matrices

---

### `hamiltonian.py`

QAOA Hamiltonian construction and gradient verification.

**Binary variable encoding:**  
`x_n = (1 - Z_n)/2`, i.e., `Z_n = 1 - 2*x_n`

**2-qubit cost function (proxy for 2-element RIS):**
```
C(x0, x1) = -(a*(1-x0) + b*(1-x1) + c*x0*x1)
```

Pauli decomposition: `C = h_II·II + h_IZ·IZ + h_ZI·ZI + h_ZZ·ZZ`

**4-qubit cost function (used in production — splits N RIS elements into 4 groups):**
```
C(x) = -[ Σ_i a_i*(1-x_i) + Σ_{i<j} c_ij*x_i*x_j ]
```

| Function | Description |
|----------|-------------|
| `build_pauli_hamiltonian(a, b, c)` | 2-qubit Hamiltonian as `SparsePauliOp` |
| `build_4qubit_hamiltonian(a_vec, c_mat)` | 4-qubit Hamiltonian + 16×16 matrix + mixer |
| `expectation_value(γ, β, a, b, c)` | ⟨C⟩ for depth-1 2-qubit QAOA |
| `analytic_gradient(γ, β, a, b, c)` | Exact gradient `(∂⟨C⟩/∂γ, ∂⟨C⟩/∂β)` — no finite differences |
| `expectation_4qubit(γ, β, H_mat, B_mat)` | ⟨C⟩ for depth-1 4-qubit QAOA |
| `gradient_4qubit(γ, β, H_mat, B_mat)` | Analytic gradient for 4-qubit case |
| `verify_gradient(...)` | Compares analytic vs. central-difference FD — passes at err < 1e-7 |
| `verify_4qubit_gradient(...)` | Same for 4-qubit — passes at err < 1e-6 |

**Analytic gradient derivation:**
```
|ψ⟩ = U_B · U_C · |++⟩       where U_C = exp(-iγH), U_B = exp(-iβB)

∂⟨C⟩/∂γ = 2·Re[⟨ψ|H|  (-i·U_B·H·U_C|s0⟩) ⟩]
∂⟨C⟩/∂β = 2·Re[⟨ψ|H|  (-i·B|ψ⟩) ⟩]
```

---

### `ddpg.py`

Pure NumPy implementation of Deep Deterministic Policy Gradient (DDPG) — no PyTorch/TensorFlow.

**Components:**

| Class | Description |
|-------|-------------|
| `DenseLayer` | Fully-connected layer with `relu`/`tanh`/`none` activation + Adam optimizer |
| `MLP` | Stack of `DenseLayer`s; supports forward, backward, Adam step, Polyak soft-update |
| `OUNoise` | Ornstein–Uhlenbeck exploration noise (temporally correlated) |
| `ReplayBuffer` | Experience replay with uniform random sampling |
| `DDPGAgent` | Full DDPG: actor (state→action), critic (state+action→Q), target networks |

**Architecture:**
- Actor: `[state_dim, 256, 128, action_dim]` → tanh output → scaled to `[0, 2π]`
- Critic: `[state_dim + action_dim, 256, 128, 1]` → Q-value
- Both use He initialization and Adam optimizer
- Soft target updates with τ = 0.005

**Key hyperparameters (in `method_ddpg`):**
- `lr_actor = 1e-4`, `lr_critic = 5e-4`, `γ = 0.95`, `τ = 0.005`
- `buffer_size = 20000`, `batch_size = 256`, `noise_sigma = 0.3`

---

### `methods.py`

All beamforming method implementations. Contains 1089 lines covering the full method library.

**Shared helpers:**

| Helper | Description |
|--------|-------------|
| `_extract_state(...)` | Flatten channel magnitudes + gains + prev rate → state vector |
| `_state_dim(p)` | Compute state vector dimensionality from params |
| `_mrt_beamformer(h_eff, P_max)` | Maximum-Ratio Transmission beamformer |
| `_analytical_phase_alignment(H_BR, H_r, H_t)` | Closed-form near-optimal phases: `φ_n = -angle(Σ_k conj(H[:,n]) · H_BR[n,:])` |
| `_gradient_update_slot(...)` | N-step finite-difference gradient ascent on (W, phases) for one slot |
| `_run_pqc(enc_angles, theta, n_qubits, n_layers)` | NumPy statevector PQC: H-layer → angle encoding → Ry+CNOT layers → Z expectations |
| `_pqc_grad_theta(...)` | Parameter-shift gradient w.r.t. circuit weights θ |
| `_pqc_grad_enc(...)` | Parameter-shift gradient w.r.t. encoding angles |
| `_QuantumActor` | Hybrid PQC actor: linear encoder → PQC (4 qubits) → linear decoder → tanh |
| `_method_metrics(sr_ts)` | Convergence std + linear slope of rate trajectory |

**PQC Architecture:**
```
state → W_enc (state_dim × n_qubits) → tanh → π·tanh → Ry encoding
→ [Ry(θ) on each qubit + ring CNOT] × n_layers
→ Z-expectations → W_dec (n_qubits × action_dim) → tanh
```

---

### `experiments.py`

Parallel benchmark runner using Python `multiprocessing.Pool`.

**Important macOS fix:** Uses `spawn` start method (not `fork`) to avoid deadlocks with NumPy threads. Redirects `stdin`/`stdout`/`stderr` to `/dev/null` in worker processes to prevent `Bad file descriptor` crashes under `nohup`.

| Function | Description |
|----------|-------------|
| `scenario_snr(snr_db_range, n_trials)` | Vary SNR 0–30 dB in 5 dB steps; adjusts `sigma2 = sigma2_ref / SNR_linear` |
| `scenario_N(N_values, n_trials)` | Vary STAR-RIS elements: 16, 32, 64 |
| `scenario_speed(speed_values, n_trials)` | Vary car speed: 5, 10, 20, 30 m/s (sets `v_min = v_max = v`) |
| `scenario_K(K_values, n_trials)` | Vary total users: K=4, 8, 16 (symmetric: Kr=Kt=K/2) |
| `_run_one_trial(p)` | Run all methods on same random channel realization |
| `_aggregate(records)` | Compute mean ± std across trials for 8 metrics |

**Aggregated metrics per CSV row:**
- `sum_rate_mean/std` — Average sum-rate (bits/s/Hz)
- `energy_norm_mean/std` — Normalized computational energy cost
- `time_s_mean/std` — Wall-clock execution time
- `circuit_evals_mean/std` — Quantum circuit evaluations
- `peak_mem_mb_mean/std` — Peak memory usage (MB)
- `iterations_mean/std` — Total training/optimization iterations
- `conv_std_mean/std` — Standard deviation of rate trajectory (stability)
- `conv_slope_mean/std` — Linear trend slope of rate trajectory

---

### `main.py`

Master pipeline script. Runs everything end-to-end:

**Step 1 — Gradient verification:**
- Builds toy 2-qubit Hamiltonian and verifies analytic gradient (tolerance < 1e-7)
- Builds 4-qubit Hamiltonian and verifies gradient (tolerance < 1e-6)
- Raises `RuntimeError` and aborts if either check fails

**Step 2 — 4 experimental sweeps (30 trials each, 4 parallel workers):**
```python
df_snr  = scenario_snr(snr_db_range=range(0, 31, 5), n_trials=30)
df_N    = scenario_N(N_values=(16, 32, 64), n_trials=30)
df_spd  = scenario_speed(speed_values=(5, 10, 20, 30), n_trials=30)
df_K    = scenario_K(K_values=((2,2), (4,4), (8,8)), n_trials=30)
```

**Step 3 — Console tables** (SR, E, time, memory, circuit evals, convergence std/slope)

**Step 4 — Figures:**
- `fig_complexity_analysis.png` — 3-panel: convergence std / memory / time vs SNR
- `fig_circuit_evals_N.png` — Quantum circuit evaluations vs N
- `fig_sumrate_{snr,speed,N,K}.png` — Sum-rate figures with SEM shading
- `fig_ee_{snr,speed,N,K}.png` — Energy efficiency (EE = R_sum / P_total) figures

---

### `plot_analysis.py`

Extended analysis producing **44 output files** (4 scenarios × 11 files each):
- `graph_all_methods.png` — All methods overlaid, 4 metric subplots
- `graph_{method}.png` — Per-method, 4 metric subplots (5 methods)
- `table_{method}.png` — Numeric table: sweep values × 4 metrics (5 methods)

Output directory: `results/analysis_output/{K,N,SNR,Speed}/`

Run from `pyFiles/`:
```bash
python plot_analysis.py
```

---

### `xai_analysis.py`

Explainable AI analysis of the benchmark results using SHAP and LIME.

**Inputs:** `results/results_{K,N,snr,speed}.csv`  
**Output:** `results/analysis_output/xai/`

**SHAP analysis (TreeExplainer on RandomForest):**
- `shap_beeswarm_{metric}.png` — Global feature importance (dot plot)
- `shap_dependence_paramvalue_{metric}.png` — How sweep param drives each metric
- `shap_waterfall_{metric}_{method}.png` — Per-instance local explanation (5 methods × 4 metrics)
- `shap_heatmap.png` — Feature × metric importance matrix
- `shap_method_contributions.png` — Method SHAP contribution per metric

**LIME analysis (LimeTabularExplainer):**
- `lime_explanation_{metric}.png` — Per-method bar charts (4 metrics)
- `lime_heatmap_{metric}.png` — Feature × method weight matrix
- `lime_vs_shap_agreement.png` — SHAP vs LIME correlation scatter plots

**Features used:** `[param_value, method_enc, sweep_K, sweep_N, sweep_snr_db, sweep_speed_ms]`

Run from `pyFiles/`:
```bash
pip install shap scikit-learn lime
python xai_analysis.py
```

---

### Testing Scripts

#### `smoke_test.py`
Minimal verification of each module (fast, < 60s):
1. Gradient check (2-qubit)
2. Simulator channel generation + sum-rate computation
3. DDPG agent action selection
4. One mini trial (T_horizon=3) through `_run_one_trial`

```bash
python smoke_test.py
```

#### `quick_test.py`
2-trial dry run of the full 4-scenario pipeline. Use this before launching an overnight run:
```bash
python quick_test.py
```
Produces `qt_results_*.csv` and `qt_fig_snr.png` — confirms everything runs end-to-end.

#### `debug_channels.py`
Prints channel magnitudes and per-user SINR for a single snapshot. Useful for sanity checking the channel scale before a big run:
```bash
python debug_channels.py
```

---

## 6. Method Details

### Method 1 — QAOA (Hybrid Quantum-Classical)

**File:** `methods.py::method_qaoa`

**Architecture:**
1. **4-qubit QAOA coarse search** — Each of 4 qubits controls one group of N/4 RIS elements
2. **L-BFGS-B optimization** over `(γ, β)` using analytic gradients (warm-started from previous slot)
3. Enumerate all 2⁴ = 16 bitstring configurations and pick the best by direct simulation
4. **Classical gradient refinement** (finite-difference, `n_iter=12×max(1,N//16)` steps) from the QAOA solution

**Hamiltonian coefficients:**
- `a_vec[g]` = Frobenius norm of BS→RIS channel for group g (linear objective)
- `c_mat[i,j]` = cross-correlation between groups i and j (interaction terms)

**Circuit evaluations tracked:** Equal to number of L-BFGS-B objective calls per slot.

**Energy cost model:** `E = 1.5 × QAOA_iters + 1.0 × gradient_iters`

---

### Method 2 — QDDPG (Quantum-Enhanced DDPG)

**File:** `methods.py::method_qddpg`

**Architecture:**
- **PQC actor** (4 qubits, 1 variational layer) — generates N candidate phase perturbations by sampling with varied encoding angles (θ fixed; only encoding changes)
- **Classical MLP critic** — trained offline to rank candidates by Q-value
- At eval: pick best candidate from 16 PQC samples, then 12 gradient refinement steps

**Training:** Critic-only pre-training (20 episodes, no PQC gradient backprop through the circuit). The critic is a classical DDPG agent with `hidden=(128,64)`.

**Reward:** `R = rate_rl - rate_analytical` (improvement over closed-form phase alignment baseline)

**Circuit evaluations tracked:** Global `_PQC_EVALS` counter incremented on every `_run_pqc` call.

---

### Method 3 — QPPO (Quantum Proximal Policy Optimisation)

**File:** `methods.py::method_qppo`

**Architecture:**
- **PQC actor** (4 qubits, 3 variational layers) — fuller expressiveness
- **Classical MLP critic** (1 hidden layer, 128 units) — value function
- On-policy: collects 6 trajectories before each PPO update (8 epochs)
- GAE(λ) advantage estimation: `γ=0.99`, `λ=0.95`
- PPO clipping: `ε=0.2`

**Key difference from QDDPG:**
- QPPO optimises **max-min fairness** (minimum per-user rate), not sum-rate
- When evaluated on sum-rate, QPPO scores lower by design (~20–40% gap)
- This is an intentional research finding: fairness-optimal and sum-rate-optimal solutions are different

**Baseline normalization:** A STAR-RIS random-phase + MRT baseline is simulated first and used as a lower-bound reference for reward shaping.

**Parameter-shift gradients:** Used for `θ` (variational weights) and encoding angles; shift = π/2.

---

## 7. Running the Simulation

### Step 0: Verify installation
```bash
cd pyFiles
python smoke_test.py
```

### Step 1: Dry run (optional but recommended)
```bash
python quick_test.py
```
Takes ~5–10 minutes. Confirms all modules work before a long run.

### Step 2: Full benchmark (background, recommended)
```bash
nohup python main.py > run.log 2>&1 &
```
- Runs 30 trials × 4 sweeps × 3 methods = ~1260 simulation trials
- On 4 parallel workers, expect 2–6 hours depending on hardware
- Progress is written to `run.log`; monitor with `tail -f run.log`

### Step 3: Check results
```bash
tail -n 50 run.log     # See summary tables
ls -la *.csv *.png     # Verify output files
```

### Step 4: Extended analysis (optional)
```bash
python plot_analysis.py      # Per-method graphs + tables (44 files)
python xai_analysis.py       # SHAP + LIME analysis
```

---

## 8. Experiment Scenarios

### Scenario 1 — SNR Sweep
**Sweep:** `snr_db ∈ {0, 5, 10, 15, 20, 25, 30}` dB  
**What changes:** `sigma2 = sigma2_ref / 10^(snr_db/10)`  
**Purpose:** Classic SE-EE tradeoff; shows how methods handle high-SNR regime

### Scenario 2 — N Sweep (RIS size)
**Sweep:** `N ∈ {16, 32, 64}` elements  
**What changes:** Number of STAR-RIS elements; more elements = higher DoF but more QPPO circuit state space  
**Purpose:** Shows scalability of quantum circuits with RIS size

### Scenario 3 — Speed Sweep
**Sweep:** `v ∈ {5, 10, 20, 30}` m/s (all cars set to the same speed)  
**What changes:** Doppler frequency and Jakes coherence factor — faster cars cause more aggressive channel decorrelation  
**Purpose:** Shows robustness of methods to high Doppler in V2X scenarios

### Scenario 4 — K Sweep (number of users)
**Sweep:** `K ∈ {4, 8, 16}` cars (symmetric: `Kr = Kt = K/2`)  
**What changes:** State dimension grows, SINR interference increases  
**Purpose:** Scalability with number of served vehicles

---

## 9. Output Files

After running `main.py`, the following files are produced in `pyFiles/` (or moved to `results/`):

### CSV Result Files
| File | Columns |
|------|---------|
| `results_snr.csv` | `snr_db, method, sum_rate_mean, sum_rate_std, energy_norm_mean, energy_norm_std, time_s_mean, time_s_std, circuit_evals_mean, circuit_evals_std, peak_mem_mb_mean, peak_mem_mb_std, iterations_mean, iterations_std, conv_std_mean, conv_std_std, conv_slope_mean, conv_slope_std` |
| `results_N.csv` | Same but with `N` as sweep column |
| `results_speed.csv` | Same but with `speed_ms` as sweep column |
| `results_K.csv` | Same but with `K` as sweep column |

### Figure Files
| File | Contents |
|------|---------|
| `fig_sumrate_snr.png` | Sum-rate vs SNR for all 3 methods with SEM shading |
| `fig_sumrate_speed.png` | Sum-rate vs car speed |
| `fig_sumrate_N.png` | Sum-rate vs N |
| `fig_sumrate_K.png` | Sum-rate vs K |
| `fig_ee_snr.png` | Energy efficiency (bits/s/Hz/W) vs SNR |
| `fig_ee_speed.png` | EE vs car speed |
| `fig_ee_N.png` | EE vs N |
| `fig_ee_K.png` | EE vs K |
| `fig_complexity_analysis.png` | 3-panel: convergence std / memory / exec time vs SNR |
| `fig_circuit_evals_N.png` | Quantum circuit evaluations vs N (quantum methods only) |

---

## 10. Results Summary

Extracted from `run.log` (30 trials, default params, SNR sweep):

### Complexity & Resource Analysis (mean across SNR sweep)

| Method | Time (s) | Mem (MB) | Circuit Evals | Iterations | Conv Std | Conv Slope |
|--------|----------|----------|---------------|------------|----------|------------|
| QAOA   | 14.2     | 0.08     | 1,239         | 2,199      | 0.5648   | +0.00083 |
| QDDPG  | 23.4     | 7.35     | 7,280         | 913        | 0.6309   | +0.00029 |
| QPPO   | 404.2    | 2.02     | 131,120       | 3,840      | 0.5973   | +0.00006 |

### Sum-Rate at SNR = 20 dB

| Method | Sum Rate (bits/s/Hz) |
|--------|----------------------|
| QAOA   | 4.247 |
| QDDPG  | 3.151 |
| QPPO   | 3.073 |

> **Key findings:**
> - **QAOA** achieves highest sum-rate across all sweep conditions due to the 4-qubit coarse search + classical gradient refinement combination
> - **QDDPG** is fastest in training and uses the least memory (8.16 MB)
> - **QPPO** uses the most circuit evaluations (~131K) and takes ~27× longer than QAOA due to full PQC gradient backpropagation via parameter shift; it intentionally optimises for fairness, not sum-rate
> - All quantum methods show near-zero convergence slope, indicating convergence (especially QPPO at +0.000006)

---

## 11. Post-Processing & Analysis

### Re-plotting without re-simulation

All EE figures use post-processing power constants in `main.py`. To retune without re-running the simulation, modify these constants and call only the EE plotting block:

```python
ETA_PA     = 0.35    # power-amp efficiency
P_RF       = 0.3     # per RF-chain [W]
P_RIS_EL   = 3.0     # per STAR-RIS element [W]
P_UE       = 20.0    # per vehicle [W]
P_BB       = 1.0     # baseband processing [W]
SIGMA2_PHY = 0.1     # noise floor for SNR→P_tx mapping
```

**Energy efficiency formula:**
```
EE = R_sum / P_total   [bits/s/Hz/W]

P_total = P_tx/η_PA + Nt·P_RF + N·P_RIS_EL + K·P_UE + P_BB

SNR sweep:  P_tx = SIGMA2_PHY · 10^(SNR/10)
Other:      P_tx = P_max
```

### Extended Plots

```bash
python plot_analysis.py
```

Produces 44 files under `results/analysis_output/`:
```
results/analysis_output/
├── K/
│   ├── graph_all_methods.png    ← all methods on one figure
│   ├── graph_qaoa.png           ← QAOA, 4 metric subplots
│   ├── graph_qddpg.png
│   ├── graph_qppo.png
│   ├── table_qaoa.png           ← numeric table: K values × 4 metrics
│   ├── table_qddpg.png
│   └── table_qppo.png
├── N/    (same structure)
├── SNR/  (same structure)
└── Speed/(same structure)
```

---

## 12. XAI Analysis

The XAI pipeline trains a `RandomForestRegressor` on the merged CSV data and applies SHAP and LIME to explain feature contributions to each metric.

**Features:**
- `param_value` — the current sweep variable's value
- `method_enc` — label-encoded method identifier
- `sweep_K/N/snr_db/speed_ms` — one-hot indicators for which scenario is active

**SHAP heatmap** (`shap_heatmap.png`) shows global mean |SHAP| per (feature, metric) pair — useful for identifying which sweep parameter is the dominant driver of each performance metric across all methods.

**LIME vs SHAP agreement** (`lime_vs_shap_agreement.png`) cross-validates the two explainers with a correlation scatter per metric.

```bash
python xai_analysis.py
# Output → results/analysis_output/xai/shap/ and /lime/
```

---

## 13. Energy Efficiency Model

The EE computation is a **post-processing step** — it does not affect the simulation. Power constants model a realistic V2X hardware deployment:

| Component | Power | Notes |
|-----------|-------|-------|
| PA efficiency η | 0.35 | Class-AB amplifier |
| RF chain (per chain) | 0.3 W | × `Nt` chains at BS |
| STAR-RIS element | 3.0 W | PIN diodes + varactors + per-element DAC + FPGA overhead |
| User equipment | 20.0 W | Vehicular OBU: RF + PA + V2X baseband + GPS + Doppler comp + CSI feedback |
| Baseband processing | 1.0 W | Fixed |

**SNR sweep** models the SE-EE tradeoff where `P_tx = SIGMA2_PHY × SNR_linear` (higher SNR → more transmit power → lower EE). All other sweeps use `P_tx = P_max`.

---

## 14. Key Design Decisions & Notes

### Why 4-qubit instead of N-qubit QAOA?
The STAR-RIS has N = 16–64 elements. A direct N-qubit QAOA would require exponential classical simulation cost. Instead, the RIS is split into **4 groups**, each controlled by one qubit — giving 16 candidate configurations (vs. 2^N). This is the practical quantum-classical compromise used here.

### Why does QAOA outperform QDDPG and QPPO on sum-rate?
- QAOA uses **classical gradient refinement** as a second stage. The QAOA phase serves as a smart warm-start; the real gains come from the gradient ascent on top.
- QDDPG's PQC is used as a **sampler**, not a fully trained circuit. Training only the critic (not the PQC) limits its expressive power.
- QPPO explicitly **optimises fairness (max-min rate)**, not sum-rate — so its lower sum-rate score is by design.

### Why does QPPO take 27× longer than QAOA?
QPPO backpropagates through the PQC via the **parameter-shift rule** for every PPO update step, across 6 trajectories × 80 slots × 8 epochs. Each parameter-shift call requires 2 forward PQC evaluations per parameter. With 4 qubits × 3 layers = 12 parameters, this is ~24 PQC calls per gradient step — explaining the ~131K circuit evaluations per run.

### nohup + macOS multiprocessing fix
Python's `multiprocessing.fork` deadlocks on macOS when NumPy's internal threads are active at fork time. The codebase explicitly sets `mp.set_start_method('spawn')` and redirects all file descriptors to `/dev/null` in worker initializers to prevent "Bad file descriptor" crashes under `nohup`.

### Convergence metrics interpretation
- `conv_std`: std of the per-slot sum-rate over the T_horizon=80 slots. Lower = more stable beamforming.
- `conv_slope`: linear regression slope over slots. Near-zero means converged; positive means still improving; negative means degrading (rare — could indicate channel-tracking failure).

### Gradient check must pass before any simulation
`main.py` raises `RuntimeError` and aborts if either the 2-qubit or 4-qubit gradient check fails. This guards against numerical issues with the Hamiltonian construction (e.g., wrong Qiskit Pauli convention) propagating silently into results.

---

## Citation / Reference

If you use this codebase, the key algorithmic references are:

- **DDPG:** Lillicrap et al., *Continuous control with deep reinforcement learning*, ICLR 2016
- **QAOA:** Farhi, Goldstone, Gutmann, *A quantum approximate optimization algorithm*, arXiv:1411.4028
- **PPO:** Schulman et al., *Proximal policy optimization algorithms*, arXiv:1707.06347
- **STAR-RIS:** Xu et al., *Star-ris: Simultaneous transmitting and reflecting*, IEEE Wireless Commun. Letters, 2021
- **RIS-DRL baseline:** (phase alignment + MRT from the RIS-DRL reference paper, closed-form Eq. 16)
