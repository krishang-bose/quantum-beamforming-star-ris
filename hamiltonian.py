"""
hamiltonian.py — 2-element STAR-RIS toy Hamiltonian

Binary variable encoding:
    x_n = (1 - Z_n) / 2      so  Z_n = 1 - 2*x_n

Cost function (minimise negative sum-rate proxy):
    C(x0, x1) = -(a*(1-x0) + b*(1-x1) + c*x0*x1)

Expanding with  x_n = (I - Z_n)/2:

    x0      = (I - Z0)/2
    x1      = (I - Z1)/2
    1 - x0  = (I + Z0)/2
    1 - x1  = (I + Z1)/2
    x0*x1   = (I-Z0)(I-Z1)/4
             = (II - IZ - ZI + ZZ)/4

So:
    C = -[ a*(I+Z0)/2  +  b*(I+Z1)/2  +  c*(II-IZ-ZI+ZZ)/4 ]

Collecting by Pauli string (Qiskit: rightmost char = qubit 0):
    II  :  -(a/2 + b/2 + c/4)
    IZ  :  -(a/2 - c/4)          qubit-0 Z
    ZI  :  -(b/2 - c/4)          qubit-1 Z
    ZZ  :  -(c/4)
"""

import numpy as np
import scipy.linalg as la
from qiskit.quantum_info import SparsePauliOp, Statevector


# ── Pauli matrices (2×2) ──────────────────────────────────────────────────────

_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)

# 2-qubit basis matrices (4×4)
_II = np.kron(_I, _I)
_IZ = np.kron(_I, _Z)   # qubit-0 Z  (Qiskit convention: rightmost = q0)
_ZI = np.kron(_Z, _I)   # qubit-1 Z
_ZZ = np.kron(_Z, _Z)

# Mixer: B = X_0 + X_1
_IX = np.kron(_I, _X)
_XI = np.kron(_X, _I)
_B  = _IX + _XI          # 4×4 mixer matrix


# ═══════════════════════════════════════════════════════════════════════════════
# Pauli decomposition
# ═══════════════════════════════════════════════════════════════════════════════

def pauli_coefficients(a: float, b: float, c: float) -> dict:
    """
    Return the four Pauli coefficients as a dict.
    These are the weights in C = h_II*II + h_IZ*IZ + h_ZI*ZI + h_ZZ*ZZ
    """
    return {
        'II': -(a / 2 + b / 2 + c / 4),
        'IZ': -(a / 2 - c / 4),
        'ZI': -(b / 2 - c / 4),
        'ZZ': -(c / 4),
    }


def build_pauli_hamiltonian(a: float, b: float, c: float) -> SparsePauliOp:
    """
    Return the 2-qubit cost Hamiltonian as a Qiskit SparsePauliOp.
    Qiskit convention: rightmost character in string = qubit 0.
    """
    coeffs = pauli_coefficients(a, b, c)
    return SparsePauliOp.from_list([
        ('II', coeffs['II']),
        ('IZ', coeffs['IZ']),
        ('ZI', coeffs['ZI']),
        ('ZZ', coeffs['ZZ']),
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# Matrix-form Hamiltonian & QAOA expectation
# ═══════════════════════════════════════════════════════════════════════════════

def _hamiltonian_matrix(a: float, b: float, c: float) -> np.ndarray:
    """4×4 Hamiltonian matrix in the computational basis."""
    coeffs = pauli_coefficients(a, b, c)
    return (coeffs['II'] * _II +
            coeffs['IZ'] * _IZ +
            coeffs['ZI'] * _ZI +
            coeffs['ZZ'] * _ZZ)


def _evolve_and_measure(gamma: float, beta: float,
                        H: np.ndarray, B: np.ndarray) -> float:
    """
    Apply depth-1 QAOA to |++> and return <C>.

    |psi> = exp(-i beta B) exp(-i gamma H) |++>
    <C>   = <psi| H |psi>
    """
    sv0 = Statevector.from_label('++').data          # shape (4,)
    U_C = la.expm(-1j * gamma * H)
    U_B = la.expm(-1j * beta  * B)
    psi = U_B @ U_C @ sv0
    return float(np.real(psi.conj() @ H @ psi))


def expectation_value(gamma: float, beta: float,
                      a: float, b: float, c: float) -> float:
    """<C> for depth-1 QAOA from |++>."""
    H = _hamiltonian_matrix(a, b, c)
    return _evolve_and_measure(gamma, beta, H, _B)


# ═══════════════════════════════════════════════════════════════════════════════
# Analytic gradient
# ═══════════════════════════════════════════════════════════════════════════════

def analytic_gradient(gamma: float, beta: float,
                      a: float, b: float, c: float):
    """
    Exact gradient of <C>(gamma, beta) via analytic statevector
    differentiation.  No finite differences, no parameter shifts.

    d<C>/d_gamma:
        |psi>          = U_B @ U_C @ |sv0>
        d|psi>/d_gamma = U_B @ (-i H) @ U_C @ |sv0>
                       = -i * U_B @ H @ U_C @ |sv0>
        d<C>/d_gamma   = 2 * Re[ <psi| H | d_psi/d_gamma > ]

    Implemented directly as matrix-vector products:

        phi      = U_C @ sv0
        psi      = U_B @ phi
        d_psi_dg = -1j * U_B @ H @ phi        (= U_B (-iH) U_C sv0)
        d_psi_db = -1j * B @ psi              (= (-iB) U_B U_C sv0)

        d<C>/d_gamma = 2 Re[ (H @ psi).conj().T @ d_psi_dg ]
        d<C>/d_beta  = 2 Re[ (H @ psi).conj().T @ d_psi_db ]

    Returns: (d_gamma, d_beta)
    """
    H   = _hamiltonian_matrix(a, b, c)
    B   = _B
    sv0 = Statevector.from_label('++').data

    U_C = la.expm(-1j * gamma * H)
    U_B = la.expm(-1j * beta  * B)

    phi      = U_C @ sv0                  # after cost layer
    psi      = U_B @ phi                  # final state

    d_psi_dg = -1j * (U_B @ (H @ phi))   # d|psi>/d_gamma
    d_psi_db = -1j * (B  @ psi)           # d|psi>/d_beta

    H_psi    = H @ psi                    # H|psi>

    d_gamma = 2.0 * float(np.real(H_psi.conj() @ d_psi_dg))
    d_beta  = 2.0 * float(np.real(H_psi.conj() @ d_psi_db))

    return d_gamma, d_beta


# ═══════════════════════════════════════════════════════════════════════════════
# Verification
# ═══════════════════════════════════════════════════════════════════════════════

def verify_gradient(gamma: float, beta: float,
                    a: float, b: float, c: float,
                    eps: float = 1e-5):
    """
    Compare analytic gradient against central-difference FD.
    Passes when max absolute error < 1e-7.
    """
    ag, ab = analytic_gradient(gamma, beta, a, b, c)

    fd_g = (expectation_value(gamma + eps, beta, a, b, c)
          - expectation_value(gamma - eps, beta, a, b, c)) / (2 * eps)
    fd_b = (expectation_value(gamma, beta + eps, a, b, c)
          - expectation_value(gamma, beta - eps, a, b, c)) / (2 * eps)

    err_g = abs(ag - fd_g)
    err_b = abs(ab - fd_b)

    print(f"  Gradient check:")
    print(f"    d/d_gamma  analytic={ag:.8f}   FD={fd_g:.8f}   err={err_g:.2e}")
    print(f"    d/d_beta   analytic={ab:.8f}   FD={fd_b:.8f}   err={err_b:.2e}")

    passed = err_g < 1e-7 and err_b < 1e-7
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed
