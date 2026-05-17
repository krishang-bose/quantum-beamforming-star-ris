"""
hamiltonian.py — STAR-RIS QAOA Hamiltonian (2-qubit and 4-qubit)

Binary variable encoding:
    x_n = (1 - Z_n) / 2      so  Z_n = 1 - 2*x_n

2-qubit cost function (minimise negative sum-rate proxy):
    C(x0, x1) = -(a*(1-x0) + b*(1-x1) + c*x0*x1)

4-qubit cost function (split RIS into 4 groups):
    C(x0,x1,x2,x3) = -(a0*(1-x0) + a1*(1-x1) + a2*(1-x2) + a3*(1-x3)
                       + sum c_ij * x_i * x_j)

The 4-qubit version provides much finer-grained control over the
RIS phase configuration, improving optimisation especially with many users.
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
# 2-qubit Pauli decomposition (original)
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
# 4-qubit Hamiltonian
# ═══════════════════════════════════════════════════════════════════════════════

def build_4qubit_hamiltonian(a_vec, c_mat):
    """
    Build a 4-qubit cost Hamiltonian for QAOA.

    Cost function:
        C(x) = -[ sum_i a_i * (1-x_i) + sum_{i<j} c_ij * x_i * x_j ]

    Using  x_i = (I - Z_i)/2  and  (1-x_i) = (I + Z_i)/2:

        C = -[ sum_i a_i * (I+Z_i)/2
             + sum_{i<j} c_ij * (I-Z_i)(I-Z_j)/4 ]

    Expanding:
        const term (IIII coeff): -(sum a_i)/2 - (sum c_ij)/4
        Z_i coeff:               -(a_i/2) + (sum_j c_ij)/4
        Z_i Z_j coeff:           -(c_ij/4)

    Args:
        a_vec: (4,) — linear coefficients for each qubit group
        c_mat: (4,4) — interaction coefficients (upper triangular used)

    Returns:
        SparsePauliOp — 4-qubit Hamiltonian
        np.ndarray    — 16×16 matrix form
        np.ndarray    — 16×16 mixer matrix
    """
    n = 4
    a = np.asarray(a_vec, dtype=float)
    c = np.asarray(c_mat, dtype=float)

    # Pauli labels for Z on qubit i: 'IIZI' etc (Qiskit: rightmost = q0)
    # So Z on qubit i is at position i from the right
    terms = []

    # Constant (IIII) term
    h0 = -np.sum(a) / 2.0
    for i in range(n):
        for j in range(i + 1, n):
            h0 -= c[i, j] / 4.0
    terms.append(('IIII', float(h0)))

    # Single-Z terms
    for i in range(n):
        hi = -a[i] / 2.0
        for j in range(n):
            if j != i:
                hi += c[min(i, j), max(i, j)] / 4.0
        label = ['I'] * n
        label[i] = 'Z'
        # Qiskit convention: rightmost = q0, so label[0] is rightmost
        pauli_str = ''.join(reversed(label))
        terms.append((pauli_str, float(hi)))

    # Two-body ZZ terms
    for i in range(n):
        for j in range(i + 1, n):
            hij = -c[i, j] / 4.0
            if abs(hij) > 1e-15:
                label = ['I'] * n
                label[i] = 'Z'
                label[j] = 'Z'
                pauli_str = ''.join(reversed(label))
                terms.append((pauli_str, float(hij)))

    H_op = SparsePauliOp.from_list(terms)

    # Build 16×16 matrix
    H_mat = H_op.to_matrix().toarray() if hasattr(H_op.to_matrix(), 'toarray') \
        else np.array(H_op.to_matrix())

    # 4-qubit mixer: B = X_0 + X_1 + X_2 + X_3
    B_mat = np.zeros((2**n, 2**n), dtype=complex)
    for q in range(n):
        label = ['I'] * n
        label[q] = 'X'
        pauli_str = ''.join(reversed(label))
        x_op = SparsePauliOp.from_list([(pauli_str, 1.0)])
        x_mat = x_op.to_matrix()
        if hasattr(x_mat, 'toarray'):
            x_mat = x_mat.toarray()
        B_mat += np.array(x_mat)

    return H_op, np.array(H_mat, dtype=complex), np.array(B_mat, dtype=complex)


def expectation_4qubit(gamma, beta, H_mat, B_mat):
    """
    Depth-1 QAOA expectation value for 4-qubit Hamiltonian.

    |psi> = exp(-i beta B) exp(-i gamma H) |++++>
    <C>   = <psi| H |psi>
    """
    n = 4
    # |++++> state
    sv0 = np.ones(2**n, dtype=complex) / (2**(n/2))

    U_C = la.expm(-1j * gamma * H_mat)
    U_B = la.expm(-1j * beta * B_mat)

    psi = U_B @ U_C @ sv0
    return float(np.real(psi.conj() @ H_mat @ psi))


def gradient_4qubit(gamma, beta, H_mat, B_mat):
    """
    Analytic gradient of <C>(gamma, beta) for 4-qubit QAOA.
    Same derivation as 2-qubit version.

    Returns: (d_gamma, d_beta)
    """
    n = 4
    sv0 = np.ones(2**n, dtype=complex) / (2**(n/2))

    U_C = la.expm(-1j * gamma * H_mat)
    U_B = la.expm(-1j * beta * B_mat)

    phi = U_C @ sv0
    psi = U_B @ phi

    d_psi_dg = -1j * (U_B @ (H_mat @ phi))
    d_psi_db = -1j * (B_mat @ psi)

    H_psi = H_mat @ psi

    d_gamma = 2.0 * float(np.real(H_psi.conj() @ d_psi_dg))
    d_beta  = 2.0 * float(np.real(H_psi.conj() @ d_psi_db))

    return d_gamma, d_beta


# ═══════════════════════════════════════════════════════════════════════════════
# Matrix-form Hamiltonian & QAOA expectation (2-qubit, original)
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
# Analytic gradient (2-qubit)
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


def verify_4qubit_gradient(gamma: float, beta: float,
                           a_vec, c_mat, eps: float = 1e-5):
    """
    Verify 4-qubit gradient with finite differences.
    """
    _, H_mat, B_mat = build_4qubit_hamiltonian(a_vec, c_mat)
    ag, ab = gradient_4qubit(gamma, beta, H_mat, B_mat)

    fd_g = (expectation_4qubit(gamma + eps, beta, H_mat, B_mat)
          - expectation_4qubit(gamma - eps, beta, H_mat, B_mat)) / (2 * eps)
    fd_b = (expectation_4qubit(gamma, beta + eps, H_mat, B_mat)
          - expectation_4qubit(gamma, beta - eps, H_mat, B_mat)) / (2 * eps)

    err_g = abs(ag - fd_g)
    err_b = abs(ab - fd_b)

    print(f"  4-qubit gradient check:")
    print(f"    d/d_gamma  analytic={ag:.8f}   FD={fd_g:.8f}   err={err_g:.2e}")
    print(f"    d/d_beta   analytic={ab:.8f}   FD={fd_b:.8f}   err={err_b:.2e}")

    passed = err_g < 1e-6 and err_b < 1e-6
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed
