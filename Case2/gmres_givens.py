"""
GMRES Version 2: Progressive Givens Rotation QR Factorisation
==============================================================
Instead of solving the full least-squares problem from scratch at each step,
we maintain a progressive QR factorisation of H_tilde_m using Givens rotations.

At step j:
  1. Apply all previous Givens rotations (c_i, s_i) for i = 0, ..., j-1
     to the new column h_j of H_tilde.
  2. Compute a new Givens rotation (c_j, s_j) to zero out H[j+1, j].
  3. Apply this rotation to the right-hand side vector g.
  4. The residual norm is |g[j+1]|.
  5. Solve the upper-triangular system R_m * y = g[:m] via back-substitution
     and build x_m = x0 + V_m * y.

References:
    - Saad, "Iterative Methods for Sparse Linear Systems", Algorithm 6.10
    - Trefethen & Bau, "Numerical Linear Algebra", Lecture 33
"""

import numpy as np


def _givens_rotation(a, b):
    """
    Compute Givens rotation parameters (c, s) such that:
        | c  s | | a |   | r |
        |-s  c | | b | = | 0 |
    where r = sqrt(a^2 + b^2).
    """
    if abs(b) < 1e-15:
        return 1.0, 0.0
    elif abs(a) < 1e-15:
        return 0.0, np.sign(b)
    else:
        r = np.sqrt(a**2 + b**2)
        return a / r, b / r


def gmres_givens(A, b, x0=None, tol=1e-6, maxiter=None):
    """
    GMRES with progressive Givens rotation QR factorisation.

    At each iteration, the QR factorisation of H_tilde is updated incrementally,
    and the approximate solution x_m is built to verify convergence.

    Parameters
    ----------
    A : ndarray or sparse matrix, shape (n, n)
    b : ndarray, shape (n,)
    x0 : ndarray, shape (n,), optional
    tol : float, optional
    maxiter : int, optional

    Returns
    -------
    x : ndarray, shape (n,)
    residual_norms : list of float
    num_iter : int
    """
    n = len(b)
    if x0 is None:
        x0 = np.zeros(n)
    if maxiter is None:
        maxiter = n

    r0 = b - A @ x0
    beta = np.linalg.norm(r0)

    if beta < tol:
        return x0.copy(), [beta], 0

    V = np.zeros((n, maxiter + 1))
    H = np.zeros((maxiter + 1, maxiter))

    V[:, 0] = r0 / beta

    # Storage for Givens rotation parameters
    cs = np.zeros(maxiter)
    sn = np.zeros(maxiter)

    # Right-hand side vector (progressively transformed)
    g = np.zeros(maxiter + 1)
    g[0] = beta

    residual_norms = [beta]

    for j in range(maxiter):
        # --- Arnoldi step ---
        w = A @ V[:, j]
        for i in range(j + 1):
            H[i, j] = np.dot(w, V[:, i])
            w = w - H[i, j] * V[:, i]
        H[j + 1, j] = np.linalg.norm(w)

        breakdown = H[j + 1, j] < 1e-14  # save before Givens modifies H

        if not breakdown:
            V[:, j + 1] = w / H[j + 1, j]

        # --- Apply previous Givens rotations to column j ---
        for i in range(j):
            temp = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
            H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j]
            H[i, j] = temp

        # --- Compute new Givens rotation to eliminate H[j+1, j] ---
        cs[j], sn[j] = _givens_rotation(H[j, j], H[j + 1, j])

        # Apply to H
        H[j, j] = cs[j] * H[j, j] + sn[j] * H[j + 1, j]
        H[j + 1, j] = 0.0

        # Apply to right-hand side
        temp = cs[j] * g[j] + sn[j] * g[j + 1]
        g[j + 1] = -sn[j] * g[j] + cs[j] * g[j + 1]
        g[j] = temp

        # Residual norm from Givens process
        res_norm = abs(g[j + 1])
        residual_norms.append(res_norm)

        if res_norm < tol or breakdown:
            # Build approximate solution via back-substitution
            m = j + 1
            y = np.zeros(m)
            for i in range(m - 1, -1, -1):
                y[i] = (g[i] - H[i, i + 1:m] @ y[i + 1:m]) / H[i, i]
            x = x0 + V[:, :m] @ y
            return x, residual_norms, m

    # Reached maxiter — build solution
    m = maxiter
    y = np.zeros(m)
    for i in range(m - 1, -1, -1):
        y[i] = (g[i] - H[i, i + 1:m] @ y[i + 1:m]) / H[i, i]
    x = x0 + V[:, :m] @ y

    return x, residual_norms, m
