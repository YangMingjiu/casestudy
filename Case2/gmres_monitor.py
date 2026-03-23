"""
GMRES Version 3: Residual Norm Monitoring Without Building the Solution
========================================================================
This version further optimises the Givens-rotation GMRES by deferring
construction of the approximate solution x_m.

Key idea:
  - During the Arnoldi + Givens loop, we ONLY monitor the residual norm
    via |g[j+1]|. We do NOT form the solution vector x_m at any point
    inside the loop.
  - Only after the loop terminates (either by convergence or maxiter)
    do we perform the back-substitution and compute x_m = x0 + V_m * y.

This avoids the cost of back-substitution and the matrix-vector product
V_m * y at every iteration — those are deferred to a single post-loop step.

References:
    - Saad, "Iterative Methods for Sparse Linear Systems", Algorithm 6.11
    - Trefethen & Bau, "Numerical Linear Algebra", Lecture 33
"""

import numpy as np


def _givens_rotation(a, b):
    """
    Compute Givens rotation parameters (c, s) such that:
        | c  s | | a |   | r |
        |-s  c | | b | = | 0 |
    """
    if abs(b) < 1e-15:
        return 1.0, 0.0
    elif abs(a) < 1e-15:
        return 0.0, np.sign(b)
    else:
        r = np.sqrt(a**2 + b**2)
        return a / r, b / r


def gmres_monitor(A, b, x0=None, tol=1e-6, maxiter=None):
    """
    GMRES with deferred solution construction.

    The main loop only performs the Arnoldi process and progressive Givens
    rotations to monitor the residual norm |g[m+1]|. The approximate
    solution x_m is built ONLY ONCE after the loop terminates.

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

    # Pre-allocate storage
    V = np.zeros((n, maxiter + 1))
    H = np.zeros((maxiter + 1, maxiter))
    cs = np.zeros(maxiter)
    sn = np.zeros(maxiter)
    g = np.zeros(maxiter + 1)

    V[:, 0] = r0 / beta
    g[0] = beta

    residual_norms = [beta]
    m = maxiter  # will be updated if convergence occurs

    # =========================================================
    # Main loop: Arnoldi + Givens rotations ONLY
    # The solution x_m is NOT built inside this loop.
    # =========================================================
    for j in range(maxiter):
        # --- Arnoldi step ---
        w = A @ V[:, j]
        for i in range(j + 1):
            H[i, j] = np.dot(w, V[:, i])
            w = w - H[i, j] * V[:, i]
        H[j + 1, j] = np.linalg.norm(w)

        breakdown = H[j + 1, j] < 1e-14  # check before Givens zeroes it

        if not breakdown:
            V[:, j + 1] = w / H[j + 1, j]

        # --- Apply all previous Givens rotations to new column ---
        for i in range(j):
            temp = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
            H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j]
            H[i, j] = temp

        # --- New Givens rotation ---
        cs[j], sn[j] = _givens_rotation(H[j, j], H[j + 1, j])

        # Apply to H (zero out sub-diagonal)
        H[j, j] = cs[j] * H[j, j] + sn[j] * H[j + 1, j]
        H[j + 1, j] = 0.0

        # Apply to right-hand side
        temp = cs[j] * g[j] + sn[j] * g[j + 1]
        g[j + 1] = -sn[j] * g[j] + cs[j] * g[j + 1]
        g[j] = temp

        # --- Monitor residual norm (NO solution construction here) ---
        res_norm = abs(g[j + 1])
        residual_norms.append(res_norm)

        # Check convergence or breakdown
        if res_norm < tol:
            m = j + 1
            break

        if breakdown:
            # Lucky breakdown
            m = j + 1
            break

    # =========================================================
    # Post-loop: build the solution ONLY ONCE
    # Back-substitution on the upper-triangular system R * y = g
    # =========================================================
    y = np.zeros(m)
    for i in range(m - 1, -1, -1):
        y[i] = g[i]
        for k in range(i + 1, m):
            y[i] -= H[i, k] * y[k]
        y[i] /= H[i, i]

    x = x0 + V[:, :m] @ y

    return x, residual_norms, m
