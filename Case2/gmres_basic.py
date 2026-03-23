"""
GMRES Version 1: Basic Implementation
======================================
Solves Ax = b using GMRES with direct least-squares minimisation at each step.

At iteration m, we minimise:
    ||H_tilde_m * y - beta * e1||_2

where H_tilde_m is the (m+1) x m upper Hessenberg matrix from the Arnoldi process,
beta = ||r0||, and e1 is the first canonical basis vector.

References:
    - Saad, "Iterative Methods for Sparse Linear Systems", Chapter 6
    - Trefethen & Bau, "Numerical Linear Algebra", Lecture 33
"""

import numpy as np


def gmres_basic(A, b, x0=None, tol=1e-6, maxiter=None):
    """
    Basic GMRES: full least-squares solve at every iteration.

    Parameters
    ----------
    A : ndarray or sparse matrix, shape (n, n)
        The coefficient matrix.
    b : ndarray, shape (n,)
        The right-hand side vector.
    x0 : ndarray, shape (n,), optional
        Initial guess (default: zero vector).
    tol : float, optional
        Convergence tolerance on the residual norm (default: 1e-6).
    maxiter : int, optional
        Maximum number of iterations (default: n).

    Returns
    -------
    x : ndarray, shape (n,)
        The approximate solution.
    residual_norms : list of float
        Residual norm history (length maxiter+1, starting with ||r0||).
    num_iter : int
        Number of iterations performed.
    """
    n = len(b)
    if x0 is None:
        x0 = np.zeros(n)
    if maxiter is None:
        maxiter = n

    # Initial residual
    r0 = b - A @ x0
    beta = np.linalg.norm(r0)

    if beta < tol:
        return x0.copy(), [beta], 0

    # Allocate storage for Arnoldi basis and Hessenberg matrix
    V = np.zeros((n, maxiter + 1))       # orthonormal basis vectors
    H = np.zeros((maxiter + 1, maxiter))  # upper Hessenberg matrix

    V[:, 0] = r0 / beta
    residual_norms = [beta]

    for j in range(maxiter):
        # --- Arnoldi step ---
        w = A @ V[:, j]

        # Modified Gram-Schmidt orthogonalisation
        for i in range(j + 1):
            H[i, j] = np.dot(w, V[:, i])
            w = w - H[i, j] * V[:, i]

        H[j + 1, j] = np.linalg.norm(w)

        if H[j + 1, j] > 1e-14:
            V[:, j + 1] = w / H[j + 1, j]
        else:
            # Lucky breakdown: Krylov subspace is invariant
            V[:, j + 1] = 0.0

        # --- Solve the least-squares problem ---
        # min || H_tilde_m * y - beta * e1 ||
        e1 = np.zeros(j + 2)
        e1[0] = beta
        y, _, _, _ = np.linalg.lstsq(H[:j + 2, :j + 1], e1, rcond=None)

        # Compute residual norm from the least-squares residual
        res_norm = np.linalg.norm(H[:j + 2, :j + 1] @ y - e1)
        residual_norms.append(res_norm)

        if res_norm < tol:
            # Build and return the approximate solution
            x = x0 + V[:, :j + 1] @ y
            return x, residual_norms, j + 1

        if H[j + 1, j] < 1e-14:
            # Lucky breakdown — solution is exact in this subspace
            x = x0 + V[:, :j + 1] @ y
            return x, residual_norms, j + 1

    # Reached maxiter without convergence — return best approximation
    e1 = np.zeros(maxiter + 1)
    e1[0] = beta
    y, _, _, _ = np.linalg.lstsq(H[:maxiter + 1, :maxiter], e1, rcond=None)
    x = x0 + V[:, :maxiter] @ y

    return x, residual_norms, maxiter
