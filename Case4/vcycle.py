"""
Multigrid V-cycle building blocks
=================================
- restriction(n_fine):       2D full-weighting restriction matrix
- weighted_jacobi(...):      damped Jacobi smoother (omega = 2/3)
- setup_hierarchy(n_fine):   build A and R matrices on every level
- vcycle(...):               recursive V-cycle solver

Restriction (1D full-weighting):  r_c(I) = 1/4 r_f(2I) + 1/2 r_f(2I+1) + 1/4 r_f(2I+2)
2D restriction:                   R = R_1d kron R_1d
Prolongation:                     P = 4 * R^T  (bilinear interpolation)
"""

import numpy as np
from scipy.sparse import csr_matrix, kron
from scipy.sparse.linalg import spsolve

from poisson2d import Poisson2D


def restriction(n_fine):
    """
    2D full-weighting restriction matrix from an n_fine x n_fine fine grid
    to an n_coarse x n_coarse coarse grid, where n_coarse = (n_fine - 1) // 2.

    Returns a sparse matrix of shape (n_coarse**2, n_fine**2).
    """
    n_coarse = (n_fine - 1) // 2

    # 1D full-weighting: row I has nonzeros at fine cols 2I, 2I+1, 2I+2
    rows, cols, data = [], [], []
    for I in range(n_coarse):
        rows += [I, I, I]
        cols += [2*I, 2*I + 1, 2*I + 2]
        data += [0.25, 0.5, 0.25]
    R1 = csr_matrix((data, (rows, cols)), shape=(n_coarse, n_fine))

    return kron(R1, R1, format='csr')


def weighted_jacobi(A, f, u, omega=2/3, nu=2):
    """
    nu sweeps of weighted Jacobi:
        u <- u + omega * D^{-1} (f - A u),   D = diag(A).
    """
    Dinv = 1.0 / A.diagonal()
    for _ in range(nu):
        u = u + omega * Dinv * (f - A @ u)
    return u


def setup_hierarchy(n_fine):
    """
    Build A and R matrices on every grid level, finest first.

    Levels: n_fine = 2^p - 1  ->  2^(p-1) - 1  ->  ...  ->  3  ->  1.
    The coarsest level (n=1) holds a 1x1 system that is solved directly.

    Returns
    -------
    A_levels : list of sparse matrices    (length = p)
    R_levels : list of sparse matrices    (length = p - 1)
               R_levels[i] maps level i -> level i+1
    """
    A_levels, R_levels = [], []
    n = n_fine
    while True:
        A_levels.append(Poisson2D(n))
        if n == 1:
            break
        R_levels.append(restriction(n))
        n = (n - 1) // 2
    return A_levels, R_levels


def vcycle(A_levels, R_levels, f, level=0, nu1=2, nu2=2, u=None):
    """
    Recursive multigrid V-cycle.

    Parameters
    ----------
    A_levels : list of sparse matrices
        Hierarchy of system matrices (finest first).
    R_levels : list of sparse matrices
        Hierarchy of restriction operators.
    f : ndarray
        Right-hand side at the current level.
    level : int
        Index of the current grid level (0 = finest).
    nu1, nu2 : int
        Number of pre- and post-smoothing sweeps.
    u : ndarray or None
        Initial guess on this level; defaults to zero.

    Returns
    -------
    u : ndarray
        Updated approximate solution at this level.
    """
    if u is None:
        u = np.zeros_like(f)

    # Coarsest level: solve directly
    if level == len(A_levels) - 1:
        return spsolve(A_levels[level], f)

    # 1. Pre-smoothing
    u = weighted_jacobi(A_levels[level], f, u, omega=2/3, nu=nu1)

    # 2. Compute residual
    r = f - A_levels[level] @ u

    # 3. Restrict residual to coarse grid
    rc = R_levels[level] @ r

    # 4. Solve coarse-grid error equation recursively (zero initial guess)
    ec = vcycle(A_levels, R_levels, rc, level + 1, nu1, nu2)

    # 5. Prolongate (P = 4 * R^T) and correct
    u = u + 4.0 * (R_levels[level].T @ ec)

    # 6. Post-smoothing
    u = weighted_jacobi(A_levels[level], f, u, omega=2/3, nu=nu2)

    return u
