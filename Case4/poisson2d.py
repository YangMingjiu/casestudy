"""
Poisson2D(n)
============
Sparse 5-point finite-difference discretisation of the negative 2D Laplacian
on a uniform n x n interior grid in (0,1)^2 with homogeneous Dirichlet BCs.

Constructed via Kronecker product:
    A = (1/h^2) * (kron(I_n, T) + kron(T, I_n)),  T = tridiag(-1, 2, -1).
"""

import numpy as np
from scipy.sparse import diags, eye, kron


def Poisson2D(n):
    """
    Build the n^2 x n^2 sparse matrix A for -Delta u = f on (0,1)^2
    with homogeneous Dirichlet BCs and n interior points per direction.

    Parameters
    ----------
    n : int
        Number of interior points per direction (typically n = 2**p - 1).

    Returns
    -------
    A : scipy.sparse.csr_matrix
        Sparse n^2 x n^2 matrix.
    """
    h = 1.0 / (n + 1)
    e = np.ones(n)
    T = diags([-e[1:], 2*e, -e[1:]],
              offsets=[-1, 0, 1], shape=(n, n), format='csr')
    I = eye(n, format='csr')
    A = (kron(I, T, format='csr') + kron(T, I, format='csr')) / h**2
    return A.tocsr()
