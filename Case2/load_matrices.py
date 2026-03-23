"""
Utility functions for loading test matrices.
=============================================
Provides helpers to:
  1. Download matrices from the SuiteSparse Matrix Collection (Matrix Market format).
  2. Generate standard test problems (convection-diffusion, diagonal-dominant, etc.).
"""

import numpy as np
import scipy.sparse as sp
import os
import urllib.request
import tarfile
import tempfile
from scipy.io import mmread


# ---------------------------------------------------------------
# SuiteSparse Matrix Collection download
# ---------------------------------------------------------------
def load_suitesparse(group, name, cache_dir="suitesparse_cache"):
    """
    Download and load a matrix from the SuiteSparse Matrix Collection
    in Matrix Market format.

    Parameters
    ----------
    group : str
        Matrix group (e.g., 'HB', 'Bai').
    name : str
        Matrix name (e.g., 'nos4', 'orsirr_1').
    cache_dir : str
        Local directory to cache downloaded files.

    Returns
    -------
    A : scipy.sparse.csr_matrix
        The sparse matrix.
    """
    os.makedirs(cache_dir, exist_ok=True)
    mtx_path = os.path.join(cache_dir, f"{name}.mtx")

    if not os.path.exists(mtx_path):
        url = f"https://sparse.tamu.edu/MM/{group}/{name}.tar.gz"
        print(f"Downloading {group}/{name} from SuiteSparse ...")
        tar_path = os.path.join(cache_dir, f"{name}.tar.gz")
        urllib.request.urlretrieve(url, tar_path)

        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=cache_dir)

        # The .mtx file is usually inside a subdirectory named after the matrix
        extracted_mtx = os.path.join(cache_dir, name, f"{name}.mtx")
        if os.path.exists(extracted_mtx):
            os.rename(extracted_mtx, mtx_path)
        else:
            raise FileNotFoundError(
                f"Could not find {name}.mtx in the extracted archive."
            )

    A = mmread(mtx_path)
    return sp.csr_matrix(A)


# ---------------------------------------------------------------
# Programmatic test-problem generators
# ---------------------------------------------------------------
def make_convdiff_2d(n_side, epsilon=0.01):
    """
    2D convection-diffusion operator on a unit square:
        -epsilon * Laplacian(u) + du/dx = f
    discretised with central differences on an n_side x n_side grid.

    Parameters
    ----------
    n_side : int
        Number of interior grid points per side.
    epsilon : float
        Diffusion coefficient.

    Returns
    -------
    A : scipy.sparse.csr_matrix, shape (N, N) with N = n_side^2
    """
    N = n_side * n_side
    h = 1.0 / (n_side + 1)

    # 1D Laplacian
    e = np.ones(n_side)
    T = sp.diags([-e, 2 * e, -e], [-1, 0, 1], shape=(n_side, n_side))
    I = sp.eye(n_side)

    # 2D Laplacian: kron(I, T) + kron(T, I)
    Lap = sp.kron(I, T) + sp.kron(T, I)

    # 1D convection (central difference for du/dx)
    C1d = sp.diags([-e, e], [-1, 1], shape=(n_side, n_side))
    Conv = sp.kron(I, C1d)

    A = (epsilon / h**2) * Lap + (1.0 / (2.0 * h)) * Conv
    return sp.csr_matrix(A)


def make_diag_dominant(n, density=0.05, seed=42):
    """
    Random sparse matrix made diagonally dominant.

    Parameters
    ----------
    n : int
        Matrix size.
    density : float
        Approximate fraction of non-zeros in the off-diagonal part.
    seed : int
        Random seed.

    Returns
    -------
    A : scipy.sparse.csr_matrix, shape (n, n)
    """
    rng = np.random.default_rng(seed)
    B = sp.random(n, n, density=density, random_state=rng, format="csr")
    B = B + B.T  # symmetrise off-diagonal
    # Make strictly diagonally dominant
    diag_vals = np.array(np.abs(B).sum(axis=1)).ravel() + 1.0
    A = B + sp.diags(diag_vals)
    return sp.csr_matrix(A)


def make_shifted_laplacian_1d(n, shift=5.0):
    """
    1D Laplacian plus a shift: A = tridiag(-1, 2, -1) + shift * I.
    A simple SPD test matrix.

    Parameters
    ----------
    n : int
        Matrix size.
    shift : float
        Diagonal shift (larger shift => easier problem).

    Returns
    -------
    A : scipy.sparse.csr_matrix, shape (n, n)
    """
    e = np.ones(n)
    A = sp.diags([-e, (2 + shift) * e, -e], [-1, 0, 1], shape=(n, n))
    return sp.csr_matrix(A)
