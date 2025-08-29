import warnings
from typing import Any, Final, Literal, Optional, Set, Tuple, TypeAlias

import numpy as np
from numpy.linalg import eig, eigh
from pyamg import smoothed_aggregation_solver
from scipy.sparse.linalg import eigs, eigsh, lobpcg

from src.array import DenseArray

EigenSolver: TypeAlias = Literal["dense", "arpack", "lobpcg", "amg"]
SYM_EIGEN_SOLVERS: Final[Set[EigenSolver]] = {"amg", "lobpcg"}
NON_SYM_EIGEN_SOLVERS: Final[Set[EigenSolver]] = {"dense", "arpack"}

_AMG_KW = {"strength", "aggregate", "smooth", "max_levels", "max_coarse"}
_ZERO_EIGVAL_TOL = 1e-9


def eigen_decomp(
    arr: DenseArray,
    ncomp: Optional[int] = None,
    eigen_solver: EigenSolver = "dense",
    is_symmetric: bool = False,
    largest: bool = True,
    **kwargs: Any,
) -> Tuple[DenseArray, DenseArray]:
    """
    Compute the eigenvalue decomposition of a matrix using various solvers.

    :param arr: Array of input points. Can be sparse or dense.
    :param ncomp: Optional number of eigenvalues and eigenvectors to return. If None, all are returned. (default: None)
    :param eigen_solver: Which solver to use. One of {"dense", "arpack", "amg", "lobpcg"}. (default: "dense")
    :param is_symmetric: Whether the matrix is symmetric/Hermitian (uses optimized solvers if True). (default: False)
    :param largest: If True, return the largest eigenvalues. Otherwise, return the smallest. (default: True)
    :param kwargs: Additional keyword arguments passed to the underlying solver functions.

    :return: Tuple of eigenvalues and eigenvectors.
    """

    match eigen_solver:
        case "dense":
            if not isinstance(arr, np.ndarray):
                raise ValueError("'dense' eigen solver requires dense matrices!")

            eig_func = eigh if is_symmetric else eig
            eigvals, eigvecs = eig_func(arr, **kwargs)

            if largest:  # eigh always returns eigenvalues in ascending order
                eigvals = eigvals[..., ::-1]  # reverse order the e-values
                eigvecs = eigvecs[..., ::-1]  # reverse order the vectors

        case "arpack":

            if is_symmetric:
                which = "LM" if largest else "SM"
                eig_func = eigsh
            else:
                which = "LR" if largest else "SR"
                eig_func = eigs

            eigvals, eigvecs = eig_func(arr, k=ncomp, which=which, **kwargs)

        case "amg" | "lobpcg":

            M = None

            if eigen_solver == "amg":

                if isinstance(arr, np.ndarray):
                    warnings.warn("AMG works better for sparse matrices")

                # separate amg kwargs
                amg_kwargs = {kw: kwargs.pop(kw) for kw in _AMG_KW if kw in kwargs}

                # Use AMG to get a preconditioner and speed up the eigenvalue problem.
                M = smoothed_aggregation_solver(arr, **amg_kwargs).aspreconditioner()

            n_find = arr.shape[-1] if ncomp is None else ncomp
            n_find = min(arr.shape[-2], 5 + 2 * n_find)

            X = np.random.rand(arr.shape[-2], n_find)
            eigvals, eigvecs = lobpcg(arr, X, M=M, largest=largest, **kwargs)

        case _:
            raise ValueError(f"Unknown eigen solver {eigen_solver}!")

    eigvals = np.real(eigvals)[..., :ncomp]
    if is_symmetric:
        eigvals[np.abs(eigvals) < _ZERO_EIGVAL_TOL] = 0.0

    return eigvals, np.real(eigvecs)[..., :ncomp]
