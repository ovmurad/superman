from typing import Any, Tuple

import numpy as np

from src.array import DenseArray
from src.array.linalg import SYM_EIGEN_SOLVERS, EigenSolver, eigen_decomp
from src.geometry.normalize import normalize
from src.geometry.matrix import (
    NON_SYM_LAPLACIAN_TYPES,
    SYM_LAPLACIAN_TYPES,
    AffinityMatrix,
)
from src.geometry.matrix.laplacian import LaplacianMatrix, eps_adjustment
from src.geometry.eigen_system import EigenSystem
from src.object import LaplacianType
from src.object.metadata import Metadata

_DIAG_ADD = 2.0


def laplacian_embedding(
    mat: AffinityMatrix | LaplacianMatrix,
    ncomp: int,
    lap_type: LaplacianType = "geometric",
    eigen_solver: EigenSolver = "dense",
    drop_first: bool = True,
    in_place: bool = False,
    **kwargs: Any,
) -> EigenSystem:
    """
    Compute a Laplacian embedding of a graph using either an affinity matrix
    or a Laplacian matrix.

    :param mat: The input matrix, either an AffinityMatrix or a LaplacianMatrix.
    :param ncomp: Number of components (eigenvectors) to compute.
    :param lap_type: Type of Laplacian to construct if `mat` is an AffinityMatrix. (default: "geometric").
    :param eigen_solver: Eigenvalue solver to use. (default: "dense").
    :param drop_first: Whether to drop the first eigenvector (typically trivial). (default: True)
    :param in_place: Whether to modify the input matrix in place. (default: False)
    :param kwargs: Additional keyword arguments passed to the eigen decomposition function.

    :return: A tuple of eigenvalues and eigenvectors.
    :raises ValueError: If the input matrix type is not recognized.
    """

    if isinstance(mat, AffinityMatrix):
        return _aff_laplacian_embedding(
            mat, ncomp, lap_type, eigen_solver, drop_first, in_place, **kwargs
        )
    elif isinstance(mat, LaplacianMatrix):
        return _lap_laplacian_embedding(mat, ncomp, eigen_solver, drop_first, **kwargs)
    raise ValueError(f"Matrix of type {type(mat)} not recognized!")


def _lap_laplacian_embedding(
    lap: LaplacianMatrix,
    ncomp: int,
    eigen_solver: EigenSolver = "dense",
    drop_first: bool = True,
    **kwargs: Any,
) -> EigenSystem:
    """
    Compute the Laplacian embedding given a precomputed Laplacian matrix.

    :param lap: A LaplacianMatrix object.
    :param ncomp: Number of components (eigenvectors) to compute.
    :param eigen_solver: Eigenvalue solver to use. (default: "dense").
    :param drop_first: Whether to drop the first eigenvector (typically trivial). (default: True).
    :param kwargs: Additional keyword arguments passed to the eigen decomposition function.

    :return: A tuple of eigenvalues and eigenvectors.
    """

    md: Metadata = lap.metadata

    eigvals, eigvecs = eigen_decomp(
        arr=lap.as_nparray() * -1.0 if lap.metadata.aff_minus_id else lap.as_nparray(),
        ncomp=ncomp + int(drop_first),
        eigen_solver=eigen_solver,
        is_symmetric=lap.metadata.lap_type in SYM_LAPLACIAN_TYPES,
        largest=False,
        **kwargs,
    )

    if drop_first:
        eigvals = eigvals[1:]
        eigvecs = eigvecs[:, 1:]

    return EigenSystem((eigvals, eigvecs), metadata=md)


def _aff_laplacian_embedding(
    aff: AffinityMatrix,
    ncomp: int,
    lap_type: LaplacianType = "geometric",
    eigen_solver: EigenSolver = "amg",
    drop_first: bool = True,
    in_place: bool = False,
    **kwargs: Any,
) -> EigenSystem:
    """
    Compute the Laplacian embedding from an affinity matrix.

    :param aff: AffinityMatrix representing pairwise similarities.
    :param ncomp: Number of components (eigenvectors) to compute.
    :param lap_type: Type of Laplacian to construct. (default: "geometric").
    :param eigen_solver: Eigenvalue solver to use. (default: "dense").
    :param drop_first: Whether to drop the first eigenvector (typically trivial). (default: True).
    :param in_place: Whether to modify the affinity matrix in place. (default: False).
    :param kwargs: Additional keyword arguments passed to the eigen decomposition function.
    :return: A tuple of eigenvalues and eigenvectors.
    """

    md: Metadata = aff.metadata

    degrees = None

    if eigen_solver in SYM_EIGEN_SOLVERS and lap_type in NON_SYM_LAPLACIAN_TYPES:

        if lap_type == "geometric":
            aff = normalize(aff, sym_norm=True, in_place=in_place)
            in_place = True

        degrees = aff.sum(axis=1, keepdims=True)

        lap_type = "symmetric"

    lap = aff.laplacian(
        lap_type=lap_type,
        diag_add=_DIAG_ADD + 1.0,
        aff_minus_id=False,
        in_place=in_place,
    )

    eigvals, eigvecs = eigen_decomp(
        arr=lap.as_nparray(),
        ncomp=ncomp + int(drop_first),
        eigen_solver=eigen_solver,
        is_symmetric=lap_type in SYM_LAPLACIAN_TYPES,
        largest=False,
        **kwargs,
    )

    eigvals -= (
        _DIAG_ADD
        if aff.metadata.eps is None
        else (_DIAG_ADD * eps_adjustment(aff.metadata.eps))
    )

    if degrees is not None:
        eigvecs /= degrees.sqrt()
        eigvecs /= np.linalg.norm(eigvecs, axis=0)

    if drop_first:
        eigvals = eigvals[1:]
        eigvecs = eigvecs[:, 1:]

    return EigenSystem((DenseArray(eigvals), DenseArray(eigvecs)), metadata=md)
