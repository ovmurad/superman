from typing import Any, Tuple

import numpy as np

from src.array import DenseArray
from src.array.linalg import SYM_EIGEN_SOLVERS, EigenSolver, eigen_decomp
from src.geometry import normalize
from src.geometry.matrix import (
    NON_SYM_LAPLACIAN_TYPES,
    SYM_LAPLACIAN_TYPES,
    AffinityMatrix,
)
from src.geometry.matrix.laplacian import LaplacianMatrix, eps_adjustment
from src.object import LaplacianType

_DIAG_ADD = 2.0


def laplacian_embedding(
    mat: AffinityMatrix | LaplacianMatrix,
    ncomp: int,
    lap_type: LaplacianType = "geometric",
    eigen_solver: EigenSolver = "amg",
    drop_first: bool = True,
    in_place: bool = False,
    **kwargs: Any,
) -> Tuple[DenseArray, DenseArray]:
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
    eigen_solver: EigenSolver = "amg",
    drop_first: bool = True,
    **kwargs: Any,
) -> Tuple[DenseArray, DenseArray]:
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

    return eigvals, eigvecs


def _aff_laplacian_embedding(
    aff: AffinityMatrix,
    ncomp: int,
    lap_type: LaplacianType = "geometric",
    eigen_solver: EigenSolver = "amg",
    drop_first: bool = True,
    in_place: bool = False,
    **kwargs: Any,
) -> Tuple[DenseArray, DenseArray]:
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

    return eigvals, eigvecs
