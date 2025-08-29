from typing import Tuple

import numpy as np

from src.array import DenseArray
from src.array.linalg import SYM_EIGEN_SOLVERS, EigenSolver, eigen_decomp
from src.geometry import normalize
from src.geometry.matrix import (
    NON_SYM_LAPLACIAN_TYPES,
    SYM_LAPLACIAN_TYPES,
    AffinityMatrix,
)
from src.geometry.matrix.laplacian import eps_adjustment
from src.object import LaplacianType

_DIAG_ADD = 2.0


def laplacian_embedding(
    aff: AffinityMatrix,
    ncomp: int,
    lap_type: LaplacianType = "geometric",
    eigen_solver: EigenSolver = "amg",
    drop_first: bool = True,
    check_connected: bool = True,
    in_place: bool = False,
    **kwargs,
) -> Tuple[DenseArray, DenseArray]:
    degrees = None

    if eigen_solver in SYM_EIGEN_SOLVERS and lap_type in NON_SYM_LAPLACIAN_TYPES:

        if lap_type == "geometric":
            aff = normalize(aff, sym_norm=True, in_place=in_place)
            in_place = True

        degrees = aff.sum(axis=1, keepdims=True)

        lap_type: LaplacianType = "symmetric"

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
        eigvecs /= np.sqrt(degrees)
        eigvecs /= np.linalg.norm(eigvecs, axis=0)

    if drop_first:
        eigvals = eigvals[1:]
        eigvecs = eigvecs[:, 1:]

    return eigvals, eigvecs
