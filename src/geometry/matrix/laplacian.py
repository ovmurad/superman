from __future__ import annotations
from typing import Optional

import numpy as np

from ...object.geometry_matrix import LaplacianType

from abc import ABC, abstractmethod

import numpy as np
from src.array.base import BaseArray
from src.array.dense.dense import DenseArray
from src.array.sparse.csr import CsrArray
from src.object.geometry_matrix import GeometryMatrixMixin


class LaplacianMatrixMixin(GeometryMatrixMixin, ABC):
    fixed_dtype = np.bool_

    def __new__(cls, *args, **kwargs):
        if cls is LaplacianMatrix:
            if 'shape' in kwargs:
                return CsrLaplacianMatrix(*args, **kwargs)
            return DenseLaplacianMatrix(*args, **kwargs)
        return super().__new__(cls)


class LaplacianMatrix(LaplacianMatrixMixin, BaseArray, ABC):
    pass


class DenseLaplacianMatrix(LaplacianMatrix, DenseArray):
    pass


class CsrLaplacianMatrix(LaplacianMatrix, CsrArray):
    pass


def eps_adjustment(eps: float) -> float:
    return 4.0 / (eps**2)


def _prepare_arr_degrees_and_out(
    arr: BaseArray[np.float64],
    axis: int,
    degree_exp: float,
    keepdims: bool,
) -> BaseArray[np.float64]:

    degrees = arr.sum(axis=axis, keepdims=keepdims) ** degree_exp

    return arr, degrees


def _normalize_de(
    arr: BaseArray[np.float64],
    axis: Optional[int] = 1,
    degree_exp: float = 1.0,
    sym_norm: bool = False,
    in_place: bool = False,
) -> BaseArray[np.float64]:

    degrees = arr.sum(axis=axis, keepdims=True) ** degree_exp

    if in_place:
        arr /= degrees
    else:
        arr = arr / degrees

    if sym_norm and axis is not None:
        arr.__itruediv__()

    return arr


# TODO replace with proper method
def _normalize_arr(
    arr: MatrixArray[np.float64],
    axis: Optional[int] = 1,
    degree_exp: float = 1.0,
    sym_norm: bool = False,
    in_place: bool = False,
) -> MatrixArray[np.float64]:
    if arr.is_sparse:
        raise NotImplementedError()
    return _normalize_de(arr, axis, degree_exp, sym_norm, in_place)


def laplacian(
    aff_mat: AffinityMatrix,
    lap_type: LaplacianType = "geometric",
    diag_add: float = 1.0,
    aff_minus_id: bool = True,
    in_place: bool = False,
) -> LaplacianMatrix:
    """
    Corresponds to laplacian.py in cryo_experiments. Note that eps will be store in aff_mat.metadata. If eps
    is None, raise Error, so can ignore the branches in the previous code where eps is None. I don't think aff_minus_id
    is ever set to False, so ignore that branch as well.
    Papers:
        - geometric and symmetric: https://www.sciencedirect.com/science/article/pii/S1063520306000546
        - random walk: https://www2.imm.dtu.dk/projects/manifold/Papers/Laplacian.pdf

    :param aff_mat:
    :param lap_type:
    :return:
    """

    eps = aff_mat.metadata.eps
    if eps is None:
        raise ValueError("Affinity matrix does not have a bandwidth `eps`!")

    aff = aff_mat.data

    match lap_type:
        case "geometric":
            aff = _normalize_arr(aff, sym_norm=True, in_place=in_place)
            lap_arr: MatrixArray[np.float64] = _normalize_arr(
                aff, sym_norm=False, in_place=True
            )
        case "random_walk":
            lap_arr: MatrixArray[np.float64] = _normalize_arr(
                aff, sym_norm=False, in_place=in_place
            )
        case "symmetric":
            lap_arr: MatrixArray[np.float64] = _normalize_arr(
                aff, sym_norm=True, in_place=in_place, degree_exp=0.5
            )
        case _:
            raise ValueError(f"Unknown laplacian type: {lap_type}!")

    if not aff_minus_id:
        lap_arr *= -1.0
    else:
        diag_add *= -1.0

    np.fill_diagonal(lap_arr, np.diag(lap_arr) + diag_add)

    lap_arr *= eps_adjustment(eps)

    return LaplacianMatrix(
        lap_arr,
        aff_mat.metadata.dist_type,
        aff_mat.metadata.aff_type,
        lap_type,
        aff_mat.metadata.radius,
        eps,
        aff_mat.metadata.name,
    )
