from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from src.array.base import BaseArray
from src.array.dense.dense import DenseArray
from src.array.linalg.normalize import normalize
from src.array.sparse.csr import CsrArray
from src.geometry.matrix.adjacency import AdjacencyMatrix
from src.geometry.matrix.laplacian import LaplacianMatrix, eps_adjustment
from src.object.metadata import LaplacianType

from ...object.geometry_matrix import (
    GeometryMatrixMixin,
)


class AffinityMatrixMixin(GeometryMatrixMixin, ABC):
    def __new__(cls, *args, **kwargs):
        if cls is AffinityMatrix:
            if 'shape' in kwargs:
                return CsrAffinityMatrix(*args, **kwargs)
            return DenseAffinityMatrix(*args, **kwargs)
        return super().__new__(cls)


class AffinityMatrix(AffinityMatrixMixin, BaseArray, ABC):
    def laplacian(
        aff: AffinityMatrix,
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

        eps = aff.metadata.eps
        if eps is None:
            raise ValueError("Affinity matrix does not have a bandwidth `eps`!")

        match lap_type:
            case "geometric":
                aff = normalize(aff, sym_norm=True, in_place=in_place)
                lap: LaplacianMatrix = LaplacianMatrix(normalize(
                    aff, sym_norm=False, in_place=True), lap_type = "geometric")
            case "random_walk":
                lap: LaplacianMatrix =  LaplacianMatrix(normalize(
                    aff, sym_norm=False, in_place=in_place
                ), lap_type = "random_walk")
            case "symmetric":
                lap: LaplacianMatrix = LaplacianMatrix(normalize(
                    aff, sym_norm=True, in_place=in_place, degree_exp=0.5
                ), lap_type = "symmetric")
            case _:
                raise ValueError(f"Unknown laplacian type: {lap_type}!")

        if not aff_minus_id:
            lap *= -1.0
        else:
            diag_add *= -1.0

        lap.fill_diagonal(lap.diagonal() + diag_add)

        lap *= eps_adjustment(eps)

        return lap


class DenseAffinityMatrix(AffinityMatrix, DenseArray):
    def _execute_adjacency(self, copy: bool) -> AdjacencyMatrix:
        return self != 0


class CsrAffinityMatrix(AffinityMatrix, CsrArray):
    def _execute_adjacency(self, copy: bool) -> AdjacencyMatrix:
        raise NotImplementedError()
