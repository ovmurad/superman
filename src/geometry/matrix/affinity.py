from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from src.array.base import BaseArray
from src.array.dense.dense import DenseArray
from src.array.sparse.csr import CsrArray
from src.geometry.matrix.adjacency import AdjacencyMatrix

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


class AffinityMatrix(BaseArray, AffinityMatrixMixin, ABC):
    def adjacency(
        self,
        copy: bool = False,
    ) -> AdjacencyMatrix:

        return self._execute_adjacency(copy)

    @abstractmethod
    def _execute_adjacency(self, copy: bool) -> AdjacencyMatrix:
        pass


class DenseAffinityMatrix(DenseArray, AffinityMatrix):
    def _execute_adjacency(self, copy: bool) -> AdjacencyMatrix:
        return AdjacencyMatrix(self != 0)


class CsrAffinityMatrix(CsrArray, AffinityMatrix):
    def _execute_adjacency(self, copy: bool) -> AdjacencyMatrix:
        raise NotImplementedError()
