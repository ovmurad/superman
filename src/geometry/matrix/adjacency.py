from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np
from src.array.base import BaseArray
from src.array.dense.dense import DenseArray
from src.array.sparse.csr import CsrArray
from src.object.geometry_matrix import GeometryMatrixMixin


class AdjacencyMatrixMixin(GeometryMatrixMixin, ABC):
    fixed_dtype = np.bool_

    def __new__(cls, *args, **kwargs):
        if cls is AdjacencyMatrix:
            if 'shape' in kwargs:
                return CsrAdjacencyMatrix(*args, **kwargs)
            return DenseAdjacencyMatrix(*args, **kwargs)
        return super().__new__(cls)


class AdjacencyMatrix(AdjacencyMatrixMixin, ABC):
    pass


class DenseAdjacencyMatrix(DenseArray, AdjacencyMatrix):
    pass


class CsrAdjacencyMatrix(CsrArray, AdjacencyMatrix):
    pass